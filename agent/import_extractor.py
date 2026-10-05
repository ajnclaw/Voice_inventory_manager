# import_extractor.py
#
# Turns an uploaded file (plain text/CSV, PDF, or a photo of a
# handwritten note) into a list of candidate inventory actions, WITHOUT
# writing anything to inventory.db yet -- see agent_server.py's
# POST /upload (extract only) vs POST /import/confirm (actually apply).
# A bad OCR/handwriting read should never silently become a wrong
# number in the real ledger; the owner reviews the extracted list in
# the web UI first.
#
# Handwriting/photos go straight to the same multimodal LLM already
# powering chat (gpt-4o-mini understands images natively) -- no
# separate OCR engine. PDFs go through PyMuPDF: if the PDF has a real
# text layer, extract it directly (cheap, reliable); if it's a scan
# with no text layer, render its pages to images and fall through to
# the same image-reading path as a photo.

import base64
import json

from .config import DEFAULT_MODEL
from .llm_client import chat
from . import inventory_db

# If the owner tells the UI upfront whether an upload is a purchase
# bill or a sale bill, that's strictly better information than anything
# the model could infer from the document itself (a buy bill and a
# sale bill can look structurally identical -- same item/qty/rate/
# amount table -- the only real signal is who the shop is in the
# transaction, which is exactly the kind of thing OCR/vision can get
# wrong on a real photo). When given, this overrides the model's own
# judgment entirely rather than just being a suggestion.
HINT_INSTRUCTIONS = {
    "buy": (
        "The owner has confirmed this document is a PURCHASE/supplier "
        "bill -- stock coming IN to the shop. Treat EVERY item on it "
        "as action='restock', regardless of any other wording or "
        "layout in the document."
    ),
    "sale": (
        "The owner has confirmed this document is a SALE bill/invoice "
        "to a customer -- stock going OUT of the shop. Treat EVERY "
        "item on it as action='sale', regardless of any other wording "
        "or layout in the document."
    ),
}

EXTRACT_SYSTEM_PROMPT = """
You are reading a shop owner's note, file, or photo to find inventory
actions to extract -- sales, restocks, corrections, or new items.

Return ONLY valid JSON, no markdown, no explanation outside the JSON,
in exactly this shape:

{
  "items": [
    {
      "item": "<item name as written/best read>",
      "action": "restock" | "sale" | "adjustment" | "new_item",
      "quantity": <number>,
      "unit_price": <number or null>,
      "reason": "<required for action=adjustment, otherwise null>"
    }
  ],
  "summary": "<one short plain-text sentence describing what you found, for the owner to read -- no markdown>"
}

Rules:
- "restock" = stock coming IN (received, bought, delivery, purchase).
  "quantity" is how many came in.
- "sale" = stock going OUT to a customer. "quantity" is how many were
  sold.
- "adjustment" = correcting a count that doesn't match reality (found
  damaged, miscounted, lost). You do NOT know the current stock count,
  so "quantity" here must be the CHANGE, not a final total: negative
  if items are missing/damaged/lost (e.g. "found 1 damaged" ->
  quantity: -1), positive if extra were found (e.g. "found 2 extra
  unlabeled" -> quantity: 2). ALWAYS include a "reason" for these.
- "new_item" = this is clearly a brand-new product being added to the
  catalog with a starting quantity, not a transaction against an
  existing one.
- A sale bill / invoice / receipt (a header like "Invoice"/"Bill", a
  customer name, an itemized list with quantity and rate/amount
  columns, a total) means EVERY line item on it is a "sale" -- the
  shop sold these to that customer, stock is going OUT. Don't skip
  bill line items just because they're in a table/column layout rather
  than a sentence -- read quantity from the Qty column and unit_price
  from the Rate column (not the Amount/line-total column, which is
  quantity times rate, not the per-unit price). Ignore the "Total"
  row itself -- it is not a separate item.
- If the text/image is in Hinglish or Hindi, understand it the same
  way as English -- extract the item name as written, don't translate
  it into a different language.
- If you genuinely can't make out an item or a quantity, skip that
  line rather than guessing a number -- it's better to miss an item
  (the owner notices it's absent and re-enters it) than to invent a
  wrong quantity that silently corrupts the real stock count.
- If nothing usable is found, return {"items": [], "summary": "..."}
  explaining why (e.g. "the photo was too blurry to read").
"""


def _call_extraction(content, logger=None, hint=None):
    system_content = EXTRACT_SYSTEM_PROMPT

    hint_instruction = HINT_INSTRUCTIONS.get(hint)

    if hint_instruction:
        system_content += "\n\nIMPORTANT, overrides anything above:\n" + hint_instruction

    response = chat(
        "import_extractor",
        DEFAULT_MODEL,
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": content},
        ],
        logger=logger,
    )

    raw = response["message"]["content"] or "{}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "items": [],
            "summary": "Couldn't make sense of that file -- try a clearer photo or a plain text list.",
        }

    data.setdefault("items", [])
    data.setdefault("summary", "")

    return data


def extract_from_text(text, logger=None, hint=None):
    prompt = "Extract inventory actions from this text:\n\n" + text
    return _call_extraction(prompt, logger=logger, hint=hint)


def extract_from_image(image_bytes, mime_type="image/jpeg", logger=None, hint=None):
    b64 = base64.b64encode(image_bytes).decode("ascii")

    content = [
        {
            "type": "text",
            "text": "Extract inventory actions from this photo/note. It may be handwritten.",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
        },
    ]

    return _call_extraction(content, logger=logger, hint=hint)


def extract_from_pdf(pdf_bytes, logger=None, hint=None):
    import pymupdf as fitz  # lazy import so every other code path in
    # this project (the normal chat flow) doesn't pay the import cost
    # or need it installed just to run. `pymupdf` is the current import
    # name -- `fitz` still works but is deprecated.

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        text_parts = [page.get_text() for page in doc]
        full_text = "\n".join(text_parts).strip()

        # A real text-layer PDF: cheap, reliable, no vision call needed.
        if len(full_text) > 20:
            return extract_from_text(full_text, logger=logger, hint=hint)

        # No usable text layer (a scanned/photographed PDF) -- fall
        # back to reading pages as images instead, same path as a
        # photo upload. Capped to avoid a large multi-page document
        # turning into a huge, slow multi-image vision call.
        merged = {"items": [], "summary": ""}
        max_pages = 3

        for page in doc[:max_pages]:
            pixmap = page.get_pixmap(dpi=150)
            image_bytes = pixmap.tobytes("png")

            result = extract_from_image(
                image_bytes, mime_type="image/png", logger=logger, hint=hint
            )
            merged["items"].extend(result.get("items", []))

            if result.get("summary"):
                merged["summary"] += (
                    " " if merged["summary"] else ""
                ) + result["summary"]

        if not merged["summary"]:
            merged["summary"] = "Scanned PDF with no extractable text or readable content."

        return merged
    finally:
        doc.close()


def apply_import_items(items, tool_manager):
    """
    Actually calls the write tools for a confirmed list of extracted
    items (see extract_from_*). Runs only AFTER the owner has reviewed
    and confirmed the preview in the web UI -- this is the only place
    in the whole import flow that touches real inventory data. Each
    item is applied independently (one bad item doesn't block the
    rest), same per-item success/failure reporting as a normal tool
    call.
    """
    results = []

    for entry in items:
        item_name = entry.get("item", "")
        action = entry.get("action")
        quantity = entry.get("quantity")
        unit_price = entry.get("unit_price")
        reason = entry.get("reason")

        # Best-effort name resolution: if there's exactly one existing
        # catalog match, use its exact stored name rather than
        # whatever phrasing the extraction produced -- same
        # resolve-before-act principle as the normal chat flow's
        # search_items-first instruction, just without an LLM in the
        # loop here to call that tool itself. Also captures the item's
        # real current quantity, needed below for "adjustment".
        current_quantity = None

        if action != "new_item" and item_name:
            matches = inventory_db.search_items(item_name)

            if len(matches) == 1:
                item_name = matches[0]["name"]
                current_quantity = matches[0]["current_quantity"]

        not_found_message = (
            f"'{item_name}' isn't in your catalog yet. Add it first "
            f"(e.g. say \"add {item_name}, starting stock X\" in chat), "
            f"then re-import this line."
        )

        try:
            if action == "restock":
                # Unlike a sale or adjustment, a restock against an
                # unknown item is completely ordinary -- "received your
                # first-ever shipment of something new" -- there's no
                # existing stock level it needs to reconcile against,
                # so auto-create it rather than failing. This is exactly
                # the situation bootstrapping a catalog from real bills
                # hits constantly.
                if current_quantity is None:
                    tool_manager.execute(
                        "add_item",
                        {"name": item_name, "initial_quantity": 0},
                    )

                tool_result = tool_manager.execute(
                    "record_restock",
                    {
                        "item": item_name,
                        "quantity": quantity,
                        "unit_cost": unit_price,
                        "note": "Bulk import",
                    },
                )
            elif action == "sale":
                if current_quantity is None:
                    # A sale against an item that was never added can't
                    # be applied safely -- there's no stock on record to
                    # sell from, and guessing a starting quantity would
                    # be inventing data, not reading it.
                    tool_result = {"success": False, "error": not_found_message}
                else:
                    tool_result = tool_manager.execute(
                        "record_sale",
                        {
                            "item": item_name,
                            "quantity": quantity,
                            "unit_price": unit_price,
                            "note": "Bulk import",
                        },
                    )
            elif action == "adjustment":
                if current_quantity is None:
                    # Can't safely turn a delta ("found 1 damaged") into
                    # an absolute new count without knowing where the
                    # item actually started -- name didn't resolve to
                    # exactly one existing item, so refuse rather than
                    # guess at the real stock number.
                    tool_result = {"success": False, "error": not_found_message}
                else:
                    tool_result = tool_manager.execute(
                        "adjust_stock",
                        {
                            "item": item_name,
                            "new_quantity": current_quantity + quantity,
                            "reason": reason or "Bulk import correction",
                        },
                    )
            elif action == "new_item":
                tool_result = tool_manager.execute(
                    "add_item",
                    {
                        "name": item_name,
                        "initial_quantity": quantity,
                        "cost_price": unit_price,
                    },
                )
            else:
                tool_result = {
                    "success": False,
                    "error": f"Unknown action: {action}",
                }
        except Exception as exc:
            tool_result = {"success": False, "error": str(exc)}

        results.append(
            {
                "item": item_name,
                "action": action,
                "success": tool_result.get("success", False),
                "output": tool_result.get("output"),
                "error": tool_result.get("error"),
            }
        )

    return results
