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
    "catalog": (
        "The owner has confirmed this document is a SUPPLIER PRICE "
        "LIST / CATALOG, not a record of an actual transaction -- it "
        "lists products a supplier can provide, not stock that has "
        "been bought or sold yet. Treat EVERY line as action='new_item' "
        "(even if some items might already exist in the catalog -- "
        "duplicates get caught and reported individually when applied, "
        "that's fine). Use the dealer/buy price column as unit_price. "
        "For quantity, use whatever minimum-order-quantity-style column "
        "is present (often labeled MOQ, Min Qty, Pack Size, or similar) "
        "as a provisional starting stock count -- the owner knows this "
        "isn't the real on-hand quantity and will correct each one "
        "later; it's a deliberate placeholder, not a guess you're "
        "making up. If no such column exists at all, use 0 rather than "
        "inventing a number. A real catalog row has a price -- if "
        "something has no price at all, check whether it's a product-"
        "family/section heading (e.g. the engine or machine model this "
        "whole page of parts belongs to) -- if so, that's the "
        "\"category\" for the items under it, not an item itself (see "
        "the category rule below). Otherwise it's letterhead; leave it "
        "out entirely."
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
      "category": "<see category rule below, or null>",
      "line_number": "<the Sr No / serial number / row code printed next to this row, exactly as written (e.g. \"1\", \"23\", \"EA07\"), or null if the document has no such numbering>",
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
- Documents have letterhead that is NOT a line item and carries no
  useful information -- company name, address, phone/mobile numbers,
  dates, signatures, "Total"/"Subtotal" rows. Skip all of it entirely;
  never emit it as an item.
- A SECTION or PRODUCT-FAMILY heading is different from letterhead --
  it IS useful, just not as its own item. Text like "170F 7.5 HP Power
  Weeder Engine Spare" or "Brake Parts" sitting above/among a group of
  items describes what THOSE items are (which engine/machine/product
  line they belong to) -- capture it as the "category" field on every
  item it applies to, rather than discarding it or emitting it as a
  fake priceless item of its own. If a new heading appears partway
  through the document, it applies to the items that follow it, not
  the ones before it. If there's no such heading anywhere, leave
  "category" null -- don't invent one.
- Either way: a real line item essentially always has a price next to
  it. Text with no price, no quantity, and no column values around it
  is either letterhead (skip it) or a category heading (capture it on
  the surrounding items' "category" field) -- never emit it as a
  standalone item with a null price.
- If a row has a Sr No / serial number / item code printed next to it
  (a numbered or lettered-and-numbered column at the start of the
  row, e.g. "1", "23", "EA07"), capture it ONLY in "line_number",
  exactly as written -- never as part of "item" too. It's a row
  position in this document, not part of the product's actual name;
  "item" should read naturally on its own ("Piston Ring Set 63 CC
  /68 CC", not "EA07 Piston Ring Set 63 CC /68 CC"). line_number is
  the exact identifier a separate, independent pass over the
  document's pictures uses to match each product photo to the right
  row -- getting it right (or leaving it null when there genuinely
  isn't one) matters more than it might look, so don't guess or
  invent a number that isn't actually printed there.
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

    # Backstop for the "line_number isn't part of the name" rule --
    # the prompt instruction alone proved unreliable in testing (same
    # pattern as every other backstop in this file): confirmed real
    # case, items came back as "EA01 Recoil Stater ..." with
    # line_number ALSO correctly set to "EA01", i.e. the code ended up
    # in both places despite being told not to duplicate it. Strip it
    # from the front of the name in code whenever it's there, rather
    # than trusting the model never to repeat it.
    for item in data["items"]:
        line_number = (item.get("line_number") or "").strip()
        name = (item.get("item") or "").strip()

        if line_number and name.lower().startswith(line_number.lower()):
            remainder = name[len(line_number):].lstrip(" -.:)")

            if remainder:
                item["item"] = remainder

    if hint == "catalog":
        # Backstop, not just a prompt instruction (which alone proved
        # unreliable in testing -- a real supplier PDF's footer/
        # letterhead text, e.g. a company name sitting right next to
        # the item rows, got emitted as a fake item with no price
        # despite being explicitly told not to). A real catalog row
        # always has a price; anything without one is near-certainly
        # misread boilerplate, not a product -- drop it in code rather
        # than hoping the model never slips.
        before = len(data["items"])
        data["items"] = [
            item for item in data["items"] if item.get("unit_price") is not None
        ]
        dropped = before - len(data["items"])

        if dropped:
            data["summary"] += (
                f" ({dropped} line(s) with no price were skipped as likely "
                f"not real products.)"
            )

        # Backstop for a confirmed real case: a document's FIRST
        # category heading sometimes doesn't apply to the handful of
        # item rows immediately above it in the model's view, even
        # though they sit under that same heading on the actual page
        # (the raw text-extraction order a multi-column/boxed-heading
        # PDF layout produces doesn't always match visual top-to-bottom
        # reading order, so the heading text can come after those rows
        # in what the model reads). The prompt rule alone ("applies to
        # items that follow it") is correct for every heading that
        # appears MID-document, which is the common case this is
        # deliberately not touching -- it only back-fills a leading run
        # of uncategorized items up to the document's very first
        # category, since those are the only ones a text-order glitch
        # like this could explain.
        first_category = next(
            (item.get("category") for item in data["items"] if item.get("category")),
            None,
        )

        if first_category:
            for item in data["items"]:
                if item.get("category"):
                    break
                item["category"] = first_category

    data["items"] = _flag_name_conflicts(data["items"])
    data["items"] = _flag_existing_matches(data["items"])

    return data


def _flag_name_conflicts(items):
    """
    Marks every item whose name collides with another item's in this
    same batch -- add_item enforces unique names, so confirming two
    same-named items as-is would silently fail the second one at
    apply time no matter what (even with identical prices: same name
    + same everything is still a UNIQUE constraint violation, not a
    harmless no-op). Flagging them here lets the web UI ask the owner
    to tell them apart before applying anything, rather than after a
    failed write.

    Each flagged item also gets a "suggested_name" the owner can
    accept as-is or edit, built from whatever actually differs within
    the group -- price (the strong real-world signal confirmed on an
    actual supplier catalog: two genuinely different parts,
    distinguished only by a picture the text extraction can't see,
    both just called "170F FLYWHEEL FAN" at different prices), line
    number, or category, in that order. If nothing distinguishes them
    at all (a true duplicate, most likely the same line read twice),
    falls back to a plain ordinal so the suggestion is at least
    unique -- the owner can still edit it to something better.
    """
    by_name = {}

    for index, item in enumerate(items):
        name = (item.get("item") or "").strip().lower()
        by_name.setdefault(name, []).append(index)

    for indices in by_name.values():
        if len(indices) < 2:
            continue

        for position, i in enumerate(indices):
            items[i]["name_conflict"] = True
            items[i]["suggested_name"] = _suggest_distinct_name(
                items, indices, i, position
            )

    return items


def _suggest_distinct_name(items, indices, i, position):
    item = items[i]
    base = (item.get("item") or "").strip()

    prices = {items[j].get("unit_price") for j in indices}
    line_numbers = {items[j].get("line_number") for j in indices}
    categories = {items[j].get("category") for j in indices}

    parts = []

    if len(prices) > 1 and item.get("unit_price") is not None:
        parts.append(f"Rs {item['unit_price']}")

    if len(line_numbers) > 1 and item.get("line_number"):
        parts.append(item["line_number"])

    if len(categories) > 1 and item.get("category"):
        parts.append(item["category"])

    if not parts:
        parts.append(f"duplicate {position + 1}")

    return f"{base} ({', '.join(parts)})"


def _flag_existing_matches(items):
    """
    Flags any action='new_item' row whose name exactly matches an item
    already in the live catalog. add_item's UNIQUE constraint would
    fail this one at confirm time regardless, but the more important
    reason to catch it here: whether this is really the SAME product
    (a price/restock update from a new price list) or a coincidentally
    identical name for a genuinely different one isn't something this
    system can reliably decide on its own -- price alone isn't proof
    (both a real update and a different product can show a different
    price), category alone isn't proof, and there's no "which supplier/
    import this came from" trail on existing items to compare against.
    Rather than guess, this surfaces a snapshot of the existing item
    alongside the new row so the owner can tell at a glance and make
    the one call only they actually have the context for -- see
    web/index.html's existing-item-match UI (two explicit choices:
    update the existing item, or keep this as a new, renamed item).
    """
    existing_by_name = {
        item["name"].strip().lower(): item for item in inventory_db.list_inventory()
    }

    for item in items:
        if item.get("action") != "new_item":
            continue

        name = (item.get("item") or "").strip().lower()
        existing = existing_by_name.get(name)

        if existing:
            item["existing_item_match"] = True
            item["existing_item"] = {
                "name": existing["name"],
                "category": existing["category"],
                "current_quantity": existing["current_quantity"],
                "cost_price": existing["cost_price"],
                "image_path": existing["image_path"],
            }

    return items


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


def _attach_images(result, pdf_bytes):
    """
    Best-effort: matches each extracted item to a product picture from
    the PDF (see pdf_images.py -- pure page-geometry matching,
    independent of the LLM's output order). Never raises -- a PDF that
    doesn't look like a picture-column table just yields no matches,
    which is a normal, silent no-op here, not a failure of the actual
    extraction.

    Consumes images per distinct name in document order rather than a
    plain name->image dict -- when the PDF repeats the same
    description for two different rows (confirmed real case: two
    unrelated parts both called "170F FLYWHEEL FAN" at different
    prices), the Nth extracted item with that name gets the Nth image
    with that name, not whichever one happened to be extracted last.
    """
    try:
        from .pdf_images import extract_row_images, normalize

        image_pairs = extract_row_images(pdf_bytes)
    except Exception:
        return

    if not image_pairs:
        return

    # Two lookups: by the row's literal Sr No/line number (primary --
    # an exact, explicit identifier the model copies verbatim off the
    # page, not something inferred from text similarity) and by
    # normalized description (fallback, for sources with no numbering
    # at all, e.g. a handwritten note). Matching product photos by
    # fuzzy text similarity was tried and explicitly rejected -- a
    # short generic description can look like a substring of a
    # different, unrelated item's name, and silently assigning the
    # wrong picture to the wrong item is worse than assigning none.
    by_line_number = {}
    by_description = {}

    for line_number, name, path in image_pairs:
        if line_number:
            by_line_number.setdefault(normalize(line_number), []).append(path)
        by_description.setdefault(name, []).append(path)

    # Both dicts can reference the same underlying path (every row has
    # both a line number and a description) -- track what's already
    # been handed out so a line-number match and a description match
    # can never both claim the same picture for two different items.
    consumed = set()

    def _take(queue):
        while queue:
            candidate = queue.pop(0)
            if candidate not in consumed:
                return candidate
        return None

    for item in result.get("items", []):
        line_number = normalize(item.get("line_number") or "")
        path = _take(by_line_number.get(line_number, [])) if line_number else None

        if path is None:
            path = _take(by_description.get(normalize(item.get("item", "")), []))

        if path:
            consumed.add(path)
            item["image_path"] = path


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
            result = extract_from_text(full_text, logger=logger, hint=hint)
            _attach_images(result, pdf_bytes)
            return result

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
                        "category": entry.get("category"),
                        "image_path": entry.get("image_path"),
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
