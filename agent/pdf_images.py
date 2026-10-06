# pdf_images.py
#
# Extracts per-item product images from a supplier catalog PDF, matched
# to each row by its position on the page -- a pure PDF-geometry
# problem, independent of anything the LLM does with the text. The
# result is keyed by the RAW item description text read directly off
# the page; import_extractor.py matches that back to its own
# (LLM-cleaned) item names by normalized string comparison, rather than
# trusting that the LLM's output order lines up position-for-position
# with the document (it doesn't always, especially once a boilerplate
# line gets dropped).

import re
import uuid

from .config import PROJECT_ROOT

IMAGE_DIR = PROJECT_ROOT / "product_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def normalize(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def extract_row_images(pdf_bytes):
    """
    Returns a LIST of (sr_no, normalized_description,
    "product_images/<file>") triples, one per table row that has a
    Sr-No-style leading column and a picture positioned in roughly the
    same row, in document order (page by page, top to bottom). A list,
    not a dict keyed by name -- a real supplier PDF can and does
    repeat the same description for two genuinely different parts
    (confirmed: two unrelated parts both just called "170F FLYWHEEL
    FAN" at different prices, distinguished only by their picture).
    Deduplicating into a dict would silently drop one of the two
    images.

    sr_no is the literal text of that leading column exactly as
    printed ("1", "23", "EA07", ...) -- the primary, exact identifier
    import_extractor.py's _attach_images matches against the model's
    own "line_number" field, rather than comparing description text
    for similarity (confirmed real case for why not: the model
    sometimes folds the Sr No into its cleaned-up item name, e.g.
    "EA01 Recoil Stater ..." vs this module's bare "Recoil Stater
    ...", which breaks an exact text match; a *fuzzy* text match to
    paper over that was tried and explicitly rejected as too easy to
    mix up two different items with similar-looking names).
    """
    import pymupdf as fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    result = []

    try:
        for page in doc:
            rows = _find_rows(page)

            if not rows:
                continue

            row_images = _match_row_images(page, rows)

            for sr_no, description, _y_center in rows:
                rect = row_images.get(sr_no)

                if rect is None:
                    continue

                try:
                    # Rendered as a pixmap of the matched region rather
                    # than extracted as a single raw embedded image --
                    # see _match_row_images for why: a row's picture is
                    # often composed of more than one image object, and
                    # this captures all of them together, flattened
                    # exactly as the page itself renders that area.
                    pixmap = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
                except Exception:
                    continue

                filename = f"{uuid.uuid4().hex}.png"
                (IMAGE_DIR / filename).write_bytes(pixmap.tobytes("png"))

                result.append(
                    (sr_no, normalize(description), f"product_images/{filename}")
                )
    finally:
        doc.close()

    return result


_SR_NO_PATTERN = re.compile(r"^[A-Za-z]{0,3}\d{1,4}[A-Za-z]{0,2}$")
_SR_NO_PREFIX_PATTERN = re.compile(r"^([A-Za-z]{0,3}\d{1,4}[A-Za-z]{0,2})\s+(\S.+)$")


def _find_rows(page):
    """
    Finds table rows by locating short "Sr No"-style spans in the left
    part of the page, then pairing each with the longest text span on
    the same line (the item description). A generic heuristic for
    "Sr No | Description | ..." table layouts, not hardcoded to one
    document's exact pixel coordinates.

    The Sr No column isn't always plain digits -- confirmed on a real
    supplier PDF using codes like "EA01", "EA02" instead of "1", "2",
    which a plain text.isdigit() check matched zero rows for and
    silently skipped every image in the whole document. The pattern
    allows an optional short letter prefix/suffix around the digits
    rather than requiring pure numbers.

    Usually the Sr No and description are separate text spans on the
    same line, paired up by y-position below. But confirmed on that
    same real PDF: one single row had them landed in ONE merged span
    ("EA20 Exilator wire  63 cc /68 CC") instead of the usual two --
    an inconsistency in how that particular line got typeset/exported,
    not something a "pair with a neighboring span" approach can catch
    since there's no separate span to pair with. Handled as a second
    case: a span starting with a valid Sr No followed by more text is
    treated as both the code and the description in one.
    """
    spans = []

    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            spans.extend(line.get("spans", []))

    rows = []

    for span in spans:
        text = span["text"].strip()
        x0 = span["bbox"][0]

        if x0 >= page.rect.width * 0.15:
            continue

        y_center = (span["bbox"][1] + span["bbox"][3]) / 2

        if _SR_NO_PATTERN.match(text):
            same_row = [
                s for s in spans
                if s is not span
                and abs(((s["bbox"][1] + s["bbox"][3]) / 2) - y_center) < 6
            ]

            if not same_row:
                continue

            description = max(same_row, key=lambda s: len(s["text"]))["text"]
            rows.append((text, description, y_center))
            continue

        merged = _SR_NO_PREFIX_PATTERN.match(text)

        if merged:
            rows.append((merged.group(1), merged.group(2), y_center))

    return rows


def _match_row_images(page, rows):
    """
    Maps each row's Sr No to the on-page rectangle covering every image
    placement that belongs to it, merged into one union rect.

    Previously this picked only the single nearest xref per row and
    extracted that one embedded image's raw bytes -- but a catalog
    PDF's product photo is frequently built from more than one image
    object layered or tiled together (confirmed on a real supplier
    PDF: a "gasket full" picture that came out as a thin cropped
    sliver, because only one of several overlapping image objects
    making up that photo was ever grabbed). Assigning every placement
    to its nearest row (instead of every row to its single nearest
    placement) and unioning the rects lets extract_row_images render
    the whole matched area as one flattened pixmap.
    """
    placements = []

    for img in page.get_images(full=True):
        xref = img[0]
        placements.extend(page.get_image_rects(xref))

    if not placements or not rows:
        return {}

    # Images confined above the first row are letterhead/logos, not
    # product pictures -- exclude them using the topmost row as cutoff.
    header_cutoff = min(y for _, _, y in rows) - 40
    placements = [r for r in placements if r.y0 >= header_cutoff]

    row_rects = {}

    for rect in placements:
        y_center = (rect.y0 + rect.y1) / 2
        best_sr, best_dist = None, None

        for sr_no, _description, row_y in rows:
            dist = 0 if rect.y0 <= row_y <= rect.y1 else abs(row_y - y_center)

            if best_dist is None or dist < best_dist:
                best_sr, best_dist = sr_no, dist

        if best_dist is not None and best_dist < 40:
            row_rects.setdefault(best_sr, []).append(rect)

    result = {}

    for sr_no, rects in row_rects.items():
        union = rects[0]
        for rect in rects[1:]:
            union |= rect
        result[sr_no] = union

    return result
