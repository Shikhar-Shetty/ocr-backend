#!/usr/bin/env python3
"""
AI-Powered Product Catalogue OCR & Image Extraction Pipeline for Technic ERP  (v3)

v3 is a stabilisation release of v2. It keeps v2's useful ideas (line-level text, document
profile, calibrated confidence, product-level merge, OCR fallbacks) but fixes the things that made
results worse than v1:

  * IMAGES NO LONGER GO MISSING
      - same-page de-duplication only removes *exact overlapping* copies (same bytes, same place);
        two different tiles that merely look alike are both kept.
      - images with a soft mask / CMYK / odd formats are decoded through PyMuPDF pixmaps, so they
        are no longer saved black or corrupt.
      - margin trimming never corrupts a file (it falls back to the original bytes).
      - the QR filter only fires when a QR code is actually *decoded* on a small, square image.
      - "decorative image" filtering is limited to small assets that repeat on most pages
        (logos / badges), and uses position as well as content.
  * IMAGE <-> PRODUCT MATCHING IS IMAGE-CENTRIC
      - every non-decorative tile image becomes a card; text blocks are attached to the best image
        (many text blocks can belong to one image, e.g. name block + spec block). Text that sits
        ABOVE an image is only used when nothing better exists.
      - text-only cards are created only for blocks that really look like a product (code / specs).
      - tile-like images that received no text are kept as low-confidence cards (region OCR is tried)
        instead of silently vanishing. Other leftovers go to `orphanImages` in the extras file.
  * NAME RANKING stays "title-likeness" based (font ratio, bold, word count, case, stop-words,
    punctuation, position relative to image / SKU) but the fragile learned-offset prior was removed
    and multi-line names are preferred over their first line.
  * LAYOUT DETECTION no longer flips a single-product page (big photo + swatches) into a grid.

OUTPUT CONTRACT: by default the JSON printed to stdout / extraction_result.json has EXACTLY the same
shape as v1 (same keys on products, images, candidates, pages and top level), so no backend, database
or UI change is needed. Extra diagnostics (orphan images, merged duplicates, confidence breakdown,
document profile) go to a sidecar file `extraction_extras.json`. Pass --extended-output to inline
them into the main payload instead.
"""

import os
import sys
import json
import re
import math
import hashlib
import argparse
import traceback
import statistics
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    print(json.dumps({"error": "PyMuPDF (fitz) is not installed."}), file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print(json.dumps({"error": "Pillow (PIL) is not installed."}), file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_OPENCV = HAS_NUMPY
except ImportError:
    HAS_OPENCV = False

try:
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.0")
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

KNOWN_BRANDS = [
    "Kajaria", "Somany", "Simpolo", "Nitco", "Qutone", "Johnson",
    "Orientbell", "RAK Ceramics", "Varmora", "Sunhearrt", "Asian Granito",
    "AGL", "Cera", "Hindware", "Velzone", "Velzone Granito", "Technic", "Prestige",
]

KNOWN_SURFACES = [
    "High Gloss", "Super Gloss", "Glossy", "Gloss", "Polished", "Matt", "Matte",
    "Carving", "Satin", "Sugar", "Rustic", "Lappato", "Bookmatch", "Silk",
    "Metallic", "Structure", "Wood", "Punch", "Rock", "Anti-Skid",
]

KNOWN_MATERIALS = [
    "Glazed Vitrified", "Polished Vitrified", "Double Charge", "Full Body Vitrified",
    "Vitrified", "Ceramic", "Porcelain", "Digital Wall", "GVT", "PGVT",
]

KNOWN_BRANDS_LOWER = {b.lower() for b in KNOWN_BRANDS}
KNOWN_SURFACES_LOWER = {s.lower() for s in KNOWN_SURFACES}
KNOWN_MATERIALS_LOWER = {m.lower() for m in KNOWN_MATERIALS}

# --- Dimensions -----------------------------------------------------------------------
# A tile size is "A x B" with an optional thickness "x T", optional units on any side and
# "x", "X", "×" or "*" as the separator: 800x1600, 800 X 1600 MM, 80x160cm, 800mm x 1600mm,
# 800x1600x9mm, 2x2 ft, 12"x18". Named groups: a, ua, b, ub (unit), t, ut (thickness).
_DIM_NUM = r"\d{1,4}(?:\.\d{1,2})?"
_DIM_UNIT = r"(?:mm|cm|mtrs?|inches|inch|in|feet|ft|'|\")"
_DIM_SEP = r"[xX×*]"
SIZE_REGEX = re.compile(
    rf"(?<![A-Za-z0-9.])(?P<a>{_DIM_NUM})\s*(?P<ua>{_DIM_UNIT})?\s*{_DIM_SEP}\s*"
    rf"(?P<b>{_DIM_NUM})\s*(?P<ub>{_DIM_UNIT})?"
    rf"(?:\s*{_DIM_SEP}\s*(?P<t>\d{{1,2}}(?:\.\d{{1,2}})?)\s*(?P<ut>mm|cm)?)?"
    rf"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Half of a size that was split by a line break or a bad OCR read: "800x" / "x1600".
DIM_FRAGMENT_REGEX = re.compile(
    rf"(?<![A-Za-z0-9.])(?:{_DIM_NUM}\s*(?:mm|cm)?\s*{_DIM_SEP}(?![A-Za-z0-9])"
    rf"|{_DIM_SEP}\s*\d{{2,4}}(?:\.\d{{1,2}})?\s*(?:mm|cm)?(?![A-Za-z0-9]))",
    re.IGNORECASE,
)
# A whole token that is nothing but a size, a size fragment or a measurement.
DIM_TOKEN_REGEX = re.compile(
    rf"^{_DIM_SEP}?\d{{1,4}}(?:\.\d{{1,2}})?(?:mm|cm|in|inch|ft)?"
    rf"(?:{_DIM_SEP}\d{{0,4}}(?:\.\d{{1,2}})?(?:mm|cm|in|inch|ft)?){{0,2}}$",
    re.IGNORECASE,
)
# Plausible tile edge length per unit (anything outside is not a tile size).
DIM_LIMITS = {"mm": (50.0, 4000.0), "cm": (5.0, 400.0), "inch": (2.0, 120.0), "ft": (1.0, 12.0)}

CODE_REGEX = re.compile(r"\b([A-Z]{1,6}[-_/ ]?[0-9]{2,6}(?:[-_/ ][A-Z0-9]+)?)\b")
SKU_LABEL_REGEX = re.compile(
    r"\b(?:sku|item|item\s*no|model|model\s*no|article|article\s*no|ref(?:erence)?|code)"
    r"\s*(?:no\.?|number)?\s*[:#.-]?\s*([A-Z0-9][A-Z0-9_/\-]{2,})\b",
    re.IGNORECASE,
)
GENERAL_SKU_REGEX = re.compile(
    r"\b(?=[A-Z0-9_/\-]*[A-Z])(?=[A-Z0-9_/\-]*\d)[A-Z0-9][A-Z0-9_/\-]{2,}\b",
    re.IGNORECASE,
)
NUMERIC_CODE_LINE_REGEX = re.compile(r"^\s*(?:#\s*|no\.?\s*)?(\d{4,8})\s*$", re.IGNORECASE)
MEASUREMENT_PHRASE_REGEX = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:mm|cm|mtrs?|inches|inch|in|ft|feet|pcs|pc|sqm|sqft|kg)(?![A-Za-z])",
    re.IGNORECASE,
)
UNIT_TOKEN_REGEX = re.compile(
    r"^\d+(?:\.\d+)?(?:mm|cm|pcs|pc|pkts?|sqm|sqft|kg|gm|g|mtr|ft|in|x\d+.*)$", re.IGNORECASE
)

URL_REGEX = re.compile(
    r"(?:https?://|www\.)\S+|(?:[a-z0-9-]+\.)+[a-z]{2,}(?:[/:?#]\S*)?", re.IGNORECASE
)
EMAIL_REGEX = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.IGNORECASE)
ISO_CODE_REGEX = re.compile(r"\b(?:ISO\s*[-:]?\s*)?\d{4,5}(?:[-:]\d+)?\b", re.IGNORECASE)

LABEL_RE = re.compile(
    r"^(?:size|surface|finish|material|mrp|price|hsn|gst|pcs|pieces|coverage|pack|packing|"
    r"box|unit|colour|color|category|collection|series|edition|random|sku|code)\b"
)

IRRELEVANT_TEXT_PATTERNS = (
    "catalogue", "catalog", "brochure", "architectural", "manufacturer",
    "company", "corporate", "head office", "branch", "contact", "website",
    "email", "www.", "ceramic industries", "private limited", "pvt ltd",
    "limited", "iso", "gstin", "disclaimer", "all rights reserved",
    "phone", "tel", "fax", "mail",
)
CORPORATE_RE = re.compile(
    "|".join(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])" for p in IRRELEVANT_TEXT_PATTERNS),
    re.IGNORECASE,
)

INSTRUCTIONAL_PATTERNS = (
    r"\b(?:please|kindly|note|refer|contact|visit|call|scan|use|ensure|available)\b",
    r"\b(?:we|our|the company|this catalogue|this catalog)\s+(?:offer|provide|manufacture|bring|are|is)\b",
    r"\b(?:terms|conditions|disclaimer|warranty|installation|delivery)\b",
)

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "with", "in", "on", "at", "to", "from",
    "by", "is", "are", "be", "this", "that", "these", "those", "our", "your", "its",
    "it", "as", "into", "which", "can", "will", "has", "have", "also", "more", "all",
}
DESCRIPTION_WORDS = {
    "is", "are", "offers", "offer", "features", "provides", "designed", "ideal",
    "suitable", "perfect", "enhance", "enhances", "available", "brings", "creates",
    "adds", "makes", "gives", "ensures", "made", "used", "perfectly",
}

NON_NAME_EXACT = {
    "bookmatch", "glossy", "series", "collection", "surface", "glossy surface",
    "glazed vitrified tiles", "glazed vitrified", "vitrified", "tiles", "floor tiles",
    "wall tiles", "size:", "size", "random", "polished vitrified tiles", "ceramic tiles",
    "porcelain tiles", "matt", "matte", "satin", "rustic", "floor", "wall",
    # page navigation, never a product name
    "index", "tap to", "tap to index", "back to index", "go to index", "contents", "home",
    "next", "previous", "next page", "previous page", "click here",
}
NON_NAME_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"^random\b",
        r"^\d+\s*(?:pcs|pieces|pc)\b",
        r"\b(?:per|/)\s*box\b",
        r"^(?:floor|wall|outdoor|parking)\s*tiles?$",
        r"^tiles?$",
        r"^(?:sq\.?\s*(?:m|ft)|sqm|sqft)\b",
        r"^(?:mrp|rs\.?|₹)\s*[:.-]?\s*\d",
    )
]

PCS_BOX_REGEX = re.compile(r"(\d+)\s*(?:pcs|pieces|pc|pkts?)\b", re.IGNORECASE)
COVERAGE_REGEX = re.compile(
    r"(\d+(?:\.\d+)?)\s*(sq\.?\s*m(?:tr)?|sqm|sq\.?\s*ft|sqft)\b", re.IGNORECASE
)
PRICE_REGEX = re.compile(r"(?:MRP|Price|Rs\.?|₹)\s*[:.-]?\s*(\d+(?:[.,]\d{2})?)", re.IGNORECASE)
PAGE_NUMBER_REGEX = re.compile(r"^(?:page\s*)?\d{1,3}(?:\s*/\s*\d{1,3})?$", re.IGNORECASE)

# Words that describe HOW a tile is shown, not WHICH tile it is. "Calacatta Gold Close-up" and
# "Calacatta Gold - Room View" are both the swatch "Calacatta Gold".
VIEW_QUALIFIER_REGEX = re.compile(
    r"\b(?:"
    r"close\s*-?\s*up(?:\s*view)?|closeup|zoom(?:ed)?(?:\s*-?\s*in)?|preview|"
    r"room\s*(?:view|shot|scene|image)|in\s*-?\s*situ|"
    r"details?\s*(?:view|shot)|texture\s*(?:view|shot)|surface\s*view|"
    r"application\s*(?:view|image|shot)|installed\s*view|installation\s*view|"
    r"lifestyle\s*(?:view|image|shot)?|ambience\s*(?:view|image|shot)?|mood\s*(?:shot|image|board)|"
    r"(?:random\s*)?faces?(?:\s*-?\s*\d{1,2})?|"
    r"(?:full|front|top|side|angle|corner)\s*view"
    r")\b",
    re.IGNORECASE,
)

SQFT_TO_SQM = 0.092903
UNRESOLVED_NAME = "Unresolved product name"
MAX_GROUP_COST = 2.2  # text block further than this from every image is not attached to one

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def normalize_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_url_like(text: str) -> bool:
    value = text.strip().strip(".,;:()[]{}<>")
    return bool(URL_REGEX.search(value) or EMAIL_REGEX.search(value))


def is_page_number_line(text: str) -> bool:
    return bool(PAGE_NUMBER_REGEX.match(text.strip()))


# --------------------------------------------------------------------------------------
# Dimension detection
# --------------------------------------------------------------------------------------


def _canonical_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    u = unit.lower()
    if u == "mm":
        return "mm"
    if u == "cm":
        return "cm"
    if u.startswith("mtr"):
        return "m"
    if u in ("in", "inch", "inches", '"'):
        return "inch"
    if u in ("ft", "feet", "'"):
        return "ft"
    return None


def _fmt_num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def detect_unit_hint(text: str) -> Optional[str]:
    """The unit a document states for its sizes ("Size (cm)", "All sizes in mm"), if any."""
    m = re.search(r"\b(?:size|sizes|dimensions?)\b[^\n]{0,20}?\(?\s*(?:in\s+)?(mm|cm|inch(?:es)?|ft|feet)\b",
                  text, re.IGNORECASE)
    return _canonical_unit(m.group(1)) if m else None


def parse_dimension(text: str, unit_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    First plausible tile size in `text`, normalised.

    Returns {"raw", "width", "height", "unit", "size", "thickness", "unitStated"}. Explicit units win;
    otherwise `unit_hint`; otherwise the magnitude decides (>= 50 is mm, <= 12 is feet). A size whose
    unit cannot be inferred (e.g. a bare "30x60") is returned with unit None so the caller can ask for
    review instead of guessing. Implausible values (an "8x2" in a sentence) are skipped.
    """
    for m in SIZE_REGEX.finditer(text or ""):
        a, b = float(m.group("a")), float(m.group("b"))
        stated = _canonical_unit(m.group("ub") or m.group("ua"))
        unit = stated or unit_hint
        if unit is None:
            if min(a, b) >= 50:
                unit = "mm"
            elif max(a, b) <= 12:
                unit = "ft"
        if unit == "m":
            a, b, unit = a * 1000, b * 1000, "mm"
        if unit is not None:
            low, high = DIM_LIMITS[unit]
            if not (low <= a <= high and low <= b <= high):
                continue
        elif not (10 <= a <= 400 and 10 <= b <= 400):
            continue
        if unit == "cm":  # catalogues sell in cm, the ERP stores mm
            a, b, unit = a * 10, b * 10, "mm"
        thickness = None
        if m.group("t"):
            t = float(m.group("t")) * (10 if (m.group("ut") or "").lower() == "cm" else 1)
            if 3 <= t <= 30:
                thickness = f"{_fmt_num(t)} mm"
        size = f"{_fmt_num(a)}x{_fmt_num(b)}" + (f" {unit}" if unit else "")
        return {
            "raw": m.group(0).strip(), "width": a, "height": b, "unit": unit, "size": size,
            "thickness": thickness, "unitStated": stated is not None,
        }
    return None


def dimension_spans(text: str) -> List[Tuple[int, int]]:
    """Character spans of every size and size fragment in a line."""
    spans = [m.span() for m in SIZE_REGEX.finditer(text)]
    for m in DIM_FRAGMENT_REGEX.finditer(text):
        if not any(s <= m.start() and m.end() <= e for s, e in spans):
            spans.append(m.span())
    return spans


def is_dimension_token(token: str) -> bool:
    """True for '800X', 'X1600', '800X1600', '800X1600X9MM', '9MM' - a measurement, never an SKU."""
    value = token.strip().strip("-_/")
    if not DIM_TOKEN_REGEX.match(value):
        return False
    return bool(re.search(r"[xX×*]|mm|cm|in|ft", value, re.IGNORECASE))


def bbox_union(boxes: List[List[float]]) -> List[float]:
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def bbox_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_iou(a: List[float], b: List[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_contained_ratio(inner: List[float], outer: List[float]) -> float:
    """Share of `inner` area that lies inside `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = bbox_area(inner)
    return (ix * iy) / area if area > 0 else 0.0


def bbox_gap(a: List[float], b: List[float]) -> Tuple[float, float]:
    gx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    gy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return gx, gy


def char_weighted_mode(lines: List[Dict[str, Any]]) -> float:
    counter: Counter = Counter()
    for line in lines:
        if line.get("size", 0) > 0:
            counter[round(line["size"] * 2) / 2.0] += len(line["text"])
    return counter.most_common(1)[0][0] if counter else 0.0


def line_sort_key(line: Dict[str, Any]) -> Tuple[int, float]:
    return (round(((line["bbox"][1] + line["bbox"][3]) / 2) / 6), line["bbox"][0])


def make_line(text: str, bbox, size: float, bold: bool, conf: float, source: str) -> Dict[str, Any]:
    return {
        "text": text,
        "bbox": [float(v) for v in bbox],
        "size": float(size),
        "bold": bool(bold),
        "conf": float(conf),
        "source": source,
        "template": False,
    }


# --------------------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------------------

_paddle_ocr = None


def get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None and HAS_PADDLEOCR:
        try:
            _paddle_ocr = PaddleOCR(use_textline_orientation=True, lang="en")
        except Exception:
            try:
                _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
            except Exception:
                _paddle_ocr = None
    return _paddle_ocr


def run_paddle_ocr(image_path: str, target_bbox: List[float]) -> List[Dict[str, Any]]:
    """
    Run PaddleOCR on an image that covers `target_bbox` (PDF coordinate space).
    Returns LINE dicts (bbox in PDF space, OCR confidence kept, box height used as size proxy).
    """
    ocr = get_paddle_ocr()
    if not ocr:
        return []
    try:
        results = ocr.ocr(image_path)
        if not results:
            return []
        with Image.open(image_path) as im:
            iw, ih = im.size
        tx0, ty0, tx1, ty1 = target_bbox
        scale_x = (tx1 - tx0) / float(iw) if iw > 0 else 1.0
        scale_y = (ty1 - ty0) / float(ih) if ih > 0 else 1.0
        lines: List[Dict[str, Any]] = []

        def add(poly_or_box, text, score):
            text = str(text).strip()
            if not text:
                return
            if len(poly_or_box) == 4 and not hasattr(poly_or_box[0], "__len__"):
                xs = [poly_or_box[0], poly_or_box[2]]
                ys = [poly_or_box[1], poly_or_box[3]]
            else:
                xs = [pt[0] for pt in poly_or_box]
                ys = [pt[1] for pt in poly_or_box]
            bx0 = tx0 + float(min(xs)) * scale_x
            by0 = ty0 + float(min(ys)) * scale_y
            bx1 = tx0 + float(max(xs)) * scale_x
            by1 = ty0 + float(max(ys)) * scale_y
            lines.append(make_line(text, (bx0, by0, bx1, by1), (by1 - by0) * 0.8, False,
                                   float(score) if score is not None else 0.8, "ocr"))

        for res_item in results:
            if not res_item:
                continue
            # Format A: modern PaddleX / PaddleOCR dict structure
            if isinstance(res_item, dict):
                texts = res_item.get("rec_texts", [])
                polys = res_item.get("rec_polys", [])
                boxes = res_item.get("rec_boxes", [])
                scores = res_item.get("rec_scores", [])
                for idx, text in enumerate(texts):
                    score = scores[idx] if idx < len(scores) else None
                    if idx < len(polys) and polys[idx] is not None:
                        add(polys[idx], text, score)
                    elif idx < len(boxes) and boxes[idx] is not None:
                        add(list(boxes[idx]), text, score)
                continue
            # Format B: classic list of [[poly], (text, score)]
            if isinstance(res_item, (list, tuple)):
                for entry in res_item:
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        bbox_poly, text_info = entry
                        if isinstance(text_info, (list, tuple)):
                            add(bbox_poly, text_info[0], text_info[1] if len(text_info) > 1 else None)
                        else:
                            add(bbox_poly, text_info, None)
        return lines
    except Exception as e:
        print(f"PaddleOCR parsing warning: {e}", file=sys.stderr)
        return []


def merge_ocr_lines(native: List[Dict[str, Any]], ocr_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add OCR lines that are not already covered by native text lines."""
    merged = list(native)
    for ocr_line in ocr_lines:
        covered = any(
            bbox_iou(ocr_line["bbox"], nat["bbox"]) > 0.3
            or bbox_contained_ratio(ocr_line["bbox"], nat["bbox"]) > 0.6
            for nat in native
        )
        if not covered:
            merged.append(ocr_line)
    return merged


def ocr_region(preview_path: str, page_rect: fitz.Rect, region: List[float]) -> List[Dict[str, Any]]:
    """OCR a crop of the page preview (used for cards with no readable name)."""
    if not HAS_PADDLEOCR:
        return []
    tmp_path = None
    try:
        rx0 = max(0.0, region[0])
        ry0 = max(0.0, region[1])
        rx1 = min(page_rect.width, region[2])
        ry1 = min(page_rect.height, region[3])
        if rx1 - rx0 < 10 or ry1 - ry0 < 10:
            return []
        with Image.open(preview_path) as im:
            iw, ih = im.size
            sx = iw / page_rect.width
            sy = ih / page_rect.height
            crop = im.crop((int(rx0 * sx), int(ry0 * sy), int(rx1 * sx), int(ry1 * sy)))
            tmp_path = preview_path + f".region_{int(rx0)}_{int(ry0)}.png"
            crop.save(tmp_path)
        return run_paddle_ocr(tmp_path, [rx0, ry0, rx1, ry1])
    except Exception:
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# --------------------------------------------------------------------------------------
# Native line extraction
# --------------------------------------------------------------------------------------


def _span_is_bold(span: Dict[str, Any]) -> bool:
    font = str(span.get("font", "")).lower()
    return bool(int(span.get("flags", 0)) & 16) or any(
        k in font for k in ("bold", "black", "heavy", "semibold", "demi")
    )


def extract_native_lines(page: fitz.Page) -> List[Dict[str, Any]]:
    """Per-LINE extraction keeping each line's own bbox, font size and bold flag."""
    lines: List[Dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = re.sub(r"\s+", " ", "".join(str(s.get("text", "")) for s in spans)).strip()
            if not text:
                continue
            real_spans = [s for s in spans if str(s.get("text", "")).strip()]
            if not real_spans:
                continue
            size = max(float(s.get("size", 0)) for s in real_spans)
            total_chars = sum(len(str(s.get("text", "")).strip()) for s in real_spans) or 1
            bold_chars = sum(len(str(s.get("text", "")).strip()) for s in real_spans if _span_is_bold(s))
            lines.append(make_line(text, line["bbox"], size, bold_chars / total_chars >= 0.5, 1.0, "native"))
    return lines


# --------------------------------------------------------------------------------------
# Document profile (pass 1)
# --------------------------------------------------------------------------------------


class DocProfile:
    """Things that repeat across pages (template text, logos) and the learned name style."""

    def __init__(self, strip_prefixes: bool = False):
        self.strip_prefixes = strip_prefixes
        self.template_lines: set = set()
        self.template_prefixes: List[Tuple[str, ...]] = []
        self.decor_hashes: set = set()
        self.decor_positions: set = set()
        self.body_size: float = 0.0
        self.name_size: Optional[float] = None
        self.name_bold: Optional[bool] = None
        self.sampled_pages = 0

    def is_decorative_image(self, raw_sha1: str, rect: fitz.Rect, page_area: float) -> bool:
        """Only SMALL assets that repeat on most pages (logos, badges) can be decorative."""
        area_ratio = (abs(rect.x1 - rect.x0) * abs(rect.y1 - rect.y0)) / max(page_area, 1.0)
        if area_ratio > 0.08:
            return False
        if (raw_sha1, round(rect.x0 / 12), round(rect.y0 / 12)) in self.decor_positions:
            return True
        return area_ratio <= 0.03 and raw_sha1 in self.decor_hashes

    def summary(self) -> Dict[str, Any]:
        return {
            "sampledPages": self.sampled_pages,
            "templateLineCount": len(self.template_lines),
            "templateLines": sorted(self.template_lines)[:30],
            "templatePrefixes": [" ".join(p) for p in self.template_prefixes[:20]],
            "decorativeImageCount": len(self.decor_hashes) + len(self.decor_positions),
            "bodyFontSize": self.body_size,
            "learnedNameStyle": {"fontSize": self.name_size, "bold": self.name_bold},
        }


def _norm_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def build_document_profile(doc: fitz.Document, strip_prefixes: bool = False, max_sample: int = 150) -> DocProfile:
    """Find template text, template prefixes and decorative images that repeat across pages."""
    profile = DocProfile(strip_prefixes)
    total = len(doc)
    idxs = list(range(total)) if total <= max_sample else sorted({int(i * total / max_sample) for i in range(max_sample)})
    profile.sampled_pages = len(idxs)
    if len(idxs) < 4:
        return profile

    line_threshold = max(4, math.ceil(0.40 * len(idxs)))
    line_pages: Dict[str, set] = defaultdict(set)
    prefix_pages: Dict[Tuple[str, ...], set] = defaultdict(set)
    hash_pages: Dict[str, set] = defaultdict(set)
    pos_pages: Dict[Tuple[str, int, int], set] = defaultdict(set)
    size_chars: Counter = Counter()
    xref_hash: Dict[int, str] = {}

    for pi in idxs:
        try:
            page = doc[pi]
            lines = extract_native_lines(page)
        except Exception:
            continue
        for ln in lines:
            text = ln["text"]
            if ln["size"] > 0:
                size_chars[round(ln["size"] * 2) / 2.0] += len(text)
            key = normalize_text_key(text)
            # lines containing digits are specs / phones / codes: never treated as template text
            if len(key) >= 4 and not re.search(r"\d", key) and not is_page_number_line(text):
                line_pages[key].add(pi)
            if strip_prefixes:
                words = text.split()
                norm = [_norm_word(w) for w in words]
                for k in (2, 3):
                    if len(words) > k:
                        prefix = tuple(norm[:k])
                        if all(w.isalpha() and len(w) >= 2 for w in prefix):
                            prefix_pages[prefix].add(pi)
        try:
            for info in page.get_images(full=True):
                xref = info[0]
                if xref not in xref_hash:
                    data = doc.extract_image(xref)
                    if not data or not data.get("image"):
                        continue
                    xref_hash[xref] = hashlib.sha1(data["image"]).hexdigest()
                sha = xref_hash[xref]
                hash_pages[sha].add(pi)
                for rect in page.get_image_rects(xref):
                    pos_pages[(sha, round(rect.x0 / 12), round(rect.y0 / 12))].add(pi)
        except Exception:
            continue

    profile.template_lines = {k for k, pages in line_pages.items() if len(pages) >= line_threshold}
    if strip_prefixes:
        prefix_threshold = max(5, math.ceil(0.40 * len(idxs)))
        prefixes = [p for p, pages in prefix_pages.items()
                    if len(pages) >= prefix_threshold and not all(w in STOPWORDS for w in p)]
        prefixes.sort(key=lambda p: (-len(p), p))
        profile.template_prefixes = prefixes
    profile.decor_hashes = {h for h, pages in hash_pages.items() if len(pages) >= max(6, math.ceil(0.6 * len(idxs)))}
    profile.decor_positions = {k for k, pages in pos_pages.items() if len(pages) >= max(6, math.ceil(0.5 * len(idxs)))}
    profile.body_size = size_chars.most_common(1)[0][0] if size_chars else 0.0
    return profile


def strip_template_prefix(text: str, prefixes: List[Tuple[str, ...]]) -> Tuple[str, str]:
    words = text.split()
    norm = [_norm_word(w) for w in words]
    for prefix in prefixes:
        k = len(prefix)
        if len(words) > k and tuple(norm[:k]) == prefix:
            return " ".join(words[k:]), " ".join(words[:k])
    return text, ""


def mark_template_lines(lines: List[Dict[str, Any]], profile: DocProfile) -> None:
    for ln in lines:
        key = normalize_text_key(ln["text"])
        ln["template"] = bool(key in profile.template_lines or is_page_number_line(ln["text"]))


# --------------------------------------------------------------------------------------
# Name / SKU candidates and ranking
# --------------------------------------------------------------------------------------


def clean_name_text(text: str) -> str:
    value = SIZE_REGEX.sub(" ", text)
    value = DIM_FRAGMENT_REGEX.sub(" ", value)
    value = PRICE_REGEX.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip(" -–|:,/")


def is_name_eligible(text: str) -> bool:
    if not (3 <= len(text) <= 80):
        return False
    if is_url_like(text) or ISO_CODE_REGEX.fullmatch(text):
        return False
    low = text.lower()
    if LABEL_RE.match(low) or CODE_REGEX.fullmatch(text):
        return False
    # A measurement ('1200mm', '9 MM', '800x') is never a name: it needs a real word of 2+ letters.
    rest = MEASUREMENT_PHRASE_REGEX.sub(" ", text)
    words = [w for w in rest.split() if not (is_dimension_token(w) or UNIT_TOKEN_REGEX.match(w))]
    if not any(re.search(r"[A-Za-z]{2,}", w) for w in words):
        return False
    if low in NON_NAME_EXACT or low in KNOWN_SURFACES_LOWER or low in KNOWN_MATERIALS_LOWER or low in KNOWN_BRANDS_LOWER:
        return False
    if any(p.search(low) for p in NON_NAME_PATTERNS):
        return False
    return bool(re.search(r"[A-Za-z]", text)) and not is_page_number_line(text)


def make_name_candidate(line: Dict[str, Any], profile: Optional[DocProfile], page_h: float) -> Optional[Dict[str, Any]]:
    if line.get("template"):
        return None
    text = clean_name_text(line["text"])
    stripped = ""
    if profile and profile.strip_prefixes and profile.template_prefixes:
        text2, prefix = strip_template_prefix(text, profile.template_prefixes)
        if prefix and len(text2.strip()) >= 3:
            text, stripped = text2.strip(), prefix
    if not is_name_eligible(text):
        return None
    cy = (line["bbox"][1] + line["bbox"][3]) / 2.0
    return {
        "text": text,
        "original": line["text"],
        "strippedPrefix": stripped,
        "bbox": list(line["bbox"]),
        "size": line.get("size", 0.0),
        "bold": line.get("bold", False),
        "conf": line.get("conf", 1.0),
        "source": line.get("source", "native"),
        "corporate": bool(CORPORATE_RE.search(text)),
        "edge": bool(page_h and (cy < 0.05 * page_h or cy > 0.95 * page_h)),
        "multiline": False,
    }


def _stack_aligned(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ax0, _, ax1, _ = a["bbox"]
    bx0, _, bx1, _ = b["bbox"]
    overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    min_w = max(1.0, min(ax1 - ax0, bx1 - bx0))
    return overlap / min_w >= 0.2 or abs(ax0 - bx0) <= 8 or abs((ax0 + ax1) / 2 - (bx0 + bx1) / 2) <= 10


def build_name_candidates(lines: List[Dict[str, Any]], profile: Optional[DocProfile], page_h: float) -> List[Dict[str, Any]]:
    """Single-line candidates plus stacked same-style lines (e.g. 'Calacatta' / 'Gold')."""
    singles = [c for c in (make_name_candidate(l, profile, page_h) for l in lines) if c]
    ordered = sorted(singles, key=lambda c: (c["bbox"][1], c["bbox"][0]))
    merged: List[Dict[str, Any]] = []
    for i, first in enumerate(ordered):
        chain = [first]
        for second in ordered[i + 1:i + 3]:
            prev = chain[-1]
            gap = second["bbox"][1] - prev["bbox"][3]
            height = max(1.0, prev["bbox"][3] - prev["bbox"][1])
            if prev["size"] > 0 and second["size"] > 0:
                same_size = abs(prev["size"] - second["size"]) <= 0.6
            else:
                same_size = abs((prev["bbox"][3] - prev["bbox"][1]) - (second["bbox"][3] - second["bbox"][1])) <= 3
            if (
                prev["bold"] != second["bold"] or not same_size or gap > height or gap < -height * 0.5
                or not _stack_aligned(prev, second) or sku_tokens_from_line(second["text"])
            ):
                break
            chain.append(second)
            text = " ".join(c["text"] for c in chain)
            if len(text.split()) > 6:
                break
            merged.append({
                "text": text,
                "original": " ".join(c["original"] for c in chain),
                "strippedPrefix": chain[0]["strippedPrefix"],
                "bbox": bbox_union([c["bbox"] for c in chain]),
                "size": max(c["size"] for c in chain),
                "bold": all(c["bold"] for c in chain),
                "conf": min(c["conf"] for c in chain),
                "source": chain[0]["source"],
                "corporate": any(c["corporate"] for c in chain),
                "edge": any(c["edge"] for c in chain),
                "multiline": True,
            })
    return singles + merged


def image_relation_bonus(box: List[float], image_box: List[float]) -> float:
    """Names usually sit below or beside their image, rarely above."""
    ix0, iy0, ix1, iy1 = image_box
    iw, ih = max(1.0, ix1 - ix0), max(1.0, iy1 - iy0)
    gx, gy = bbox_gap(box, image_box)
    xov = max(0.0, min(box[2], ix1) - max(box[0], ix0)) / max(1.0, min(box[2] - box[0], iw))
    yov = max(0.0, min(box[3], iy1) - max(box[1], iy0)) / max(1.0, min(box[3] - box[1], ih))
    if xov > 0.3 and yov > 0.3:
        return 0.3
    if box[1] >= iy1 - 0.15 * ih and xov > 0.15:
        return 1.0 / (1.0 + gy / 60.0)
    if yov > 0.3:
        return 0.6 / (1.0 + gx / 80.0)
    if box[3] <= iy0 + 0.15 * ih:
        return -0.4
    return 0.0


def score_name(c: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    text = c["text"]
    words = text.split()
    nw = len(words)
    low = text.lower()
    profile: Optional[DocProfile] = ctx.get("profile")
    score = 0.0

    # Looks like a title: short
    if nw <= 4:
        score += 1.0
    elif nw <= 6:
        score += 0.5
    elif nw > 8:
        score -= 1.5

    # Font size relative to body text, bold
    size, body = c.get("size", 0.0), ctx.get("body", 0.0)
    if size > 0 and body > 0:
        weight = 1.0 if c.get("source") != "ocr" else 0.6
        score += weight * clamp((size / body - 1.0) * 2.0, -0.6, 2.0)
    if c.get("bold"):
        score += 0.6
    if c.get("multiline"):
        score += 0.4  # a stacked name beats its own first line

    # Case
    alpha = [w for w in words if w and w[0].isalpha()]
    if text.isupper() and nw <= 5:
        score += 0.4
    elif alpha and sum(1 for w in alpha if w[0].isupper()) / len(alpha) >= 0.75:
        score += 0.5
    elif alpha and alpha[0][0].islower():
        score -= 0.1

    # Sentence-like text
    stop_ratio = sum(1 for w in words if w.lower() in STOPWORDS) / max(1, nw)
    if nw >= 4 and stop_ratio > 0.4:
        score -= 0.9
    if nw >= 5 and any(w.lower().strip(".,") in DESCRIPTION_WORDS for w in words):
        score -= 1.0
    score -= min(1.2, 0.6 * text.count(","))
    if text.endswith((".", ":", ";")):
        score -= 0.8
    if text.count(".") > 1:
        score -= 0.8
    if nw >= 3 and any(re.search(p, text, re.IGNORECASE) for p in INSTRUCTIONAL_PATTERNS):
        score -= 2.0

    digits = sum(ch.isdigit() for ch in text)
    if digits:
        score -= 0.1
        if digits / max(1, len(text)) > 0.5:
            score -= 1.5

    # Style learned from this document's confident cards (native text only; OCR size is a proxy)
    if profile and c.get("source") != "ocr":
        if profile.name_size and size > 0:
            if abs(size - profile.name_size) <= 0.6:
                score += 0.5
            elif size < profile.name_size * 0.7:
                score -= 0.2
        if profile.name_bold is not None and c.get("bold") == profile.name_bold:
            score += 0.15

    # Spatial context
    image_box = ctx.get("image")
    if image_box:
        score += image_relation_bonus(c["bbox"], image_box)
    # Names sit right above their spec block ("SIZE: ...", "FINISH: ...")
    height = max(1.0, c["bbox"][3] - c["bbox"][1])
    for sb in ctx.get("spec_boxes", ()):
        dy = sb[1] - c["bbox"][3]
        aligned = (min(c["bbox"][2], sb[2]) - max(c["bbox"][0], sb[0]) > 0
                   or abs(c["bbox"][0] - sb[0]) <= 10)
        if aligned and -0.3 * height <= dy <= 3.5 * height:
            score += 0.9 / (1.0 + max(dy, 0.0) / (2.0 * height))
            break
    sku_box = ctx.get("sku_bbox")
    if sku_box:
        d = math.hypot(
            (c["bbox"][0] + c["bbox"][2]) / 2 - (sku_box[0] + sku_box[2]) / 2,
            (c["bbox"][1] + c["bbox"][3]) / 2 - (sku_box[1] + sku_box[3]) / 2,
        )
        score += 1.5 / (1.0 + d / 100.0)

    if c.get("corporate"):
        score -= 1.2
    if c.get("edge"):
        score -= 0.5
    if c.get("source") == "ocr" and c.get("conf", 1.0) < 0.6:
        score -= 0.5
    if ctx.get("exclude_text") and low == ctx["exclude_text"]:
        score -= 5.0
    return score


def is_valid_sku(token: str, line_text: str = "", span: Optional[Tuple[int, int]] = None) -> bool:
    """
    SKU validation. A code must contain a digit and must not be a measurement: it may not look like
    a size or size fragment on its own ('800X' from '800x 1600', '800X1600X9MM', '9MM'), and it may
    not sit entirely inside a size printed on its line. A real code that merely CONTAINS a size
    ('VZ-800X1600-CAL') is still valid.
    """
    if not (3 <= len(token) <= 20) or not re.search(r"\d", token):
        return False
    if is_url_like(token) or ISO_CODE_REGEX.fullmatch(token) or UNIT_TOKEN_REGEX.match(token):
        return False
    if SIZE_REGEX.fullmatch(token) or is_dimension_token(token):
        return False
    if span and line_text:
        start, end = span
        if any(s <= start and end <= e for s, e in dimension_spans(line_text)):
            return False
    return True


def numeric_code_from_line(text: str) -> Optional[str]:
    """
    A design code printed on its own line as a bare number ('20001'). Many tile catalogues identify
    designs only by such a number. Years (1900-2099) are not codes; neither are sizes or page numbers
    (those are 1-3 digits or contain units and are rejected elsewhere).
    """
    m = NUMERIC_CODE_LINE_REGEX.match(text or "")
    if not m:
        return None
    code = m.group(1)
    if len(code) == 4 and 1900 <= int(code) <= 2099:
        return None
    return code


def sku_tokens_from_line(text: str) -> List[Tuple[str, bool]]:
    numeric = numeric_code_from_line(text)
    if numeric:
        return [(numeric, False)]
    labeled_matches = list(SKU_LABEL_REGEX.finditer(text))
    if labeled_matches:
        labeled, found = True, [(m.group(1), m.span(1)) for m in labeled_matches]
    else:
        labeled, found = False, [(m.group(0), m.span(0)) for m in GENERAL_SKU_REGEX.finditer(text)]
    out = []
    for raw, (start, end) in found:
        token = raw.strip("-_/")
        offset = raw.find(token)
        if not is_valid_sku(token, text, (start + offset, start + offset + len(token))):
            continue
        out.append((token, labeled))
    return out


def _is_prominent(line: Dict[str, Any], body: float) -> bool:
    """Set larger than body text or in a heavier weight - how catalogues print a design code."""
    size = line.get("size", 0.0)
    return bool(line.get("bold")) or (body > 0 and size >= body * 1.15)


def extract_sku_candidates(lines: List[Dict[str, Any]], body: float = 0.0) -> List[Dict[str, Any]]:
    seen = set()
    candidates = []
    for line in lines:
        if line.get("template"):
            continue
        for token, labeled in sku_tokens_from_line(line["text"]):
            numeric = token.isdigit()
            # A bare number is only a code when it is printed like one (big or bold), never body text.
            if numeric and not _is_prominent(line, body):
                continue
            key = token.upper()
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "text": token, "bbox": list(line["bbox"]), "labeled": labeled, "numeric": numeric,
                "source": line.get("source", "native"), "conf": line.get("conf", 1.0),
                "size": line.get("size", 0.0), "bold": line.get("bold", False), "multiline": False,
            })
    return candidates


def score_sku(c: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    token = c["text"]
    score = 0.0
    if c.get("labeled"):
        score += 2.5
    if re.search(r"[A-Za-z]", token) and re.search(r"\d", token):
        score += 1.0
    if c.get("numeric"):
        score += 1.0  # a standalone, prominently printed design number (already filtered for that)
    if re.search(r"[-_/]", token):
        score += 0.5
    if 4 <= len(token) <= 12:
        score += 0.5
    image_box = ctx.get("image")
    if image_box:
        d = math.hypot(
            (c["bbox"][0] + c["bbox"][2]) / 2 - (image_box[0] + image_box[2]) / 2,
            (c["bbox"][1] + c["bbox"][3]) / 2 - (image_box[1] + image_box[3]) / 2,
        )
        score += 0.5 / (1.0 + d / 150.0)
    if c.get("source") == "ocr" and c.get("conf", 1.0) < 0.6:
        score -= 0.5
    return score


def _texts_overlap(a: str, b: str) -> bool:
    """True when one candidate is just a part of the other (e.g. 'Calacatta' vs 'Calacatta Gold')."""
    a, b = normalize_text_key(a), normalize_text_key(b)
    return bool(a and b and (a in b or b in a))


def rank_candidates(candidates: List[Dict[str, Any]], ctx: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    scorer = score_name if kind == "name" else score_sku
    scored = sorted(
        ((scorer(c, ctx), c) for c in candidates),
        key=lambda item: (-item[0], item[1]["bbox"][1], item[1]["bbox"][0]),
    )
    seen, unique = set(), []
    for score, c in scored:
        key = c["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((score, c))
    unique = unique[:3]
    results = []
    for score, c in unique:
        conf = sigmoid((score - 1.8) / 1.0) if kind == "name" else sigmoid((score - 1.0) / 0.9)
        results.append({
            "value": c["text"],
            "confidence": round(conf, 2),
            "score": round(score, 2),
            "bbox": c["bbox"],
            "multiline": bool(c.get("multiline")),
            "size": c.get("size", 0.0),
            "bold": c.get("bold", False),
            "source": c.get("source", "native"),
            "original": c.get("original", c["text"]),
            "strippedPrefix": c.get("strippedPrefix", ""),
        })
    # Penalise only genuinely competing candidates (not a name vs its own first line)
    if len(unique) > 1 and unique[0][0] - unique[1][0] < 0.4 and not _texts_overlap(results[0]["value"], results[1]["value"]):
        results[0]["confidence"] = round(results[0]["confidence"] * 0.7, 2)
    return results


def sku_status(value: Optional[str], confidence: float) -> str:
    if not value:
        return "NOT_PRESENT"
    return "EXTRACTED" if confidence >= 0.55 else "NEEDS_REVIEW"


# --------------------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------------------

SAFE_EXTS = ("png", "jpg", "jpeg", "webp")


def load_embedded_image(doc: fitz.Document, xref: int, smask: int) -> Tuple[Optional[bytes], str]:
    """
    Return (bytes, ext) of an embedded image in a browser-safe format.
    Images with a soft mask, CMYK/other colour spaces or exotic codecs are decoded through a
    pixmap so they are not saved black, inverted or unreadable.
    """
    try:
        extracted = doc.extract_image(xref)
    except Exception:
        extracted = None
    if not extracted or not extracted.get("image"):
        return None, "png"
    ext = str(extracted.get("ext", "png")).lower()
    needs_decode = bool(smask) or ext not in SAFE_EXTS or int(extracted.get("colorspace", 3) or 3) not in (1, 3)
    if not needs_decode:
        return extracted["image"], ext
    try:
        pix = fitz.Pixmap(doc, xref)
        if pix.colorspace is not None and pix.n - pix.alpha > 3:  # CMYK etc. -> RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if smask and not pix.alpha:
            try:
                pix = fitz.Pixmap(pix, fitz.Pixmap(doc, smask))
            except Exception:
                pass
        return pix.tobytes("png"), "png"
    except Exception:
        return (extracted["image"], ext) if ext in SAFE_EXTS else (None, "png")


def trim_image_file(path: str, padding: int = 4) -> Tuple[int, int]:
    """
    Trim blank (white/transparent) margins in place and return the final pixel size.
    Never corrupts the file: if anything fails the original file stays untouched.
    """
    with Image.open(path) as im:
        im.load()
        w, h = im.size
        original = im.copy()
    try:
        if original.mode in ("RGBA", "LA", "P"):
            rgba = original.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            rgb = original.convert("RGB")
        if HAS_NUMPY:
            arr = np.asarray(rgb)
            mask = (arr < 245).any(axis=2)
            if not mask.any():
                return w, h
            ys, xs = np.where(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        else:
            bbox = Image.eval(rgb, lambda v: 255 if v < 245 else 0).convert("L").getbbox()
            if not bbox:
                return w, h
        x0, y0 = max(0, bbox[0] - padding), max(0, bbox[1] - padding)
        x1, y1 = min(w, bbox[2] + padding), min(h, bbox[3] + padding)
        new_area, old_area = (x1 - x0) * (y1 - y0), w * h
        # trim only when it removes a meaningful margin and keeps most of the picture
        if new_area >= old_area * 0.98 or new_area < old_area * 0.25:
            return w, h
        cropped = original.crop((x0, y0, x1, y1))
        if path.lower().endswith((".jpg", ".jpeg")) and cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        tmp = path + ".tmp"
        fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}.get(os.path.splitext(path)[1][1:].lower(), "PNG")
        cropped.save(tmp, format=fmt)
        os.replace(tmp, path)
        return cropped.size
    except Exception:
        if os.path.exists(path + ".tmp"):
            try:
                os.remove(path + ".tmp")
            except Exception:
                pass
        return w, h


def is_qr_code_image(image_path: str, pw: int, ph: int, rendered_side: Optional[float] = None) -> bool:
    """
    True for QR codes, including ones too small or blurry to decode. A QR is flagged when:
      - a code is decoded from a roughly square image, OR
      - the image is square, strictly black-and-white (two-tone) and either shows QR finder
        patterns or is drawn small on the page (tile swatches are never two-tone like that).
    """
    if not (HAS_OPENCV and HAS_NUMPY):
        return False
    if max(pw, ph) / max(min(pw, ph), 1) > 1.25:
        return False
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False
        detector = cv2.QRCodeDetector()
        if max(pw, ph) <= 1200:
            try:
                decoded, _, _ = detector.detectAndDecode(image)
                if decoded:
                    return True
            except Exception:
                pass  # decoder can fail on tiny / noisy images; fall through to the two-tone test
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_NEAREST)  # nearest keeps edges crisp
        dark = float((small < 70).mean())
        light = float((small > 185).mean())
        two_tone = dark + light >= 0.9 and dark >= 0.12 and light >= 0.2
        if not two_tone:
            return False
        if rendered_side is not None and rendered_side <= 150:
            return True
        try:
            found = detector.detect(image)
            return bool(found[0]) if isinstance(found, tuple) else bool(found)
        except Exception:
            return False
    except Exception:
        return False


def image_hashes(image_path: str) -> Tuple[str, Optional[str]]:
    with open(image_path, "rb") as image_file:
        sha256 = hashlib.sha256(image_file.read()).hexdigest()
    if not HAS_OPENCV:
        return sha256, None
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return sha256, None
        resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")
        dct = cv2.dct(resized)[:8, :8]
        threshold = float(np.median(dct[1:, 1:]))
        bits = "".join("1" if value > threshold else "0" for value in dct.flatten())
        return sha256, f"{int(bits, 2):016x}"
    except Exception:
        return sha256, None


def deduplicate_images(images: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Remove only EXACT duplicates on the same page: identical bytes drawn at (almost) the same place.
    Similar-looking or identical tiles at different positions are different products and are kept.
    """
    canonical: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []
    for image in images:
        match = next((e for e in canonical
                      if e.get("sha256") == image.get("sha256") and bbox_iou(e["bbox"], image["bbox"]) > 0.5), None)
        if match is None:
            canonical.append(image)
            continue
        # keep the sharper placement
        preferred, other = (image, match) if image["pixelWidth"] * image["pixelHeight"] > match["pixelWidth"] * match["pixelHeight"] else (match, image)
        if preferred is not match:
            canonical.remove(match)
            canonical.append(preferred)
        duplicates.append({**other, "isDuplicate": True, "duplicateOf": preferred["id"]})
    return canonical, duplicates


def extract_embedded_images(
    doc: fitz.Document, page: fitz.Page, page_no: int, job_id: str, out_dir: str,
    profile: Optional[DocProfile] = None,
) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """Priority 1: embedded images, every placement on the page, with decor/background filtering."""
    images: List[Dict[str, Any]] = []
    bleed: List[Dict[str, Any]] = []
    ignored = 0
    page_area = max(1.0, page.rect.width * page.rect.height)
    seen_xrefs = set()

    for img_idx, img_info in enumerate(page.get_images(full=True)):
        xref, smask = img_info[0], img_info[1]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            raw_bytes, ext = load_embedded_image(doc, xref, smask)
            if not raw_bytes:
                continue
            raw_sha1 = hashlib.sha1(raw_bytes).hexdigest()

            saved: Optional[Dict[str, Any]] = None  # one file shared by all placements of this xref
            for r_idx, rect in enumerate(rects):
                rect_w = abs(rect.x1 - rect.x0)
                rect_h = abs(rect.y1 - rect.y0)
                if rect_w < 45 or rect_h < 45:
                    continue
                if rect.y0 < 120 and rect_w < 180 and rect_h < 100 and (rect.x0 > 700 or rect.x1 < 250):
                    continue
                if max(rect_w, rect_h) / max(min(rect_w, rect_h), 1) > 10.0:
                    continue
                if profile and profile.is_decorative_image(raw_sha1, rect, page_area):
                    ignored += 1
                    continue
                if saved is None:
                    filename = f"img_{job_id}_p{page_no}_{xref}_{img_idx}.{ext}"
                    file_path = os.path.join(out_dir, filename)
                    with open(file_path, "wb") as f:
                        f.write(raw_bytes)
                    try:
                        pw, ph = trim_image_file(file_path)
                    except Exception:
                        os.remove(file_path)  # unreadable image data
                        break
                    if pw < 40 or ph < 40 or is_qr_code_image(file_path, pw, ph, max(rect_w, rect_h)):
                        os.remove(file_path)
                        break
                    sha256, phash = image_hashes(file_path)
                    saved = {"filename": filename, "pw": pw, "ph": ph, "sha256": sha256, "phash": phash}

                is_bleed = (rect_w * rect_h) / page_area > 0.92  # near full-page: background art
                (bleed if is_bleed else images).append({
                    "id": f"img_p{page_no}_{xref}_{img_idx}_{r_idx}",
                    "xref": xref,
                    "filename": saved["filename"],
                    "storagePath": f"uploads/catalogues/{job_id}/images/{saved['filename']}",
                    "url": f"/api/files/catalogues/{job_id}/images/{saved['filename']}",
                    "sourcePage": page_no,
                    "bbox": [round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1)],
                    "pixelWidth": saved["pw"],
                    "pixelHeight": saved["ph"],
                    "method": "EMBEDDED",
                    "aspectRatio": round(saved["pw"] / max(saved["ph"], 1), 2),
                    "sha256": saved["sha256"],
                    "phash": saved["phash"],
                })
        except Exception:
            continue
    return images, ignored, bleed


def extract_composite_crops(
    page_img_path: str, page_no: int, job_id: str, out_dir: str, page_rect: fitz.Rect
) -> List[Dict[str, Any]]:
    """Priority 2: detect tile regions on raster/composite pages (with overlap suppression)."""
    crops: List[Dict[str, Any]] = []
    if not HAS_OPENCV:
        return crops
    try:
        img = cv2.imread(page_img_path)
        if img is None:
            return crops
        h, w, _ = img.shape
        scale_x = (page_rect.x1 - page_rect.x0) / float(w)
        scale_y = (page_rect.y1 - page_rect.y0) / float(h)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 7, 50, 50)
        edges = cv2.Canny(blurred, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (w * h) * 0.02
        max_area = (w * h) * 0.75

        boxes = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if not (min_area <= cw * ch <= max_area):
                continue
            if not (0.25 <= float(cw) / float(max(ch, 1)) <= 4.0):
                continue
            boxes.append([x, y, x + cw, y + ch])

        # non-max suppression: keep larger box when two overlap heavily
        boxes.sort(key=lambda b: -bbox_area(b))
        kept: List[List[float]] = []
        for b in boxes:
            if all(bbox_iou(b, k) < 0.5 and bbox_contained_ratio(b, k) < 0.8 for k in kept):
                kept.append(b)
        kept.sort(key=lambda b: (b[1], b[0]))

        for crop_idx, (x0, y0, x1, y1) in enumerate(kept):
            cw, ch = x1 - x0, y1 - y0
            tile_crop = img[y0:y1, x0:x1]
            if tile_crop.size == 0:
                continue
            crop_filename = f"crop_{job_id}_p{page_no}_{crop_idx}.jpg"
            crop_path = os.path.join(out_dir, crop_filename)
            cv2.imwrite(crop_path, tile_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if is_qr_code_image(crop_path, int(cw), int(ch)):
                os.remove(crop_path)
                continue
            sha256, phash = image_hashes(crop_path)
            crops.append({
                "id": f"crop_p{page_no}_{crop_idx}",
                "xref": None,
                "filename": crop_filename,
                "storagePath": f"uploads/catalogues/{job_id}/images/{crop_filename}",
                "url": f"/api/files/catalogues/{job_id}/images/{crop_filename}",
                "sourcePage": page_no,
                "bbox": [round(x0 * scale_x, 1), round(y0 * scale_y, 1), round(x1 * scale_x, 1), round(y1 * scale_y, 1)],
                "pixelWidth": int(cw),
                "pixelHeight": int(ch),
                "method": "PAGE_CROP",
                "aspectRatio": round(cw / max(ch, 1), 2),
                "sha256": sha256,
                "phash": phash,
            })
    except Exception:
        pass
    return crops


# --------------------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------------------


def extract_tile_specifications(text: str, unit_hint: Optional[str] = None) -> Dict[str, Any]:
    specs: Dict[str, Any] = {
        "size": None, "sizeRaw": None, "sizeUnitKnown": True, "thickness": None,
        "surface": None, "material": None, "color": None,
        "piecesPerBox": None, "coveragePerBox": None, "mrp": None, "sellingPrice": None,
        "hsnCode": "6907", "gstRate": 18, "unit": "box",
    }

    dimension = parse_dimension(text, unit_hint or detect_unit_hint(text))
    if dimension:
        specs["size"] = dimension["size"]
        specs["sizeRaw"] = dimension["raw"]
        specs["sizeUnitKnown"] = dimension["unit"] is not None
        specs["thickness"] = dimension["thickness"]

    surf_explicit = re.search(r"Surface\s*(?:/\s*Finish)?\s*[:.-]\s*([^\n·,;]+)", text, re.IGNORECASE)
    if surf_explicit:
        specs["surface"] = surf_explicit.group(1).strip()
    else:
        for surface in sorted(KNOWN_SURFACES, key=len, reverse=True):
            if re.search(r"\b" + re.escape(surface) + r"\b", text, re.IGNORECASE):
                specs["surface"] = surface
                break

    mat_explicit = re.search(r"Material\s*[:.-]\s*([^\n·,;]+)", text, re.IGNORECASE)
    if mat_explicit:
        specs["material"] = mat_explicit.group(1).strip()
    else:
        for mat in sorted(KNOWN_MATERIALS, key=len, reverse=True):
            if re.search(r"\b" + re.escape(mat) + r"\b", text, re.IGNORECASE):
                specs["material"] = mat
                break

    pcs_match = PCS_BOX_REGEX.search(text)
    if pcs_match:
        specs["piecesPerBox"] = int(pcs_match.group(1))

    cov_match = COVERAGE_REGEX.search(text)
    if cov_match:
        value = float(cov_match.group(1))
        if re.search(r"ft", cov_match.group(2), re.IGNORECASE):
            value = round(value * SQFT_TO_SQM, 3)
        specs["coveragePerBox"] = value

    price_match = PRICE_REGEX.search(text)
    if price_match:
        try:
            p_val = float(price_match.group(1).replace(",", ""))
            specs["mrp"] = p_val
            specs["sellingPrice"] = p_val
        except ValueError:
            pass
    return specs


def detect_brand(text: str, fallback_brand: Optional[str] = None) -> Optional[str]:
    for brand in KNOWN_BRANDS:
        if re.search(r"\b" + re.escape(brand) + r"\b", text, re.IGNORECASE):
            return brand
    return fallback_brand


# --------------------------------------------------------------------------------------
# Layout: grouping lines, assigning text blocks to images
# --------------------------------------------------------------------------------------


def cluster_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Union-find clustering of lines into text blocks."""
    n = len(lines)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        a = lines[i]["bbox"]
        for j in range(i + 1, n):
            b = lines[j]["bbox"]
            ha, hb = a[3] - a[1], b[3] - b[1]
            h = max(ha, hb, 1.0)
            v_overlap = min(a[3], b[3]) - max(a[1], b[1])
            x_overlap = min(a[2], b[2]) - max(a[0], b[0])
            min_w = max(1.0, min(a[2] - a[0], b[2] - b[0]))
            if v_overlap > 0.5 * min(ha, hb):
                # same row: link when horizontally close ("Size:" ... "600x1200")
                if max(0.0, max(a[0], b[0]) - min(a[2], b[2])) <= max(8.0, 0.8 * h):
                    union(i, j)
                continue
            v_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
            if v_gap <= max(10.0, 1.6 * h) and (
                x_overlap / min_w >= 0.25 or abs(a[0] - b[0]) <= 8
                or abs((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) <= 10
            ):
                union(i, j)

    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for i in range(n):
        buckets[find(i)].append(lines[i])
    groups = []
    for members in buckets.values():
        members.sort(key=line_sort_key)
        groups.append({"lines": members, "bbox": bbox_union([m["bbox"] for m in members])})
    groups.sort(key=lambda g: (g["bbox"][1], g["bbox"][0]))
    return groups


def group_signals(group: Dict[str, Any], profile: DocProfile, page_h: float) -> Dict[str, bool]:
    has_code = has_spec = name_like = False
    for ln in group["lines"]:
        if ln.get("template"):
            continue
        t = ln["text"]
        if SIZE_REGEX.search(t) or PCS_BOX_REGEX.search(t) or COVERAGE_REGEX.search(t) or PRICE_REGEX.search(t):
            has_spec = True
        if sku_tokens_from_line(t):
            has_code = True
        cand = make_name_candidate(ln, profile, page_h)
        if cand:
            nw = len(cand["text"].split())
            if nw <= 8 and not cand["text"].endswith((".", ":", ";")) and not any(
                re.search(p, cand["text"], re.IGNORECASE) for p in INSTRUCTIONAL_PATTERNS
            ):
                name_like = True
    return {"code": has_code, "spec": has_spec, "name": name_like}


def match_cost(gbox: List[float], ibox: List[float]) -> Tuple[float, str]:
    """
    Lower is better. Returns (cost, relation) where relation is where the text sits relative to the
    image: 'inside', 'below', 'beside', 'above' or 'other'. Text below / beside an image is good,
    text above is penalised (it normally belongs to the image above it).
    """
    gx0, gy0, gx1, gy1 = gbox
    ix0, iy0, ix1, iy1 = ibox
    iw, ih = max(1.0, ix1 - ix0), max(1.0, iy1 - iy0)
    gapx, gapy = bbox_gap(gbox, ibox)
    gap = math.hypot(gapx, gapy) / max(iw, ih)
    xov = max(0.0, min(gx1, ix1) - max(gx0, ix0)) / max(1.0, min(gx1 - gx0, iw))
    yov = max(0.0, min(gy1, iy1) - max(gy0, iy0)) / max(1.0, min(gy1 - gy0, ih))
    if xov > 0.3 and yov > 0.3:
        direction, relation = 0.2, "inside"
    elif gy0 >= iy1 - 0.15 * ih and xov > 0.15:
        direction, relation = 0.0, "below"
    elif yov > 0.3 and (gx0 >= ix1 - 0.1 * iw or gx1 <= ix0 + 0.1 * iw):
        direction, relation = 0.35, "beside"
    elif gy1 <= iy0 + 0.15 * ih:
        direction, relation = 1.0, "above"
    else:
        direction, relation = 0.7, "other"
    column = 0.0 if (xov > 0.2 or yov > 0.3) else 0.6
    return min(6.0, gap * 1.5 + direction + column), relation


def cost_to_conf(cost: float) -> float:
    return round(clamp(math.exp(-cost / 2.5), 0.05, 0.97), 2)


def assign_groups_to_images(
    groups: List[Dict[str, Any]], images: List[Dict[str, Any]]
) -> Tuple[Dict[int, List[int]], Dict[int, float], List[int]]:
    """
    Image-centric assignment: every text block goes to its best image (many blocks per image allowed,
    e.g. a name block and a spec block). Returns (image_idx -> group idxs, image_idx -> confidence,
    unassigned group idxs).
    """
    per_image: Dict[int, List[Tuple[float, str, int, bool]]] = defaultdict(list)
    unassigned: List[int] = []
    for gi, g in enumerate(groups):
        scored = sorted((*match_cost(g["bbox"], img["bbox"]), ii) for ii, img in enumerate(images))
        if not scored or scored[0][0] > MAX_GROUP_COST:
            unassigned.append(gi)
            continue
        cost, relation, ii = scored[0]
        ambiguous = len(scored) > 1 and scored[1][0] - cost < 0.15 and cost < 1.0
        per_image[ii].append((cost, relation, gi, ambiguous))

    members: Dict[int, List[int]] = {}
    conf: Dict[int, float] = {}
    for ii, items in per_image.items():
        # text above the image is used only if nothing sits below / beside / inside it
        if any(rel != "above" for _, rel, _, _ in items):
            unassigned.extend(gi for _, rel, gi, _ in items if rel == "above")
            items = [it for it in items if it[1] != "above"]
        members[ii] = [gi for _, _, gi, _ in items]
        best = min(items, key=lambda it: it[0])
        c = cost_to_conf(best[0])
        conf[ii] = round(c * 0.8, 2) if best[3] else c
    return members, conf, unassigned


def detect_layout(images: List[Dict[str, Any]], n_codes: int, has_size: bool, n_name_groups: int,
                  n_spec_groups: int) -> str:
    """single = one product on the page (e.g. big photo + swatches); grid = many products."""
    areas = [bbox_area(img["bbox"]) for img in images]
    similar = 0
    if areas:
        med = statistics.median(areas)
        similar = sum(1 for a in areas if 0.6 * med <= a <= 1.6 * med)
    if n_codes >= 2:
        return "grid"
    if has_size and n_codes <= 1 and n_spec_groups <= 1:
        return "single"  # one spec block on the page = one product, however many swatches it shows
    if similar >= 2 and n_name_groups >= 2:
        return "grid"
    if has_size and n_codes <= 1:
        return "single"
    if similar >= 3:
        return "grid"
    return "single" if len(images) <= 1 else "grid"


def flat_fraction(path: str) -> float:
    """Share of 8x8 blocks that are almost uniform. High for 3D corner/edge shots, low for flat texture swatches."""
    if not HAS_NUMPY:
        return 0.0
    try:
        with Image.open(path) as im:
            gray = im.convert("L")
            gray.thumbnail((256, 256))
            arr = np.asarray(gray, dtype="float32")
        h, w = (arr.shape[0] // 8) * 8, (arr.shape[1] // 8) * 8
        if h < 8 or w < 8:
            return 0.0
        blocks = arr[:h, :w].reshape(h // 8, 8, w // 8, 8).std(axis=(1, 3))
        return float((blocks < 2.5).mean())
    except Exception:
        return 0.0


def rank_single_page_images(images: List[Dict[str, Any]], size_text: Optional[str], page_w: float,
                            out_dir: str) -> List[Tuple[float, Dict[str, Any]]]:
    """Best swatch first: its shape matches the tile size, it is a flat texture (not a 3D corner shot)."""
    size_ratio = None
    dimension = parse_dimension(size_text or "")
    if dimension and dimension["height"] > 0:
        size_ratio = dimension["width"] / dimension["height"]
    max_area = max((img["pixelWidth"] * img["pixelHeight"] for img in images), default=1) or 1
    ranked = []
    for img in images:
        ar = max(img.get("aspectRatio", 1.0), 0.01)
        score = 0.3 * (img["pixelWidth"] * img["pixelHeight"]) / max_area
        if size_ratio:
            score -= 1.5 * min(abs(math.log(ar / size_ratio)), abs(math.log(ar * size_ratio)))
        score -= 2.0 * flat_fraction(os.path.join(out_dir, img["filename"]))
        if img["bbox"][0] > page_w * 0.45 and 1.3 <= ar <= 2.7:
            score += 0.5  # wide swatch on the right half (common template)
        ranked.append((score, img))
    ranked.sort(key=lambda item: -item[0])
    return ranked


def region_around_image(ibox: List[float]) -> List[float]:
    iw, ih = ibox[2] - ibox[0], ibox[3] - ibox[1]
    return [ibox[0] - 0.1 * iw, ibox[3] - 2, ibox[2] + 0.1 * iw, ibox[3] + 0.6 * ih]


# --------------------------------------------------------------------------------------
# Page preparation (pass 2) - builds cards, no final naming yet
# --------------------------------------------------------------------------------------


def _new_card(lines, bbox, image, image_conf, layout, alts=None) -> Dict[str, Any]:
    return {"lines": lines, "bbox": bbox, "image": image, "imageConf": image_conf,
            "alts": alts or [], "layout": layout}


def prepare_page(
    doc: fitz.Document, page_no: int, job_id: str, out_dir: str, profile: DocProfile, use_ocr: bool = True
) -> Dict[str, Any]:
    page = doc[page_no - 1]
    page_rect = page.rect

    preview_name = f"page_{job_id}_p{page_no}.png"
    preview_path = os.path.join(out_dir, preview_name)
    page.get_pixmap(dpi=150).save(preview_path)
    preview_url = f"/api/files/catalogues/{job_id}/images/{preview_name}"

    lines = extract_native_lines(page)
    native_chars = sum(len(l["text"]) for l in lines)

    images, ignored, bleed = extract_embedded_images(doc, page, page_no, job_id, out_dir, profile)
    if not images:
        if bleed and native_chars >= 50:
            images = bleed  # normal page with one big photo: keep the real embedded image
        else:
            images = extract_composite_crops(preview_path, page_no, job_id, out_dir, page_rect)
            if not images and bleed:
                images = bleed
    images, duplicate_images = deduplicate_images(images)

    # OCR: no native text, OR images exist but no readable name text (names baked into art)
    ocr_used = False
    mark_template_lines(lines, profile)
    if use_ocr and HAS_PADDLEOCR:
        eligible = sum(1 for l in lines if make_name_candidate(l, profile, page_rect.height))
        if native_chars < 50 or (len(images) >= 2 and eligible == 0):
            ocr_lines = run_paddle_ocr(preview_path, [0.0, 0.0, page_rect.width, page_rect.height])
            if ocr_lines:
                lines = ocr_lines if native_chars < 50 else merge_ocr_lines(lines, ocr_lines)
                ocr_used = True
                mark_template_lines(lines, profile)

    total_text = "\n".join(l["text"] for l in lines).strip()
    body_size = char_weighted_mode([l for l in lines if not l["template"]]) or profile.body_size
    if not body_size:
        heights = [l["bbox"][3] - l["bbox"][1] for l in lines]
        body_size = statistics.median(heights) * 0.8 if heights else 10.0

    state: Dict[str, Any] = {
        "pageNo": page_no, "width": page_rect.width, "height": page_rect.height,
        "isScanned": native_chars < 40 and not ocr_used, "ocrUsed": ocr_used,
        "previewUrl": preview_url, "previewPath": preview_path, "lines": lines,
        "totalText": total_text, "images": images, "duplicateImages": duplicate_images,
        "ignoredImages": ignored, "bodySize": body_size, "cards": [], "layout": "none",
        "nonProduct": False, "orphanImageIds": [],
    }

    # Non-product pages (cover / divider / index)
    size_on_page = SIZE_REGEX.search(total_text)
    low = total_text.lower()
    non_product = False
    if any(k in low for k in ["index", "table of contents", "collection overview", "design index"]):
        non_product = True
    elif not size_on_page and len(lines) <= 8 and any(w in low for w in ["collection", "edition", "series", "surface"]):
        non_product = True
    elif len(total_text) < 30 and not images:
        non_product = True
    if non_product and not size_on_page:
        state["nonProduct"] = True
        return state

    groups = cluster_lines(lines)
    kept = []
    for g in groups:
        sig = group_signals(g, profile, page_rect.height)
        text_len = sum(len(l["text"]) for l in g["lines"] if not l["template"])
        if not (sig["code"] or sig["spec"] or sig["name"]):
            continue
        if text_len > 400 and not sig["code"] and not sig["spec"]:
            continue  # description paragraph
        g["signals"] = sig
        kept.append(g)
    n_codes = len({t.upper() for l in lines if not l["template"] for t, _ in sku_tokens_from_line(l["text"])})
    n_name_groups = sum(1 for g in kept if g["signals"]["name"])
    n_spec_groups = sum(1 for g in kept if g["signals"]["spec"])
    layout = detect_layout(images, n_codes, bool(size_on_page), n_name_groups, n_spec_groups)
    state["layout"] = layout

    # ---------------- single-product page ----------------
    if layout == "single":
        ranked_imgs = rank_single_page_images(images, extract_tile_specifications(total_text)["size"], page_rect.width, out_dir)
        best = ranked_imgs[0][1] if ranked_imgs else None
        margin = ranked_imgs[0][0] - ranked_imgs[1][0] if len(ranked_imgs) > 1 else 1.0
        alts = [(img, 0.70) for _, img in ranked_imgs[1:]]
        img_conf = (0.85 if margin >= 0.3 else 0.70) if best else 0.0
        card = _new_card(lines, bbox_union([l["bbox"] for l in lines]) if lines else [0, 0, 1, 1],
                         best, img_conf, "single", alts)

        # Name not clearly readable from the text layer: OCR the page once and re-read it
        if use_ocr and HAS_PADDLEOCR and not ocr_used:
            ranked = rank_candidates(build_name_candidates(lines, profile, page_rect.height),
                                     _card_ctx(card, state, profile), "name")
            if not ranked or ranked[0]["confidence"] < 0.55:
                ocr_lines = run_paddle_ocr(preview_path, [0.0, 0.0, page_rect.width, page_rect.height])
                if ocr_lines:
                    lines = merge_ocr_lines(lines, ocr_lines)
                    mark_template_lines(lines, profile)
                    card["lines"], state["lines"] = lines, lines
                    card["bbox"] = bbox_union([l["bbox"] for l in lines])
                    state["totalText"] = "\n".join(l["text"] for l in lines).strip()
                    state["ocrUsed"] = True
        state["cards"].append(card)
        return state

    # ---------------- grid page ----------------
    cards: List[Dict[str, Any]] = []
    members, confs, unassigned = assign_groups_to_images(kept, images) if images else ({}, {}, list(range(len(kept))))

    for ii in sorted(members, key=lambda i: (images[i]["bbox"][1], images[i]["bbox"][0])):
        gs = [kept[gi] for gi in members[ii]]
        cards.append(_new_card(
            [l for g in gs for l in g["lines"]], bbox_union([g["bbox"] for g in gs]),
            images[ii], confs[ii], "grid"))

    # Tile-like images that received no text are kept (region OCR is tried later), not dropped.
    assigned_ids = {c["image"]["id"] for c in cards}
    if assigned_ids:
        med_area = statistics.median(bbox_area(c["image"]["bbox"]) for c in cards)
        for img in images:
            if img["id"] not in assigned_ids and 0.5 * med_area <= bbox_area(img["bbox"]) <= 2.0 * med_area:
                cards.append(_new_card([], list(img["bbox"]), img, 0.3, "grid"))
                assigned_ids.add(img["id"])
    elif images:  # no text matched any image: one card per image
        cards = [_new_card([], list(img["bbox"]), img, 0.3, "grid") for img in images]
        assigned_ids = {img["id"] for img in images}

    # Leftover text blocks: attach to the nearest card, or make a text-only card if it is a real product
    for gi in unassigned:
        g = kept[gi]
        sig = g["signals"]
        best_card, best_gap = None, 1e9
        for card in cards:
            gx, gy = bbox_gap(g["bbox"], card["bbox"])
            d = math.hypot(gx, gy)
            if d < best_gap:
                best_card, best_gap = card, d
        if best_card is not None and best_gap <= 70 and not sig["name"]:
            best_card["lines"].extend(g["lines"])
            best_card["bbox"] = bbox_union([best_card["bbox"], g["bbox"]])
        elif sig["code"] or sig["spec"] or (sig["name"] and not images):
            cards.append(_new_card(list(g["lines"]), list(g["bbox"]), None, 0.0, "grid"))

    # Per-card region OCR when no plausible name text exists
    if use_ocr and HAS_PADDLEOCR and not ocr_used:
        budget = 12
        for card in cards:
            if budget <= 0:
                break
            if card["image"] is None or any(make_name_candidate(l, profile, page_rect.height) for l in card["lines"]):
                continue
            budget -= 1
            new_lines = ocr_region(preview_path, page_rect, region_around_image(card["image"]["bbox"]))
            new_lines = [l for l in new_lines if not any(bbox_iou(l["bbox"], e["bbox"]) > 0.3 for e in card["lines"])]
            if new_lines:
                mark_template_lines(new_lines, profile)
                card["lines"].extend(new_lines)
                card["bbox"] = bbox_union([card["bbox"]] + [l["bbox"] for l in new_lines])
                state["ocrUsed"] = True

    # Alternate images only for weak cards (offering them for every card caused wrong swaps)
    for card in cards:
        if card["image"] is not None and card["imageConf"] >= 0.6:
            continue
        ranked = sorted(
            ((match_cost(card["bbox"], img["bbox"])[0], img) for img in images
             if card["image"] is None or img["id"] != card["image"]["id"]),
            key=lambda item: item[0])
        card["alts"] = [(img, round(cost_to_conf(c) * 0.8, 2)) for c, img in ranked[:2] if c <= 3.5]

    cards.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    state["cards"] = cards
    state["orphanImageIds"] = [img["id"] for img in images if img["id"] not in assigned_ids]
    return state


# --------------------------------------------------------------------------------------
# Learning the document's name style from confident cards
# --------------------------------------------------------------------------------------


def _card_ctx(card: Dict[str, Any], state: Dict[str, Any], profile: DocProfile, sku_bbox=None) -> Dict[str, Any]:
    img = card.get("image")
    return {
        "body": state["bodySize"] or profile.body_size or 10.0,
        "image": img["bbox"] if img else None,
        "sku_bbox": sku_bbox,
        "profile": profile,
    }


def learn_name_style(states: List[Dict[str, Any]], profile: DocProfile) -> None:
    """Learn the dominant name font size / weight from cards whose name is already clear."""
    sizes: List[float] = []
    bolds: List[bool] = []
    for st in states:
        if st.get("nonProduct"):
            continue
        for card in st["cards"]:
            cands = build_name_candidates(card["lines"], profile, st["height"])
            if not cands:
                continue
            ranked = rank_candidates(cands, _card_ctx(card, st, profile), "name")
            if not ranked or ranked[0]["confidence"] < 0.8 or ranked[0]["source"] != "native":
                continue
            if ranked[0]["size"] > 0:
                sizes.append(round(ranked[0]["size"] * 2) / 2.0)
                bolds.append(bool(ranked[0]["bold"]))
    if len(sizes) >= 8:
        size, count = Counter(sizes).most_common(1)[0]
        if count / len(sizes) >= 0.5:
            profile.name_size = size
            profile.name_bold = sum(bolds) / len(bolds) >= 0.5


# --------------------------------------------------------------------------------------
# Product building (pass 3)
# --------------------------------------------------------------------------------------


def stacked_label_value(lines: List[Dict[str, Any]], label: str) -> Optional[str]:
    """
    Value printed directly BELOW a label in a spec strip ("RANDOM" over "06", "SURFACE" over "MATT").
    Reading order puts all labels first and all values after, so a regex over the text cannot pair them.
    """
    labels = [l for l in lines if normalize_text_key(l["text"]) == label]
    for lab in labels:
        lx0, ly0, lx1, ly1 = lab["bbox"]
        height = max(1.0, ly1 - ly0)
        mid = (ly0 + ly1) / 2.0
        # Below = starts under the label's midline (line boxes overlap with tight leading) and no more
        # than about one and a half lines further down, left-aligned with the label.
        below = [
            l for l in lines
            if l is not lab and l["bbox"][1] > mid and l["bbox"][1] - ly1 <= height * 1.5
            and abs(l["bbox"][0] - lx0) <= max(8.0, 0.3 * (lx1 - lx0))
        ]
        if below:
            return min(below, key=lambda l: l["bbox"][1])["text"].strip()
    return None


def build_product(card: Dict[str, Any], state: Dict[str, Any], profile: DocProfile, doc_brand: Optional[str]) -> Dict[str, Any]:
    page_no = state["pageNo"]
    lines = card["lines"]
    card_text = "\n".join(l["text"] for l in sorted(lines, key=line_sort_key))
    total_text = state["totalText"]
    page_brand = detect_brand(total_text, doc_brand)
    best_img = card.get("image")
    img_conf = card.get("imageConf", 0.0)

    # SKU first (its position helps the name ranking)
    base_ctx = _card_ctx(card, state, profile)
    sku_rank = rank_candidates(extract_sku_candidates(lines, base_ctx["body"]), base_ctx, "sku")
    sku_text = sku_rank[0]["value"] if sku_rank else None
    sku_conf = sku_rank[0]["confidence"] if sku_rank else 0.0
    sku = sku_text.upper() if sku_text else None
    sku_state = sku_status(sku, sku_conf)

    ctx = _card_ctx(card, state, profile, sku_rank[0]["bbox"] if sku_rank else None)
    ctx["spec_boxes"] = [l["bbox"] for l in lines if not l.get("template") and (
        SIZE_REGEX.search(l["text"]) or PCS_BOX_REGEX.search(l["text"]) or LABEL_RE.match(l["text"].lower()))]
    if sku:
        ctx["exclude_text"] = sku.lower()
    # The SKU is an identifier, never a name: a candidate that is just the code is not considered.
    sku_key = normalize_text_key(sku or "")
    name_cands = [c for c in build_name_candidates(lines, profile, state["height"])
                  if not sku_key or normalize_text_key(c["text"]) != sku_key]
    if not name_cands and not sku:
        # Everything was template text and there is no code to link this card to another page with:
        # fall back to the template text rather than leave the name empty.
        relaxed = [{**l, "template": False} for l in lines if not is_page_number_line(l["text"])]
        name_cands = build_name_candidates(relaxed, profile, state["height"])
    # With an SKU and no name text, the name stays unresolved: grouping by SKU later borrows a real
    # name from another page that carries the same code, and page furniture ("TAP TO INDEX") is
    # never promoted to a name.
    name_rank = rank_candidates(name_cands, ctx, "name")
    if name_rank:
        name, name_conf = name_rank[0]["value"], name_rank[0]["confidence"]
        name_original = name_rank[0]["original"]
        stripped_prefix = name_rank[0]["strippedPrefix"]
    else:
        name, name_conf, name_original, stripped_prefix = UNRESOLVED_NAME, 0.0, UNRESOLVED_NAME, ""
    name = name[:60]
    if card.get("layout") == "single" and name_rank and name != UNRESOLVED_NAME:
        margin = name_rank[0]["score"] - (name_rank[1]["score"] if len(name_rank) > 1 else -9.0)
        if margin >= 0.8 and name_rank[0]["score"] >= 1.0:
            name_conf = max(name_conf, 0.6)  # the only product on the page and a clear winner

    unit_hint = detect_unit_hint(card_text) or detect_unit_hint(total_text)
    specs = extract_tile_specifications(card_text, unit_hint)
    if not specs["size"]:
        page_specs = extract_tile_specifications(total_text, unit_hint)
        for key in ("size", "sizeRaw", "sizeUnitKnown", "thickness", "surface", "material"):
            if not specs.get(key) and page_specs.get(key) is not None:
                specs[key] = page_specs[key]

    category = "Floor Tiles"
    subcategory = specs.get("material") or ("Glazed Vitrified" if card.get("layout") == "single" else "Vitrified Tiles")
    scope_low = (card_text + "\n" + total_text).lower()
    if "wall" in scope_low:
        category, subcategory = "Wall Tiles", "Ceramic Wall"
    elif "outdoor" in card_text.lower() or "parking" in card_text.lower() or (
        card.get("layout") == "single" and ("outdoor" in scope_low or "parking" in scope_low)
    ):
        category, subcategory = "Outdoor Tiles", "Heavy Duty"
    elif "slab" in scope_low or (specs.get("size") and "1600" in specs["size"]):
        category, subcategory = "Slab Tiles", "Large Format"

    # Calibrated confidence: weakest of the components that exist
    parts = [name_conf, img_conf if best_img else 0.3]
    if sku:
        parts.append(sku_conf)
    overall = round(min(parts), 2)

    warnings = []
    if sku_state != "EXTRACTED":
        warnings.append("Product SKU could not be confidently identified; manual review required.")
    if name == UNRESOLVED_NAME or name_conf < 0.55:
        warnings.append("Product name could not be confidently identified; manual review required.")
    if not best_img:
        warnings.append("No matching product image found on page.")
    elif img_conf < 0.6:
        warnings.append(f"Low image matching confidence ({int(img_conf * 100)}%); verify image.")
    if not specs["size"]:
        warnings.append("Size/dimension not detected in text.")
    elif not specs.get("sizeUnitKnown", True):
        warnings.append(f"Size '{specs['sizeRaw']}' has no unit (mm / cm / inch); verify the size.")
    if stripped_prefix:
        warnings.append(f"Template prefix '{stripped_prefix}' was removed from the name (original kept).")
    if len(name_rank) > 1 and name_rank[0]["score"] - name_rank[1]["score"] < 0.4 \
            and not _texts_overlap(name_rank[0]["value"], name_rank[1]["value"]):
        warnings.append("Several similar name candidates; verify the product name.")

    images_payload = []
    if best_img:
        images_payload.append({**best_img, "isPrimary": True, "confidence": img_conf})
    # On a single-product page every other image (faces, close-ups) shows this same tile, so they
    # belong to the swatch. On a grid page they are only guesses at another product's picture.
    same_product = card.get("layout") == "single"
    for alt_img, alt_conf in card.get("alts", []):
        images_payload.append({**alt_img, "isPrimary": False, "isAlternate": not same_product,
                               "isSwatchImage": same_product, "confidence": alt_conf})

    attributes = []
    if specs.get("size"):
        attributes.append({"name": "Size", "value": specs["size"]})
    if specs.get("thickness"):
        attributes.append({"name": "Thickness", "value": specs["thickness"]})
    if specs.get("surface"):
        attributes.append({"name": "Surface / Finish", "value": specs["surface"]})
    if specs.get("material"):
        attributes.append({"name": "Material", "value": specs["material"]})
    if specs.get("piecesPerBox"):
        attributes.append({"name": "Pieces per Box", "value": str(specs["piecesPerBox"])})
    if specs.get("coveragePerBox"):
        attributes.append({"name": "Coverage per Box", "value": f"{specs['coveragePerBox']} Sqm"})
    rf_match = re.search(r"\bRandom\s*[-:]?\s*(\d+)\b", card_text, re.IGNORECASE)
    faces = rf_match.group(1) if rf_match else stacked_label_value(lines, "random")
    if faces and faces.isdigit():
        attributes.append({"name": "Random Faces", "value": f"{int(faces)} Faces"})

    return {
        "sourcePage": page_no,
        "sourcePages": [page_no],
        "productCode": sku,
        "productName": name,
        "productNameOriginal": name_original,
        "strippedNamePrefix": stripped_prefix,
        "brandName": page_brand,
        "categoryName": category,
        "subcategory": subcategory,
        "size": specs.get("size"),
        "sizeRaw": specs.get("sizeRaw"),
        "surface": specs.get("surface"),
        "color": specs.get("color"),
        "material": specs.get("material"),
        "unit": specs.get("unit") or "box",
        "piecesPerBox": specs.get("piecesPerBox"),
        "coveragePerBox": specs.get("coveragePerBox"),
        "mrp": specs.get("mrp") or 0.0,
        "purchasePrice": 0.0,
        "sellingPrice": specs.get("sellingPrice") or 0.0,
        "gstRate": specs.get("gstRate") or 18.0,
        "hsnCode": specs.get("hsnCode") or "6907",
        "description": f"{name} - {specs.get('size') or ''} {specs.get('surface') or ''}".strip(),
        "confidence": overall,
        "confidenceBreakdown": {
            "name": round(name_conf, 2),
            "image": round(img_conf if best_img else 0.0, 2),
            "sku": round(sku_conf, 2) if sku else None,
        },
        "needsReview": bool(overall < 0.6 or warnings),
        "layoutMode": card.get("layout"),
        "reviewWarnings": warnings,
        "ocrNameCandidates": [
            {k: r[k] for k in ("value", "confidence", "bbox", "multiline", "original", "strippedPrefix")}
            for r in name_rank
        ],
        "ocrSkuCandidates": [
            {k: r[k] for k in ("value", "confidence", "bbox", "multiline")} for r in sku_rank
        ],
        "skuStatus": sku_state,
        "attributes": attributes,
        "rawText": card_text,
        "primaryImageUrl": best_img["url"] if best_img else None,
        "images": images_payload,
    }


def finalize_page(state: Dict[str, Any], profile: DocProfile, doc_brand: Optional[str]) -> Dict[str, Any]:
    products = []
    if not state.get("nonProduct"):
        products = [build_product(card, state, profile, doc_brand) for card in state["cards"]]
    orphan_ids = set(state.get("orphanImageIds", []))
    orphans = [
        {k: img[k] for k in ("id", "filename", "url", "sourcePage", "bbox", "pixelWidth", "pixelHeight", "method", "sha256")}
        for img in state["images"] if img["id"] in orphan_ids
    ]
    return {
        "pageNo": state["pageNo"],
        "isScanned": state["isScanned"],
        "ocrUsed": state.get("ocrUsed", False),
        "layout": state.get("layout"),
        "previewUrl": state["previewUrl"],
        "brand": detect_brand(state["totalText"], doc_brand),
        "products": products,
        "orphanImages": orphans,
        "extractedImagesCount": len(state["images"]),
    }


# --------------------------------------------------------------------------------------
# Document-level post-processing
# --------------------------------------------------------------------------------------


SWATCH_NAME_MIN_CONFIDENCE = 0.55


def canonical_sku(code: Optional[str]) -> Optional[str]:
    sku = re.sub(r"[^A-Z0-9]", "", (code or "").upper())
    return sku if len(sku) >= 3 else None


def strip_view_qualifiers(name: str) -> str:
    """'Calacatta Gold (Close-up)' -> 'Calacatta Gold'. Returns the input if nothing usable is left."""
    stripped = VIEW_QUALIFIER_REGEX.sub(" ", name or "")
    stripped = re.sub(r"[\(\[\{]\s*[\)\]\}]", " ", stripped)  # brackets left empty
    stripped = re.sub(r"\s+", " ", stripped).strip(" -–|:,/()[]")
    return stripped if len(stripped) >= 3 else (name or "")


def swatch_name_key(p: Dict[str, Any]) -> Optional[str]:
    name = p.get("productName") or ""
    if not name or name == UNRESOLVED_NAME:
        return None
    key = normalize_text_key(strip_view_qualifiers(name))
    return key or None


def _size_key(p: Dict[str, Any]) -> Optional[str]:
    return re.sub(r"[^0-9x]", "", (p.get("size") or "").lower()) or None


def _surface_key(p: Dict[str, Any]) -> Optional[str]:
    return normalize_text_key(p.get("surface") or "") or None


def product_key(p: Dict[str, Any]) -> Optional[str]:
    sku = canonical_sku(p.get("productCode"))
    if sku:
        return "sku:" + sku
    name = swatch_name_key(p)
    return f"nm:{name}|{_size_key(p) or ''}" if name else None


def _names_compatible(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    na, nb = swatch_name_key(a), swatch_name_key(b)
    if not na or not nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= 0.7


def _swatch_rank(p: Dict[str, Any]) -> Tuple:
    """Which record of a swatch group leads it: has an SKU, a resolved name, confidence, an image."""
    return (
        0 if canonical_sku(p.get("productCode")) else 1,
        # The record printed with the specs (size) and the flat tile faces leads, so a face - not a
        # room preview - is the swatch's primary image. A name found elsewhere is borrowed later.
        0 if p.get("size") else 1,
        -sum(1 for img in p.get("images", []) if not img.get("isAlternate")),
        0 if swatch_name_key(p) else 1,
        -float(p.get("confidence") or 0.0),
        0 if p.get("primaryImageUrl") else 1,
        p.get("sourcePage", 0),
    )


# Scalar fields a swatch borrows from its other records when its leader lacks them.
_FILLABLE_FIELDS = ("size", "sizeRaw", "surface", "material", "color", "piecesPerBox",
                    "coveragePerBox", "brandName")
_PRICE_FIELDS = ("mrp", "sellingPrice")


def _merge_swatch(leader: Dict[str, Any], members: List[Dict[str, Any]]) -> None:
    """Fold `members` into `leader` in place: all their images, plus any details the leader lacks."""
    # Images: leader's primary, then every member's own image(s), then the leader's alternates.
    # A member's *alternates* are guesses at other tiles on its page, so they are not carried over.
    def image_key(img: Dict[str, Any]) -> str:
        return img.get("sha256") or img.get("url") or img.get("id") or ""

    primary = [img for img in leader["images"] if img.get("isPrimary")]
    own = [img for img in leader["images"] if not img.get("isPrimary") and not img.get("isAlternate")]
    alternates = [img for img in leader["images"] if img.get("isAlternate")]
    ordered: List[Dict[str, Any]] = primary + own
    seen = {image_key(img) for img in ordered}
    for member in members:
        for img in member["images"]:
            if img.get("isAlternate") or image_key(img) in seen:
                continue
            seen.add(image_key(img))
            ordered.append({**img, "isPrimary": False, "isAlternate": False, "isSwatchImage": True})
    for img in alternates:
        if image_key(img) not in seen:
            seen.add(image_key(img))
            ordered.append(img)
    leader["images"] = ordered
    if not leader.get("primaryImageUrl") and len(ordered) > 0:
        ordered[0]["isPrimary"] = True
        leader["primaryImageUrl"] = ordered[0].get("url")

    # Name: a leader without a readable name takes the best one in the group.
    if not swatch_name_key(leader):
        named = [m for m in members if swatch_name_key(m)]
        if named:
            best = max(named, key=lambda m: m.get("confidenceBreakdown", {}).get("name") or 0.0)
            for field in ("productName", "productNameOriginal", "ocrNameCandidates", "strippedNamePrefix"):
                if field in best:
                    leader[field] = best[field]
            name_conf = (best.get("confidenceBreakdown") or {}).get("name")
            if name_conf is not None:
                leader.setdefault("confidenceBreakdown", {})["name"] = name_conf
            if name_conf is None or name_conf >= 0.55:
                leader["reviewWarnings"] = [w for w in leader["reviewWarnings"] if "name could not" not in w]
            leader["reviewWarnings"].append(
                f"Name '{best['productName']}' taken from page {best.get('sourcePage')}, which shares this SKU."
            )

    # SKU: a leader without one takes it (with its candidates and status) from a member.
    if not canonical_sku(leader.get("productCode")):
        coded = next((m for m in members if canonical_sku(m.get("productCode"))), None)
        if coded:
            for field in ("productCode", "ocrSkuCandidates", "skuStatus"):
                leader[field] = coded.get(field)
            if leader.get("skuStatus") == "EXTRACTED":
                leader["reviewWarnings"] = [w for w in leader["reviewWarnings"] if "SKU could not" not in w]

    for field in _FILLABLE_FIELDS:
        if not leader.get(field):
            donor = next((m for m in members if m.get(field)), None)
            if donor:
                leader[field] = donor[field]
    for field in _PRICE_FIELDS:
        if not leader.get(field):
            donor = next((m for m in members if m.get(field)), None)
            if donor:
                leader[field] = donor[field]
    if leader.get("size"):
        leader["reviewWarnings"] = [w for w in leader["reviewWarnings"] if "Size/dimension not detected" not in w]

    have = {a["name"] for a in leader.get("attributes", [])}
    for member in members:
        for attr in member.get("attributes", []):
            if attr["name"] not in have:
                leader.setdefault("attributes", []).append(attr)
                have.add(attr["name"])

    texts = [leader.get("rawText") or ""] + [m.get("rawText") or "" for m in members]
    leader["rawText"] = "\n---\n".join(dict.fromkeys(t for t in texts if t))
    pages = set(leader.get("sourcePages", [leader["sourcePage"]]))
    for member in members:
        pages.update(member.get("sourcePages", [member["sourcePage"]]))
    leader["sourcePages"] = sorted(pages)
    leader["description"] = (
        f"{leader.get('productName')} - {leader.get('size') or ''} {leader.get('surface') or ''}".strip()
    )


def series_words(products: List[Dict[str, Any]]) -> set:
    """
    Words that open two or more product names ('Balistone' in 'Balistone Ash', 'Balistone Fog'): the
    series or collection a catalogue prints before each design name. A word that is itself a whole
    product name is never treated as a series.
    """
    keys = [swatch_name_key(p) for p in products]
    keys = [k for k in keys if k]
    whole = set(keys)
    leading: Counter = Counter()
    for key in set(keys):
        words = key.split()
        if len(words) >= 2:
            leading[words[0]] += 1
    return {w for w, count in leading.items() if count >= 2 and w not in whole and len(w) >= 3}


def strip_series(key: str, series: set) -> str:
    """'balistone royal blue' -> 'royal blue' when 'balistone' is a series word."""
    words = key.split()
    while len(words) > 1 and words[0] in series:
        words = words[1:]
    return " ".join(words)


SERIES_LINK_MAX_PAGE_GAP = 1


def _prefer_series_name(leader: Dict[str, Any], members: List[Dict[str, Any]], series: set) -> None:
    """'ROYAL BLUE' + 'Balistone Royal Blue' -> the swatch is named 'Balistone Royal Blue'."""
    key = swatch_name_key(leader)
    if not key or not series:
        return
    fuller = [m for m in members
              if (mk := swatch_name_key(m)) and mk != key and strip_series(mk, series) == key]
    if fuller:
        best = max(fuller, key=lambda m: (m.get("confidenceBreakdown") or {}).get("name") or 0.0)
        leader["productName"] = best["productName"]
        leader["productNameOriginal"] = best.get("productNameOriginal") or best["productName"]
        leader["description"] = (
            f"{leader['productName']} - {leader.get('size') or ''} {leader.get('surface') or ''}".strip()
        )


def dedupe_products(products: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Group records into swatches: one product per design, carrying every image of it.

    A catalogue often shows one tile several times - a room preview, a close-up, random faces - and
    each picture used to become its own product. Records are grouped when:
      * they share an SKU (always - the SKU identifies the product), or
      * they share a name once view words are removed ('Calacatta Gold Close-up' == 'Calacatta Gold'),
        provided they do not carry different SKUs, sizes or surfaces (those are separate products).
    Groups never mix two different SKUs. Folded-in records are returned, not lost.
    """
    n = len(products)
    parent = list(range(n))
    skus: List[set] = []
    sizes: List[set] = []
    surfaces: List[set] = []
    for p in products:
        p["productKey"] = product_key(p)
        sku, size, surface = canonical_sku(p.get("productCode")), _size_key(p), _surface_key(p)
        skus.append({sku} if sku else set())
        sizes.append({size} if size else set())
        surfaces.append({surface} if surface else set())

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int, by_sku: bool) -> None:
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if len(skus[ri] | skus[rj]) > 1:
            return  # two different SKUs are two products
        if not by_sku and (len(sizes[ri] | sizes[rj]) > 1 or len(surfaces[ri] | surfaces[rj]) > 1):
            return  # same name, different size or finish: separate products
        parent[rj] = ri
        skus[ri] |= skus[rj]
        sizes[ri] |= sizes[rj]
        surfaces[ri] |= surfaces[rj]

    # SKU links first (strongest evidence), then name links.
    by_sku: Dict[str, List[int]] = defaultdict(list)
    by_name: Dict[str, List[int]] = defaultdict(list)
    for idx, p in enumerate(products):
        sku = canonical_sku(p.get("productCode"))
        if sku:
            by_sku[sku].append(idx)
        name = swatch_name_key(p)
        # Only a confidently read name may link records: a misread name (e.g. a size fragment that
        # repeats on every page) would otherwise collapse unrelated designs into one swatch.
        name_conf = (p.get("confidenceBreakdown") or {}).get("name") or 0.0
        if name and name_conf >= SWATCH_NAME_MIN_CONFIDENCE:
            by_name[name].append(idx)
    for idxs in by_sku.values():
        for other in idxs[1:]:
            union(idxs[0], other, by_sku=True)
    for idxs in by_name.values():
        for other in idxs[1:]:
            union(idxs[0], other, by_sku=False)

    # Series links: a preview page titled 'Balistone Royal Blue' next to the swatch page whose name
    # reads 'ROYAL BLUE' (the series line is page furniture there). Deliberately narrow: only records
    # on adjacent pages, only a full name against its bare form (two differently prefixed names such
    # as 'Statuario Ash' / 'Balistone Ash' never link), and the size / finish / SKU guards still apply.
    series = series_words(products)
    if series:
        confident = [
            (idx, name) for name, idxs in by_name.items() for idx in idxs
        ]
        for i, name_i in confident:
            for j, name_j in confident:
                if i >= j or name_i == name_j:
                    continue
                gap = abs((products[i].get("sourcePage") or 0) - (products[j].get("sourcePage") or 0))
                if gap > SERIES_LINK_MAX_PAGE_GAP:
                    continue
                longer, shorter = (name_i, name_j) if len(name_i) > len(name_j) else (name_j, name_i)
                if strip_series(shorter, series) == shorter and strip_series(longer, series) == shorter:
                    union(i, j, by_sku=False)

    components: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        components[find(idx)].append(idx)

    drop = set()
    removed: List[Dict[str, Any]] = []
    for idxs in components.values():
        if len(idxs) < 2:
            continue
        idxs.sort(key=lambda i: _swatch_rank(products[i]))
        leader = products[idxs[0]]
        members = [products[i] for i in idxs[1:]]
        leader_sku = canonical_sku(leader.get("productCode"))
        names_differ = bool(leader_sku) and any(
            canonical_sku(m.get("productCode")) == leader_sku and not _names_compatible(leader, m) for m in members
        )
        before = len(leader["images"])
        _merge_swatch(leader, members)
        added = len(leader["images"]) - before
        _prefer_series_name(leader, members, series)
        leader["productKey"] = product_key(leader)
        for i, member in zip(idxs[1:], members):
            drop.add(i)
            removed.append({**member, "isDuplicate": True, "duplicateOfKey": leader["productKey"],
                            "duplicateOfName": leader["productName"]})
        pages = ", ".join(map(str, leader["sourcePages"]))
        leader["reviewWarnings"].append(
            f"Grouped {len(members) + 1} records into one swatch ({added} extra image(s), page(s) {pages})."
        )
        if names_differ:
            leader["reviewWarnings"].append(
                "Records with this SKU had different names; verify the name and images of this swatch."
            )
            leader["needsReview"] = True

    kept = [p for i, p in enumerate(products) if i not in drop]
    # The swatch is the design, not the shot: drop view words from every name (original kept).
    for p in kept:
        name = p.get("productName") or ""
        clean = strip_view_qualifiers(name)
        if name and name != UNRESOLVED_NAME and clean != name:
            p["productNameOriginal"] = p.get("productNameOriginal") or name
            p["productName"] = clean
            p["description"] = f"{clean} - {p.get('size') or ''} {p.get('surface') or ''}".strip()
    return kept, removed


def flag_shared_images(products: List[Dict[str, Any]]) -> None:
    counter: Counter = Counter()
    for p in products:
        for img in p["images"]:
            if img.get("isPrimary") and img.get("sha256"):
                counter[img["sha256"]] += 1
    for p in products:
        for img in p["images"]:
            if img.get("isPrimary") and counter.get(img.get("sha256"), 0) > 1:
                p["reviewWarnings"].append(
                    f"Same primary image is used by {counter[img['sha256']]} products; verify image."
                )
                p["needsReview"] = True
                break


def name_prefix_hints(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Common leading words across names: the UI can offer a one-click 'strip prefix from all'."""
    names = [p["productName"] for p in products if p["productName"] != UNRESOLVED_NAME]
    if len(names) < 4:
        return []
    counts: Counter = Counter()
    for n in names:
        w = n.split()
        for k in (1, 2, 3):
            if len(w) > k:
                counts[" ".join(w[:k])] += 1
    threshold = max(4, math.ceil(0.4 * len(names)))
    hints = [{"prefix": p, "count": c} for p, c in counts.items() if c >= threshold]
    hints.sort(key=lambda h: (-len(h["prefix"].split()), -h["count"]))
    out: List[Dict[str, Any]] = []
    for h in hints:
        if not any(o["prefix"].startswith(h["prefix"]) and o["count"] >= h["count"] for o in out):
            out.append(h)
    return out[:5]


# --------------------------------------------------------------------------------------
# v1-compatible output
# --------------------------------------------------------------------------------------

V1_PRODUCT_KEYS = [
    "sourcePage", "productCode", "productName", "brandName", "categoryName", "subcategory",
    "size", "surface", "color", "material", "unit", "piecesPerBox", "coveragePerBox", "mrp",
    "purchasePrice", "sellingPrice", "gstRate", "hsnCode", "description", "confidence",
    "reviewWarnings", "ocrNameCandidates", "ocrSkuCandidates", "skuStatus", "attributes",
    "rawText", "primaryImageUrl", "images",
]
V1_IMAGE_KEYS = [
    "id", "filename", "storagePath", "url", "sourcePage", "bbox", "pixelWidth", "pixelHeight",
    "method", "aspectRatio", "sha256", "phash", "isPrimary", "confidence",
]
V1_CANDIDATE_KEYS = ["value", "confidence", "bbox", "multiline"]
V1_PAGE_KEYS = ["pageNo", "isScanned", "previewUrl", "brand", "productCount", "imageCount"]


def to_v1_product(p: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: p[k] for k in V1_PRODUCT_KEYS if k in p}
    out["images"] = [{k: img[k] for k in V1_IMAGE_KEYS if k in img} for img in p.get("images", [])]
    out["ocrNameCandidates"] = [{k: c[k] for k in V1_CANDIDATE_KEYS if k in c} for c in p.get("ocrNameCandidates", [])]
    out["ocrSkuCandidates"] = [{k: c[k] for k in V1_CANDIDATE_KEYS if k in c} for c in p.get("ocrSkuCandidates", [])]
    out["reviewWarnings"] = list(p.get("reviewWarnings", []))
    return out


def to_v1_page(pg: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in pg:
        return {k: pg[k] for k in ("pageNo", "error", "productCount", "imageCount") if k in pg}
    return {k: pg[k] for k in V1_PAGE_KEYS if k in pg}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ERP Catalogue OCR and Image Extractor (v3)")
    parser.add_argument("--pdf", required=True, help="Path to source PDF catalogue")
    parser.add_argument("--output-dir", required=True, help="Directory to save extracted images and results")
    parser.add_argument("--job-id", required=True, help="Unique import job ID")
    parser.add_argument("--start-page", type=int, default=1, help="Start page (1-based)")
    parser.add_argument("--end-page", type=int, default=0, help="End page (1-based, 0 = all)")
    parser.add_argument("--pages", default="", help="Comma-separated pages to extract (1-based)")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR even if PaddleOCR is installed")
    parser.add_argument("--strip-template-prefix", action="store_true",
                        help="Opt-in: strip recurring template prefixes (e.g. 'Product Showcase') from names")
    parser.add_argument("--extended-output", action="store_true",
                        help="Inline the extra fields into the main JSON (default: v1-identical shape, "
                             "extras go to extraction_extras.json)")
    parser.add_argument("--profile-sample", type=int, default=150,
                        help="Max pages sampled for the document profile pass")
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(pdf_path):
        print(json.dumps({"error": f"File not found: {pdf_path}"}))
        sys.exit(1)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(json.dumps({"error": f"Cannot open PDF: {str(e)}"}))
        sys.exit(1)

    total_pages = len(doc)
    start_p = max(1, args.start_page)
    end_p = total_pages if args.end_page <= 0 else min(total_pages, args.end_page)
    if args.pages.strip():
        selected_pages = sorted({
            int(value) for value in args.pages.split(",")
            if value.strip().isdigit() and 1 <= int(value) <= total_pages
        })
        if not selected_pages:
            print(json.dumps({"error": "No valid pages were selected"}))
            sys.exit(1)
    else:
        selected_pages = list(range(start_p, end_p + 1))

    doc_brand = detect_brand(os.path.basename(pdf_path))
    if not doc_brand and doc.metadata:
        doc_brand = detect_brand(str(doc.metadata))
    if not doc_brand:
        for p_idx in range(min(5, total_pages)):
            doc_brand = detect_brand(doc[p_idx].get_text())
            if doc_brand:
                break

    # Pass 1: document profile (template text, decorative images, body font)
    try:
        profile = build_document_profile(doc, args.strip_template_prefix, args.profile_sample)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        profile = DocProfile(args.strip_template_prefix)

    # Pass 2: per-page extraction (render, text, images, cards)
    states: Dict[int, Dict[str, Any]] = {}
    errors: Dict[int, str] = {}
    for p in selected_pages:
        try:
            states[p] = prepare_page(doc, p, args.job_id, out_dir, profile, use_ocr=not args.no_ocr)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            errors[p] = str(e)

    # Learn this document's name style from confident cards, then finalize every page
    try:
        learn_name_style(list(states.values()), profile)
    except Exception:
        traceback.print_exc(file=sys.stderr)

    pages_result: List[Dict[str, Any]] = []
    all_products: List[Dict[str, Any]] = []
    all_orphans: List[Dict[str, Any]] = []
    for p in selected_pages:
        if p in errors:
            pages_result.append({"pageNo": p, "error": errors[p], "productCount": 0, "imageCount": 0})
            continue
        try:
            page_data = finalize_page(states[p], profile, doc_brand)
            pages_result.append({
                "pageNo": page_data["pageNo"],
                "isScanned": page_data["isScanned"],
                "ocrUsed": page_data["ocrUsed"],
                "layout": page_data["layout"],
                "previewUrl": page_data["previewUrl"],
                "brand": page_data["brand"],
                "productCount": len(page_data["products"]),
                "imageCount": page_data["extractedImagesCount"],
                "orphanImageCount": len(page_data["orphanImages"]),
            })
            all_products.extend(page_data["products"])
            all_orphans.extend(page_data["orphanImages"])
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            pages_result.append({"pageNo": p, "error": str(e), "productCount": 0, "imageCount": 0})

    raw_count = len(all_products)
    all_products, duplicate_products = dedupe_products(all_products)
    flag_shared_images(all_products)

    # Products without any image are not shown; they are kept in the extras file for reference.
    skipped_no_image = [p for p in all_products if not p.get("primaryImageUrl")]
    all_products = [p for p in all_products if p.get("primaryImageUrl")]
    shown_per_page = Counter(p["sourcePage"] for p in all_products)
    for pg in pages_result:
        if "error" not in pg:
            pg["productCount"] = shown_per_page.get(pg["pageNo"], 0)

    extras = {
        "rawProductCount": raw_count,
        "duplicateProducts": duplicate_products,
        "skippedNoImageProducts": skipped_no_image,
        "orphanImages": all_orphans,
        "nameBulkEditHints": name_prefix_hints(all_products),
        "documentProfile": profile.summary(),
        "capabilities": {"opencv": HAS_OPENCV, "paddleocr": HAS_PADDLEOCR and not args.no_ocr},
        "productExtras": [
            {
                "sourcePage": prod["sourcePage"], "productCode": prod["productCode"],
                "productName": prod["productName"],
                "productNameOriginal": prod.get("productNameOriginal"),
                "strippedNamePrefix": prod.get("strippedNamePrefix"),
                "sizeRaw": prod.get("sizeRaw"),
                "confidenceBreakdown": prod.get("confidenceBreakdown"),
                "needsReview": prod.get("needsReview"),
                "sourcePages": prod.get("sourcePages"),
            }
            for prod in all_products
        ],
    }

    if args.extended_output:
        output_payload = {
            "jobId": args.job_id, "totalPages": total_pages,
            "processedStartPage": start_p, "processedEndPage": end_p,
            "documentBrand": doc_brand, "totalProducts": len(all_products),
            "pages": pages_result, "products": all_products, **extras,
        }
    else:
        output_payload = {
            "jobId": args.job_id,
            "totalPages": total_pages,
            "processedStartPage": start_p,
            "processedEndPage": end_p,
            "documentBrand": doc_brand,
            "totalProducts": len(all_products),
            "pages": [to_v1_page(pg) for pg in pages_result],
            "products": [to_v1_product(prod) for prod in all_products],
        }
        with open(os.path.join(out_dir, "extraction_extras.json"), "w", encoding="utf-8") as f:
            json.dump(extras, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "extraction_result.json"), "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(output_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
