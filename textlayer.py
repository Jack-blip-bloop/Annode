"""Getting the words (and where they are) off each page.

Text PDFs: the PDF's own text (exact positions, no guessing).
Scans and photos: Tesseract OCR on a rendered image of the page.

Every word is a dict {text, x, y, w, h, line} in PDF points, measured from the top-left of the visible page.
`line` numbers the text lines on the page in reading order (0, 1, 2...)."""
import difflib
import os
import re
import shutil
import unicodedata

import pymupdf
from PIL import Image

MIN_TEXT_WORDS = 12          # fewer real words than this on a page with pictures: it's a scan, so OCR it
OCR_MIN_CONF = 35            # Tesseract's confidence (0-100) below which a "word" is usually a smudge


def _tesseract():
    import pytesseract
    cmd = shutil.which("tesseract")
    if not cmd:
        for c in (os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
                  r"C:\Program Files\Tesseract-OCR\tesseract.exe", "/usr/bin/tesseract", "/usr/local/bin/tesseract",
                  "/opt/homebrew/bin/tesseract"):
            if os.path.exists(c):
                cmd = c
                break
    if not cmd:
        raise RuntimeError("Tesseract OCR isn't installed, so scanned pages can't be read.")
    pytesseract.pytesseract.tesseract_cmd = cmd
    return pytesseract


def page_words(page, ocr_dpi=300):
    """(words, how) for one page. how = 'text' (the PDF's own text), 'ocr' (a scan, read by OCR) or 'none'."""
    ox, oy = page.rect.x0, page.rect.y0
    raw = [w for w in page.get_text("words") if w[4].strip()]
    vertical = _vertical_lines(page)
    words, lines = [], {}
    for x0, y0, x1, y1, text, block, line, _ in raw:
        if (block, line) in vertical:
            continue                         # text printed sideways (margin stamps, rotated tables): don't mark it
        li = lines.setdefault((block, line), len(lines))
        words.append({"text": text, "x": x0 - ox, "y": y0 - oy, "w": x1 - x0, "h": y1 - y0, "line": li})
    if len(words) >= MIN_TEXT_WORDS or not page.get_images():
        return words, ("text" if words else "none")
    ocr = ocr_page(page, ocr_dpi)
    if len(ocr) > len(words):
        return ocr, "ocr"
    return words, ("text" if words else "none")


def _vertical_lines(page):
    """(block, line) numbers of text lines that don't run left to right."""
    out = set()
    try:
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for li, ln in enumerate(b.get("lines", [])):
                dx, dy = ln.get("dir", (1, 0))
                if abs(dy) > 0.3 or dx < 0:
                    out.add((b["number"], li))
    except Exception:
        pass
    return out


def ocr_page(page, dpi=300):
    """Read a scanned page with Tesseract. Positions are converted back to PDF points."""
    tess = _tesseract()
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    d = tess.image_to_data(img, output_type=tess.Output.DICT, config="--psm 3")
    k = page.rect.width / pix.width
    words, lines = [], {}
    for i, t in enumerate(d["text"]):
        t = (t or "").strip()
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if not t or conf < OCR_MIN_CONF:
            continue
        li = lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), len(lines))
        words.append({"text": t, "x": d["left"][i] * k, "y": d["top"][i] * k,
                      "w": d["width"][i] * k, "h": d["height"][i] * k, "line": li})
    return words


def numbered_lines(words):
    """The page's text as '[line] words...' rows, for the AI."""
    rows = {}
    for w in words:
        rows.setdefault(w["line"], []).append(w["text"])
    return "\n".join(f"[{li}] " + " ".join(ws) for li, ws in sorted(rows.items()))


def median_height(words, default=12.0):
    hs = sorted(w["h"] for w in words if w["h"] > 0)
    return hs[len(hs) // 2] if hs else default


# ---------------------------------------------------------------- finding a quote on the page
def norm(s):
    s = unicodedata.normalize("NFKC", s).lower().replace("\u00ad", "")
    return re.sub(r"\W+", "", s)


def find_quote(words, quote, hint_line=None):
    """The words of `quote` on the page (tolerates small OCR/typing differences), or None. When the same words appear
    more than once, the one on (or next to) hint_line wins."""
    qt = [t for t in (norm(w) for w in quote.split()) if t][:40]
    if not qt or not words:
        return None
    toks = [norm(w["text"]) for w in words]
    qs, n = " ".join(qt), len(qt)
    best, best_score = None, 0.0
    sm = difflib.SequenceMatcher
    for size in sorted({max(1, n - 1), n, n + 1}):
        for i in range(0, len(words) - size + 1):
            if sm(None, qt[0], toks[i]).ratio() < 0.6 and sm(None, qt[-1], toks[i + size - 1]).ratio() < 0.6:
                continue
            score = sm(None, qs, " ".join(toks[i:i + size])).ratio()
            if hint_line is not None and abs(words[i]["line"] - hint_line) <= 1:
                score += 0.04
            if score > best_score:
                best, best_score = (i, i + size), score
    if best and best_score >= 0.8:
        return words[best[0]:best[1]]
    return None


def segments(words):
    """One box per text line for these words: [(x1, y1, x2, y2)]."""
    by_line = {}
    for w in words:
        by_line.setdefault(w["line"], []).append(w)
    out = []
    for li in sorted(by_line):
        ws = by_line[li]
        out.append((min(w["x"] for w in ws), min(w["y"] for w in ws),
                    max(w["x"] + w["w"] for w in ws), max(w["y"] + w["h"] for w in ws)))
    return out
