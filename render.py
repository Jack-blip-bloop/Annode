"""Drawing the annotations into a new PDF.

Two steps, so a preview can be shown and things unticked before downloading:
  make_plan(...)  -> every mark, note, margin symbol and the summary box, each with an id and its exact position
  draw(...)       -> the annotated PDF, leaving out any ids you unticked

The original pages go in untouched (an exact copy of each page, text still selectable) with a margin column added on
the right. Everything the AI adds is on its own layer called "AI annotations", so it can be told apart from a
student's own notes and switched off in any PDF app that shows layers."""
import os
import random

import pymupdf

from . import shapes
from .ai import ROLE_ORDER, role_styles
from .textlayer import find_quote, median_height, norm, segments

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_FILE = os.path.join(HERE, "fonts", "PatrickHand-Regular.ttf")   # SIL Open Font License (fonts/OFL.txt)
LAYER_NAME = "AI annotations"
DEFAULT_COLORS = {"ink": "#6b3db8", "highlight": "#cbb3f5", "notes": "#6b3db8"}
DEFAULT_ROLE_COLORS = {"main_idea": "#c62828", "key_term": "#1f4fbf", "definition": "#2e7d32", "fact": "#7b3fb5",
                       "question": "#e07000", "key_quote": "#ad1457"}
HAND_SIZE = {"small": 0.85, "normal": 1.0, "large": 1.2}
ASCII = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": " - ",
                       "…": "...", " ": " "})


def rgb(hexcolor, fallback="#6b3db8"):
    h = (hexcolor or fallback).lstrip("#")
    try:
        return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return rgb(fallback)


def handwriting_available():
    return os.path.exists(os.path.join(HERE, "handwriting.json"))


class Hand:
    """Writing notes: in your own handwriting (if it's been set up and switched on) or in a handwriting font."""

    def __init__(self, mine=False, messy="normal"):
        self.buffer = open(FONT_FILE, "rb").read()
        self.font = pymupdf.Font(fontbuffer=self.buffer)
        self.writer = None
        if mine and handwriting_available():
            from . import handwriting as hw
            lib = hw.load_library()
            if lib:
                self.writer = hw.Writer(lib, variation={"neat": 0.55, "normal": 1.0, "messy": 1.55}.get(messy, 1.0))

    def clean(self, text):
        text = " ".join(str(text).translate(ASCII).split())
        if self.writer:
            return "".join(c if (c == " " or not self.writer.missing(c)) else "" for c in text)
        return "".join(c if self.font.has_glyph(ord(c)) else "?" for c in text)

    def layout(self, text, size, width):
        """('strokes', strokes with the top-left at 0,0, w, h) in your handwriting, or ('font', lines, w, h)."""
        if self.writer:
            u = size * 0.42                  # x-height in points
            s = self.writer.write(text, max_width=max(width / u, 4))
            if s:
                xs = [p[0] for st in s for p in st]
                ys = [p[1] for st in s for p in st]
                x0, y0 = min(xs), min(ys)
                return ("strokes", [[((x - x0) * u, (y - y0) * u) for x, y in st] for st in s],
                        (max(xs) - x0) * u, (max(ys) - y0) * u)
        lines, cur = [], ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if cur and self.font.text_length(trial, size) > width:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return ("font", lines, max((self.font.text_length(l, size) for l in lines), default=0), len(lines) * size * 1.12)


def margin_width(page_w):
    return max(170.0, min(250.0, 0.34 * page_w))


def page_figures(page):
    """Rectangles of the pictures and drawn figures on a page (lines from the margin must not cross them)."""
    ox, oy = page.rect.x0, page.rect.y0
    rects = []
    try:
        for info in page.get_image_info():
            rects.append(pymupdf.Rect(info["bbox"]))
        for d in page.get_drawings():
            r = d.get("rect")
            if r and r.width * r.height > 300 and r.width < page.rect.width * 0.95:
                rects.append(pymupdf.Rect(r))
    except Exception:
        pass
    return [r - (ox, oy, ox, oy) for r in rects if r.is_valid and not r.is_empty]


# ---------------------------------------------------------------- 1. the plan
def _word_match(tok, term):
    import difflib
    if not tok or not term:
        return False
    if tok == term or tok in (term + "s", term + "es") or term in (tok + "s", tok + "es"):
        return True
    return len(term) >= 5 and difflib.SequenceMatcher(None, tok, term).ratio() >= 0.88


def _overlaps_any(segs, taken):
    for a in segs:
        for b in taken:
            h = min(a[3] - a[1], b[3] - b[1])
            if min(a[3], b[3]) - max(a[1], b[1]) > 0.5 * h and min(a[2], b[2]) - max(a[0], b[0]) > 0:
                return True
    return False


def make_plan(src, page_data, request, results, summary, options):
    """Every annotation, with an id, page and exact position. Quotes the AI made up (not on the page) are dropped."""
    rs = role_styles(options)
    targeted = request.get("mode") == "targeted"
    obey_style = targeted or bool((options.get("teacher") or "").strip())
    enabled = options.get("styles") or ["underline"]
    plan, missed = [], 0

    def add(**item):
        item["id"] = len(plan)
        plan.append(item)

    if summary and options.get("summary_box") and not targeted:
        lines = []
        if str(summary.get("main_idea", "")).strip():
            lines.append("Main idea: " + " ".join(str(summary["main_idea"]).split()))
        lines += ["- " + " ".join(str(t).split()) for t in (summary.get("key_terms") or [])[:3] if str(t).strip()]
        if str(summary.get("question", "")).strip():
            q = " ".join(str(summary["question"]).split())
            lines.append(q if q.lower().startswith("q:") else "Q: " + q)
        if lines:
            add(type="summary", page=0, lines=lines, label="Summary box")

    for pno in range(len(src)):
        words = page_data.get(pno) or []
        res = results.get(pno) or {}
        taken = []
        for m in res.get("marks") or []:
            quote = str(m.get("quote", "")).strip()
            hit = find_quote(words, quote, m.get("line") if isinstance(m.get("line"), int) else None)
            if not hit:
                missed += 1
                continue
            segs = segments(hit)
            if _overlaps_any(segs, taken):
                continue
            taken += segs
            role = m.get("role") if m.get("role") in rs else "key_term"
            style = m["style"] if obey_style and m.get("style") in shapes.MARKS else rs[role]
            if style in ("circle", "box") and (len(segs) > 1 or len(quote.split()) > 6) and not obey_style:
                style = next((s for s in ("underline", "highlight", "double") if s in enabled), "underline")
            add(type="mark", page=pno, role=role, style=style, segs=segs, text=quote,
                label=f"{style.capitalize()}: {quote[:60]}")
        if targeted:                         # the program also finds every place the search words appear
            terms = []
            for t in request.get("search_terms") or []:
                tt = [x for x in (norm(w) for w in str(t).split()) if x]
                if tt and tt not in terms and len(tt) <= 6:
                    terms.append(tt)
            styles = [p["style"] for p in plan if p["type"] == "mark"]
            style = max(set(styles), key=styles.count) if styles else rs["key_term"]
            toks = [norm(w["text"]) for w in words]
            for tt in terms:
                for i in range(len(toks) - len(tt) + 1):
                    if all(_word_match(toks[i + k], tt[k]) for k in range(len(tt))):
                        segs = segments(words[i:i + len(tt)])
                        if not _overlaps_any(segs, taken):
                            taken += segs
                            text = " ".join(w["text"] for w in words[i:i + len(tt)])
                            add(type="mark", page=pno, role="key_term", style=style, segs=segs, text=text,
                                label=f"{style.capitalize()}: {text[:60]}")
        if options.get("notes", True) or targeted:
            for n in res.get("notes") or []:
                text = " ".join(str(n.get("text", "")).split())
                if sum(c.isalnum() for c in text) < 3:
                    continue                 # just a "!" or "?" on its own: that's what margin symbols are for
                a = find_quote(words, str(n.get("anchor_quote", "")), n.get("line") if isinstance(n.get("line"), int) else None)
                y = segments(a)[0][1] if a else None
                if y is None:
                    same = [w for w in words if w["line"] == n.get("line")]
                    y = min(w["y"] for w in same) if same else None
                add(type="note", page=pno, text=text, anchor=segments(a) if a else None, want_y=y,
                    label=f"Note: {text[:60]}")
        symbol_lines = set()
        for s in res.get("symbols") or []:
            sym = s.get("symbol")
            if sym not in shapes.SYMBOLS:
                continue
            a = find_quote(words, str(s.get("quote", "")), s.get("line") if isinstance(s.get("line"), int) else None)
            if not a or a[0]["line"] in symbol_lines:
                continue                     # not on the page, or that line already has a symbol
            li = a[0]["line"]
            symbol_lines.add(li)
            line_words = [w for w in words if w["line"] == li]
            box = [min(w["x"] for w in line_words), min(w["y"] for w in line_words),
                   max(w["x"] + w["w"] for w in line_words), max(w["y"] + w["h"] for w in line_words)]
            # A PDF can store one printed line in pieces, so use the left edge of the paragraph around it: nearby
            # lines in the same column (they overlap it left-to-right).
            lh = box[3] - box[1]
            near = [w for w in words if abs((w["y"] + w["h"] / 2) - (box[1] + box[3]) / 2) < 3 * lh
                    and w["x"] < box[2] and w["x"] + w["w"] > box[0] - 0.5 * (box[2] - box[0])]
            if near:
                box[0] = min(box[0], min(w["x"] for w in near))
            name = {"?": "?", "!": "!", "star": "Star", "arrow": "Arrow"}[sym]
            add(type="symbol", page=pno, symbol=sym, line_box=box, label=f"{name} beside: {a[0]['text']} ...")
    return plan, missed


# ---------------------------------------------------------------- 2. drawing
def _draw(shape, strokes, width, color, opacity=1.0, oc=0):
    for s in strokes:
        if len(s) > 1:
            shape.draw_polyline(s)
            shape.finish(width=width, color=color, closePath=False, lineCap=1, lineJoin=1, stroke_opacity=opacity, oc=oc)


def _layout_column(items, page_h, top, gap):
    """Stack margin items level with what they're about, pushed down (or up) so none overlap."""
    items.sort(key=lambda it: it["want"])
    y = top
    for it in items:
        it["y"] = max(y, it["want"])
        y = it["y"] + it["h"] + gap
    over = y - gap - (page_h - 22)
    if over > 0:
        floor = page_h - 22
        for it in reversed(items):
            it["y"] = min(it["y"], floor - it["h"])
            floor = it["y"] - gap
    return [it for it in items if it["y"] >= top - 1]


def draw(src, page_data, plan, options, exclude=(), seed=1):
    """The annotated PDF (bytes), without the plan items whose ids are in `exclude`."""
    exclude = set(exclude)
    colors = dict(DEFAULT_COLORS, **{k: v for k, v in (options.get("colors") or {}).items() if v})
    role_colors = dict(DEFAULT_ROLE_COLORS, **{k: v for k, v in (options.get("role_colors") or {}).items() if v})
    by_type = bool(options.get("color_by_type"))
    looks = options.get("mark_style", "normal")
    hand = Hand(mine=bool(options.get("hand_notes")) and handwriting_available(), messy=options.get("hand_style", "normal"))
    hs = HAND_SIZE.get(options.get("hand_size"), 1.0)
    connectors = options.get("connectors", True)

    out = pymupdf.open()
    oc = out.add_ocg(LAYER_NAME, on=True)
    items = [p for p in plan if p["id"] not in exclude]
    for pno in range(len(src)):
        sp = src[pno]
        w, h = sp.rect.width, sp.rect.height
        mw = margin_width(w)
        page = out.new_page(width=w + mw, height=h)
        page.show_pdf_page(pymupdf.Rect(0, 0, w, h), src, pno)
        words = page_data.get(pno) or []
        figures = page_figures(sp)
        mh = median_height(words)
        scale = max(0.45, min(1.6, mh / 13.0))
        pen = max(0.7, 0.85 * scale)
        page.insert_font(fontname="hand", fontbuffer=hand.buffer)
        shape = page.new_shape()
        _draw(shape, [[(w + 5, 18), (w + 5, h - 18)]], 0.5, (0.86, 0.84, 0.9), oc=oc)
        mine = [p for p in items if p["page"] == pno]

        for m in (p for p in mine if p["type"] == "mark"):
            random.seed(seed * 100003 + m["id"])
            if m["style"] == "highlight":
                col = rgb(colors["highlight"])
                for b in m["segs"]:
                    _draw(shape, shapes.highlight(b, scale), (b[3] - b[1]) * 0.95, col, opacity=0.45, oc=oc)
                continue
            col = rgb(role_colors.get(m["role"]) if by_type else colors["ink"])
            boxes = m["segs"]
            if m["style"] in ("circle", "box") and len(boxes) > 1:
                boxes = [(min(b[0] for b in boxes), boxes[0][1], max(b[2] for b in boxes), boxes[-1][3])]
            for b in boxes:
                _draw(shape, shapes.look(shapes.MARKS[m["style"]](b, scale), looks, scale), pen, col, 0.9, oc)

        for s in (p for p in mine if p["type"] == "symbol"):
            random.seed(seed * 100003 + s["id"])
            x1, y1, x2, y2 = s["line_box"]
            size = (y2 - y1) * (1.3 if s["symbol"] == "star" else 1.05)
            cy = (y1 + y2) / 2
            if x1 - size - 5 > 4:
                b = (x1 - size - 5, cy - size / 2, x1 - 5, cy + size / 2)      # in the left margin
            elif x2 + size + 6 < w:
                b = (x2 + 5, cy - size / 2, x2 + 5 + size, cy + size / 2)      # just after the line
            else:
                continue
            col = rgb(role_colors["question"] if by_type and s["symbol"] == "?" else colors["ink"])
            _draw(shape, shapes.SYMBOLS[s["symbol"]](b, scale), pen, col, 0.95, oc)

        # the margin column: the summary box (first page) then the notes, level with what they're about
        size = max(9.5, min(13.0, mh * 0.95)) * hs
        col_x, col_w = w + 14, mw - 24
        note_col = rgb(colors["notes"])
        top = 24.0
        for sm in (p for p in mine if p["type"] == "summary"):
            random.seed(seed * 100003 + sm["id"])
            y = top + 6
            head = hand.layout("Summary", size * 1.05, col_w)
            blocks = [head] + [hand.layout(hand.clean(l), size * 0.95, col_w - 10) for l in sm["lines"]]
            pad = 6
            ys = []
            for blk in blocks:
                ys.append(y)
                y += blk[3] + size * 0.45
            _write_blocks(page, shape, blocks, [col_x + 2] * len(blocks), ys, size, note_col, pen, oc)
            _draw(shape, shapes.box((col_x - pad + 2, top, col_x + col_w + 2, y - size * 0.2), scale * 0.8), pen,
                  note_col, 0.9, oc)
            top = y + size
        notes = []
        for n in (p for p in mine if p["type"] == "note"):
            random.seed(seed * 100003 + n["id"])
            blk = hand.layout(hand.clean(n["text"]), size, col_w)
            want = n["want_y"] if n["want_y"] is not None else h * 0.5
            notes.append({"n": n, "blk": blk, "h": blk[3], "want": want - size * 0.25})
        for it in _layout_column(notes, h, top, size * 0.9):
            n, blk = it["n"], it["blk"]
            _write_blocks(page, shape, [blk], [col_x], [it["y"]], size, note_col, pen, oc)
            a = n["anchor"]
            if a:
                random.seed(seed * 100003 + n["id"] + 7)
                _draw(shape, [s for b in a for s in shapes.dotted_underline(b, scale)], pen * 0.8, note_col, 0.7, oc)
                ax, ay = a[-1][2], (a[-1][1] + a[-1][3]) / 2
                blocked = any(o["x"] > ax + 2 and o["y"] < ay < o["y"] + o["h"] for o in words) or \
                    any(r.x1 > ax + 2 and r.y0 - 3 < ay < r.y1 + 3 for r in figures)
                if connectors and not blocked:
                    _draw(shape, shapes.connector(ax + 3, ay, col_x - 4, it["y"] + min(blk[3], size) * 0.5, scale),
                          pen * 0.7, note_col, 0.55, oc)
        shape.commit(overlay=True)
        if pno == 0:
            page.insert_text((col_x, h - 10), "AI notes (layer: AI annotations)", fontsize=6.5, fontname="helv",
                             color=(0.55, 0.5, 0.62), oc=oc)
    out.set_metadata({"producer": "Study Annotator"})
    data = out.tobytes(garbage=3, deflate=True)
    out.close()
    return data


def _write_blocks(page, shape, blocks, xs, ys, size, color, pen, oc):
    for blk, x, y in zip(blocks, xs, ys):
        kind, content = blk[0], blk[1]
        if kind == "strokes":
            _draw(shape, [[(x + px, y + py) for px, py in st] for st in content], max(0.6, pen * 0.85), color, 1.0, oc)
        else:
            for i, ln in enumerate(content):
                page.insert_text((x, y + size * (0.85 + 1.12 * i)), ln, fontsize=size, fontname="hand", color=color, oc=oc)


def previews(pdf_bytes, dpi=60):
    """A PNG of every page of the annotated PDF (for the preview)."""
    d = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out = [p.get_pixmap(dpi=dpi).tobytes("png") for p in d]
    d.close()
    return out
