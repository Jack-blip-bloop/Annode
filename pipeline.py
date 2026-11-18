"""PDF or page image in, annotated PDF out. Everything happens in memory: nothing is written to disk.

    annotate(data, filename, options)          one go: returns (pdf bytes, stats)
    job = prepare(data, filename, options)     read + ask the AI + plan (for a preview)
    pdf = render(job, exclude={ids})           draw it, leaving out what was unticked (no AI call; can repeat)
    job.close()"""
import io
import time
from concurrent.futures import ThreadPoolExecutor

import pymupdf
from PIL import Image, ImageOps

from . import ai
from .render import draw, make_plan

IMAGE_TYPES = {"png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "gif"}
PAGE_WIDTH = 612.0           # a photo or scan image becomes a letter-width page

# The same choices as the OneNote annotator's menu (minus the screen-only ones: scrolling, zoom, check & fix, undo)
DEFAULT_OPTIONS = {
    "focus": "",                       # what to annotate for, or a specific request ("every time clothes are mentioned")
    "teacher": "",                     # the teacher's annotation instructions (they win over the defaults)
    "styles": ["underline", "circle", "highlight"],
    "amount": "normal",                # few / normal / lots
    "notes": True,                     # margin notes
    "hand_notes": False,               # notes in MY handwriting (if it's been set up), else a handwriting font
    "notes_amount": "normal",          # few / normal / many
    "summary_box": False,
    "symbols": False,                  # ? ! star arrow in the margin
    "color_by_type": False,            # a different colour for each kind of idea
    "colors": {},                      # ink / highlight / notes  (#rrggbb)
    "role_colors": {},                 # kind of idea -> #rrggbb
    "hand_size": "normal",             # small / normal / large
    "hand_style": "normal",            # neat / normal / messy (my handwriting)
    "mark_style": "normal",            # neat / normal / messy (the marks)
    "connectors": True,                # lines from margin notes back to their text
}


class UserError(Exception):
    """A problem with the file itself (shown to the person as is)."""


def open_input(data, filename=""):
    """The upload as a PDF document (a photo/scan image becomes a one-page PDF)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in IMAGE_TYPES or (not data.startswith(b"%PDF") and ext != "pdf"):
        try:
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img).convert("RGB")     # phone photos: apply their rotation
        except Exception:
            raise UserError("That file isn't a PDF or a picture I can read. Upload a PDF, PNG or JPG.")
        if img.width > 5000 or img.height > 7000:
            img.thumbnail((5000, 7000))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        out = pymupdf.open()
        page = out.new_page(width=PAGE_WIDTH, height=PAGE_WIDTH * img.height / img.width)
        page.insert_image(page.rect, stream=buf.getvalue())
        return out, True
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise UserError("That PDF couldn't be opened. It may be damaged.")
    if doc.needs_pass:
        raise UserError("That PDF is password-protected. Remove the password and upload it again.")
    if len(doc) == 0:
        raise UserError("That PDF has no pages.")
    return doc, False


class Job:
    """Everything needed to (re)draw one reading. Lives in memory only."""

    def __init__(self, src, page_data, plan, options, stats):
        self.src, self.page_data, self.plan, self.options, self.stats = src, page_data, plan, options, stats

    def close(self):
        try:
            self.src.close()
        except Exception:
            pass


def prepare(data, filename="", options=None, max_pages=None, api_key=None, progress=lambda pct, msg: None):
    t0 = time.time()
    opts = dict(DEFAULT_OPTIONS, **(options or {}))
    src, from_image = open_input(data, filename)
    try:
        if max_pages and len(src) > max_pages:
            raise UserError(f"That file has {len(src)} pages; the limit is {max_pages}.")
        for p in src:
            if p.rotation:
                p.remove_rotation()

        progress(3, "Reading the pages...")
        page_data, how = {}, {}

        def read(pno):
            from .textlayer import page_words
            p = src[pno]
            dpi = 300
            if from_image:
                imgs = p.get_images()
                dpi = 300 if not imgs else max(200, min(400, int(72 * imgs[0][2] / p.rect.width)))
            return pno, page_words(p, dpi)

        with ThreadPoolExecutor(max_workers=4) as ex:
            for i, (pno, (words, kind)) in enumerate(ex.map(read, range(len(src)))):
                page_data[pno], how[pno] = words, kind
                progress(3 + int(32 * (i + 1) / len(src)), f"Reading page {i + 1} of {len(src)}...")
        if sum(len(w) for w in page_data.values()) < 20:
            raise UserError("I couldn't find any text to annotate in that file.")

        def ai_progress(done, total):
            progress(35 + int(50 * done / max(total, 1)), "Thinking about what matters..." if done < total else
                     "Planning the annotations...")
        request, results, summary, usage = ai.annotate_pages(sorted(page_data.items()), opts, api_key=api_key,
                                                             progress=ai_progress)
        plan, missed = make_plan(src, page_data, request, results, summary, opts)
        stats = {
            "pages": len(src),
            "pages_ocr": sum(1 for k in how.values() if k == "ocr"),
            "pages_empty": sum(1 for k in how.values() if k == "none"),
            "mode": request.get("mode", "general"),
            "marks": sum(1 for p in plan if p["type"] == "mark"),
            "notes": sum(1 for p in plan if p["type"] == "note"),
            "symbols": sum(1 for p in plan if p["type"] == "symbol"),
            "summary": any(p["type"] == "summary" for p in plan),
            "missed_quotes": missed,
            "ai_calls": usage.calls, "tokens_in": usage.tokens_in, "tokens_out": usage.tokens_out,
            "cost_usd": round(usage.cost_usd, 6),
            "seconds": round(time.time() - t0, 1),
        }
        return Job(src, page_data, plan, opts, stats)
    except BaseException:
        src.close()
        raise


def render(job, exclude=()):
    return draw(job.src, job.page_data, job.plan, job.options, exclude)


def annotate(data, filename="", options=None, max_pages=None, api_key=None, progress=lambda pct, msg: None, **kw):
    """One go: (annotated pdf bytes, stats). Extra keyword arguments (focus=..., amount=...) are options too."""
    opts = dict(options or {}, **kw)
    job = prepare(data, filename, opts, max_pages, api_key, progress)
    try:
        progress(90, "Drawing the annotations...")
        pdf = render(job)
        progress(100, "Done")
        st = dict(job.stats)
        st["questions"] = st["notes"]
        return pdf, st
    finally:
        job.close()
