"""Annotate every file in a folder and report what happened.

    python tests/batch_test.py [folder] [--pages N]

Writes the annotated PDFs and a picture of each one's first page to <folder>/annotated/, and prints a table:
pages, OCR pages, marks, questions, quotes the AI made up (couldn't be found on the page), seconds and cost.
Use it on your own readings too: put them in a folder and point this at it. Nothing leaves your computer except the
page text sent to the AI."""
import argparse
import glob
import os
import sys
import traceback

import pymupdf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from annotate_pdf import UserError, annotate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folder", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs"))
ap.add_argument("--pages", type=int, default=0, help="only the first N pages of each file")
a = ap.parse_args()
out_dir = os.path.join(a.folder, "annotated")
os.makedirs(out_dir, exist_ok=True)
files = sorted(f for f in glob.glob(os.path.join(a.folder, "*")) if os.path.isfile(f))

rows, total_cost, total_pages = [], 0.0, 0
print(f"{'file':38} {'result':8} {'pg':>3} {'ocr':>3} {'marks':>5} {'qs':>3} {'lost':>4} {'sec':>5} {'cost $':>8}")
for path in files:
    name = os.path.basename(path)
    data = open(path, "rb").read()
    if a.pages and data.startswith(b"%PDF"):
        try:
            d = pymupdf.open(stream=data, filetype="pdf")
            if not d.needs_pass and len(d) > a.pages:
                t = pymupdf.open()
                t.insert_pdf(d, to_page=a.pages - 1)
                data = t.tobytes()
        except Exception:
            pass
    try:
        pdf, st = annotate(data, name)
        base = os.path.splitext(name)[0]
        open(os.path.join(out_dir, base + "-annotated.pdf"), "wb").write(pdf)
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        doc[0].get_pixmap(dpi=70).save(os.path.join(out_dir, base + "-p1.png"))
        flag = "ok"
        text_pages = st["pages"] - st["pages_empty"]
        if st["marks"] == 0 and text_pages:
            flag = "NO MARKS"
        elif st["missed_quotes"] > max(2, 0.2 * (st["marks"] + st["missed_quotes"])):
            flag = "LOST"
        total_cost += st["cost_usd"]
        total_pages += st["pages"]
        print(f"{name[:38]:38} {flag:8} {st['pages']:>3} {st['pages_ocr']:>3} {st['marks']:>5} {st['questions']:>3} "
              f"{st['missed_quotes']:>4} {st['seconds']:>5} {st['cost_usd']:>8.5f}", flush=True)
    except UserError as e:
        print(f"{name[:38]:38} refused  {str(e)[:70]}", flush=True)
    except Exception as e:
        print(f"{name[:38]:38} CRASH    {type(e).__name__}: {str(e)[:60]}", flush=True)
        traceback.print_exc(limit=3)
if total_pages:
    print(f"\n{total_pages} pages, ${total_cost:.4f} total, ${total_cost / total_pages:.5f} per page "
          f"(~${total_cost / total_pages * 1000:.2f} per 1000 pages)")
