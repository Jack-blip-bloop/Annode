"""python -m annotate_pdf reading.pdf [-o out.pdf] [--focus "..."] [--teacher "..."] [--amount few|normal|lots]
                                    [--options '{"summary_box": true, "symbols": true, ...}']"""
import argparse
import json
import os
import sys

from .pipeline import UserError, annotate


def main():
    ap = argparse.ArgumentParser(description="Annotate a reading (PDF or page image) for studying.")
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--focus", default="")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--amount", default="normal", choices=["few", "normal", "lots"])
    ap.add_argument("--options", default="{}", help="any other options as JSON (see DEFAULT_OPTIONS in pipeline.py)")
    a = ap.parse_args()
    out = a.output or os.path.splitext(a.input)[0] + "-annotated.pdf"
    opts = dict(json.loads(a.options), focus=a.focus, teacher=a.teacher, amount=a.amount)
    with open(a.input, "rb") as f:
        data = f.read()
    try:
        pdf, stats = annotate(data, os.path.basename(a.input), opts,
                              progress=lambda pct, msg: print(f"{pct:3d}% {msg}", file=sys.stderr))
    except UserError as e:
        sys.exit(str(e))
    with open(out, "wb") as f:
        f.write(pdf)
    print(json.dumps(stats, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
