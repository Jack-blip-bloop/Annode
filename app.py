"""The web upload tool: upload a reading, choose how to annotate it, (preview it), download it.

    python web/app.py            (then open http://127.0.0.1:5000)   - or use launch.pyw / the desktop shortcut

Privacy by design:
  - Uploaded files and the annotated results are only ever held in memory. Nothing is written to disk. A job is
    deleted the moment its result is downloaded, or after RESULT_TTL_MIN minutes without activity.
  - No accounts. The free daily limit is counted per visitor, where a visitor is a salted one-way hash of their IP
    address (the IP itself is never stored). Settings and presets live in the visitor's own browser.
  - usage.db keeps only numbers per day and visitor (pages, AI tokens, cost, seconds) so AI costs stay visible.

Settings (environment variables): FREE_PAGES_PER_DAY, MAX_PAGES_PER_FILE, MAX_UPLOAD_MB, ADMIN_KEY, TRUST_PROXY,
APP_NAME, CONTACT_EMAIL, and the engine's STUDY_ANNOTATOR_GEMINI_KEY / PRICE_IN_PER_M / PRICE_OUT_PER_M."""
import datetime as dt
import hashlib
import io
import json
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pymupdf
from flask import Flask, abort, jsonify, render_template, request, send_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from annotate_pdf import DEFAULT_OPTIONS, UserError, prepare, render  # noqa: E402
from annotate_pdf.pipeline import IMAGE_TYPES  # noqa: E402
from annotate_pdf.render import handwriting_available, previews  # noqa: E402

APP_NAME = os.environ.get("APP_NAME", "Study Annotator")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")
FREE_PAGES_PER_DAY = int(os.environ.get("FREE_PAGES_PER_DAY", "15"))
MAX_PAGES_PER_FILE = int(os.environ.get("MAX_PAGES_PER_FILE", "40"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
RESULT_TTL_MIN = 15
TRUST_PROXY = os.environ.get("TRUST_PROXY") == "1"     # only behind a proxy you control (it sets X-Forwarded-For)
DB_PATH = os.environ.get("USAGE_DB", os.path.join(HERE, "usage.db"))
SALT_PATH = os.path.join(HERE, ".visitor_salt")
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

ADMIN_KEY = os.environ.get("ADMIN_KEY") or secrets.token_urlsafe(16)
WORKERS = ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "2")))
JOBS, JOBS_LOCK = {}, threading.Lock()


# ---------------------------------------------------------------- visitors and usage (numbers only)
def _salt():
    if not os.path.exists(SALT_PATH):
        with open(SALT_PATH, "w") as f:
            f.write(secrets.token_hex(32))
    return open(SALT_PATH).read().strip()


SALT = _salt()


def visitor_id():
    ip = request.remote_addr or ""
    if TRUST_PROXY and request.headers.get("X-Forwarded-For"):
        ip = request.headers["X-Forwarded-For"].split(",")[0].strip()
    return hashlib.sha256((SALT + ip).encode()).hexdigest()[:16]


def db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS usage (
        day TEXT, visitor TEXT, jobs INTEGER DEFAULT 0, pages INTEGER DEFAULT 0, pages_ocr INTEGER DEFAULT 0,
        tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0, seconds REAL DEFAULT 0,
        failures INTEGER DEFAULT 0, PRIMARY KEY (day, visitor))""")
    return con


def today():
    return dt.date.today().isoformat()


def pages_used_today(visitor):
    with db() as con:
        row = con.execute("SELECT pages FROM usage WHERE day=? AND visitor=?", (today(), visitor)).fetchone()
    used = row[0] if row else 0
    with JOBS_LOCK:
        used += sum(j["pages"] for j in JOBS.values() if j["visitor"] == visitor and j["status"] == "working")
    return used


def record(visitor, stats=None, failed=False):
    s = stats or {}
    with db() as con:
        con.execute("INSERT OR IGNORE INTO usage (day, visitor) VALUES (?, ?)", (today(), visitor))
        con.execute("""UPDATE usage SET jobs=jobs+1, pages=pages+?, pages_ocr=pages_ocr+?, tokens_in=tokens_in+?,
                       tokens_out=tokens_out+?, cost_usd=cost_usd+?, seconds=seconds+?, failures=failures+?
                       WHERE day=? AND visitor=?""",
                    (s.get("pages", 0), s.get("pages_ocr", 0), s.get("tokens_in", 0), s.get("tokens_out", 0),
                     s.get("cost_usd", 0.0), s.get("seconds", 0.0), 1 if failed else 0, today(), visitor))


# ---------------------------------------------------------------- options from the form (never trusted as is)
def clean_options(raw):
    try:
        o = json.loads(raw or "{}")
        if not isinstance(o, dict):
            o = {}
    except ValueError:
        o = {}
    pick = lambda k, allowed, d: o.get(k) if o.get(k) in allowed else d
    colors = {k: v for k, v in (o.get("colors") or {}).items() if k in ("ink", "highlight", "notes") and HEX.match(str(v))}
    roles = {k: v for k, v in (o.get("role_colors") or {}).items()
             if k in ("main_idea", "key_term", "definition", "fact", "question", "key_quote") and HEX.match(str(v))}
    styles = [s for s in (o.get("styles") or []) if s in ("underline", "double", "squiggle", "circle", "box", "highlight")]
    return dict(DEFAULT_OPTIONS,
                focus=str(o.get("focus") or "").strip()[:300], teacher=str(o.get("teacher") or "").strip()[:3000],
                styles=styles or ["underline"], amount=pick("amount", ("few", "normal", "lots"), "normal"),
                notes=bool(o.get("notes", True)), hand_notes=bool(o.get("hand_notes")) and handwriting_available(),
                notes_amount=pick("notes_amount", ("few", "normal", "many"), "normal"),
                summary_box=bool(o.get("summary_box")), symbols=bool(o.get("symbols")),
                color_by_type=bool(o.get("color_by_type")), colors=colors, role_colors=roles,
                hand_size=pick("hand_size", ("small", "normal", "large"), "normal"),
                hand_style=pick("hand_style", ("neat", "normal", "messy"), "normal"),
                mark_style=pick("mark_style", ("neat", "normal", "messy"), "normal"),
                connectors=bool(o.get("connectors", True)))


# ---------------------------------------------------------------- jobs (in memory only)
def count_pages(data, filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in IMAGE_TYPES:
        return 1
    try:
        d = pymupdf.open(stream=data, filetype="pdf")
        n = 0 if d.needs_pass else len(d)
        d.close()
        return n
    except Exception:
        return 0


def _draw_job(job, exclude=()):
    job["pdf"] = render(job["engine"], exclude)
    job["exclude"] = sorted(exclude)
    if job["preview"]:
        job["pngs"] = previews(job["pdf"], dpi=60)
    job["version"] = job.get("version", 0) + 1


def run_job(job_id, data, filename, options):
    job = JOBS[job_id]

    def progress(pct, msg):
        job["pct"], job["msg"] = min(pct, 95), msg

    try:
        engine = prepare(data, filename, options, max_pages=MAX_PAGES_PER_FILE, progress=progress)
        job["engine"] = engine
        progress(92, "Drawing the annotations...")
        _draw_job(job)
        st = engine.stats
        record(job["visitor"], st)
        job.update(status="review" if job["preview"] else "done", pct=100, msg="Done", touched=time.time(),
                   stats={k: st[k] for k in ("pages", "marks", "notes", "symbols", "summary", "mode")})
    except UserError as e:
        job.update(status="error", msg=str(e), touched=time.time())
    except Exception as e:
        app.logger.error("job failed: %s: %s", type(e).__name__, str(e)[:200])   # never the file or its text
        job.update(status="error", msg="Something went wrong annotating that file. Please try again.",
                   touched=time.time())
        record(job["visitor"], failed=True)
    finally:
        del data


def forget(job_id):
    with JOBS_LOCK:
        j = JOBS.pop(job_id, None)
    if j and j.get("engine"):
        j["engine"].close()


def sweeper():
    """Forget jobs nobody has touched for RESULT_TTL_MIN minutes."""
    while True:
        time.sleep(30)
        cutoff = time.time() - RESULT_TTL_MIN * 60
        for jid in [j for j, v in list(JOBS.items()) if v.get("touched") and v["touched"] < cutoff]:
            forget(jid)


threading.Thread(target=sweeper, daemon=True).start()


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    v = visitor_id()
    return render_template("index.html", app_name=APP_NAME, free=FREE_PAGES_PER_DAY,
                           left=max(0, FREE_PAGES_PER_DAY - pages_used_today(v)), max_pages=MAX_PAGES_PER_FILE,
                           max_mb=MAX_UPLOAD_MB, my_hand=handwriting_available())


@app.get("/privacy")
def privacy():
    return render_template("privacy.html", app_name=APP_NAME, contact=CONTACT_EMAIL, ttl=RESULT_TTL_MIN,
                           updated="November 17, 2026")


@app.get("/terms")
def terms():
    return render_template("terms.html", app_name=APP_NAME, contact=CONTACT_EMAIL, free=FREE_PAGES_PER_DAY,
                           updated="November 17, 2026")


@app.post("/api/annotate")
def start():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Choose a PDF or a photo of a page first."), 400
    filename = os.path.basename(f.filename)[-120:]
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext != "pdf" and ext not in IMAGE_TYPES:
        return jsonify(error="Upload a PDF, or a photo of a page (PNG or JPG)."), 400
    data = f.read()
    pages = count_pages(data, filename)
    if pages == 0:
        return jsonify(error="That file couldn't be opened (is it password-protected or damaged?)."), 400
    if pages > MAX_PAGES_PER_FILE:
        return jsonify(error=f"That file has {pages} pages. The limit is {MAX_PAGES_PER_FILE} per file."), 400
    v = visitor_id()
    with JOBS_LOCK:
        if any(j["visitor"] == v and j["status"] == "working" for j in JOBS.values()):
            return jsonify(error="You already have a file being annotated. Wait for it to finish."), 429
    left = FREE_PAGES_PER_DAY - pages_used_today(v)
    if pages > left:
        return jsonify(error=(f"That's {pages} pages, and you have {max(left, 0)} free pages left today. "
                              "The limit resets at midnight.")), 429
    options = clean_options(request.form.get("options"))
    preview = request.form.get("preview") == "1"
    job_id = secrets.token_urlsafe(18)
    with JOBS_LOCK:
        JOBS[job_id] = {"visitor": v, "status": "working", "pct": 0, "msg": "Waiting to start...", "pages": pages,
                        "preview": preview, "name": os.path.splitext(filename)[0][:80] + "-annotated.pdf"}
    WORKERS.submit(run_job, job_id, data, filename, options)
    return jsonify(job=job_id, pages=pages)


def _job(job_id):
    j = JOBS.get(job_id)
    if not j or j["visitor"] != visitor_id():
        abort(404)
    if j.get("touched"):
        j["touched"] = time.time()
    return j


@app.get("/api/jobs/<job_id>")
def status(job_id):
    j = _job(job_id)
    out = dict(status=j["status"], pct=j["pct"], msg=j["msg"], stats=j.get("stats"),
               left=max(0, FREE_PAGES_PER_DAY - pages_used_today(j["visitor"])))
    if j["status"] == "review":
        out["items"] = [{"id": p["id"], "page": p["page"], "type": p["type"], "label": p["label"]}
                        for p in j["engine"].plan]
        out["exclude"], out["version"], out["page_count"] = j["exclude"], j["version"], len(j["pngs"])
    return jsonify(out)


@app.get("/api/jobs/<job_id>/page/<int:n>")
def page_png(job_id, n):
    j = _job(job_id)
    if j["status"] != "review" or not 0 <= n < len(j.get("pngs") or []):
        abort(404)
    return send_file(io.BytesIO(j["pngs"][n]), mimetype="image/png", max_age=0)


@app.post("/api/jobs/<job_id>/redraw")
def redraw(job_id):
    j = _job(job_id)
    if j["status"] != "review":
        abort(404)
    ids = {p["id"] for p in j["engine"].plan}
    exclude = {int(i) for i in (request.get_json(silent=True) or {}).get("exclude", []) if str(i).isdigit() and int(i) in ids}
    _draw_job(j, exclude)
    return jsonify(ok=True, version=j["version"])


@app.get("/api/jobs/<job_id>/download")
def download(job_id):
    j = _job(job_id)
    if j["status"] not in ("done", "review") or not j.get("pdf"):
        abort(404)
    data, name = j["pdf"], j["name"]
    forget(job_id)                           # gone from memory as soon as it's handed over
    return send_file(io.BytesIO(data), mimetype="application/pdf", as_attachment=True, download_name=name)


@app.get("/admin/usage")
def admin_usage():
    if not secrets.compare_digest(request.args.get("key", ""), ADMIN_KEY):
        abort(404)
    with db() as con:
        days = con.execute("""SELECT day, COUNT(DISTINCT visitor), SUM(jobs), SUM(pages), SUM(pages_ocr),
                              SUM(tokens_in), SUM(tokens_out), SUM(cost_usd), SUM(seconds), SUM(failures)
                              FROM usage GROUP BY day ORDER BY day DESC LIMIT 60""").fetchall()
        top = con.execute("""SELECT visitor, SUM(pages), SUM(cost_usd) FROM usage WHERE day >= ?
                             GROUP BY visitor ORDER BY SUM(cost_usd) DESC LIMIT 20""",
                          ((dt.date.today() - dt.timedelta(days=30)).isoformat(),)).fetchall()
    return render_template("admin.html", app_name=APP_NAME, days=days, top=top)


@app.errorhandler(413)
def too_big(_):
    return jsonify(error=f"That file is bigger than {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    print(f"\n  {APP_NAME} running at http://127.0.0.1:5000")
    print(f"  Usage and AI costs: http://127.0.0.1:5000/admin/usage?key={ADMIN_KEY}\n")
    app.run(host="127.0.0.1", port=5000, threaded=True)
