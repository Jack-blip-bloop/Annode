"""
Your handwriting, for the OneNote annotator.

  pythonw handwriting.py setup                       record your handwriting (one page of boxes, one character per box)
  python  handwriting.py preview "some text" out.png  write text in your handwriting into a picture (for checking)
  python  handwriting.py rebuild                      rebuild the letters from the saved recording (after code changes)

How it works
  1. Setup shows one page with a box for every character. You write one character in each box, sitting on the line.
     Because every box belongs to one character, there is nothing to guess about which stroke is which letter.
  2. Every letter is stored relative to the baseline, in units of your x-height (the height of an 'x').
  3. To write, each letter is freshly altered every time: size, slant, tilt, a smooth warp of its shape, longer/shorter
     and differently curved strokes, a little shake, plus drift along the line. So no two letters come out the same.
"""
import json
import math
import os
import random
import sys
import time

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(APP_DIR, "handwriting_raw.json")   # what you wrote, exactly as recorded
LIB_PATH = os.path.join(APP_DIR, "handwriting.json")       # the letters, ready to write with

# One box per character, in rows of 13
GRID_ROWS = [
    "abcdefghijklm",
    "nopqrstuvwxyz",
    "ABCDEFGHIJKLM",
    "NOPQRSTUVWXYZ",
    "0123456789.,!",
    "?'\"-:;()%&$/+",
    "=@#*",
]
GRID_CHARS = [c for row in GRID_ROWS for c in row]

XH_CHARS = set("acemnorsuvwxz")                                    # letters that are exactly x-height tall
BASE_CHARS = set("acemnorsuvwxzbdfhiklt") | set("ABCDEFGHIKLMNOPRSTUVWXYZ") | set("0123456789")  # sit on the baseline


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _median(v, default):
    v = sorted(v)
    return v[len(v) // 2] if v else default


def build_library(raw):
    """Turn the recorded boxes into letters stored relative to the baseline, in x-height units.
    raw = {"cells": [{"ch", "base" (y of the box's line, px), "xh" (guide x-height, px), "strokes": [[[x, y, t], ...]]}]}"""
    entries = []
    for c in raw.get("cells", []):
        strokes = [s for s in c.get("strokes", []) if s]
        if not strokes:
            continue
        entries.append({"ch": c["ch"], "strokes": strokes, "base": c["base"], "xh": c.get("xh", 30.0),
                        "box": _bbox([p for s in strokes for p in s])})
    if not entries:
        raise ValueError("Nothing was written in the boxes")

    xh = [e["box"][3] - e["box"][1] for e in entries if e["ch"] in XH_CHARS]
    U = _median(xh, entries[0]["xh"])
    # If you tend to write a little above or below the guide line, measure that once and use it for every box,
    # so letters still line up with each other.
    offset = _median([e["box"][3] - e["base"] for e in entries if e["ch"] in BASE_CHARS], 0.0)

    chars, widths = {}, []
    for e in entries:
        x0, b = e["box"][0], e["base"] + offset
        strokes = [[[round((p[0] - x0) / U, 4), round((p[1] - b) / U, 4)] for p in s] for s in e["strokes"]]
        w = (e["box"][2] - e["box"][0]) / U
        chars.setdefault(e["ch"], []).append({"strokes": strokes, "w": round(w, 4)})
        if e["ch"] in XH_CHARS:
            widths.append(w)
    # Spacing can't be measured from separate boxes, so it's set from how wide your letters are.
    wm = float(_median(widths, 0.9))
    return {"version": 2, "unit_px": round(float(U), 2), "chars": chars,
            "gap": [round(0.36 * wm, 4), round(0.08 * wm, 4)],
            "space": [round(1.45 * wm, 4), round(0.18 * wm, 4)], "variation": 1.0}


def load_library():
    try:
        with open(LIB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------- writing (with fresh variation every time)
def _resample(stroke, n):
    """n points evenly spaced along a stroke."""
    p = np.asarray(stroke, dtype=float)
    if len(p) == 1:
        return np.repeat(p, n, axis=0)
    seg = np.hypot(*np.diff(p, axis=0).T)
    L = np.concatenate([[0], np.cumsum(seg)])
    if L[-1] < 1e-9:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0, L[-1], n)
    return np.stack([np.interp(t, L, p[:, 0]), np.interp(t, L, p[:, 1])], axis=1)


def _length(p):
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def _smooth1d(n, amp, waves=2):
    i = np.arange(n)
    out = np.zeros(n)
    for _ in range(waves):
        out += np.sin(2 * math.pi * random.uniform(0.3, 1.5) * i / max(n - 1, 1) + random.uniform(0, 2 * math.pi))
    return out * amp / waves


class Writer:
    """Writes text with the recorded letters, altering every letter each time so it reads as real handwriting."""

    def __init__(self, lib, variation=None):
        self.lib = lib
        self.chars = lib["chars"]
        self.v = float(variation if variation is not None else lib.get("variation", 1.0))
        self.gap = lib.get("gap", [0.3, 0.1])
        self.space = lib.get("space", [1.0, 0.2])
        self.last_pick = {}

    def missing(self, text):
        return sorted({c for c in text if not c.isspace() and self._samples(c)[0] is None})

    def _samples(self, ch):
        if ch in self.chars and self.chars[ch]:
            return self.chars[ch], 1.0
        other = ch.upper() if ch.islower() else ch.lower()
        if ch.isalpha() and other in self.chars and self.chars[other]:
            return self.chars[other], (0.75 if ch.islower() else 1.3)   # borrow the other case, resized
        return None, 1.0

    def glyph(self, ch):
        """One freshly altered copy of a character: (strokes as Nx2 arrays with left edge at x=0, width)."""
        samples, size = self._samples(ch)
        if samples is None:
            return None, 0.6
        v = self.v
        k = random.randrange(len(samples))
        if len(samples) > 1 and k == self.last_pick.get(ch):
            k = (k + random.randrange(1, len(samples))) % len(samples)   # never the same sample twice in a row
        self.last_pick[ch] = k
        base = samples[k]
        strokes = []
        for s in base["strokes"]:
            p = np.asarray(s, dtype=float)
            n = int(min(max(_length(p) / 0.05, 6), 120))
            strokes.append(_resample(p, n))

        # blend with another of your samples of this letter (when they were written the same way)
        partners = [s for i, s in enumerate(samples) if i != k and len(s["strokes"]) == len(base["strokes"])]
        if partners and random.random() < 0.6:
            other = random.choice(partners)
            mixed, ok = [], True
            for a, s in zip(strokes, other["strokes"]):
                b = _resample(s, len(a))
                if np.abs(a - b).mean() > 0.3:
                    ok = False
                    break
                w = random.uniform(0.15, 0.5)
                mixed.append(a * (1 - w) + b * w)
            if ok:
                strokes = mixed

        pts = np.concatenate(strokes)
        cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
        cy = (pts[:, 1].min() + pts[:, 1].max()) / 2

        sx = size * (1 + random.gauss(0, 0.05 * v))
        sy = size * (1 + random.gauss(0, 0.05 * v))
        shear = random.gauss(0, 0.045 * v)
        rot = math.radians(random.gauss(0, 1.8 * v))
        cr, sr = math.cos(rot), math.sin(rot)
        warp = [(random.uniform(0.012, 0.035) * v, random.uniform(1.5, 4.0), random.uniform(0, 2 * math.pi))
                for _ in range(4)]

        out = []
        for p in strokes:
            p = p.copy()
            # longer or shorter stroke ends (extend along the stroke's own direction, or trim)
            if len(p) >= 3:
                L = max(_length(p), 1e-6)
                for end in (0, 1):
                    d = random.uniform(-0.06, 0.05) * v * L
                    if end == 0:
                        if d < 0:
                            p = _trim(p, -d, from_start=True)
                        else:
                            t = p[0] - p[min(2, len(p) - 1)]
                            p = np.vstack([p[0] + t / (np.linalg.norm(t) + 1e-9) * d, p])
                    else:
                        if d < 0:
                            p = _trim(p, -d, from_start=False)
                        else:
                            t = p[-1] - p[max(-3, -len(p))]
                            p = np.vstack([p, p[-1] + t / (np.linalg.norm(t) + 1e-9) * d])
            # a slightly different curve
            if len(p) >= 3:
                chord = p[-1] - p[0]
                cl = np.linalg.norm(chord)
                if cl > 0.15:
                    nrm = np.array([-chord[1], chord[0]]) / cl
                    s = np.linspace(0, 1, len(p))
                    p = p + np.outer(np.sin(np.pi * s) * random.uniform(-0.035, 0.035) * v * min(cl, 1.5), nrm)
            # whole-letter changes: size, slant, tilt, smooth warp
            x, y = p[:, 0] - cx, p[:, 1] - cy
            x, y = x * sx, y * sy
            x = x - shear * y
            x, y = x * cr - y * sr, x * sr + y * cr
            for a, f, ph in warp[:2]:
                x = x + a * np.sin(f * y + ph)
            for a, f, ph in warp[2:]:
                y = y + a * np.sin(f * x + ph)
            # a little shake
            x = x + _smooth1d(len(x), 0.008 * v, 3)
            y = y + _smooth1d(len(y), 0.008 * v, 3)
            q = np.stack([x + cx * sx, y + cy * sy], axis=1)
            if len(q) == 1:
                q = np.vstack([q, q + [0.02, 0.02]])
            out.append(q)
        allp = np.concatenate(out)
        left = allp[:, 0].min()
        for q in out:
            q[:, 0] -= left
        return out, float(allp[:, 0].max() - left)

    def write(self, text, max_width=None, line_pitch=2.75):
        """Lay text out in lines (wrapping at max_width units). Returns strokes as lists of (x, y) in x-height units,
        with the first baseline at y = 0 (y grows downward, like the screen)."""
        v = self.v
        words = text.split()
        lines, cur, x = [], [], 0.0
        for w in words:
            glyphs = [(c,) + self.glyph(c) for c in w]
            gaps = [max(0.04, random.gauss(self.gap[0], self.gap[1] * 0.7)) for _ in glyphs]
            ww = sum(g[2] for g in glyphs) + sum(gaps[:-1])
            sp = max(0.35, random.gauss(self.space[0], self.space[1] * 0.8)) if cur else 0.0
            if max_width and cur and x + sp + ww > max_width:
                lines.append(cur)
                cur, x, sp = [], 0.0, 0.0
            x += sp
            for (c, strokes, gw), g in zip(glyphs, gaps):
                if strokes:
                    cur.append((x, strokes))
                x += gw + g
            x -= gaps[-1] if gaps else 0
        if cur:
            lines.append(cur)

        result = []
        for li, items in enumerate(lines):
            by = li * line_pitch + random.gauss(0, 0.06 * v)
            # slow drift along the line: the baseline wanders, the size and slant change a little
            wl = [random.uniform(7, 16) for _ in range(3)]
            ph = [random.uniform(0, 2 * math.pi) for _ in range(3)]
            amp_b, amp_s, amp_k = 0.07 * v, 0.035 * v, 0.03 * v
            tilt = random.gauss(0, 0.012 * v)
            for gx, strokes in items:
                drift = amp_b * math.sin(2 * math.pi * gx / wl[0] + ph[0]) + tilt * gx
                size = 1 + amp_s * math.sin(2 * math.pi * gx / wl[1] + ph[1])
                slant = amp_k * math.sin(2 * math.pi * gx / wl[2] + ph[2])
                lift = random.gauss(0, 0.035 * v)
                for q in strokes:
                    xx = q[:, 0] * size
                    yy = q[:, 1] * size
                    xx = xx - slant * yy
                    result.append(list(zip((gx + xx).tolist(), (by + drift + lift + yy).tolist())))
        return result


def _trim(p, amount, from_start):
    """Cut `amount` (in units of length) off one end of a stroke."""
    if from_start:
        p = p[::-1]
    seg = np.hypot(*np.diff(p, axis=0).T)
    L = np.concatenate([[0], np.cumsum(seg)])
    keep = L[-1] - amount
    if keep <= L[-1] * 0.5:
        return p[::-1] if from_start else p
    i = int(np.searchsorted(L, keep))
    i = max(1, min(i, len(p) - 1))
    f = (keep - L[i - 1]) / max(seg[i - 1], 1e-9)
    end = p[i - 1] + (p[i] - p[i - 1]) * f
    q = np.vstack([p[:i], end])
    return q[::-1] if from_start else q


def render_png(strokes, path, px_per_unit=26, color=(40, 40, 160), width=2.2, margin=20):
    """Draw handwriting strokes into a picture (for previews)."""
    from PIL import Image, ImageDraw
    if not strokes:
        Image.new("RGB", (200, 80), "white").save(path)
        return
    ss = 3
    allp = np.array([p for s in strokes for p in s])
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0)
    W = int((x1 - x0) * px_per_unit + 2 * margin)
    H = int((y1 - y0) * px_per_unit + 2 * margin)
    img = Image.new("RGB", (W * ss, H * ss), "white")
    d = ImageDraw.Draw(img)
    lw = max(1, int(width * ss))
    for s in strokes:
        pts = [((x - x0) * px_per_unit * ss + margin * ss, (y - y0) * px_per_unit * ss + margin * ss) for x, y in s]
        d.line(pts, fill=color, width=lw, joint="curve")
        for x, y in (pts[0], pts[-1]):
            d.ellipse((x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2), fill=color)
    img.resize((W, H), Image.LANCZOS).save(path)


# ---------------------------------------------------------------- setup window
def setup():
    import tkinter as tk
    from tkinter import messagebox
    from PIL import ImageTk, Image
    import tempfile
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

    root = tk.Tk()
    root.title("Set up my handwriting")
    sc = root.winfo_fpixels("1i") / 96
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    cols, rows = 13, len(GRID_ROWS)
    cell = int(min(sw * 0.9 / cols, sh * 0.66 / rows))
    W, H = cols * cell + 2, rows * cell + 2
    bg, fg, accent, guide_c = "#f4f4f4", "#1a1a1a", "#b03030", "#c9d3e6"
    root.configure(bg=bg)

    tk.Label(root, text="Print one character in each box. Don't connect anything.", font=("Segoe UI", 14, "bold"),
             fg=accent, bg=bg).pack(pady=(int(8 * sc), 0))
    tk.Label(root, text="Sit letters on the solid line. Small letters (a, c, e...) reach the dashed line; tall ones go "
                        "above it, and g, j, p, q, y hang below the line.", font=("Segoe UI", 10), fg=fg, bg=bg).pack()
    canvas = tk.Canvas(root, width=W, height=H, bg="white", highlightthickness=0, cursor="pencil")
    canvas.pack(padx=int(12 * sc), pady=(int(8 * sc), 0))
    status = tk.Label(root, text="", font=("Segoe UI", 10, "bold"), fg=accent, bg=bg)
    status.pack(pady=(int(4 * sc), 0))
    bar = tk.Frame(root, bg=bg)
    bar.pack(pady=int(8 * sc))

    base_frac, xh_frac = 0.66, 0.24          # where the line and the dashed x-height guide sit inside a box
    cells = []                               # (ch, x0, y0)
    for r, row in enumerate(GRID_ROWS):
        for c, ch in enumerate(row):
            cells.append((ch, c * cell + 1, r * cell + 1))

    def draw_grid():
        lw = max(1, int(sc))
        for ch, x0, y0 in cells:
            canvas.create_rectangle(x0, y0, x0 + cell, y0 + cell, outline="#d8d8d8", width=lw)
            canvas.create_text(x0 + 4 * sc, y0 + 3 * sc, text=ch, anchor="nw", fill="#9a9a9a",
                               font=("Segoe UI", 9, "bold"))
            yb = y0 + cell * base_frac
            canvas.create_line(x0 + 6 * sc, yb, x0 + cell - 6 * sc, yb, fill=guide_c, width=lw)
            yx = yb - cell * xh_frac
            canvas.create_line(x0 + 6 * sc, yx, x0 + cell - 6 * sc, yx, fill=guide_c, width=lw, dash=(3, 3))

    state = {"strokes": [], "cur": None}

    def redraw_ink():
        canvas.delete("ink")
        for s in state["strokes"]:
            if len(s) == 1:
                x, y, _ = s[0]
                r = max(1, int(1.5 * sc))
                canvas.create_oval(x - r, y - r, x + r, y + r, fill=fg, outline=fg, tags=("ink",))
            for (x1, y1, _), (x2, y2, _) in zip(s, s[1:]):
                canvas.create_line(x1, y1, x2, y2, fill=fg, width=max(2, int(2 * sc)), capstyle="round", tags=("ink",))

    # ---- capturing the pen
    # Windows' normal mouse messages drop points when a stylus moves fast, and pen "press and hold" / flick gestures
    # can swallow the start of a stroke. So: (1) switch those pen gestures off for this window, and (2) while the pen
    # is down, also read the pen position directly ~250 times a second, so no part of a stroke is missed.
    import ctypes
    from ctypes import wintypes
    u32 = ctypes.windll.user32

    def no_pen_gestures(hwnd):
        try:
            # press-and-hold, tap feedback, barrel feedback, flicks: all off
            u32.SetPropW(wintypes.HWND(hwnd), "MicrosoftTabletPenServiceProperty",
                         wintypes.HANDLE(0x00000001 | 0x00000008 | 0x00000010 | 0x00010000))
        except Exception:
            pass

    root.update_idletasks()
    no_pen_gestures(canvas.winfo_id())
    no_pen_gestures(u32.GetParent(root.winfo_id()) or root.winfo_id())

    def pen_pos():
        pt = wintypes.POINT()
        u32.GetCursorPos(ctypes.byref(pt))
        return pt.x - canvas.winfo_rootx(), pt.y - canvas.winfo_rooty()

    def add_point(x, y):
        cur = state["cur"]
        if cur is None:
            return
        px, py, _ = cur[-1]
        if abs(x - px) < 0.75 and abs(y - py) < 0.75:
            return  # same spot
        cur.append((x, y, time.time()))
        canvas.create_line(px, py, x, y, fill=fg, width=max(2, int(2 * sc)), capstyle="round", tags=("ink",))

    def poll():
        if state["cur"] is None:
            return
        if not (u32.GetAsyncKeyState(0x01) & 0x8000):   # pen lifted, even if the "up" message got lost
            finish()
            return
        add_point(*pen_pos())
        root.after(4, poll)

    def down(e):
        if state["cur"] is not None:
            finish()
        state["cur"] = [(e.x, e.y, time.time())]
        root.after(4, poll)

    def move(e):
        add_point(e.x, e.y)

    def finish():
        cur = state["cur"]
        state["cur"] = None
        if cur:
            state["strokes"].append(cur)
            if len(cur) == 1:
                redraw_ink()
        status.config(text="")

    def up(e):
        if state["cur"] is not None:
            add_point(e.x, e.y)
        finish()

    def undo():
        if state["strokes"]:
            state["strokes"].pop()
            redraw_ink()

    def clear_all():
        if state["strokes"] and messagebox.askyesno("Handwriting", "Clear everything you've written?"):
            state["strokes"] = []
            redraw_ink()

    def collect():
        """Put each stroke in the box its middle falls in."""
        by_cell = {i: [] for i in range(len(cells))}
        for s in state["strokes"]:
            x0, y0, x1, y1 = _bbox(s)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            c, r = int(cx // cell), int(cy // cell)
            if 0 <= r < len(GRID_ROWS) and 0 <= c < len(GRID_ROWS[r]):
                i = sum(len(GRID_ROWS[k]) for k in range(r)) + c
                by_cell[i].append([[round(x, 1), round(y, 1), round(t, 3)] for x, y, t in s])
        out = []
        for i, (ch, x0, y0) in enumerate(cells):
            out.append({"ch": ch, "base": y0 + cell * base_frac, "xh": cell * xh_frac, "strokes": by_cell[i]})
        return {"version": 2, "cells": out}

    def empty_boxes(raw):
        return [c["ch"] for c in raw["cells"] if not c["strokes"]]

    def preview():
        raw = collect()
        try:
            lib = build_library(raw)
        except ValueError as ex:
            status.config(text=str(ex))
            return
        w = Writer(lib)
        text = "Main idea: grace keeps pulling me back. Question: why is pride the hardest sin to see in yourself?"
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        render_png(w.write(text, max_width=24), path, px_per_unit=int(15 * sc), width=1.6 * sc)
        top = tk.Toplevel(root)
        top.title("Preview (different every time)")
        top.configure(bg="white")
        img = ImageTk.PhotoImage(Image.open(path))
        lbl = tk.Label(top, image=img, bg="white")
        lbl.image = img
        lbl.pack()
        os.remove(path)
        miss = empty_boxes(raw)
        if miss:
            tk.Label(top, text="Empty boxes (those characters get skipped): " + " ".join(miss), fg=accent,
                     bg="white").pack(pady=4)

    def save():
        raw = collect()
        miss = empty_boxes(raw)
        if len(miss) == len(cells):
            status.config(text="Write in the boxes first.")
            return
        if miss and not messagebox.askyesno("Handwriting", "These boxes are empty and will be skipped when writing:\n\n"
                                            + " ".join(miss) + "\n\nSave anyway?"):
            return
        try:
            lib = build_library(raw)
        except ValueError as ex:
            status.config(text=str(ex))
            return
        save_json(RAW_PATH, raw)
        save_json(LIB_PATH, lib)
        messagebox.showinfo("Handwriting", "Saved. The annotator will use this handwriting.")
        root.destroy()

    canvas.bind("<ButtonPress-1>", down)
    canvas.bind("<B1-Motion>", move)
    canvas.bind("<ButtonRelease-1>", up)
    for name, cmd in (("Undo stroke", undo), ("Clear all", clear_all), ("Preview", preview), ("Save", save)):
        tk.Button(bar, text=name, command=cmd, font=("Segoe UI", 11), relief="flat", bg="#dcdcdc",
                  activebackground="#c8c8c8", padx=int(14 * sc), pady=int(4 * sc)).pack(side="left", padx=int(6 * sc))
    root.bind("<Control-z>", lambda e: undo())
    draw_grid()
    root.mainloop()


# ---------------------------------------------------------------- command line
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "setup"
    if cmd == "setup":
        setup()
    elif cmd == "preview":
        lib = load_library()
        if not lib:
            sys.exit("No handwriting saved yet: run  handwriting.py setup")
        text = sys.argv[2] if len(sys.argv) > 2 else "The quick brown fox jumps over 13 lazy dogs!"
        out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(APP_DIR, "handwriting_preview.png")
        render_png(Writer(lib).write(text, max_width=30), out)
        print("saved", out)
    elif cmd == "rebuild":
        with open(RAW_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        save_json(LIB_PATH, build_library(raw))
        print("rebuilt", LIB_PATH)


if __name__ == "__main__":
    main()
