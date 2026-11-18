"""Hand-drawn marks. Every shape comes out a little different each time, like a real pen: uneven size, tilt, wobble,
gaps and overshoots. All take a box (x1, y1, x2, y2) and a scale (about 1.0 for 12 pt text) and return a list of
strokes, each a list of (x, y) points."""
import math
import random


def _smooth(n, amp, waves=3):
    comps = [(random.uniform(0.6, 2.6), random.uniform(0, math.tau), random.uniform(0.4, 1.0)) for _ in range(waves)]
    tot = sum(c[2] for c in comps)
    return [amp * sum(a * math.sin(math.tau * f * i / max(n, 1) + ph) for f, ph, a in comps) / tot for i in range(n + 1)]


def _rotate(points, cx, cy, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in points]


def line(x1, y1, x2, y2, scale, step=6, wobble=0.6):
    """A hand-drawn straight-ish line: slow wobble plus a little shake."""
    n = max(int(math.hypot(x2 - x1, y2 - y1) / (step * max(scale, 0.3))), 4)
    w = _smooth(n, wobble * scale * random.uniform(0.7, 1.5))
    return [(x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n + w[i] + random.uniform(-0.2, 0.2) * scale)
            for i in range(n + 1)]


def underline(box, scale):
    x1, y1, x2, y2 = box
    y = y2 + random.uniform(0.8, 2.2) * scale
    xa = x1 - random.uniform(-1.5, 3) * scale
    xb = x2 + random.uniform(-2, 4) * scale
    return [line(xa, y + random.uniform(-0.6, 0.6) * scale, xb, y + random.uniform(-1.6, 1.6) * scale, scale,
                 wobble=random.uniform(0.4, 1.0))]


def dotted_underline(box, scale):
    """A light dashed underline (marks what a margin question is about)."""
    x1, y1, x2, y2 = box
    y = y2 + 1.4 * scale
    out, x, dash, gap = [], x1, 3.2 * scale, 2.4 * scale
    while x < x2:
        out.append([(x, y + random.uniform(-0.3, 0.3) * scale), (min(x2, x + dash), y + random.uniform(-0.3, 0.3) * scale)])
        x += dash + gap
    return out


def squiggle(box, scale):
    x1, y1, x2, y2 = box
    y0 = y2 + random.uniform(2, 3.5) * scale
    xa, xb = x1 - random.uniform(-1, 2) * scale, x2 + random.uniform(-2, 3) * scale
    n = max(int((xb - xa) / (1.6 * scale)), 10)
    amp, wl, drift = _smooth(n, 1.0), _smooth(n, 0.3), _smooth(n, 0.9 * scale)
    base_amp, base_wl = random.uniform(1.3, 2.3) * scale, random.uniform(7, 10) * scale
    phase, pts = random.uniform(0, math.tau), []
    for i in range(n + 1):
        x = xa + (xb - xa) * i / n
        phase += math.tau * ((xb - xa) / n) / (base_wl * (1 + wl[i]))
        pts.append((x, y0 + drift[i] + base_amp * (1 + 0.45 * amp[i]) * math.sin(phase)))
    return [pts]


def circle(box, scale):
    x1, y1, x2, y2 = box
    rx = (x2 - x1) / 2 + random.uniform(4, 8) * scale
    ry = (y2 - y1) / 2 + random.uniform(2.5, 5) * scale
    cx = (x1 + x2) / 2 + random.uniform(-1.5, 1.5) * scale
    cy = (y1 + y2) / 2 + random.uniform(-1, 1) * scale
    power, turns = random.uniform(2.2, 3.2), random.uniform(0.98, 1.13)
    tilt, spiral = random.uniform(-5, 5), random.uniform(0.0, 0.06)
    perim = 2 * math.pi * math.sqrt((rx * rx + ry * ry) / 2)
    n = max(int(perim / (4 * max(scale, 0.3))), 30)
    th0, direction = random.uniform(-math.pi, math.pi), random.choice([1, 1, 1, -1])
    wob = _smooth(n, 0.06, waves=4)
    pts = []
    for i in range(n + 1):
        f = i / n
        th = th0 + direction * turns * 2 * math.pi * f
        c, s = math.cos(th), math.sin(th)
        w = 1 + wob[i] - spiral * f
        pts.append((cx + rx * w * math.copysign(abs(c) ** (2 / power), c), cy + ry * w * math.copysign(abs(s) ** (2 / power), s)))
    return [_rotate(pts, cx, cy, tilt)]


def highlight(box, scale):
    """The centre line of a highlighter stroke (drawn thick and see-through)."""
    x1, y1, x2, y2 = box
    y = (y1 + y2) / 2 + random.uniform(-0.4, 0.6) * scale
    return [line(x1 - random.uniform(0, 2.5) * scale, y, x2 + random.uniform(0, 3) * scale,
                 y + random.uniform(-0.7, 0.7) * scale, scale, wobble=random.uniform(0.2, 0.5))]


def connector(x1, y1, x2, y2, scale):
    """A loose line from a margin note back to the text it's about."""
    return [line(x1, y1, x2, y2, scale, step=8, wobble=0.5)]


def double(box, scale):
    x1, y1, x2, y2 = box
    y = y2 + random.uniform(0.8, 1.8) * scale
    gap = random.uniform(2.4, 3.6) * scale
    return [line(x1 - random.uniform(-1, 2.5) * scale, y, x2 + random.uniform(-1.5, 3.5) * scale,
                 y + random.uniform(-1.2, 1.2) * scale, scale, wobble=random.uniform(0.4, 0.9)),
            line(x1 + random.uniform(-1.5, 3) * scale, y + gap, x2 + random.uniform(-3, 4) * scale,
                 y + gap + random.uniform(-1.5, 1.5) * scale, scale, wobble=random.uniform(0.4, 0.9))]


def box(b, scale):
    """A box drawn in one go: four slightly crooked sides, uneven corners, overshooting where it meets itself."""
    x1, y1, x2, y2 = b
    px, py = random.uniform(3, 6) * scale, random.uniform(2, 4) * scale
    j = lambda: random.uniform(-1.6, 1.6) * scale
    corners = [(x1 - px + j(), y1 - py + j()), (x2 + px + j(), y1 - py + j()),
               (x2 + px + j(), y2 + py + j()), (x1 - px + j(), y2 + py + j())]
    k = random.randrange(4)
    corners = corners[k:] + corners[:k]
    closing = (corners[0][0] + j(), corners[0][1] + j())
    nxt = corners[1]
    over = random.uniform(-0.04, 0.12)
    tail = (closing[0] + (nxt[0] - closing[0]) * over, closing[1] + (nxt[1] - closing[1]) * over)
    path, pts = [corners[0], corners[1], corners[2], corners[3], closing, tail], []
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        seg = line(ax, ay, bx, by, scale, wobble=random.uniform(0.5, 1.1))
        pts += seg if not pts else seg[1:]
    return [_rotate(pts, (x1 + x2) / 2, (y1 + y2) / 2, random.uniform(-1.5, 1.5))]


MARKS = {"underline": underline, "double": double, "squiggle": squiggle, "circle": circle, "box": box,
         "highlight": highlight}


# ---------------------------------------------------------------- margin symbols (box = the square the symbol sits in)
def _dot(x, y, scale):
    r = 0.6 * scale
    return [(x - r * 0.5, y - r * 0.3), (x + r * 0.4, y + r * 0.2), (x - r * 0.2, y + r * 0.5)]


def _jiggle(pts, scale, amt=0.2):
    return [(x + random.uniform(-amt, amt) * scale, y + random.uniform(-amt, amt) * scale) for x, y in pts]


def sym_question(b, scale):
    x1, y1, x2, y2 = b
    s = y2 - y1
    cx = (x1 + x2) / 2
    r, cy = 0.24 * s * random.uniform(0.9, 1.1), y1 + 0.27 * s
    hook = [(cx + r * math.cos(math.radians(a)), cy + r * 1.05 * math.sin(math.radians(a))) for a in range(200, 420, 12)]
    hook += [(cx + 0.02 * s, y1 + 0.5 * s), (cx, y1 + 0.66 * s)]
    return [_rotate(_jiggle(hook, scale), cx, y1 + 0.5 * s, random.uniform(-8, 8)), _dot(cx, y1 + 0.9 * s, scale)]


def sym_bang(b, scale):
    x1, y1, x2, y2 = b
    s, cx = y2 - y1, (x1 + x2) / 2
    lean = random.uniform(-0.06, 0.06) * s
    return [line(cx + lean, y1 + 0.04 * s, cx, y1 + 0.66 * s, scale, step=3, wobble=0.2), _dot(cx, y1 + 0.9 * s, scale)]


def sym_star(b, scale):
    """Drawn as the outline of a star (it doesn't cross itself, so it stays crisp when small)."""
    x1, y1, x2, y2 = b
    s = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ro, ri, a0 = 0.5 * s, 0.22 * s, -90 + random.uniform(-8, 8)
    pts = []
    for k in range(10):
        r = (ro if k % 2 == 0 else ri) * random.uniform(0.92, 1.06)
        a = math.radians(a0 + k * 36)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append(pts[0])
    out = []
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        seg = line(ax, ay, bx, by, scale, step=2, wobble=0.15)
        out += seg if not out else seg[1:]
    return [out]


def sym_arrow(b, scale):
    x1, y1, x2, y2 = b
    s = y2 - y1
    cy = (y1 + y2) / 2
    head = 0.3 * s * random.uniform(0.85, 1.15)
    return [line(x1 - 0.1 * s, cy, x2, cy, scale, step=3, wobble=0.2),
            [(x2 - head, cy - head * 0.85), (x2, cy), (x2 - head, cy + head * 0.85)]]


SYMBOLS = {"?": sym_question, "!": sym_bang, "star": sym_star, "arrow": sym_arrow}


def look(strokes, how, scale):
    """Neat / normal / messy: how hand-drawn the marks look."""
    if how == "neat":                        # smooth out the shake
        out = []
        for s in strokes:
            if len(s) < 5:
                out.append(s)
                continue
            sm = [s[0]] + [((s[i - 1][0] + s[i][0] + s[i + 1][0]) / 3, (s[i - 1][1] + s[i][1] + s[i + 1][1]) / 3)
                           for i in range(1, len(s) - 1)] + [s[-1]]
            out.append(sm)
        return out
    if how == "messy":                       # more shake, and the line drifts
        out = []
        for s in strokes:
            w = _smooth(len(s) - 1, 1.2 * scale)
            out.append([(x + random.uniform(-0.45, 0.45) * scale, y + w[i] + random.uniform(-0.45, 0.45) * scale)
                        for i, (x, y) in enumerate(s)])
        return out
    return strokes
