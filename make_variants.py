"""Make the harder test inputs from the real readings in tests/readings: scans, phone photos, rotated and cropped pages,
plus a few files that should be refused politely. Output: tests/inputs/ (the originals are copied there too, trimmed
to their first pages so the test run stays quick and cheap)."""
import glob
import io
import os
import random

import pymupdf
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, OUT = os.path.join(HERE, "readings"), os.path.join(HERE, "inputs")
PAGES = 3
os.makedirs(OUT, exist_ok=True)
random.seed(1)


def first_pages(path, n=PAGES):
    d = pymupdf.open(path)
    out = pymupdf.open()
    out.insert_pdf(d, from_page=0, to_page=min(n, len(d)) - 1)
    return out


def page_image(page, dpi, angle=0.0, noise=True):
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    if angle:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    if noise:                                   # a photocopier look: slight blur, grey paper, speckles
        img = img.filter(ImageFilter.GaussianBlur(0.6))
        px = img.load()
        for _ in range(img.width * img.height // 400):
            x, y = random.randrange(img.width), random.randrange(img.height)
            px[x, y] = random.randrange(120, 230)
        img = img.point(lambda v: int(v * 0.9 + 18))
    return img


def save_scan(doc, path, dpi=150):
    out = pymupdf.open()
    for p in doc:
        img = page_image(p, dpi, angle=random.uniform(-1.2, 1.2))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=70)
        np = out.new_page(width=p.rect.width, height=p.rect.height)
        np.insert_image(np.rect, stream=buf.getvalue())
    out.save(path)


readings = sorted(glob.glob(os.path.join(SRC, "*.pdf")))
for path in readings:
    name = os.path.splitext(os.path.basename(path))[0]
    doc = first_pages(path)
    doc.save(os.path.join(OUT, f"{name}.pdf"))                               # 1. the real PDF
    save_scan(doc, os.path.join(OUT, f"{name}-scan.pdf"))                    # 2. a scan of it
    print("made", name)

# 3. phone photos of single pages (one with an EXIF "rotate me" flag, the way phones save portrait shots)
for i, path in enumerate(readings[:4]):
    name = os.path.splitext(os.path.basename(path))[0]
    p = first_pages(path, 1)[0]
    img = page_image(p, 200, angle=random.uniform(-2.5, 2.5)).convert("RGB")
    if i % 2:
        img = img.rotate(90, expand=True)                                    # stored sideways...
        exif = Image.Exif()
        exif[0x0112] = 6                                                     # ...with "rotate 90° to view"
        img.save(os.path.join(OUT, f"{name}-photo.jpg"), quality=85, exif=exif)
    else:
        img.save(os.path.join(OUT, f"{name}-photo.png"))

# 4. awkward PDFs: rotated pages, a cropped page
d = first_pages(readings[0], 2)
d[0].set_rotation(90)
d.save(os.path.join(OUT, "edge-rotated-90.pdf"))
d = first_pages(readings[1], 2)
for p in d:
    r = p.rect
    p.set_cropbox(pymupdf.Rect(r.x0 + 40, r.y0 + 60, r.x1 - 30, r.y1 - 80))
d.save(os.path.join(OUT, "edge-cropped.pdf"))

# 5. files that must be refused politely
d = first_pages(readings[2], 1)
d.save(os.path.join(OUT, "bad-password.pdf"), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="secret")
blank = pymupdf.open()
blank.new_page()
blank.save(os.path.join(OUT, "bad-blank.pdf"))
with open(os.path.join(OUT, "bad-not-a-pdf.pdf"), "wb") as f:
    f.write(b"this is just text pretending to be a pdf\n" * 20)
print(len(os.listdir(OUT)), "test inputs in", OUT)
