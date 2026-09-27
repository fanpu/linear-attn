"""gallery/contact_sheet.png: the best images, each scaled to a common height, on cream."""
import os
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__)); G = os.path.join(ROOT, "gallery")
Image.MAX_IMAGE_PIXELS = None
rows = [["first_learned_hero.png", "specimen_first_last.png", "study_last50_labels.png"],
        ["becher_four_architectures.png"], ["agreement_plates.png"], ["agreement_table.png"]]
W, pad = 2200, 30
out = []
for r in rows:
    ims = [Image.open(os.path.join(G, f)).convert("RGB") for f in r]
    h = 700
    ims = [im.resize((int(im.width * h / im.height), h), Image.LANCZOS) for im in ims]
    tw = sum(i.width for i in ims) + pad * (len(ims) - 1)
    if tw > W:
        s = W / tw; ims = [im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS) for im in ims]
    out.append(ims)
H = sum(max(i.height for i in r) for r in out) + pad * (len(out) + 1)
sheet = Image.new("RGB", (W + 2 * pad, H), (242, 237, 225)); y = pad
for r in out:
    x = pad
    for im in r:
        sheet.paste(im, (x, y)); x += im.width + pad
    y += max(i.height for i in r) + pad
sheet.save(os.path.join(G, "contact_sheet.png"), optimize=True); print(sheet.size)
