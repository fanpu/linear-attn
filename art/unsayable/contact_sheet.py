"""gallery/contact_sheet.png from the chosen plates (all share one 3440 x 4864 frame)."""
import os, sys
from PIL import Image, ImageDraw, ImageFont
import look as L
import typeset as ts

HERE = os.path.dirname(os.path.abspath(__file__))
names = sys.argv[1:]
cols = 4 if len(names) > 6 else 3
rows = (len(names) + cols - 1) // cols
tw = 860
th = int(tw * L.H / L.W)
pad, lab = 50, 70
out = Image.new("RGB", (cols * tw + (cols + 1) * pad, rows * (th + lab) + (rows + 1) * pad), (225, 219, 206))
d = ImageDraw.Draw(out)
f = ImageFont.truetype(ts.F["text_i"], 34)
for k, n in enumerate(names):
    im = Image.open(os.path.join(HERE, "gallery", n)).convert("RGB")
    im = im.resize((tw, th), Image.LANCZOS)
    x = pad + (k % cols) * (tw + pad)
    y = pad + (k // cols) * (th + lab + pad)
    out.paste(im, (x, y))
    d.text((x, y + th + 14), n.replace(".png", ""), font=f, fill=L.INK, anchor="lt")
out.save(os.path.join(HERE, "gallery", "contact_sheet.png"), optimize=True)
print(out.size)
