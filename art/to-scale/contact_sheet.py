"""gallery/contact_sheet.png: the best eight, each fitted to a cell on cream."""
import os
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery")
ITEMS = [("banner_film_first.png", "film, first frame: 1,024 bars"), ("banner_film_last.png", "film, last frame: 39 m"),
         ("typology.png", "nine models, same scale"), ("superweight_detail_1.7b.png", "Qwen3-1.7B super weight (detail)"),
         ("superweight_field_1.7b.png", "Qwen3-1.7B: 12.6M weights, one ring"), ("committee.png", "Qwen3-0.6B: the committee of six"),
         ("fragility_detail.png", "Qwen3-0.6B fragility (detail)"), ("fragility_calibration.png", "first order vs exact")]
cw, ch, pad, lab = 760, 760, 40, 50
cols = 4
rows = (len(ITEMS) + cols - 1) // cols
W, H = cols * cw + (cols + 1) * pad, rows * (ch + lab) + (rows + 1) * pad
sheet = Image.new("RGB", (W, H), (242, 237, 227))
dr = ImageDraw.Draw(sheet)
f = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/C059-Roman.otf", 26)
for k, (fn, t) in enumerate(ITEMS):
    im = Image.open(os.path.join(G, fn)).convert("RGB")
    im.thumbnail((cw, ch), Image.LANCZOS)
    r, c = divmod(k, cols)
    x = pad + c * (cw + pad) + (cw - im.width) // 2
    y = pad + r * (ch + lab + pad) + (ch - im.height) // 2
    sheet.paste(im, (x, y))
    dr.rectangle([x - 1, y - 1, x + im.width, y + im.height], outline=(200, 195, 188))
    dr.text((pad + c * (cw + pad), pad + r * (ch + lab + pad) + ch + 10), t, font=f, fill=(27, 26, 31))
sheet.save(os.path.join(G, "contact_sheet.png"), optimize=True)
print(sheet.size)
