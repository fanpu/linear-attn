"""Contact sheet of the best images (declared: thumbnails letterboxed on cream, labels below)."""
from PIL import Image, ImageDraw
from common import *
items = [("diptych_adam_vs_sgd_r15.png", "Two shadows of the same data (Adam vs SGD)"),
         ("typology_mnist_adam_raw_rw0_s0_panel.png", "Typology, 20 rounds, Adam, raw pixels"),
         ("object_mnist_adam_raw_rw0_s0_r15_wall_medium.png", "Simulated backlit sheet, 8,275 holes"),
         ("plate_nulls_r15.png", "Null controls at 3.5% of input weights"),
         ("typology_mnist_adam_norm_rw0_s0_panel.png", "Typology, 20 rounds, Adam, standardised pixels"),
         ("plate_classes_r15.png", "One mask, ten classes (a null)")]
T = 900; pad = 50; lab = 60
W = 3 * T + 4 * pad; H = 2 * (T + lab) + 3 * pad
im = Image.new("RGB", (W, H), CREAM); d = ImageDraw.Draw(im)
for k, (f, s) in enumerate(items):
    i, j = divmod(k, 3)
    t = Image.open(f"{GALLERY}/{f}").convert("RGB"); t.thumbnail((T, T), Image.LANCZOS)
    x = pad + j * (T + pad); y = pad + i * (T + lab + pad)
    im.paste(t, (x + (T - t.width) // 2, y + (T - t.height) // 2))
    text(d, (x, y + T + 15), s, 22)
im.save(f"{GALLERY}/contact_sheet.png"); print(im.size)
