"""gallery/contact_sheet.png: the best images, each scaled to a common height per row."""
import os
from PIL import Image
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery")
rows = [["one_road_atlas_tr.png", "plate_roads_cifar_te.png", "long_way_round_modadd_H.png"],
        ["other_road_tr.png", "one_road_atlas_te.png", "straight_line_becher.png"]]
H, pad, bg = 1100, 40, (242, 235, 220)
ims = [[Image.open(os.path.join(G, f)).convert("RGB") for f in r] for r in rows]
ims = [[im.resize((int(im.width * H / im.height), H), Image.LANCZOS) for im in r] for r in ims]
W = max(sum(im.width for im in r) + pad * (len(r) + 1) for r in ims)
sheet = Image.new("RGB", (W, len(rows) * (H + pad) + pad), bg)
for j, r in enumerate(ims):
    x = pad + (W - (sum(im.width for im in r) + pad * (len(r) + 1))) // 2
    for im in r:
        sheet.paste(im, (x, pad + j * (H + pad))); x += im.width + pad
sheet.save(os.path.join(G, "contact_sheet.png")); print(sheet.size)
