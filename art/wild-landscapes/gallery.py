"""Assemble gallery/ from cache/renders (JPEG for the big stills, PNG contact sheet)."""
import os, shutil
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "cache", "renders")
G = os.path.join(HERE, "gallery")
os.makedirs(G, exist_ok=True)


def im(n):
    return Image.open(os.path.join(RD, n)).convert("RGB")


def row(names, gap=0):
    ims = [im(n) for n in names]
    h = min(i.height for i in ims)
    ims = [i.resize((round(i.width * h / i.height), h), Image.LANCZOS) for i in ims]
    c = Image.new("RGB", (sum(i.width for i in ims) + gap * (len(ims) - 1), h))
    x = 0
    for i in ims:
        c.paste(i, (x, 0)); x += i.width + gap
    return c


def save(img, name, q=90):
    p = os.path.join(G, name); img.save(p, quality=q, subsampling=0); return p


out = {}
out["hero"] = save(im("rnn_T30_hero_a.png"), "hero_rnn_crater.jpg")
out["hero_b"] = save(im("rnn_T30_hero_b.png"), "rnn_crater_raking.jpg")
out["horizon"] = save(row(["rnn_horizon_T10.png", "rnn_horizon_T30.png", "rnn_horizon_T100.png"]), "rnn_horizon_triptych.jpg")
out["tiny5"] = save(im("tiny_p4_s5.png"), "tiny_parity4_shattered.jpg")
out["tiny08"] = save(im("tiny_p4_s0.8.png"), "tiny_parity4_knot.jpg")
out["three"] = save(row(["three_sq_a0.png", "three_sq_a1.png"], gap=24), "three_minima_raw_vs_aligned.jpg")
out["hess"] = save(row(["rand_resnet56_s1_r128.png", "hess_resnet56_s0.05_r128.png"]), "study_random_vs_hessian.jpg")
out["gpt"] = save(row(["rand_resnet56_s1_r128.png", "gpt_adamw_s1_r128.png"]), "study_convnet_vs_transformer.jpg")
out["skip"] = save(row(["resnet56_noshort_s3_r128.png", "resnet56_s3_r128.png"]), "study_noskip_vs_skip_zoomout.jpg")
out["nsclose"] = save(im("resnet56_noshort_s3_r256_close.png"), "resnet56_noskip_crater.jpg", q=88)
shutil.copy(os.path.join(HERE, "pilot", "rnn_rough.png"), os.path.join(G, "study_structure_function.png"))

# contact sheet: the best images, labelled
tiles = [("hero_rnn_crater.jpg", "unrolled RNN, T=30: chaos highlands around the trained crater"),
         ("rnn_horizon_triptych.jpg", "same plane, T = 10 / 30 / 100"),
         ("tiny_parity4_shattered.jpg", "parity-4, 4 hidden units: 317 Boolean functions on one plane"),
         ("three_minima_raw_vs_aligned.jpg", "three MLPs: raw plane vs permutation-aligned"),
         ("study_random_vs_hessian.jpg", "ResNet-56: random vs top-2 Hessian directions"),
         ("rnn_crater_raking.jpg", "unrolled RNN, raking light"),
         ("study_noskip_vs_skip_zoomout.jpg", "ResNet-56 no-skip vs skip, +-3 units"),
         ("study_convnet_vs_transformer.jpg", "ResNet-56 vs small GPT, same treatment")]
W, cols = 1100, 2
try:
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
except Exception:
    fnt = ImageFont.load_default()
cells = []
for f, cap in tiles:
    a = Image.open(os.path.join(G, f)).convert("RGB")
    a = a.resize((W, round(a.height * W / a.width)), Image.LANCZOS)
    cells.append((a, cap))
rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
H = sum(max(c[0].height for c in r) + 50 for r in rows) + 20
sheet = Image.new("RGB", (cols * W + (cols + 1) * 20, H), (12, 12, 14))
d = ImageDraw.Draw(sheet); y = 20
for r in rows:
    x = 20; rh = max(c[0].height for c in r)
    for a, cap in r:
        sheet.paste(a, (x, y)); d.text((x, y + a.height + 10), cap, fill=(220, 214, 200), font=fnt); x += W + 20
    y += rh + 50
sheet.save(os.path.join(G, "contact_sheet.png"), optimize=True)
tot = sum(os.path.getsize(os.path.join(G, f)) for f in os.listdir(G))
print({k: os.path.basename(v) for k, v in out.items()}, "gallery MB", round(tot / 1e6, 1))
