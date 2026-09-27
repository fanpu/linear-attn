"""Export renders from cache/renders to gallery/ (JPEG), build diptych, grain study and contact sheet."""
import os, sys
from PIL import Image, ImageDraw, ImageFont
from dl_common import CACHE, GALLERY

RD = os.path.join(CACHE, "renders")
FULL = {"F4_noise_acc_top", "F3_noise_acc"}          # heroes kept at full 5120 px
os.makedirs(GALLERY, exist_ok=True)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except OSError:
    font = small = ImageFont.load_default()

ims = {}
for f in sorted(os.listdir(RD)):
    if not f.endswith(".png"):
        continue
    n = f[:-4]; im = Image.open(os.path.join(RD, f)).convert("RGB"); ims[n] = im
    out = im if n in FULL else im.resize((2560, im.height * 2560 // im.width), Image.LANCZOS)
    out.save(os.path.join(GALLERY, n + ".jpg"), quality=90, optimize=True)


def label(img, text, xy=(24, 20)):
    d = ImageDraw.Draw(img); d.text(xy, text, fill=(225, 225, 225), font=font)
    return img


# diptych: fixed minibatch (C) vs fresh minibatch per point (B), same camera, light, colour, K
if "F2_fixed_C" in ims and "F3_noise_acc" in ims:
    w = 2400
    a = ims["F2_fixed_C"].resize((w, w * 9 // 16), Image.LANCZOS)
    b = ims["F3_noise_acc"].resize((w, w * 9 // 16), Image.LANCZOS)
    D = Image.new("RGB", (2 * w + 24, a.height), (0, 0, 0)); D.paste(a, (0, 0)); D.paste(b, (w + 24, 0))
    label(D, "one fixed minibatch of 16 at every point (BN train)", (30, 24))
    label(D, "a fresh minibatch of 16 at every point (BN train)  -- Ideami protocol", (w + 54, 24))
    D.save(os.path.join(GALLERY, "diptych_fixed_vs_fresh.jpg"), quality=90, optimize=True)

# grain study: same camera, 1024^2 vs every 4th point (256^2), identical crop
if "F4_noise_acc_top" in ims and "F7_noise_acc_top_sub4" in ims:
    box = (1760, 1000, 3360, 1900)
    a = ims["F4_noise_acc_top"].crop(box); b = ims["F7_noise_acc_top_sub4"].crop(box)
    D = Image.new("RGB", (a.width * 2 + 20, a.height), (0, 0, 0)); D.paste(a, (0, 0)); D.paste(b, (a.width + 20, 0))
    label(D, "1024² grid (spacing 0.002)"); label(D, "same measurement, every 4th point (256²)", (a.width + 44, 20))
    D.save(os.path.join(GALLERY, "study_grain_1024_vs_256.jpg"), quality=90, optimize=True)

# x-ray pair: fixed minibatch (C) above, fresh minibatch (B) below, same 256 rows, same camera
if "F8_xray_C" in ims and "F8_xray_B" in ims:
    w = 2560
    a = ims["F8_xray_C"].resize((w, w * 9 // 16), Image.LANCZOS); b = ims["F8_xray_B"].resize((w, w * 9 // 16), Image.LANCZOS)
    a = a.crop((0, int(a.height * 0.2), w, int(a.height * 0.85))); b = b.crop((0, int(b.height * 0.2), w, int(b.height * 0.85)))
    D = Image.new("RGB", (w, a.height + b.height), (0, 0, 0)); D.paste(a, (0, 0)); D.paste(b, (0, a.height))
    label(D, "x-ray, fixed minibatch of 16 (C)"); label(D, "x-ray, fresh minibatch of 16 per point (B)", (24, a.height + 20))
    D.save(os.path.join(GALLERY, "xray_pair_fixed_vs_fresh.jpg"), quality=90, optimize=True)

# contact sheet
order = [("F4_noise_acc_top", "fresh bs16 per point, colour = accuracy"),
         ("F3_noise_acc", "fresh bs16 per point (B)"), ("F2_fixed_C", "fixed bs16 (C): texture gone"),
         ("F1_truth_acc", "fixed 1024 imgs, BN eval (A)"), ("F6_truth_fid", "A, colour = function identity"),
         ("F5_noise_fid", "B, colour = function identity"), ("F8_xray_C", "x-ray of C (fixed bs16)"), ("F8_xray_B", "x-ray of B (fresh bs16)")]
order = [(n, t) for n, t in order if n in ims]
tw, th = 1000, 563; cols = 2; rows = (len(order) + 1) // 2
S = Image.new("RGB", (cols * tw + (cols + 1) * 16, rows * (th + 40) + 16), (8, 8, 9))
for i, (n, t) in enumerate(order):
    x = 16 + (i % cols) * (tw + 16); y = 16 + (i // cols) * (th + 40)
    S.paste(ims[n].resize((tw, th), Image.LANCZOS), (x, y))
    ImageDraw.Draw(S).text((x, y + th + 8), f"{n}: {t}", fill=(200, 200, 200), font=small)
S.save(os.path.join(GALLERY, "contact_sheet.png"), optimize=True)
tot = sum(os.path.getsize(os.path.join(GALLERY, f)) for f in os.listdir(GALLERY))
print("gallery MB", tot / 1e6)
