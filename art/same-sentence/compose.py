"""Gallery composites: the four-measure null quartet (common scale), 1:1 details, contact sheet."""
import os
from PIL import Image, ImageDraw
import numpy as np
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__)); C = f"{HERE}/cache"; G = f"{HERE}/gallery"
CREAM = (244, 238, 225)

# quartet: the four null/measure plates side by side at their true common scale
ims = [Image.open(f"{C}/null_{k}.png") for k in ["chars", "bytes", "qwen3", "gpt2"]]
gap = 300; W = sum(i.width for i in ims) + gap * 3; H = max(i.height for i in ims)
q = Image.new("RGB", (W, H), CREAM); x = 0
for i in ims:
    q.paste(i, (x, 0)); x += i.width + gap
q.resize((9000, round(H * 9000 / W)), Image.LANCZOS).save(f"{G}/null_quartet.png", optimize=True)

# 1:1 details of the Qwen3 plate
P = Image.open(f"{C}/plate_qwen3.png")
P.crop((0, 0, 3000, 1900)).save(f"{G}/detail_qwen3_head.png", optimize=True)
P.crop((0, P.height - 1900, 3000, P.height)).save(f"{G}/detail_qwen3_tail.png", optimize=True)
P.crop((P.width - 7000, P.height - 370, P.width, P.height - 140)).save(f"{G}/detail_qwen3_end.png", optimize=True)
Q2 = Image.open(f"{C}/plate_gpt2.png")
Q2.crop((0, 0, 6600, 2500)).resize((4620, 1750), Image.LANCZOS).save(f"{G}/detail_gpt2_head.png", optimize=True)
# diptych: GPT-2 (2019) above Qwen3 (2025), same scale, left-aligned
gap = 400
D = Image.new("RGB", (Q2.width, Q2.height + P.height + gap), CREAM)
D.paste(Q2, (0, 0)); D.paste(P, (0, Q2.height + gap))
D.resize((6000, round(D.height * 6000 / D.width)), Image.LANCZOS).save(f"{G}/diptych_gpt2_qwen3.png", optimize=True)
Q = Image.open(f"{C}/plate_pairs.png")
Q.crop((0, 0, 3000, 1900)).save(f"{G}/detail_pairs_head.png", optimize=True)

# contact sheet
tiles = ["plate_qwen3.png", "detail_qwen3_head.png", "diptych_gpt2_qwen3.png", "detail_qwen3_tail.png",
         "detail_gpt2_head.png", "detail_pairs_head.png", "null_quartet.png", "detail_qwen3_end.png"]
cw, ch, pad = 1400, 900, 40
sheet = Image.new("RGB", (2 * cw + 3 * pad, 4 * ch + 5 * pad), (230, 224, 210))
for k, t in enumerate(tiles):
    im = Image.open(f"{G}/{t}"); im.thumbnail((cw, ch), Image.LANCZOS)
    cx = pad + (k % 2) * (cw + pad); cy = pad + (k // 2) * (ch + pad)
    bg = Image.new("RGB", (cw, ch), CREAM); bg.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
    sheet.paste(bg, (cx, cy))
sheet.save(f"{G}/contact_sheet.png", optimize=True)
print("ok", sheet.size)
