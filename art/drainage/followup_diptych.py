"""Follow-up diptych: The Watershed of Qwen3-0.6B-Base (left) beside the post-trained Qwen3-0.6B (right).
Both panels are the unchanged render_hourglass.py output at the same master size (7200 x 10200) and
the same line weight / opacity, downsampled identically (LANCZOS) to the gallery size.

python followup_diptych.py cache/followup_hourglass_base_full.png cache/hourglass_full.png
  -> gallery/followup_watershed_base.png (2880 px) and gallery/followup_watershed_diptych.png
"""
import sys
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
PAPER, GREY, INK = (243, 238, 226), (118, 112, 102), (27, 26, 23)
base, chat = Image.open(sys.argv[1]).convert("RGB"), Image.open(sys.argv[2]).convert("RGB")
w, h = 2880, 4080
b = base.resize((w, h), Image.LANCZOS)
b.save("gallery/followup_watershed_base.png", optimize=True)
c = chat.resize((w, h), Image.LANCZOS)
gut, foot = 120, 200
D = Image.new("RGB", (2 * w + 3 * gut, h + foot + gut), PAPER)
D.paste(b, (gut, gut // 2))
D.paste(c, (2 * gut + w, gut // 2))
d = ImageDraw.Draw(D)
f = ImageFont.truetype("/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-Italic.ttf", 64)
fs = ImageFont.truetype("/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-Regular.ttf", 40)
y = h + gut // 2 + 40
d.text((gut + 115, y), "Qwen3-0.6B-Base: before post-training", font=f, fill=INK)
d.text((2 * gut + w + 115, y), "Qwen3-0.6B: after post-training", font=f, fill=INK)
d.text((gut + 115, y + 90), "Same engine, same 151,643 starts, bf16, greedy, 256-token cap; identical rendering.", font=fs, fill=GREY)
D = D.resize((D.width * 3 // 5, D.height * 3 // 5), Image.LANCZOS)  # declared: gallery copy at 60 %; masters in cache/
D.save("gallery/followup_watershed_diptych.png", optimize=True)
print("ok", D.size)
