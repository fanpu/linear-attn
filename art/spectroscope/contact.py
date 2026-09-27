"""Contact sheet of the best images (thumbnails, Lanczos downsampling for the sheet only)."""
import os
from PIL import Image, ImageDraw
from render import GAL, font, INK, PAPER
items = [('hero_dark_g1.png', 'hero, gamma 1 (faithful)'), ('hero_dark_g2.png', 'hero, gamma 2 (declared)'),
         ('study_adamw_g1.png', 'AdamW study, gamma 1'), ('study_muon_g2.png', 'Muon study, gamma 2'),
         ('study_sgd_g1.png', 'SGD study, gamma 1'), ('interleaved_adamw_muon_sgd_null_g2.png', 'interleaved rows, gamma 2'),
         ('key.png', 'key'), ('inset_spectra.png', 'measured spectra inset')]
TW, TH, pad = 620, 620, 28
cols = 4
rows = (len(items) + cols - 1) // cols
sheet = Image.new('RGB', (cols * (TW + pad) + pad, rows * (TH + pad + 30) + pad), PAPER)
d = ImageDraw.Draw(sheet)
for k, (f, cap) in enumerate(items):
    im = Image.open(os.path.join(GAL, f)).convert('RGB')
    im.thumbnail((TW, TH), Image.LANCZOS)
    x = pad + (k % cols) * (TW + pad); y = pad + (k // cols) * (TH + pad + 30)
    sheet.paste(im, (x + (TW - im.width) // 2, y + (TH - im.height) // 2))
    d.text((x, y + TH + 6), cap, fill=INK, font=font(18))
sheet.save(os.path.join(GAL, 'contact_sheet.png'), optimize=True)
print(os.path.join(GAL, 'contact_sheet.png'), sheet.size)
