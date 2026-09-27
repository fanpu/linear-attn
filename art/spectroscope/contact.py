"""Contact sheet of the best images (thumbnails, Lanczos downsampling for the sheet only)."""
import os
from PIL import Image, ImageDraw
from render import GAL, font, INK, PAPER
items = [('still_glass_main_g1_profile_final.png', 'film: stained glass, last frame, gamma 1'),
         ('still_glass_main_g2_profile_final.png', 'film: stained glass, gamma 2 (declared)'),
         ('still_glass_wd_g1_profile_final.png', 'film: stained glass, 10k steps + wd, gamma 1'),
         ('still_spectrograph_main_final.png', 'film: spectrograph, last frame'),
         ('still_spectrograph_wd_final.png', 'film: spectrograph, 10k steps + wd'),
         ('hero_dark_g1.png', 'hero strips, gamma 1 (faithful)'),
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
