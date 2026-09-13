"""Page layout helpers: scientific-plate pages with coordinates (PIL)."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SERIF = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
SERIF_B = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
PAPER = (244, 240, 230)
INKC = (28, 26, 30)
GREY = (110, 104, 98)


def font(path, size):
    return ImageFont.truetype(path, size)


def roman(n):
    vals = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
            (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    s = ''
    for v, r in vals:
        while n >= v:
            s += r; n -= v
    return s


def sci(x, digits=4):
    return f'{x:.{digits}e}'


def fmt_log_centre(c, hw):
    """Enough decimals of log10 centre that the field is resolved to ~1/1000 width."""
    nd = max(2, int(math.ceil(-math.log10(max(hw, 1e-300) * 2 / 1000))))
    nd = min(nd, 17)
    return f'{c:.{nd}f}'


def _paste_vtext(page, text, fnt, fill, paper, x_right, y, align='centre'):
    """Paste text rotated 90 deg (reading bottom-to-top) with its right edge at x_right.
    align: 'start' = text begins at y and runs upward, 'end' = text ends at y,
    'centre' = centred on y."""
    if not text:
        return
    bb = fnt.getbbox(text)
    w, h = bb[2] - bb[0] + 4, bb[3] - bb[1] + 8
    tmp = Image.new('RGBA', (w, h), paper + (0,))
    ImageDraw.Draw(tmp).text((2 - bb[0], 4 - bb[1]), text, font=fnt, fill=fill)
    tmp = tmp.rotate(90, expand=True)          # now h wide, w tall; text reads upward
    if align == 'start':
        top = int(y - w)
    elif align == 'end':
        top = int(y)
    else:
        top = int(y - w / 2)
    page.paste(tmp, (int(x_right - h), top), tmp)


def plate_page(img, meta, lines, title, plate_no=None, locator=None, W=2400, H=3150,
               img_px=2048, paper=PAPER, ink=INKC, tick_labels=True):
    """img: uint8 RGB (top row = high eta1). meta: dict with c0,c1,hw (log10 units) and
    axis labels. lines: caption lines (list of str). locator: optional (uint8 RGB, rect)
    where rect=(x0,y0,x1,y1) fractions in the locator image to outline."""
    page = Image.new('RGB', (W, H), paper)
    d = ImageDraw.Draw(page)
    f_title = font(SERIF_B, 54)
    f_no = font(SERIF, 40)
    f_cap = font(SERIF, 34)
    f_mono = font(MONO, 28)
    f_tick = font(MONO, 24)
    ml = (W - img_px) // 2
    mt = 190
    if plate_no is not None:
        d.text((W // 2, 70), f'PLATE {roman(plate_no)}', font=f_no, fill=ink, anchor='mt')
    d.text((W // 2, 125), title, font=f_title, fill=ink, anchor='mt') if False else None
    im = Image.fromarray(img).resize((img_px, img_px), Image.NEAREST)
    page.paste(im, (ml, mt))
    d.rectangle([ml - 3, mt - 3, ml + img_px + 2, mt + img_px + 2], outline=ink, width=3)
    # ticks: 5 per side at 0, .25, .5, .75, 1 of the field
    c0, c1, hw = meta['c0'], meta['c1'], meta['hw']
    hwy = meta.get('hwy', hw)
    for i, f in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
        x = ml + f * img_px
        d.line([x, mt + img_px + 3, x, mt + img_px + 22], fill=ink, width=3)
        y = mt + img_px - f * img_px
        d.line([ml - 22, y, ml - 3, y], fill=ink, width=3)
        if tick_labels and f in (0, 0.5, 1.0):
            off = (f * 2 - 1) * hw
            offy = (f * 2 - 1) * hwy
            lx = f'{off:+.3g}' if f != 0.5 else fmt_log_centre(c0, hw)
            ly = f'{offy:+.3g}' if f != 0.5 else fmt_log_centre(c1, hwy)
            anc = {0: 'lt', 0.5: 'mt', 1.0: 'rt'}[f]
            d.text((x, mt + img_px + 28), lx, font=f_tick, fill=GREY, anchor=anc)
            _paste_vtext(page, ly, f_tick, GREY, paper, x_right=ml - 30, y=y,
                         align={0: 'start', 0.5: 'centre', 1.0: 'end'}[f])
    d.text((ml, mt + img_px + 70), meta.get('xlabel', ''), font=f_cap, fill=ink, anchor='lt')
    _paste_vtext(page, meta.get('ylabel', ''), f_cap, ink, paper, x_right=ml - 70, y=mt + img_px / 2,
                 align='centre')
    # caption
    y = mt + img_px + 135
    d.text((ml, y), title, font=f_title, fill=ink, anchor='lt')
    y += 80
    for ln in lines:
        fnt = f_mono if ln.startswith('~') else f_cap
        txt = ln[1:] if ln.startswith('~') else ln
        d.text((ml, y), txt, font=fnt, fill=ink if fnt is f_cap else GREY, anchor='lt')
        y += 46 if fnt is f_cap else 40
    if locator is not None:
        limg, rect = locator
        L = 300
        li = Image.fromarray(limg).resize((L, L), Image.NEAREST)
        lx, ly = ml + img_px - L, H - L - 70
        page.paste(li, (lx, ly))
        dd = ImageDraw.Draw(page)
        dd.rectangle([lx - 2, ly - 2, lx + L + 1, ly + L + 1], outline=ink, width=2)
        x0, y0, x1, y1 = rect
        # rect fractions in data coords (y up) -> image coords (y down)
        dd.rectangle([lx + x0 * L, ly + (1 - y1) * L, lx + x1 * L, ly + (1 - y0) * L],
                     outline=(230, 40, 60), width=3)
        dd.text((lx - 16, ly + L), 'previous plate', font=font(SERIF, 26), fill=GREY, anchor='rb')
    return page
