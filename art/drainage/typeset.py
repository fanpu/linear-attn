"""Minimal multi-script typesetting on PIL with per-character font fallback (cmap lookup)."""
import functools

from fontTools.ttLib import TTFont, TTCollection
from PIL import Image, ImageDraw, ImageFont

FONTS = {
    "italic": "/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-Italic.ttf",
    "light_italic": "/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-LightItalic.ttf",
    "medium_italic": "/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-MediumItalic.ttf",
    "roman": "/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-Regular.ttf",
    "light": "/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-Light.ttf",
    "medium": "/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-Medium.ttf",
    "semibold": "/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-SemiBold.ttf",
    "caps": "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf",
}
FALLBACK = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 0),
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0),
]
import glob as _g
for pat in ["/usr/share/fonts/truetype/tlwg/Loma.ttf", "/usr/share/fonts/truetype/kacst/KacstBook.ttf",
            "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
            "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf",
            "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf",
            "/usr/share/fonts/truetype/lohit-telugu/Lohit-Telugu.ttf",
            "/usr/share/fonts/truetype/lohit-gujarati/Lohit-Gujarati.ttf",
            "/usr/share/fonts/truetype/lohit-kannada/Lohit-Kannada.ttf",
            "/usr/share/fonts/truetype/lohit-malayalam/Lohit-Malayalam.ttf",
            "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf",
            "/usr/share/fonts/truetype/Sarai/Sarai.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"]:
    for f in _g.glob(pat):
        FALLBACK.append((f, 0))


@functools.lru_cache(None)
def cmap(path, idx=0):
    if path.endswith(".ttc"):
        f = TTCollection(path).fonts[idx]
    else:
        f = TTFont(path, lazy=True)
    return set(f.getBestCmap().keys())


@functools.lru_cache(None)
def font(path, size, idx=0):
    return ImageFont.truetype(path, size=max(1.0, round(float(size), 1)), index=idx)


def visible(s):
    """Declared transcription of invisible characters: newline -> ¶, tab -> ⇥."""
    return s.replace("\r", "␍").replace("\n", "¶").replace("\t", "⇥")


@functools.lru_cache(maxsize=500000)
def runs(text, primary):
    """Split text into (substring, fontpath, index) runs by glyph availability."""
    chain = [(primary, 0)] + FALLBACK
    out = []
    for ch in text:
        cp = ord(ch)
        pick = chain[-1]
        for p, i in chain:
            if cp in cmap(p, i):
                pick = (p, i)
                break
        if out and out[-1][1:] == pick:
            out[-1] = (out[-1][0] + ch, *pick)
        else:
            out.append((ch, *pick))
    return tuple(out)


@functools.lru_cache(maxsize=500000)
def length(text, size, style="italic"):
    return sum(font(p, size, i).getlength(s) for s, p, i in runs(text, FONTS[style]))


def _halo(h):
    return {} if h is None else dict(stroke_width=int(h[0]), stroke_fill=h[1])


def draw(d, xy, text, size, style="italic", fill=(27, 26, 23), anchor="ls", max_x=None, halo=None):
    """Draw text with fallback starting at xy (baseline-left). Stops at max_x. Returns end x."""
    x, y = xy
    for s, p, i in runs(text, FONTS[style]):
        f = font(p, size, i)
        if max_x is not None:
            w = f.getlength(s)
            if x + w > max_x:
                # truncate by characters
                acc = ""
                for ch in s:
                    if x + f.getlength(acc + ch) > max_x:
                        break
                    acc += ch
                d.text((x, y), acc, font=f, fill=fill, anchor=anchor, **_halo(halo))
                return x + f.getlength(acc)
        d.text((x, y), s, font=f, fill=fill, anchor=anchor, **_halo(halo))
        x += f.getlength(s)
    return x
