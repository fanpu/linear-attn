"""PIL typesetting with per-character font fallback (cmap lookup) and an explicit
'no glyph on this system' mark.  Adapted from art/drainage/typeset.py (copied, not imported)."""
import functools, glob, os, unicodedata
from fontTools.ttLib import TTFont, TTCollection
from PIL import ImageFont

U = "/usr/share/fonts"
F = {
    "serif": f"{U}/opentype/urw-base35/P052-Roman.otf",
    "serif_i": f"{U}/opentype/urw-base35/P052-Italic.otf",
    "serif_b": f"{U}/opentype/urw-base35/P052-Bold.otf",
    "text": f"{U}/truetype/fonts-yrsa-rasa/Yrsa-Regular.ttf",
    "text_i": f"{U}/truetype/fonts-yrsa-rasa/Yrsa-Italic.ttf",
    "text_m": f"{U}/truetype/fonts-yrsa-rasa/Yrsa-Medium.ttf",
    "text_mi": f"{U}/truetype/fonts-yrsa-rasa/Yrsa-MediumItalic.ttf",
    "text_l": f"{U}/truetype/fonts-yrsa-rasa/Yrsa-Light.ttf",
    "mono": f"{U}/truetype/dejavu/DejaVuSansMono.ttf",
    "mono_cjk": f"{U}/opentype/noto/NotoSansCJK-Regular.ttc",
    "cjk_serif": f"{U}/opentype/noto/NotoSerifCJK-Regular.ttc",
}
# serif-first fallback chain for token glyphs (declared: the first font on this machine whose cmap
# contains the code point is used; faces therefore change with script)
SERIF_CHAIN = [
    (F["serif"], 0), (f"{U}/opentype/noto/NotoSerifCJK-Regular.ttc", 1),   # JP face then others
    (f"{U}/opentype/noto/NotoSerifCJK-Regular.ttc", 0),
]
_extra = []
for pat in ["truetype/dejavu/DejaVuSans.ttf", "truetype/tlwg/Loma.ttf", "truetype/tlwg/Norasi.ttf",
            "opentype/noto/NotoSansCJK-Regular.ttc", "truetype/abyssinica/*.ttf", "truetype/lohit-*/*.ttf",
            "truetype/tibetan-machine/*.ttf", "truetype/padauk/Padauk-Regular.ttf",
            "truetype/ttf-khmeros-core/KhmerOS.ttf", "truetype/lao/*.ttf", "truetype/droid/*.ttf",
            "truetype/noto/*.ttf", "truetype/arphic/*.ttc", "truetype/sinhala/*.ttf",
            "truetype/libreoffice/*.ttf", "truetype/ubuntu/*.ttf", "truetype/*/*.ttf", "opentype/*/*.otf"]:
    for f in sorted(glob.glob(f"{U}/{pat}")):
        if (f, 0) not in SERIF_CHAIN and (f, 0) not in _extra and "Emoji" not in f:
            _extra.append((f, 0))
HERE = os.path.dirname(os.path.abspath(__file__))
# Noto faces downloaded into cache/fonts (google/fonts, OFL) for scripts this machine lacks
_dl = [(f, 0) for f in sorted(glob.glob(os.path.join(HERE, "cache/fonts/*.ttf")))]
_dl.sort(key=lambda t: ("serif" not in t[0], t[0]))
_jig = [(f, 0) for f in sorted(glob.glob(os.path.join(HERE, "cache/jigmo/Jigmo*.ttf")))]  # Han ext. B-H (Mincho)
_first = [os.path.join(HERE, "cache/fonts", n) for n in
          ["notoserif.ttf", "notoserifhebrew.ttf", "notonaskharabic.ttf", "notoserifthai.ttf"]]
_dl = [(f, 0) for f in _first if os.path.exists(f)] + [t for t in _dl if t[0] not in _first]
SERIF_CHAIN += _jig + _dl + [(f"{U}/truetype/dejavu/DejaVuSerif.ttf", 0)] + _extra
MONO_CHAIN = [(F["mono"], 0), (F["mono_cjk"], 0)] + SERIF_CHAIN


@functools.lru_cache(None)
def cmap(path, idx=0):
    try:
        f = TTCollection(path).fonts[idx] if path.endswith(".ttc") else TTFont(path, lazy=True)
        return frozenset(f.getBestCmap().keys())
    except Exception:
        return frozenset()


@functools.lru_cache(None)
def font(path, size, idx=0):
    return ImageFont.truetype(path, size=max(1, int(round(size))), index=idx)


def pick(cp, chain):
    for p, i in chain:
        if cp in cmap(p, i):
            return (p, i)
    return None


def runs(text, chain):
    """[(substring, (path, idx) or None)] — None marks code points with no glyph on this system."""
    out = []
    for ch in text:
        pk = pick(ord(ch), chain)
        if out and out[-1][1] == pk and pk is not None:
            out[-1] = (out[-1][0] + ch, pk)
        else:
            out.append((ch, pk))
    return out


def missing_label(ch):
    return f"U+{ord(ch):04X}"


def width(text, size, chain):
    w = 0
    for s, pk in runs(text, chain):
        if pk is None:
            w += size * 0.62 * len(s)
        else:
            w += font(pk[0], size, pk[1]).getlength(s)
    return w


def draw(d, xy, text, size, chain, fill, anchor="ls", missing_fill=None):
    """Draw at baseline-left.  Missing glyphs -> hairline box with the code point inside
    (declared: the system has no font for it)."""
    x, y = xy
    for s, pk in runs(text, chain):
        if pk is None:
            for ch in s:
                bw, bh = size * 0.58, size * 0.72
                d.rectangle([x + size * 0.02, y - bh, x + size * 0.02 + bw, y], outline=missing_fill or fill,
                            width=max(1, int(size / 40)))
                lab = f"{ord(ch):X}"
                fs = bw / max(3.2, len(lab) * 0.62) * 1.0
                lf = font(F["mono"], fs)
                half = (len(lab) + 1) // 2
                for li, part in enumerate([lab[:half], lab[half:]]):
                    d.text((x + size * 0.02 + bw / 2, y - bh * (0.68 - 0.36 * li)), part, font=lf,
                           fill=missing_fill or fill, anchor="mm")
                x += size * 0.62
        else:
            f = font(pk[0], size, pk[1])
            try:
                d.text((x, y), s, font=f, fill=fill, anchor=anchor)
            except OSError:   # FreeType raster overflow on a few huge-bbox glyphs: next covering font
                for p2, i2 in SERIF_CHAIN:
                    if (p2, i2) != pk and all(ord(c) in cmap(p2, i2) for c in s):
                        try:
                            d.text((x, y), s, font=font(p2, size, i2), fill=fill, anchor=anchor)
                            break
                        except OSError:
                            continue
            x += f.getlength(s)
    return x


def coverage(strings, chain=None):
    chain = chain or SERIF_CHAIN
    miss = {}
    for s in strings:
        for ch in s:
            if not ch.isspace() and pick(ord(ch), chain) is None:
                miss[ch] = miss.get(ch, 0) + 1
    return miss
