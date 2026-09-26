"""Shaping (HarfBuzz) + rasterising (FreeType) with every glyph pixel labelled by the token
that owns it.  Token ownership is exact at the byte level: a HarfBuzz cluster covers a byte
range of the UTF-8 text; each token covers a byte range; a cluster that is cut by token
boundaries is sliced horizontally in proportion to the bytes each token takes from it
(declared: proportional-by-bytes slicing, in reading direction)."""
import os, sys, functools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "cache/pylib"))
import uharfbuzz as hb  # noqa: E402
import freetype  # noqa: E402
from fontTools.ttLib import TTFont, TTCollection  # noqa: E402

FD = os.path.join(HERE, "cache/fonts")
CJK = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
SERIF = (f"{FD}/notoserif.ttf", 0)
# declared: Noto throughout (one design family built to cover every script); the serif cut
# wherever Noto has one, the sans cut otherwise; Naskh for Arabic script.
SCRIPT_FONT = {
    "Arab": ("notonaskharabic.ttf",), "Hebr": ("notoserifhebrew.ttf",), "Thaa": ("notosansthaana.ttf",),
    "Ethi": ("notoserifethiopic.ttf",), "Tibt": ("notoseriftibetan.ttf",), "Deva": ("notoserifdevanagari.ttf",),
    "Beng": ("notoserifbengali.ttf",), "Gujr": ("notoserifgujarati.ttf",), "Guru": ("notoserifgurmukhi.ttf",),
    "Knda": ("notoserifkannada.ttf",), "Mlym": ("notoserifmalayalam.ttf",), "Taml": ("notoseriftamil.ttf",),
    "Telu": ("notoseriftelugu.ttf",), "Sinh": ("notoserifsinhala.ttf",), "Khmr": ("notoserifkhmer.ttf",),
    "Laoo": ("notoseriflao.ttf",), "Thai": ("notoserifthai.ttf",), "Mymr": ("notosansmyanmar.ttf",),
    "Armn": ("notoserifarmenian.ttf",), "Geor": ("notoserifgeorgian.ttf",), "Yiii": ("notosansyi.ttf",),
    "Cans": ("notosanscanadianaboriginal.ttf",), "Java": ("notosansjavanese.ttf",), "Gran": ("notosansgrantha.ttf",),
    "Cher": ("notosanscherokee.ttf",), "Tfng": ("notosanstifinagh.ttf",), "Vaii": ("notosansvai.ttf",),
    "Mong": ("notosansmongolian.ttf",), "Syrc": ("notosanssyriac.ttf",), "Adlm": ("notosansadlam.ttf",),
    "Tavt": ("notosanstaiviet.ttf",), "Lana": ("notosanstaitham.ttf",), "Limb": ("notosanslimbu.ttf",),
    "Cakm": ("notosanschakma.ttf",), "Mand": ("notosansmandaic.ttf",), "Tglg": ("notosanstagalog.ttf",),
    "Orya": ("notosansoriya.ttf",), "Avst": ("notosansavestan.ttf",), "Rohg": ("notosanshanifirohingya.ttf",),
    "Xsux": ("notosanscuneiform.ttf",),
}
CJK_IDX = {"ja": 0, "ko": 1, "zh": 2, "zh-Hant": 3}


def chain(script, bcp47):
    c = []
    if script in ("Hans", "Hant", "Hani", "Jpan", "Kore", "Hang"):
        i = CJK_IDX.get(bcp47, CJK_IDX.get(bcp47.split("-")[0], 3 if script in ("Hant", "Hani") else 2))
        c.append((CJK, i))
        c += [(os.path.join(FD, f), 0) for f in sorted(os.listdir(FD)) if f.startswith("Jigmo")]
    for f in SCRIPT_FONT.get(script, ()):
        if os.path.exists(os.path.join(FD, f)):
            c.append((os.path.join(FD, f), 0))
    c.append(SERIF)
    c.append((CJK, 2))
    return c


@functools.lru_cache(None)
def cmap(path, idx):
    f = TTCollection(path).fonts[idx] if path.endswith(".ttc") else TTFont(path, lazy=True)
    return frozenset(f.getBestCmap())


# declared: text weight on the variable wght axis where a font has one (400 = Regular; 560 used, between Regular and SemiBold, so the lines hold at a distance)
WGHT = float(os.environ.get("SS_WGHT", 560))


@functools.lru_cache(None)
def axes(path, idx):
    f = TTCollection(path).fonts[idx] if path.endswith(".ttc") else TTFont(path, lazy=True)
    return [(a.axisTag, a.minValue, a.defaultValue, a.maxValue) for a in f["fvar"].axes] if "fvar" in f else []


def coords(path, idx):
    return {t: (min(max(WGHT, lo), hi) if t == "wght" else d) for t, lo, d, hi in axes(path, idx)}


@functools.lru_cache(None)
def hbfont(path, idx):
    face = hb.Face(hb.Blob.from_file_path(path), idx)
    font = hb.Font(face)
    c = coords(path, idx)
    if c:
        font.set_variations(c)
    return font, face.upem


@functools.lru_cache(None)
def ftface(path, idx):
    f = freetype.Face(path, index=idx)
    c = coords(path, idx)
    if c:
        f.set_var_design_coords([c[t] for t, *_ in axes(path, idx)])
    return f


def itemise(text, ch):
    """Runs of (font, char_lo, char_hi) by first-covering font in the chain.  Combining marks /
    spaces stay with the run they are in."""
    runs = []
    for i, c in enumerate(text):
        o = ord(c)
        cur = runs[-1][0] if runs else None
        if cur and o in cmap(*cur) and not c.isalpha():
            f = cur          # spaces, punctuation, marks stay in the current run
        else:
            f = next((ff for ff in ch if o in cmap(*ff)), ch[-1])
        if runs and runs[-1][0] == f:
            runs[-1][2] = i + 1
        else:
            runs.append([f, i, i + 1])
    return runs


def shape(text, script, bcp47, direction, size, features=None):
    """Glyph list (visual order, left to right): dict(font, gid, x, y, lo, hi) with x,y in px at
    `size` (em in px) and [lo,hi) the byte range of the glyph's cluster; returns (glyphs, width)."""
    ch = chain(script, bcp47)
    runs = itemise(text, ch)
    cb = np.cumsum([0] + [len(c.encode("utf-8")) for c in text])
    shaped = []
    for f, a, b in runs:
        font, upem = hbfont(*f)
        buf = hb.Buffer()
        buf.add_utf8(text[a:b].encode("utf-8"))
        buf.guess_segment_properties()
        if direction == "rtl":
            buf.direction = "rtl"
        hb.shape(font, buf, features or {})
        sc = size / upem
        gl = []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            gl.append(dict(font=f, gid=info.codepoint, cl=int(cb[a]) + info.cluster,
                           dx=pos.x_offset * sc, dy=pos.y_offset * sc, adv=pos.x_advance * sc))
        shaped.append(gl)
    if direction == "rtl":
        shaped = shaped[::-1]
    glyphs, x = [], 0.0
    for gl in shaped:
        for g in gl:
            g["x"] = x + g["dx"]; g["y"] = g["dy"]
            x += g["adv"]
            glyphs.append(g)
    starts = sorted({g["cl"] for g in glyphs})
    nb = int(cb[-1])
    nxt = {s: (starts[i + 1] if i + 1 < len(starts) else nb) for i, s in enumerate(starts)}
    for g in glyphs:
        g["lo"], g["hi"] = g["cl"], nxt[g["cl"]]
    return glyphs, x


def raster(glyphs, spans, size, kx, H, base, x0=0.0, W=None, rtl=False):
    """Rasterise glyphs scaled horizontally by kx (FreeType transform, not resampling).
    Returns alpha (H,W) float32 and tok (H,W) int32 (-1 = no ink)."""
    if W is None:
        W = int(np.ceil(x0 + max((g["x"] + g["adv"]) for g in glyphs) * kx + size * 2)) + 4
    alpha = np.zeros((H, W), np.float32)
    tok = np.full((H, W), -1, np.int32)
    lo = np.array([s[0] for s in spans]); hi = np.array([s[1] for s in spans])
    # per glyph bitmaps first (need cluster ink extents)
    bms = []
    for g in glyphs:
        f = ftface(*g["font"])
        f.set_char_size(int(round(size * 64)))
        px = x0 + g["x"] * kx
        ix = int(np.floor(px)); fr = px - ix
        f.set_transform(freetype.Matrix(int(kx * 0x10000), 0, 0, 0x10000), freetype.Vector(int(round(fr * 64)), 0))
        f.load_glyph(g["gid"], freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_NO_BITMAP)
        bm = f.glyph.bitmap
        if bm.width == 0 or bm.rows == 0:
            bms.append(None); continue
        a = np.array(bm.buffer, np.uint8).reshape(bm.rows, bm.pitch)[:, :bm.width].astype(np.float32) / 255.0
        gx = ix + f.glyph.bitmap_left
        gy = int(round(base - g["y"])) - f.glyph.bitmap_top
        bms.append((a, gx, gy))
    # cluster extents
    ext = {}
    for g, b in zip(glyphs, bms):
        if b is None:
            continue
        a, gx, gy = b
        e = ext.setdefault(g["lo"], [1e9, -1e9, g["hi"]])
        e[0] = min(e[0], gx); e[1] = max(e[1], gx + a.shape[1])
    for g, b in zip(glyphs, bms):
        if b is None:
            continue
        a, gx, gy = b
        clo, chi = g["lo"], g["hi"]
        ov = np.minimum(hi, chi) - np.maximum(lo, clo)
        ids = np.nonzero(ov > 0)[0]
        if len(ids) == 0:
            ids = np.array([int(np.searchsorted(lo, clo, "right") - 1)]); ov = np.ones(len(lo))
        e0, e1, _ = ext[clo]
        w = ov[ids].astype(float); cum = np.concatenate([[0], np.cumsum(w) / w.sum()])
        cols = np.arange(gx, gx + a.shape[1]) + 0.5
        fx = (cols - e0) / max(e1 - e0, 1e-6)
        if rtl:
            fx = 1 - fx
        k = np.clip(np.searchsorted(cum, fx, "right") - 1, 0, len(ids) - 1)
        tcol = ids[k]
        # composite
        y0, y1 = max(gy, 0), min(gy + a.shape[0], H)
        xa, xb = max(gx, 0), min(gx + a.shape[1], W)
        if y1 <= y0 or xb <= xa:
            continue
        sub = a[y0 - gy:y1 - gy, xa - gx:xb - gx]
        dst = alpha[y0:y1, xa:xb]; dt = tok[y0:y1, xa:xb]
        tc = np.broadcast_to(tcol[xa - gx:xb - gx][None, :], sub.shape)
        win = sub > dst
        dt[win] = tc[win]
        alpha[y0:y1, xa:xb] = 1 - (1 - dst) * (1 - sub)
    return alpha, tok
