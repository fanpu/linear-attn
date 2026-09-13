"""Procedural textile renderers (numpy only). Every function takes a matrix of
MEASURED values in [0,1] and returns an RGB float image in [0,1]. The textile
look (thread shading, stitches, ink grain, dye bleed) is a DECLARED aesthetic
choice; the per-cell value always comes from the data.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image


def hexrgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def to_unit(A, gamma=0.5, vmax=None):
    """Clip to [0, vmax] and apply a power transform (declared)."""
    A = np.asarray(A, dtype=np.float64)
    vmax = 1.0 if vmax is None else vmax
    return np.clip(A / vmax, 0, 1) ** gamma


def save(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, optimize=False)


def _local(n, m, s):
    y = (np.arange(n * s) + 0.5) / s
    x = (np.arange(m * s) + 0.5) / s
    iy = np.floor(y).astype(int)
    ix = np.floor(x).astype(int)
    v = (y - iy)[:, None]     # within-cell coordinate across a weft (row) thread
    u = (x - ix)[None, :]     # within-cell coordinate across a warp (column) thread
    return iy, ix, u, v


# ----------------------------------------------------------------- weave ----
def weave(W, s=8, warp_lo='#2a2233', warp_hi='#f2c46d', weft='#1b2a4a', gap='#07080c',
          float_thr=0.35, seed=0, fibre=0.06, weft_jitter=0.05):
    """Jacquard-like weave. Column j = warp thread, row i = weft thread.
    Encodes the cell value W[i,j] three ways (all declared):
      * warp thread colour  lerp(warp_lo, warp_hi, W)
      * warp thread width   0.35 + 0.6 W  (of the cell)
      * over/under          plain weave (parity) everywhere, but the warp
                            floats over the weft wherever W > float_thr.
    """
    W = np.clip(np.asarray(W, float), 0, 1)
    n, m = W.shape
    rng = np.random.default_rng(seed)
    iy, ix, u, v = _local(n, m, s)
    Wp = W[iy][:, ix]
    lo, hi, wf, gp = map(hexrgb, (warp_lo, warp_hi, weft, gap))
    warp_w = 0.35 + 0.6 * Wp
    du = np.abs(u - 0.5) * 2 / warp_w                          # 0 centre .. 1 edge
    in_warp = du < 1
    dv = np.abs(v - 0.5) * 2 / 0.82
    in_weft = np.broadcast_to(dv < 1, Wp.shape)
    parity = ((iy[:, None] + ix[None, :]) % 2 == 0)
    warp_top = parity | (Wp > float_thr)
    # cylindrical shading
    sh_warp = 0.45 + 0.55 * np.sqrt(np.clip(1 - du ** 2, 0, 1))
    sh_weft = 0.45 + 0.55 * np.sqrt(np.clip(1 - dv ** 2, 0, 1))
    # thread dives under neighbours at the cell ends: darken ends along length
    end_warp = 1 - 0.35 * np.abs(v - 0.5) ** 3 * 8 * (~(Wp > float_thr))
    end_weft = 1 - 0.35 * np.abs(u - 0.5) ** 3 * 8
    # fibre noise along thread direction (declared)
    fn_warp = 1 + fibre * rng.standard_normal((n * s, m))[:, ix]
    fn_weft = 1 + fibre * rng.standard_normal((n, m * s))[iy, :]
    row_tone = (1 + weft_jitter * rng.standard_normal(n))[iy][:, None]
    warp_col = lo + (hi - lo) * Wp[..., None]
    img = np.empty((n * s, m * s, 3))
    img[:] = gp
    c_weft = wf[None, None] * (sh_weft * end_weft * fn_weft * row_tone)[..., None]
    c_warp = warp_col * (sh_warp * end_warp * fn_warp)[..., None]
    under_warp = in_warp & ~in_weft
    img[under_warp] = (c_warp * 0.55)[under_warp]
    img[in_weft] = c_weft[in_weft]
    top = in_warp & warp_top
    img[top] = c_warp[top]
    return img


# ------------------------------------------------------------ risograph ----
def riso(layers, inks, paper='#f3ede1', offsets=None, grain=0.18, seed=0, dot=1):
    """Subtractive spot-colour overprint. layers: list of HxW densities in [0,1].
    Stochastic grain threshold emulates riso screen (declared); offsets are
    deliberate misregistration in pixels (declared)."""
    rng = np.random.default_rng(seed)
    Hh, Ww = layers[0].shape
    out = np.ones((Hh, Ww, 3)) * hexrgb(paper)
    offsets = offsets or [(0, 0)] * len(layers)
    for D, ink, (dy, dx) in zip(layers, inks, offsets):
        D = np.roll(np.roll(D, dy, 0), dx, 1)
        noise = rng.random((Hh // dot + 1, Ww // dot + 1))
        noise = np.kron(noise, np.ones((dot, dot)))[:Hh, :Ww]
        cover = np.clip((D - noise) / grain + 0.5, 0, 1) if grain > 0 else D
        cover = (0.15 * D + 0.85 * cover) * np.clip(D / 0.04, 0, 1)   # blank paper stays blank
        ink = hexrgb(ink)
        out *= 1 - cover[..., None] * (1 - ink)
    return out


# --------------------------------------------------------- cross-stitch ----
def cross_stitch(Q, colors, s=12, fabric='#efe6d2', hole='#cfc2a6', th=0.16, seed=0):
    """Q: integer levels (0 = no stitch, k -> colors[k-1]). Aida cloth + X stitches.
    Quantisation of the measured value into levels is a declared choice."""
    rng = np.random.default_rng(seed)
    n, m = Q.shape
    iy, ix, u, v = _local(n, m, s)
    Qp = Q[iy][:, ix]
    img = np.ones((n * s, m * s, 3)) * hexrgb(fabric)
    # aida texture: holes at cell corners, faint weave lines
    cu, cv = np.minimum(u, 1 - u), np.minimum(v, 1 - v)
    holes = (cu ** 2 + cv ** 2) < 0.02
    img[holes] = hexrgb(hole)
    tex = 1 - 0.05 * ((np.abs(np.sin(np.pi * u * 3)) < 0.15) | (np.abs(np.sin(np.pi * v * 3)) < 0.15))
    img *= tex[..., None]
    pal = np.array([hexrgb(c) for c in colors])
    d1 = np.abs(u - v) / np.sqrt(2)
    d2 = np.abs(u + v - 1) / np.sqrt(2)
    inset = (u > 0.08) & (u < 0.92) & (v > 0.08) & (v < 0.92)
    jit = 1 + 0.06 * rng.standard_normal((n, m))[iy][:, ix]
    for k in range(len(colors)):
        mask = (Qp == k + 1) & inset
        col = pal[k]
        # bottom stroke '\' then top stroke '/'
        s1 = mask & (d1 < th / 2)
        shade1 = (0.62 + 0.3 * np.sqrt(np.clip(1 - (d1 / (th / 2)) ** 2, 0, 1))) * jit
        img[s1] = (col[None] * shade1[s1][:, None]) * 0.85
        s2 = mask & (d2 < th / 2)
        shade2 = (0.7 + 0.35 * np.sqrt(np.clip(1 - (d2 / (th / 2)) ** 2, 0, 1))) * jit
        img[s2] = col[None] * shade2[s2][:, None]
    return np.clip(img, 0, 1)


# -------------------------------------------------------------- shibori ----
def indigo(W, s=6, ink='#15306b', cloth='#f1eee6', bleed=0.45, absorb=3.0, seed=0, mottle=0.08):
    """Single-colour dye: density D = 1-exp(-absorb*W), blurred by `bleed`
    cells (dye wicking, declared), on cloth with a faint plain-weave texture."""
    rng = np.random.default_rng(seed)
    n, m = W.shape
    Wp = np.kron(np.clip(W, 0, 1), np.ones((s, s)))
    if bleed > 0:
        Wp = gaussian_filter(Wp, bleed * s)
    D = 1 - np.exp(-absorb * Wp)
    low = gaussian_filter(rng.standard_normal(Wp.shape), 12 * s / 6)
    low = low / (low.std() + 1e-9)
    D = np.clip(D * (1 + mottle * low), 0, 1)
    yy, xx = np.mgrid[0:Wp.shape[0], 0:Wp.shape[1]]
    tex = 1 - 0.06 * (((yy // 2) + (xx // 2)) % 2)
    cl, ik = hexrgb(cloth), hexrgb(ink)
    img = cl * (1 - D[..., None]) + ik * D[..., None]
    return img * tex[..., None]


# ----------------------------------------------------------------- dark ----
def cmap_img(W, cmap, s=1):
    import matplotlib
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    img = cm(np.clip(W, 0, 1))[..., :3]
    return np.kron(img, np.ones((s, s, 1))) if s > 1 else img


# --------------------------------------------------------------- layout ----
def tile(imgs, rows, cols, gutter=8, bg='#000000', outer=None):
    h, w = imgs[0].shape[:2]
    outer = gutter if outer is None else outer
    H = rows * h + (rows - 1) * gutter + 2 * outer
    Wd = cols * w + (cols - 1) * gutter + 2 * outer
    canvas = np.ones((H, Wd, 3)) * hexrgb(bg)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        y = outer + r * (h + gutter)
        x = outer + c * (w + gutter)
        canvas[y:y + h, x:x + w] = im
    return canvas


# ------------------------------------------------------------ annotation ----
FONT_MONO = '/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf'
FONT_SERIF = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'


def frame(img, title='', caption='', row_labels=None, col_labels=None, tile_hw=None, gutter=0, outer=0,
          bg='#000000', fg='#d8d2c4', margin=(130, 60, 90, 90), font_size=22, title_size=40):
    """Add margins with a title, a caption and optional row/column tick labels.
    margin = (top, right, bottom, left) in px."""
    from PIL import Image, ImageDraw, ImageFont
    top, right, bottom, left = margin
    H, W = img.shape[:2]
    canvas = np.ones((H + top + bottom, W + left + right, 3)) * hexrgb(bg)
    canvas[top:top + H, left:left + W] = img
    pim = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(pim)
    f = ImageFont.truetype(FONT_MONO, font_size)
    ft = ImageFont.truetype(FONT_SERIF, title_size)
    col = tuple(int(c * 255) for c in hexrgb(fg))
    if title:
        dr.text((left, top // 2 - title_size // 2 - 10), title, font=ft, fill=col)
    if caption:
        dr.text((left, top + H + bottom // 2 - font_size // 2), caption, font=f, fill=col)
    if tile_hw is not None:
        th, tw = tile_hw
        if row_labels is not None:
            for r, lab in enumerate(row_labels):
                y = top + outer + r * (th + gutter) + th // 2
                dr.text((left - 12, y), str(lab), font=f, fill=col, anchor='rm')
        if col_labels is not None:
            for c, lab in enumerate(col_labels):
                x = left + outer + c * (tw + gutter) + tw // 2
                dr.text((x, top - 12), str(lab), font=f, fill=col, anchor='mb')
    return np.asarray(pim).astype(float) / 255


def label_tiles(img, labels, rows, cols, tile_hw, gutter, outer, top_offset=0, left_offset=0,
                fg='#d8d2c4', size=22, below=True):
    """Write one label under (or above) every tile of a `tile` layout."""
    from PIL import Image, ImageDraw, ImageFont
    pim = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(pim)
    f = ImageFont.truetype(FONT_MONO, size)
    col = tuple(int(c * 255) for c in hexrgb(fg))
    th, tw = tile_hw
    for k, lab in enumerate(labels):
        r, c = divmod(k, cols)
        x = left_offset + outer + c * (tw + gutter)
        y = top_offset + outer + r * (th + gutter) + (th + 8 if below else -8)
        dr.text((x, y), lab, font=f, fill=col, anchor='la' if below else 'ld')
    return np.asarray(pim).astype(float) / 255


def row_bottom(imgs, gap=40, bg='#000000', pad=40):
    """Place images of different sizes in one row, bottom-aligned."""
    H = max(i.shape[0] for i in imgs) + 2 * pad
    W = sum(i.shape[1] for i in imgs) + gap * (len(imgs) - 1) + 2 * pad
    canvas = np.ones((H, W, 3)) * hexrgb(bg)
    x = pad
    boxes = []
    for im in imgs:
        y = H - pad - im.shape[0]
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        boxes.append((x, y, im.shape[1], im.shape[0]))
        x += im.shape[1] + gap
    return canvas, boxes


def text_at(img, items, fg='#d8d2c4', size=22):
    """items: list of (x, y, text, anchor)."""
    from PIL import Image, ImageDraw, ImageFont
    pim = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(pim)
    f = ImageFont.truetype(FONT_MONO, size)
    col = tuple(int(c * 255) for c in hexrgb(fg))
    for x, y, t, a in items:
        dr.text((x, y), t, font=f, fill=col, anchor=a)
    return np.asarray(pim).astype(float) / 255
