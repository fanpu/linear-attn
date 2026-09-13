"""Poincare sections and simplex trajectories of the SAF replicator learning dynamics.
Reads cache/kam_*.npz, cache/traj.npz, cache/saf_repro.npz.
python render_poincare.py [ink|series|dark|riso|simplex|plotter|butterfly|video_section|video_traj|all]"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from render_lib import GAL, PAPER, INK, RISO_BLUE, RISO_PINK, save_png, u8, to_mp4_gif, font

CHAOS_THR = 5e-3  # per-orbit largest Lyapunov exponent (1/time) above which an orbit is called chaotic
EPS_LIST = ['0.00', '0.10', '0.25', '0.40', '0.50']


def hexrgb(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255


def load_kam(tag):
    z = np.load(f'cache/kam_eps{tag}.npz')
    d = {k: z[k] for k in z.files}  # decompress once
    pts = []; lam = []
    sec, nsec = d['sec'], d['nsec']
    for o in range(len(nsec)):
        s = sec[o, :nsec[o]]
        pts.append(s[:, [0, 4]]); lam.append(np.full(len(s), d['lyap'][o]))
    return np.concatenate(pts), np.concatenate(lam), d


def density(pts, R, lo=(0.0, 0.0), hi=(0.85, 0.85)):
    H, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=R, range=[[lo[1], hi[1]], [lo[0], hi[0]]])
    return H[::-1]


def bounds(pts, pad=0.03):
    lo = pts.min(0) - pad; hi = pts.max(0) + pad
    c = 0.5 * (lo + hi); w = (hi - lo).max()
    return c - w / 2, c + w / 2


def ink_coverage(H, gain=1.0):
    """Declared: dot coverage = 1 - exp(-gain * hits), i.e. overlapping translucent ink dots."""
    return 1 - np.exp(-gain * H)


def ink(tag='0.50', R=3000, name=None):
    pts, lam, d = load_kam(tag)
    lo, hi = bounds(pts)
    H = density(pts, R, lo, hi)
    cov = ink_coverage(H, 0.9)
    paper = hexrgb(PAPER); inkc = hexrgb(INK)
    img = paper[None, None] * (1 - cov[..., None]) + inkc[None, None] * cov[..., None]
    save_png(img, f'{GAL}/{name or "poincare_ink_eps" + tag}.png')


def series(R=1100):
    fig, axs = plt.subplots(1, 5, figsize=(25, 6.3), dpi=150, facecolor=PAPER)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.12, wspace=0.05)
    allpts = np.concatenate([load_kam(t)[0] for t in EPS_LIST])
    lo, hi = bounds(allpts)
    for ax, tag in zip(axs, EPS_LIST):
        pts, lam, d = load_kam(tag)
        img = np.ones((R, R, 3)) * hexrgb(PAPER)
        for sel, col, g in ((lam <= CHAOS_THR, INK, 0.9), (lam > CHAOS_THR, '#c8102e', 0.35)):
            if sel.any():
                cov = ink_coverage(density(pts[sel], R, lo, hi), g)[..., None]
                img = img * (1 - cov * (1 - hexrgb(col)))
        ax.imshow(img, extent=[lo[0], hi[0], lo[1], hi[1]], interpolation='lanczos')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fc = np.mean(d['lyap'] > CHAOS_THR)
        ax.set_title(f'ε = {float(tag):.2f}\n{100*fc:.0f}% of {len(d["lyap"])} orbits chaotic', fontsize=14, color=INK)
    fig.text(0.02, 0.92, 'Tori break into a chaotic sea as the tie payoff ε grows  —  zero-sum RPS, replicator learning, '
             'one energy surface H = 2.8', fontsize=17, color=INK, family='DejaVu Serif')
    fig.text(0.02, 0.03, 'Poincaré section x_P − x_R + y_P − y_R = 0 (upward), axes x_R (player 1 rock) vs y_P (player 2 paper). '
             f'Black: orbits with finite-time λ ≤ {CHAOS_THR:g} at T = 4·10⁴; red: λ > {CHAOS_THR:g} (measured per orbit). '
             'Ink density = 1 − exp(−hits) (declared).', fontsize=11, color=INK)
    fig.savefig(f'{GAL}/poincare_eps_series.png', facecolor=PAPER)
    plt.close(fig)


def dark(tag='0.50', R=2400, name=None):
    """Dark ground; each dot coloured by its orbit's measured Lyapunov exponent (log scale)."""
    import cmcrameri.cm  # noqa
    import matplotlib as mpl
    pts, lam, d = load_kam(tag)
    lo, hi = bounds(pts)
    cm = mpl.colormaps['cmc.lajolla'] if 'cmc.lajolla' in mpl.colormaps else mpl.colormaps['magma_r']
    llog = np.clip((np.log10(np.maximum(lam, 1e-5)) + 4.2) / (np.log10(0.06) + 4.2), 0, 1)
    img = np.zeros((R, R, 3))
    wsum = np.zeros((R, R))
    ix = ((pts[:, 0] - lo[0]) / (hi[0] - lo[0]) * (R - 1)).astype(int)
    iy = (R - 1 - (pts[:, 1] - lo[1]) / (hi[1] - lo[1]) * (R - 1)).astype(int)
    col = np.where((lam > CHAOS_THR)[:, None], hexrgb('#9fd3ff')[None] * (0.75 + 0.25 * llog[:, None]),
                   hexrgb('#e39a55')[None] * (1 - 0.35 * llog[:, None]))
    wgt = np.where(lam > CHAOS_THR, 4.0, 1.0)
    for c in range(3):
        np.add.at(img[..., c], (iy, ix), col[:, c])
    np.add.at(wsum, (iy, ix), wgt)
    cnt = np.zeros((R, R)); np.add.at(cnt, (iy, ix), 1)
    mean = img / np.maximum(cnt, 1)[..., None]
    alpha = 1 - np.exp(-0.7 * wsum)
    bg = hexrgb('#0b0b10')
    out = bg * (1 - alpha[..., None]) + mean * alpha[..., None]
    save_png(out, f'{GAL}/{name or "poincare_dark_eps" + tag}.png')


def tern(p):
    p = np.asarray(p)
    return np.stack([p[..., 1] + 0.5 * p[..., 2], p[..., 2] * np.sqrt(3) / 2], -1)


def raster_lines(xy, R, lo, hi, width=1.0, weight=1.0):
    """Anti-aliased-ish line density via dense resampling + splat (declared: 1 px pen)."""
    seg = np.diff(xy, axis=0); L = np.linalg.norm(seg, axis=1)
    pix = (hi - lo) / R
    n = np.maximum(1, np.ceil(L / (0.5 * pix)).astype(int))
    idx = np.repeat(np.arange(len(seg)), n)
    frac = np.concatenate([np.arange(k) / k for k in n]) if len(n) < 200000 else None
    if frac is None:
        starts = np.repeat(np.cumsum(n) - n, n)
        frac = (np.arange(n.sum()) - starts) / np.repeat(n, n)
    P = xy[idx] + seg[idx] * frac[:, None]
    ix = ((P[:, 0] - lo) / (hi - lo) * (R - 1)).astype(int)
    iy = (R - 1 - (P[:, 1] - lo) / (hi - lo) * (R - 1)).astype(int)
    H = np.zeros((R, R)); ok = (ix >= 0) & (ix < R) & (iy >= 0) & (iy < R)
    np.add.at(H, (iy[ok], ix[ok]), weight * 0.5)
    if width > 1:
        H = gaussian_filter(H, width / 2.5)
    return H


def riso(R=2200, orbit=0, n=60000, name='simplex_riso_players', gain=0.5):
    """Riso 2-ink: player 1 (blue) and player 2 (fluorescent pink) on the same simplex.
    orbit index into SAF starts k = (1, 2, 5, 20); k=1 chaotic, k=5 a regular torus."""
    z = np.load('cache/traj.npz'); d = {k: z[k] for k in z.files}
    tr = d['traj_0.50'][orbit]
    lo, hi = -0.04, 1.04
    Hx = raster_lines(tern(tr[:n, :3]), R, lo, hi)
    Hy = raster_lines(tern(tr[:n, 3:]), R, lo, hi)
    cx = 1 - np.exp(-gain * Hx); cy = 1 - np.exp(-gain * Hy)
    # deliberate misregistration of the pink plate by 4 px (declared)
    cy = np.roll(cy, (4, -3), axis=(0, 1))
    paper = hexrgb('#f4efe4')
    img = paper * (1 - cx[..., None] * (1 - hexrgb(RISO_BLUE))) * (1 - cy[..., None] * (1 - hexrgb(RISO_PINK)))
    # simplex outline in blue ink
    im = Image.fromarray(u8(img)); dr = ImageDraw.Draw(im)
    V = tern(np.eye(3)); V = np.vstack([V, V[:1]])
    pv = [((v[0] - lo) / (hi - lo) * (R - 1), R - 1 - (v[1] - lo) / (hi - lo) * (R - 1)) for v in V]
    dr.line(pv, fill=tuple((hexrgb(RISO_BLUE) * 255).astype(int)), width=3)
    f = font(40)
    for lab, v in zip(['R', 'P', 'S'], pv[:3]):
        dr.text((v[0] + (-50 if lab == 'R' else 20), v[1] - (60 if lab == 'S' else -5)), lab, fill=(40, 40, 60), font=f)
    im = im.crop((0, int(R * 0.125), R, R))  # trim the empty band above the apex
    im.save(f'{GAL}/{name}.png', optimize=True)


def simplex(R=1400):
    """Scientific plate: 2x4 ternary phase portraits, rows = player, cols = orbits (chaotic -> regular)."""
    z = np.load('cache/traj.npz'); d = {k: z[k] for k in z.files}
    tr = d['traj_0.50']; lam = d['lyap_0.50']; ks = [1, 2, 5, 20]
    fig, axs = plt.subplots(2, 4, figsize=(22, 12.5), dpi=130, facecolor=PAPER)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.86, bottom=0.08, wspace=0.04, hspace=0.12)
    lo, hi = -0.03, 1.03
    for j in range(4):
        for i, (sl, col) in enumerate(((slice(0, 3), RISO_BLUE), (slice(3, 6), '#c8102e'))):
            H = raster_lines(tern(tr[j, :100000, sl]), R, lo, hi)
            cov = 1 - np.exp(-0.6 * H)
            img = hexrgb(PAPER) * (1 - cov[..., None] * (1 - hexrgb(col)))
            ax = axs[i, j]; ax.imshow(img, extent=[lo, hi, lo, hi], interpolation='lanczos')
            V = tern(np.eye(3)); V = np.vstack([V, V[:1]]); ax.plot(V[:, 0], V[:, 1], color=INK, lw=0.7)
            for lab, v, off in zip('RPS', V[:3], [(-0.03, -0.035), (0.01, -0.035), (-0.01, 0.015)]):
                ax.text(v[0] + off[0], v[1] + off[1], lab, fontsize=11, color=INK)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f'SAF start k = {ks[j]}\nλ₁ = {lam[j]*1e3:.1f}·10⁻³', fontsize=13, color=INK)
        axs[0, 0].text(-0.08, 0.45, 'player 1  (x)', rotation=90, fontsize=13, color=RISO_BLUE, transform=axs[0, 0].transAxes)
        axs[1, 0].text(-0.08, 0.45, 'player 2  (y)', rotation=90, fontsize=13, color='#c8102e', transform=axs[1, 0].transAxes)
    fig.text(0.03, 0.935, 'Learning rock–paper–scissors against each other: four starts, same game (ε_x = −ε_y = 0.5), t ∈ [0, 2000]',
             fontsize=17, color=INK, family='DejaVu Serif')
    fig.text(0.03, 0.025, 'Coupled replicator equations (SAF 2002), RK4 h = 0.005 in logit coordinates, float64. λ₁: finite-time largest '
             'Lyapunov exponent over t ≤ 4000 (regular orbits decay toward 0 as 1/T; chaotic ones plateau). Ink density declared.',
             fontsize=11, color=INK)
    fig.savefig(f'{GAL}/simplex_plate_eps0.50.png', facecolor=PAPER)
    plt.close(fig)


def plotter(R=4000):
    """One continuous line: the chaotic orbit of player 1 for t in [0, 6000], 1-px pen on white paper.
    Also writes an SVG polyline (decimated to plotter-friendly resolution)."""
    z = np.load('cache/traj.npz'); d = {k: z[k] for k in z.files}
    tr = d['plotter']  # samples every 0.05
    n = 24000  # t in [0, 1200]
    xy = tern(tr[:n, :3])
    xy = xy - np.array([0.5, (xy[:, 1].max() + xy[:, 1].min()) / 2]) + 0.5  # centre the drawing on the sheet
    lo, hi = -0.04, 1.04
    H = raster_lines(xy, R, lo, hi, weight=1.0)
    cov = 1 - np.exp(-0.9 * H)
    img = np.ones((R, R, 3)) * hexrgb('#fbfaf6') * (1 - cov[..., None] * (1 - hexrgb('#101418')))
    save_png(img, f'{GAL}/plotter_single_line.png')
    # SVG, A2-ish 420 mm square area; keep points that move > 0.15 mm
    scale = 400.0
    P = np.column_stack([10 + (xy[:, 0]) * scale, 10 + (np.sqrt(3) / 2 - xy[:, 1]) * scale])
    keep = [0]; last = P[0]
    for i in range(1, len(P)):
        if np.hypot(*(P[i] - last)) > 0.15:
            keep.append(i); last = P[i]
    P = P[keep]
    with open(f'{GAL}/plotter_single_line.svg', 'w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="420mm" height="370mm" viewBox="0 0 420 370">\n')
        f.write('<polyline fill="none" stroke="black" stroke-width="0.2" points="')
        f.write(' '.join(f'{x:.2f},{y:.2f}' for x, y in P))
        f.write('"/>\n</svg>\n')
    print('plotter svg points', len(P))


def butterfly(R=1600):
    z = np.load('cache/traj.npz'); d = {k: z[k] for k in z.files}; B = d['butterfly']  # (16, 30000, 6), dt=0.05
    lo, hi = -0.03, 1.03
    import matplotlib as mpl
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.6), dpi=140, facecolor='#0b0b10')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.08, wspace=0.02)
    cm = mpl.colormaps['cmc.batlow'] if 'cmc.batlow' in mpl.colormaps else mpl.colormaps['viridis']
    for ax, (t0, t1) in zip(axs, [(0, 4000), (4000, 12000), (12000, 30000)]):
        img = np.zeros((R, R, 3))
        for o in range(16):
            H = raster_lines(tern(B[o, t0:t1, :3]), R, lo, hi)
            cov = 1 - np.exp(-0.25 * H)
            c = np.array(cm(o / 15)[:3])
            img = img + cov[..., None] * c * 0.55
        ax.imshow(np.clip(img, 0, 1), extent=[lo, hi, lo, hi]); ax.set_axis_off()
        ax.set_title(f't ∈ [{t0*0.05:.0f}, {t1*0.05:.0f}]', color='#ddd', fontsize=14)
        dist = np.abs(B[:, t1 - 1, :3] - B[0, t1 - 1, :3]).max()
        ax.text(0.5, -0.02, f'max spread of player-1 strategy at t={t1*0.05:.0f}: {dist:.2g}', color='#aaa',
                ha='center', fontsize=11, transform=ax.transAxes)
    fig.text(0.01, 0.93, 'Butterfly effect in self-play learning: 16 player-1 strategies started 10⁻⁹ apart (logit), '
             'zero-sum RPS ε = 0.5', color='#eee', fontsize=16, family='DejaVu Serif')
    fig.savefig(f'{GAL}/butterfly_simplex_dark.png', facecolor='#0b0b10')
    plt.close(fig)


def video_section(tag='0.50', nf=420, R=1080):
    """Poincare section accumulating: crossing index grows geometrically so early structure is visible."""
    z = np.load(f'cache/kam_eps{tag}.npz'); d = {k: z[k] for k in z.files}
    sec = d['sec']; nsec = d['nsec']; lam = d['lyap']
    allp = np.concatenate([sec[o, :nsec[o]][:, [0, 4]] for o in range(len(nsec))])
    lo, hi = bounds(allp)
    tmp = 'cache/frames_section'; os.makedirs(tmp, exist_ok=True)
    N = nsec.min()
    counts = np.unique(np.round(np.geomspace(1, N, nf)).astype(int))
    prev = 0; H1 = np.zeros((R, R)); H2 = np.zeros((R, R))
    f = font(24, mono=True); k = 0
    paper = hexrgb(PAPER)
    for c in list(counts) + [counts[-1]] * 60:
        if c > prev:
            chunk = sec[:, prev:c][:, :, [0, 4]]
            reg = chunk[lam <= CHAOS_THR].reshape(-1, 2); cha = chunk[lam > CHAOS_THR].reshape(-1, 2)
            H1 += density(reg, R, lo, hi); H2 += density(cha, R, lo, hi); prev = c
        img = paper * (1 - (1 - np.exp(-1.2 * H1))[..., None] * (1 - hexrgb(INK)))
        img = img * (1 - (1 - np.exp(-0.5 * H2))[..., None] * (1 - hexrgb('#c8102e')))
        im = Image.fromarray(u8(img)); dr = ImageDraw.Draw(im)
        dr.text((24, 20), f'return {c:5d}   ε = {float(tag):.2f}   H = {float(d["H0"]):.2f}', fill=(30, 30, 30), font=f)
        im.save(f'{tmp}/{k:05d}.png'); k += 1
    to_mp4_gif(tmp, f'poincare_accumulate_eps{tag}', fps=24, gif_width=600, gif_fps=12)


def video_traj(nf=600, R=1080):
    """Both players' strategies being drawn on two simplices (riso blue / pink)."""
    z = np.load('cache/traj.npz'); d = {k: z[k] for k in z.files}; tr = d['traj_0.50'][0]  # dt 0.02
    tmp = 'cache/frames_traj'; os.makedirs(tmp, exist_ok=True)
    W = R // 2
    lo, hi = -0.05, 1.05
    total = 50000  # t = 1000
    ends = np.linspace(200, total, nf).astype(int)
    Hx = np.zeros((W, W)); Hy = np.zeros((W, W)); prev = 0
    paper = hexrgb('#f4efe4'); f = font(26, mono=True)
    V = tern(np.eye(3)); V = np.vstack([V, V[:1]])
    pv = [((v[0] - lo) / (hi - lo) * (W - 1), W - 1 - (v[1] - lo) / (hi - lo) * (W - 1)) for v in V]
    k = 0
    for e in list(ends) + [ends[-1]] * 48:
        if e > prev:
            Hx += raster_lines(tern(tr[max(prev - 1, 0):e, :3]), W, lo, hi)
            Hy += raster_lines(tern(tr[max(prev - 1, 0):e, 3:]), W, lo, hi); prev = e
        canvas = np.ones((R, R, 3)) * paper
        for off, H, col in ((0, Hx, RISO_BLUE), (W, Hy, RISO_PINK)):
            cov = 1 - np.exp(-0.5 * H)
            canvas[R // 4:R // 4 + W, off:off + W] = paper * (1 - cov[..., None] * (1 - hexrgb(col)))
        im = Image.fromarray(u8(canvas)); dr = ImageDraw.Draw(im)
        for off, col, name, cur in ((0, RISO_BLUE, 'player 1', tr[e - 1, :3]), (W, RISO_PINK, 'player 2', tr[e - 1, 3:])):
            dr.line([(x + off, y + R // 4) for x, y in pv], fill=tuple((hexrgb(col) * 255).astype(int)), width=2)
            cx, cy = tern(cur); px = (cx - lo) / (hi - lo) * (W - 1) + off; py = W - 1 - (cy - lo) / (hi - lo) * (W - 1) + R // 4
            dr.ellipse([px - 7, py - 7, px + 7, py + 7], fill=(20, 20, 30))
            dr.text((off + 30, R // 4 - 60), name, fill=tuple((hexrgb(col) * 200).astype(int)), font=f)
        dr.text((30, R - 120), f't = {e*0.02:7.1f}   zero-sum RPS, ε = 0.5, λ₁ ≈ {d["lyap_0.50"][0]:.3f}', fill=(40, 40, 40), font=f)
        im.save(f'{tmp}/{k:05d}.png'); k += 1
    to_mp4_gif(tmp, 'simplex_drawing', fps=30, gif_width=600, gif_fps=15)


if __name__ == '__main__':
    what = sys.argv[1:] or ['all']
    allw = ['ink', 'series', 'dark', 'riso', 'simplex', 'plotter', 'butterfly', 'video_section', 'video_traj']
    for w in (allw if 'all' in what else what):
        if w == 'ink':
            ink('0.50'); ink('0.00')
            if os.path.exists('cache/kam_eps0.50_H3.0.npz'):
                pass
        elif w == 'riso_torus':
            riso(orbit=2, n=100000, name='simplex_riso_torus_k5', gain=0.35)
        elif w == 'dark':
            dark('0.50'); dark('0.25')
        else:
            globals()[w]()
        print('done', w, flush=True)
