"""Spectroscope renders: strip panels, the wall, the key, spectra inset, numbers.

    .venv/bin/python spectroscope/render.py            (CPU only)

Declared choices (see README):
  * null = mean of the two fresh entry-shuffles recorded at each checkpoint (sorted eigenvalues
    averaged). MP quantiles at the measured variance are computed as a check only.
  * rows: per layer [q, q.h0..h5, k, k.h0..h5, v, v.h0..h5, o, o.h0..h5, up, down], layers 0..5
    top to bottom, then emb, pos, head. Columns: every recorded checkpoint, in order (every step to
    ~step 150, log-spaced after), so the x axis is log time after the first ~150 steps.
  * one native cell = one (matrix, checkpoint); enlargement is nearest-neighbour only.
  * gamma: 1 is faithful. gamma > 1 raises the spectral power P to gamma and renormalises, which
    exaggerates the spectrum's shape (hue direction kept, saturation increased). Always labelled.
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import spectral as S

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'cache', 'runs')
GAL = os.path.join(HERE, 'gallery')
MET = os.path.join(HERE, 'cache', 'metrics')
os.makedirs(GAL, exist_ok=True); os.makedirs(MET, exist_ok=True)

L, H = 6, 6
ROWS = []
for i in range(L):
    for t in 'qkvo':
        ROWS.append(f'L{i}.{t}')
        ROWS += [f'L{i}.{t}.h{h}' for h in range(H)]
    ROWS += [f'L{i}.up', f'L{i}.down']
ROWS += ['emb', 'pos', 'head']
WHOLE_HIDDEN = [r for r in ROWS if r.startswith('L') and '.h' not in r]
HEADS = [r for r in ROWS if '.h' in r]

INK = (28, 26, 24)
PAPER = (244, 240, 230)


def font(sz):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def load(run, gammas=(1.0,), null='shuffle'):
    """Returns dict with steps, P[gamma] (R, T, NW), plus noise-floor spectra (s1 vs s2)."""
    cache = os.path.join(MET, f'{run}_{null}.npz')
    d = np.load(os.path.join(RUNS, run + '.npz'))
    meta = json.load(open(os.path.join(RUNS, run + '.json')))
    out = dict(steps=d['steps'], meta=meta)
    if os.path.exists(cache):
        c = np.load(cache)
        P1, Pn = c['P1'], c['Pn']
    else:
        P1, Pn = [], []
        for r in ROWS:
            lam = d[r + '/lam']
            if null == 'shuffle':
                nl = 0.5 * (-np.sort(-d[r + '/s1'], -1) - np.sort(-d[r + '/s2'], -1))
            else:
                nl = mp_for(r, lam)
            P1.append(S.energy_power(lam, nl))
            Pn.append(S.energy_power(d[r + '/s1'], d[r + '/s2']))   # shuffle-vs-shuffle: the null's own noise
        P1, Pn = np.stack(P1), np.stack(Pn)
        np.savez_compressed(cache, P1=P1.astype(np.float32), Pn=Pn.astype(np.float32))
    out['P1'] = P1.astype(float); out['Pn'] = Pn.astype(float)
    return out


SHAPES = {'q': (384, 384), 'k': (384, 384), 'v': (384, 384), 'o': (384, 384), 'up': (1536, 384), 'down': (384, 1536),
          'emb': (8192, 384), 'head': (8192, 384), 'pos': (256, 384)}


def mp_for(r, lam):
    parts = r.split('.')
    if len(parts) == 3:
        n, m = 64, 384
    elif r in SHAPES:
        n, m = min(SHAPES[r]), max(SHAPES[r])
    else:
        n, m = min(SHAPES[parts[1]]), max(SHAPES[parts[1]])
    return np.broadcast_to(S.mp_null(n, m), lam.shape)


def gammafy(P, g):
    if g == 1.0:
        return P
    Q = P ** g
    return Q / np.maximum(Q.mean(-1, keepdims=True), 1e-30)


def strip(P, g, sx=4, sy=5):
    rgb = S.spectra_to_rgb(gammafy(P, g))            # (R, T, 3)
    img = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    return np.repeat(np.repeat(img, sy, 0), sx, 1)


def save(arr, name):
    Image.fromarray(arr).save(os.path.join(GAL, name), optimize=True)
    return os.path.join(GAL, name)


def key_commit(arr):
    a = arr.astype(float) / 255
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    return abs(lum.mean() - 0.5)



# ------------------------------------------------------------------ numbers
TYPES = ['q', 'k', 'v', 'o', 'up', 'down']


def rowsel(pred):
    return [i for i, r in enumerate(ROWS) if pred(r)]


def metrics(R, t=-1):
    """Colour statistics at checkpoint index t (gamma 1)."""
    P, Pn = R['P1'][:, t], R['Pn'][:, t]
    cen, cen_n = S.centroid_nm(P) - 550, S.centroid_nm(Pn) - 550
    xyv = S.xy(P)
    out = {}
    groups = {'hidden(whole)': rowsel(lambda r: r in WHOLE_HIDDEN), 'heads': rowsel(lambda r: '.h' in r)}
    for ty in TYPES:
        groups[ty] = rowsel(lambda r, ty=ty: r.startswith('L') and r.split('.')[1] == ty and '.h' not in r)
    for r in ('emb', 'pos', 'head'):
        groups[r] = [ROWS.index(r)]
    for k, idx in groups.items():
        mxy = xyv[idx].mean(0)
        out[k] = dict(shift_nm=float(cen[idx].mean()), chroma=float(S.chroma(xyv[idx]).mean()),
                      dom_nm=float(S.dominant_wavelength(mxy[None])[0]), noise_shift_sd=float(cen_n[idx].std()),
                      noise_chroma=float(S.chroma(S.xy(Pn[idx])).mean()))
    # per-head divergence: spread of head centroids within each (layer, type) vs the null's own spread
    sp, spn = [], []
    for i in range(L):
        for ty in 'qkvo':
            idx = [ROWS.index(f'L{i}.{ty}.h{h}') for h in range(H)]
            sp.append(cen[idx].std()); spn.append(cen_n[idx].std())
    out['head_spread_nm'] = float(np.mean(sp)); out['head_spread_noise_nm'] = float(np.mean(spn))
    out['head_spread_by_type'] = {ty: float(np.mean(sp[j::4])) for j, ty in enumerate('qkvo')}
    return out


# ------------------------------------------------------------------ images
def labelled(img_arr, title, sub, w_label=0, pad=24, fs=22):
    h, w = img_arr.shape[:2]
    top = fs * 3 + pad
    canvas = Image.new('RGB', (w + 2 * pad, h + top + pad), PAPER)
    canvas.paste(Image.fromarray(img_arr), (pad, top))
    d = ImageDraw.Draw(canvas)
    d.text((pad, pad // 2), title, fill=INK, font=font(fs))
    d.text((pad, pad // 2 + int(fs * 1.4)), sub, fill=(90, 86, 80), font=font(int(fs * 0.75)))
    return np.asarray(canvas)


def row_ticks(img_arr, sy, pad_left=150, fs=16):
    """Study version: layer / type labels in the left margin."""
    h, w = img_arr.shape[:2]
    canvas = Image.new('RGB', (w + pad_left, h), PAPER)
    canvas.paste(Image.fromarray(img_arr), (pad_left, 0))
    d = ImageDraw.Draw(canvas)
    for i, r in enumerate(ROWS):
        y = i * sy
        if '.h' in r or r.endswith('.down'):
            continue
        if r.startswith('L'):
            lay, ty = r.split('.')
            lab = {'q': f'layer {lay[1]}  q', 'up': 'mlp up,down'}.get(ty, ty)
        else:
            lab = {'emb': 'emb,pos,head'}.get(r, None)
            if lab is None:
                continue
        d.text((6, y - 3), lab, fill=INK, font=font(fs if r.endswith('.q') else fs - 3))
        d.line([(pad_left - 10, y), (pad_left - 2, y)], fill=INK)
    return np.asarray(canvas)


def wall(Rs, names, g, sx=4, sy=5, gap=18):
    parts = []
    for R in Rs:
        parts.append(strip(R['P1'], g, sx, sy))
        parts.append(np.full((gap, parts[-1].shape[1], 3), PAPER, np.uint8))
    return np.concatenate(parts[:-1], 0)


if __name__ == '__main__':
    runs = [r for r in (sys.argv[1:] or ['adamw', 'muon', 'sgd', 'null']) if os.path.exists(os.path.join(RUNS, r + '.npz'))]
    Rs = [load(r) for r in runs]
    allm = {}
    for run, R in zip(runs, Rs):
        m = metrics(R)
        m['final_val'] = R['meta']['final_val']
        allm[run] = m
        print(f"{run:12s} val {m['final_val']:.3f} | hidden shift {m['hidden(whole)']['shift_nm']:+.1f} nm chroma {m['hidden(whole)']['chroma']:.4f} dom {m['hidden(whole)']['dom_nm']:.0f} | heads {m['heads']['shift_nm']:+.1f} | emb {m['emb']['shift_nm']:+.1f} head {m['head']['shift_nm']:+.1f} | " +
              ' '.join(f"{t}:{m[t]['shift_nm']:+.1f}" for t in TYPES) + f" | head spread {m['head_spread_nm']:.1f} (noise {m['head_spread_noise_nm']:.1f})")
        for g in (1.0, 2.0):
            save(strip(R['P1'], g), f'strip_{run}_g{g:g}.png')
            save(labelled(row_ticks(strip(R['P1'], g), 5), f'{run}  (final val loss {m["final_val"]:.3f})',
                          f'rows: matrices (layers 0-5, each q,k,v,o whole then its 6 heads, then MLP up, down; emb, pos, head). columns: {len(R["steps"])} checkpoints, step 0 to {R["steps"][-1]}, log-spaced. gamma {g:g}' + (' (faithful)' if g == 1 else ' (declared exaggeration)')),
                 f'study_{run}_g{g:g}.png')
    # MP-null check on the final checkpoint (hidden, whole matrices)
    for run, R in zip(runs, Rs):
        d = np.load(os.path.join(RUNS, run + '.npz'))
        idx = [ROWS.index(r) for r in WHOLE_HIDDEN]
        Pm = np.stack([S.energy_power(d[r + '/lam'][-1:], mp_for(r, d[r + '/lam'][-1:]))[0] for r in WHOLE_HIDDEN])
        allm[run]['mp_null_hidden_shift_nm'] = float((S.centroid_nm(Pm) - 550).mean())
        # trajectory of the mean hidden-matrix shift
        cen = S.centroid_nm(R['P1'][idx]) - 550
        allm[run]['traj_steps'] = R['steps'].tolist()
        allm[run]['traj_hidden_shift_nm'] = cen.mean(0).round(2).tolist()
        allm[run]['min_hidden_shift_nm'] = float(cen.mean(0).min())
        allm[run]['min_hidden_shift_step'] = int(R['steps'][cen.mean(0).argmin()])
        print(run, 'MP-null hidden shift', round(allm[run]['mp_null_hidden_shift_nm'], 1), 'min traj', round(allm[run]['min_hidden_shift_nm'], 1), '@', allm[run]['min_hidden_shift_step'])
    json.dump(allm, open(os.path.join(HERE, 'metrics.json'), 'w'), indent=1)
    # interleaved: for every matrix, one row per run, in run order (the comparison made row by row)
    T = min(R['P1'].shape[1] for R in Rs)
    for g in (1.0, 2.0):
        cols = [S.spectra_to_rgb(gammafy(R['P1'][:, :T], g)) for R in Rs]
        inter = np.stack(cols, 1).reshape(-1, T, 3)          # (R*nruns, T, 3)
        img = (np.clip(inter, 0, 1) * 255).round().astype(np.uint8)
        save(np.repeat(np.repeat(img, 2, 0), 4, 1), f'interleaved_{"_".join(runs)}_g{g:g}.png')
    for g in (1.0, 2.0):
        save(wall(Rs, runs, g), f'wall_g{g:g}.png')
