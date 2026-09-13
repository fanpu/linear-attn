"""§11 verification: box counting across zoom levels, resolution check, null model, uncertainty exponent.
Writes cache/verify_<name>.json and gallery/verify_<name>.png.

  python analyze.py fact3     python analyze.py xor
"""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractal import boundary, box_counts, fit_dimension, local_slopes

C_MAIN, C_ALT, C_NULL = '#2a6fb0', '#d0582f', '#7a7570'
INKC, MUTED = '#1c1b1a', '#6b6763'
plt.rcParams.update({'font.family': 'serif', 'axes.edgecolor': MUTED, 'axes.labelcolor': INKC, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': '#e4e0d8', 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2})


def levels(tag, key='raw'):
    fs = sorted(glob.glob(f'cache/zoom/{tag}_L*.npz'), key=lambda s: int(s.split('_L')[-1][:-4]))
    out = []
    for f in fs:
        d = np.load(f); m = json.loads(str(d['meta']))
        lab = d[key]
        b = boundary(lab)
        R = lab.shape[0]
        s, c = box_counts(b, 1, R // 8)
        D, se, n = fit_dimension(s, c, 1, R // 16)
        px = 2 * m['half'] / (R - 1)
        out.append(dict(level=m['level'], half=m['half'], px=px, sizes=s.tolist(), counts=c.tolist(), D=D, se=se,
                        fit_px=[1, R // 16], boundary_frac=float(b.mean())))
    return out


def uncert(tag):
    f = f'cache/uncert/{tag}.npz'
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    eps, fr = d['eps'], d['frac']
    M = len(d['A'])
    sel = fr > 20 / M
    p = np.polyfit(np.log10(eps[sel]), np.log10(fr[sel]), 1) if sel.sum() >= 3 else [np.nan, np.nan]
    return dict(eps=eps.tolist(), frac=fr.tolist(), alpha=float(p[0]), D=float(2 - p[0]), M=int(M),
                fit_eps=[float(eps[sel].min()), float(eps[sel].max())] if sel.any() else None)


def rescheck(prefix, key='raw'):
    fs = sorted(glob.glob(f'cache/maps/{prefix}_R*.npz'), key=lambda s: int(s.split('_R')[-1][:-4]))
    if not fs:
        return None
    labs = {int(f.split('_R')[-1][:-4]): np.load(f)[key] for f in fs}
    Rs = sorted(labs)
    win = np.load(fs[0])['win']; W = win[1] - win[0]
    res = dict(R=Rs, curves={}, flip={}, identical={})
    for R in Rs:
        b = boundary(labs[R]); s, c = box_counts(b, 1, R // 8)
        res['curves'][R] = dict(eps=(s * W / (R - 1)).tolist(), counts=c.tolist(), D=fit_dimension(s, c, 1, R // 16)[0])
    for Rc, Rf in zip(Rs[:-1], Rs[1:]):
        c, f = labs[Rc], labs[Rf]
        res['identical'][f'{Rc}->{Rf}'] = float((f[::2, ::2] == c).mean())   # same float64 sample points
        # new samples (odd rows or cols): disagree with either coarse neighbour along that axis
        new_r = f[1::2, ::2]; up, dn = c[:-1, :], c[1:, :]
        new_c = f[::2, 1::2]; lf, rt = c[:, :-1], c[:, 1:]
        flip = np.concatenate([((new_r != up) | (new_r != dn)).ravel(), ((new_c != lf) | (new_c != rt)).ravel()])
        res['flip'][f'{Rc}->{Rf}'] = float(flip.mean())
    return res


def main(name):
    if name == 'fact3':
        zmain, znull, umain = 'zA', 'znull', ['f3_e1.1_full', 'f3_e1.1_L1']
        unull, rpre, title = 'f3_null', 'res_f3_L3', 'Four Roots: f = ¼(xyz−1)², slice x+y+z = √3·0.5'
        null_map = 'cache/maps/null_f3_res4097.npz'
    else:
        zmain, znull, umain = 'zX', None, ['xor_e1.2', 'xor_e1.2_canon']
        unull, rpre, title = 'xor_e0.3_null', None, 'XOR 2-2-1 tanh net, seed-4 slice'
        null_map = None
    out = dict(zoom=levels(zmain), zoom_null=levels(znull) if znull else None,
               uncert={u: uncert(u) for u in umain + [unull]}, res=rescheck(rpre) if rpre else None)
    if null_map and os.path.exists(null_map):
        lab = np.load(null_map)['raw']; b = boundary(lab); s, c = box_counts(b, 1, 512)
        out['null_map'] = dict(sizes=s.tolist(), counts=c.tolist(), D=fit_dimension(s, c, 1, 256))
    json.dump(out, open(f'cache/verify_{name}.json', 'w'), indent=1, default=float)

    fig, axs = plt.subplots(1, 4, figsize=(26, 6.4), facecolor='#fcfcfb')
    # (1) box counts per zoom level, in physical units
    ax = axs[0]
    for i, z in enumerate(out['zoom']):
        e = np.array(z['sizes']) * z['px']
        ax.loglog(e, z['counts'], color=C_MAIN, alpha=0.35 + 0.65 * i / max(len(out['zoom']) - 1, 1), lw=1.6,
                  label='large-η zoom levels' if i == 0 else None)
    for i, z in enumerate(out['zoom_null'] or []):
        e = np.array(z['sizes']) * z['px']
        ax.loglog(e, z['counts'], color=C_NULL, ls='--', lw=1.6, label='null (small η) zoom levels' if i == 0 else None)
    ax.set_xlabel('box side ε (slice units)'); ax.set_ylabel('boxes touching a basin boundary N(ε)')
    ax.legend(frameon=False); ax.set_title('box counting, every zoom level', color=INKC)
    # (2) fitted D per level
    ax = axs[1]
    Lm = [z['level'] for z in out['zoom']]
    ax.errorbar([z['half'] * 2 for z in out['zoom']], [z['D'] for z in out['zoom']], yerr=[z['se'] for z in out['zoom']],
                color=C_MAIN, marker='o', ms=8, label='large η')
    if out['zoom_null']:
        ax.errorbar([z['half'] * 2 for z in out['zoom_null']], [z['D'] for z in out['zoom_null']],
                    yerr=[z['se'] for z in out['zoom_null']], color=C_NULL, marker='s', ms=8, ls='--', label='null')
    ax.set_xscale('log'); ax.invert_xaxis(); ax.set_ylim(0.9, 2.0)
    ax.set_xlabel('window width (zooming in →)'); ax.set_ylabel('box-counting dimension D (fit 1–R/16 px)')
    ax.legend(frameon=False); ax.set_title('D is stable across the zoom', color=INKC)
    # (3) uncertainty exponent
    ax = axs[2]
    for u, col, ls in zip(umain + [unull], [C_MAIN, C_ALT, C_NULL], ['-', '-', '--']):
        r = out['uncert'].get(u)
        if not r: continue
        fr = np.array(r['frac']); e = np.array(r['eps'])
        ok = fr > 0
        ax.loglog(e[ok], fr[ok], color=col, ls=ls, marker='o', ms=5, label=f'{u}: α={r["alpha"]:.2f}, D=2−α={r["D"]:.2f}')
    ax.set_xlabel('perturbation ε / window width'); ax.set_ylabel('fraction of ε-uncertain initialisations f(ε)')
    ax.legend(frameon=False, fontsize=10); ax.set_title('uncertainty exponent (riddled ⇒ α≈0)', color=INKC)
    # (4) resolution check
    ax = axs[3]
    if out['res']:
        for R, col in zip(out['res']['R'], ['#9ec3e6', '#5a95cf', C_MAIN]):
            cv = out['res']['curves'][R]
            ax.loglog(cv['eps'], cv['counts'], color=col, marker='.', label=f'R={R}: D={cv["D"]:.3f}')
        txt = '\n'.join(f'refine {k}: new-sample flips {v:.3%}' for k, v in out['res']['flip'].items())
        ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=10, color=MUTED)
        ax.legend(frameon=False); ax.set_xlabel('box side ε (slice units)'); ax.set_ylabel('N(ε)')
        ax.set_title('resolution check (same window, 1×/2×/4×)', color=INKC)
    else:
        ax.axis('off')
    fig.suptitle(title, color=INKC, fontsize=16)
    fig.tight_layout()
    fig.savefig(f'gallery/verify_{name}.png', dpi=110, facecolor=fig.get_facecolor())
    print(json.dumps({k: (v if k != 'zoom' and k != 'zoom_null' else [(z['level'], round(z['D'], 3)) for z in v or []])
                      for k, v in out.items() if k not in ('res',)}, default=float)[:3000])
    if out['res']:
        print({R: out['res']['curves'][R]['D'] for R in out['res']['R']}, out['res']['flip'], out['res']['identical'])


if __name__ == '__main__':
    main(sys.argv[1])
