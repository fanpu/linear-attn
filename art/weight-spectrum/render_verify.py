"""Verification sheets: alpha fits, MP departure, batch-size series, and the generalization caveat.

usage: python render_verify.py [main_run]
Writes gallery/verify_*.png and cache/metrics_table.json
"""
import json, re, sys
import numpy as np
import matplotlib.pyplot as plt
from common import load_run, metrics, list_runs, fit_powerlaw, shape_NM, mp_edges, LAYERS, CACHE
import render_common as R

LC = {'FC1': '#2b4a6f', 'FC2': '#b3342b', 'FC3': '#c9962b'}


def style_ax(ax):
    ax.set_facecolor(R.PAPER)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def table():
    rows = []
    for name in list_runs():
        r = load_run(name); bs = r['meta']['bs']
        row = dict(run=name, bs=bs, seed=r['meta']['seed'], steps=int(r['step'][-1]), test_acc=float(r['test_acc'][-1]),
                   train_acc=float(r['train_acc'][-1]), wall_s=float(r['meta']['wall']))
        for L in LAYERS:
            m = metrics(name, L)
            for k in ['alpha', 'alpha_se', 'alpha_shuf', 'lmax_over_null', 'lmax_over_mp', 'n_out', 'srank', 'srank_shuf',
                      'ks_vs_null', 'n_tail']:
                row[f'{L}_{k}'] = float(m[k][-1])
            row[f'{L}_alpha_init'] = float(m['alpha'][0])
        rows.append(row)
    rows.sort(key=lambda d: (d['bs'], d['seed']))
    json.dump(rows, open(f'{CACHE}/metrics_table.json', 'w'), indent=1)
    return rows


def sheet_main(run):
    r = load_run(run); st = np.maximum(r['step'], 1)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6), facecolor=R.PAPER)
    for L in LAYERS:
        m = metrics(run, L)
        a = axs[0, 0]; a.plot(st, m['alpha'], color=LC[L], lw=1.6, label=f'{L} measured')
        a.plot(st, m['alpha_shuf'], color=LC[L], lw=0.9, ls=':', label=f'{L} shuffled null')
        axs[0, 1].plot(st, m['lmax_over_mp'], color=LC[L], lw=1.6, label=L)
        axs[0, 2].plot(st, m['n_out'], color=LC[L], lw=1.6, label=L)
        axs[1, 1].plot(st, m['srank'], color=LC[L], lw=1.6, label=f'{L}')
        axs[1, 1].plot(st, m['srank_shuf'], color=LC[L], lw=0.9, ls=':')
        axs[1, 2].plot(st, m['ks_vs_null'], color=LC[L], lw=1.6, label=L)
    axs[0, 0].set_title('power-law exponent α of ESD tail (Clauset MLE, xmin by KS)', fontsize=9)
    axs[0, 0].axhspan(2, 4, color='#b3342b', alpha=0.06); axs[0, 0].legend(fontsize=6.5, ncol=2, frameon=False)
    axs[0, 1].set_title('λmax / MP edge λ₊(σ² of current entries)', fontsize=9); axs[0, 1].axhline(1, color='k', lw=0.6)
    axs[0, 2].set_title('# eigenvalues above the shuffled-null bulk edge', fontsize=9)
    axs[1, 1].set_title('stable rank Σλ/λmax (dotted: shuffled null)', fontsize=9); axs[1, 1].set_yscale('log')
    axs[1, 2].set_title('KS distance ESD vs shuffled null (log λ)', fontsize=9)
    a = axs[1, 0]
    a.plot(st, r['test_acc'], color='#1b1a17', lw=1.6, label='test acc'); a.plot(st, r['train_acc'], color='#8a847a', lw=1.2, label='train acc (10k subset)')
    a.set_title('accuracy', fontsize=9); a.legend(fontsize=7, frameon=False); a.set_ylim(0.6, 1.0)
    for a in axs.ravel():
        a.set_xscale('log'); a.set_xlabel('SGD step', fontsize=8); style_ax(a)
    axs[0, 1].legend(fontsize=7, frameon=False)
    fig.suptitle(f'Departure from randomness over training — {run}  (MLP 784-1024³-10, FashionMNIST, SGD bs {r["meta"]["bs"]})',
                 fontsize=12, family=R.SERIF)
    fig.tight_layout()
    return R.save(fig, f'verify_over_training_{run}.png', dpi=150)


def sheet_ccdf(run):
    r = load_run(run)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=R.PAPER)
    for a, L in zip(axs, LAYERS):
        N, M = shape_NM(r, L)
        for t, c, lab in [(0, '#8a847a', 'step 0'), (-1, LC[L], 'final')]:
            lam = np.sort(r[f'{L}/lam'][t])[::-1]; ccdf = np.arange(1, lam.size + 1) / lam.size
            a.loglog(lam, ccdf, color=c, lw=1.8, label=f'{lab} measured')
            sh = np.sort(r[f'{L}/lam_shuf'][t])[::-1]
            a.loglog(sh, ccdf, color=c, lw=0.9, ls=':', label=f'{lab} shuffled')
        f = fit_powerlaw(r[f'{L}/lam'][-1])
        x = np.geomspace(f['xmin'], r[f'{L}/lam'][-1].max(), 30)
        a.loglog(x, f['n_tail'] / M * (x / f['xmin']) ** (1 - f['alpha']), color='k', lw=1.1, ls='--',
                 label=f"fit α={f['alpha']:.2f}±{f['alpha_se']:.2f}, n_tail={f['n_tail']}")
        lp = mp_edges(r[f'{L}/elem_var'][-1], N / M)[1]
        a.axvline(lp, color='#2b4a6f', lw=0.8, ls='-.', label='MP edge λ₊ (final σ²)')
        a.set_title(f'{L} ({N}×{M}) CCDF of eigenvalues', fontsize=10); a.legend(fontsize=6.5, frameon=False)
        a.set_xlabel('λ', fontsize=9); style_ax(a); a.set_xlim(np.percentile(r[f'{L}/lam'][[0, -1]], 2) * 0.7, None)
    axs[0].set_ylabel('P(Λ ≥ λ)')
    fig.tight_layout()
    return R.save(fig, f'verify_ccdf_{run}.png', dpi=150)


def sheet_batch(rows):
    """Group by batch size: thin marks = individual seeds, line = seed mean. alpha shown hollow where < 5 eigenvalues
    exceed the null edge (fit is then meaningless: see init/null alpha of 5-20)."""
    bss = sorted(set(d['bs'] for d in rows))
    grp = {b: [d for d in rows if d['bs'] == b] for b in bss}
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), facecolor=R.PAPER)
    def series(ax, key, color, label, mask_alpha=None):
        mean = np.array([np.mean([d[key] for d in grp[b]]) for b in bss])
        ax.plot(bss, mean, '-', color=color, lw=1.6, label=label, zorder=2)
        for b in bss:
            for d in grp[b]:
                ok = True if mask_alpha is None else d[f'{mask_alpha}_n_out'] >= 5
                ax.plot([b], [d[key]], 'o', ms=5, mfc=color if ok else 'none', mec=color, mew=1.0, zorder=3)
    for L in LAYERS:
        series(axs[0], f'{L}_alpha', LC[L], L, mask_alpha=L)
        series(axs[1], f'{L}_lmax_over_null', LC[L], L)
        series(axs[2], f'{L}_n_out', LC[L], L)
    series(axs[3], 'test_acc', '#1b1a17', 'test')
    series(axs[3], 'train_acc', '#8a847a', 'train (10k subset)')
    axs[0].axhspan(2, 4, color='#c9962b', alpha=0.08, lw=0); axs[0].text(9, 2.1, 'reference band 2<α<4', fontsize=7, color='#7a5a10')
    t = ['final power-law α of ESD tail (hollow: <5 eigs above null edge)', 'λmax / null bulk edge',
         '# eigenvalues above null bulk edge', 'accuracy after 30 epochs']
    for a, tt in zip(axs, t):
        a.set_xscale('log', base=2); a.set_title(tt, fontsize=9.5); a.set_xlabel('batch size (lr 0.01, mom 0.9, 30 epochs)', fontsize=9)
        a.set_xticks(bss); a.set_xticklabels([str(b) for b in bss]); a.minorticks_off()
        style_ax(a); a.legend(fontsize=7, frameon=False)
    axs[0].set_yscale('log'); axs[0].set_yticks([2, 3, 4, 6, 10, 15]); axs[0].set_yticklabels(['2', '3', '4', '6', '10', '15'])
    axs[1].set_yscale('log')
    ns = sorted(set(len(g) for g in grp.values()))
    fig.suptitle(f'Batch-size series ({"/".join(map(str, ns))} seeds per point; dots = seeds, line = mean): smaller batches grow heavier tails, '
                 'but test accuracy stays flat from 16 to 256', fontsize=11.5, family=R.SERIF)
    fig.tight_layout()
    return R.save(fig, 'verify_batch_series.png', dpi=150)


def sheet_caveat(rows):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=R.PAPER)
    abar = np.array([np.mean([d[f'{L}_alpha'] for L in LAYERS]) for d in rows])
    acc = np.array([d['test_acc'] for d in rows]); gap = np.array([d['train_acc'] - d['test_acc'] for d in rows])
    bs = np.array([d['bs'] for d in rows])
    from scipy.stats import spearmanr
    rel = np.array([all(d[f'{L}_n_out'] >= 5 for L in LAYERS) for d in rows])
    seed = np.array([d['seed'] for d in rows])
    for a, y, lab in [(axs[0], acc, 'test accuracy'), (axs[1], gap, 'train − test accuracy gap')]:
        cm = plt.get_cmap('viridis')
        col = cm((np.log2(bs) - 3) / 7)
        a.scatter(abar[rel], y[rel], c=col[rel], s=46, edgecolor='k', lw=0.4, zorder=3)
        a.scatter(abar[~rel], y[~rel], facecolor='none', edgecolor=col[~rel], s=46, lw=1.2, zorder=3)
        for x_, y_, b, s in zip(abar, y, bs, seed):
            a.annotate(f'{b}·s{s}', (x_, y_), fontsize=6.5, xytext=(4, 3), textcoords='offset points', color='#444')
        rs = spearmanr(abar[rel], y[rel])[0]; rs16 = spearmanr(abar[rel & (bs >= 16) & (bs <= 256)], y[rel & (bs >= 16) & (bs <= 256)])[0]
        a.set_title(f'{lab} vs mean α\nSpearman ρ = {rs:.2f} (filled points, n={rel.sum()});  bs 16–256 only: ρ = {rs16:.2f}', fontsize=9); print('CAVEAT', lab, 'rho', round(rs, 3), 'rho16-256', round(rs16, 3))
        a.set_xscale('log'); a.set_xticks([2, 3, 4, 6, 10]); a.set_xticklabels(['2', '3', '4', '6', '10']); a.minorticks_off(); a.set_xlabel('mean final α over FC1–FC3 (hollow: some layer has <5 eigs above null edge)', fontsize=8.5); style_ax(a)
    fig.text(0.5, 0.01, 'Caveat: one architecture, one dataset, a confounded series (batch size also changes step count and '
             'noise scale).\nHT-SR claims that α predicts generalization are contested; this is a correlation, not evidence.',
             ha='center', fontsize=8, style='italic')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return R.save(fig, 'verify_caveat_alpha_vs_generalization.png', dpi=150)


if __name__ == '__main__':
    R.set_rc('DejaVu Sans')
    main = sys.argv[1] if len(sys.argv) > 1 else 'mlp_bs16_s0'
    rows = table()
    for d in rows:
        print(d['run'], 'acc %.4f' % d['test_acc'], ' '.join(f"{L}: a={d[L+'_alpha']:.2f}±{d[L+'_alpha_se']:.2f} "
              f"(null {d[L+'_alpha_shuf']:.1f}, init {d[L+'_alpha_init']:.1f}) lmax/MP={d[L+'_lmax_over_mp']:.1f} nout={d[L+'_n_out']:.0f}" for L in LAYERS))
    if main in list_runs():
        print(sheet_main(main)); print(sheet_ccdf(main))
    if len(rows) >= 3:
        print(sheet_batch(rows)); print(sheet_caveat(rows))
