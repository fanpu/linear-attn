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
    axs[0, 2].set_title('# eigenvalues above the shuffled-null maximum', fontsize=9)
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
        a.set_xlabel('λ', fontsize=9); style_ax(a)
    axs[0].set_ylabel('P(Λ ≥ λ)')
    fig.tight_layout()
    return R.save(fig, f'verify_ccdf_{run}.png', dpi=150)


def sheet_batch(rows):
    bs = np.array([d['bs'] for d in rows])
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), facecolor=R.PAPER)
    for L in LAYERS:
        axs[0].plot(bs, [d[f'{L}_alpha'] for d in rows], 'o-', color=LC[L], lw=1.5, ms=4, label=f'{L}')
        axs[0].errorbar(bs, [d[f'{L}_alpha'] for d in rows], yerr=[d[f'{L}_alpha_se'] for d in rows], fmt='none', ecolor=LC[L], lw=0.8)
        axs[0].plot(bs, [d[f'{L}_alpha_shuf'] for d in rows], 'x:', color=LC[L], lw=0.8, ms=4)
        axs[1].plot(bs, [d[f'{L}_lmax_over_mp'] for d in rows], 'o-', color=LC[L], lw=1.5, ms=4, label=L)
        axs[2].plot(bs, [d[f'{L}_n_out'] for d in rows], 'o-', color=LC[L], lw=1.5, ms=4, label=L)
    axs[3].plot(bs, [d['test_acc'] for d in rows], 'o-', color='#1b1a17', lw=1.5, ms=4, label='test')
    axs[3].plot(bs, [d['train_acc'] for d in rows], 'o-', color='#8a847a', lw=1.2, ms=4, label='train (10k)')
    t = ['final α (x: shuffled null)', 'λmax / MP edge', '# eigenvalues above null max', 'accuracy after 30 epochs']
    for a, tt in zip(axs, t):
        a.set_xscale('log', base=2); a.set_title(tt, fontsize=10); a.set_xlabel('batch size (lr 0.01, 30 epochs)', fontsize=9)
        style_ax(a); a.legend(fontsize=7, frameon=False)
    axs[1].set_yscale('log')
    fig.suptitle('Batch-size series: smaller batches (more SGD noise, more steps) → heavier tails?', fontsize=12, family=R.SERIF)
    fig.tight_layout()
    return R.save(fig, 'verify_batch_series.png', dpi=150)


def sheet_caveat(rows):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=R.PAPER)
    abar = np.array([np.mean([d[f'{L}_alpha'] for L in LAYERS]) for d in rows])
    acc = np.array([d['test_acc'] for d in rows]); gap = np.array([d['train_acc'] - d['test_acc'] for d in rows])
    bs = np.array([d['bs'] for d in rows])
    for a, y, lab in [(axs[0], acc, 'test accuracy'), (axs[1], gap, 'train − test accuracy gap')]:
        sc = a.scatter(abar, y, c=np.log2(bs), cmap='art.cyanotype_r' if 'art.cyanotype_r' in plt.colormaps() else 'viridis', s=50, edgecolor='k', lw=0.4)
        for x_, y_, b in zip(abar, y, bs):
            a.annotate(f'bs {b}', (x_, y_), fontsize=7, xytext=(4, 3), textcoords='offset points')
        rho = np.corrcoef(abar, y)[0, 1] if len(y) > 2 else np.nan
        a.set_title(f'{lab} vs mean α  (Pearson r = {rho:.2f}, n = {len(y)})', fontsize=10)
        a.set_xlabel('mean final α over FC1–FC3', fontsize=9); style_ax(a)
    fig.text(0.5, 0.01, 'Caveat: one architecture, one dataset, a confounded series (batch size also changes step count and '
             'noise scale). HT-SR claims that α predicts generalization are contested; this is a correlation, not evidence.',
             ha='center', fontsize=8, style='italic')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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
