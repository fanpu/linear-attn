"""Render EWA-on-RPS Lyapunov planes (reads cache/rps_*.npz).
Finite-time lambda of quasi-periodic orbits sits within ~1e-3 of zero, so the split seam is placed at
lambda = THR (declared noise floor): lambda > THR is drawn on the chaotic side."""
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from render_lib import GAL, PAPER, INK, spectral, dark_diverging, save_png, flip

THR = 3e-3
LABEL = {'beta': r'intensity of choice  $\beta$', 'eps': r'tie payoff  $\varepsilon_x=-\varepsilon_y$',
         'alpha': r'memory loss  $\alpha$'}


def main():
    for f in sorted(glob.glob('cache/rps_*.npz')):
        name = os.path.basename(f)[:-4]
        d = np.load(f); L = d['L'].astype(float) - THR
        x, y = d['x'], d['y']; xk, yk = str(d['xk']), str(d['yk'])
        save_png(flip(spectral(L)), f'{GAL}/{name}_spectral.png')
        save_png(flip(spectral(L, pairing='indigo_madder')), f'{GAL}/{name}_indigo.png')
        save_png(flip(dark_diverging(L)), f'{GAL}/{name}_dark.png')
        fig = plt.figure(figsize=(11, 12.6), dpi=200, facecolor=PAPER)
        ax = fig.add_axes([0.11, 0.2, 0.84, 0.72])
        ax.imshow(flip(spectral(L)), extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', interpolation='nearest')
        ax.set_xlabel(LABEL[xk], fontsize=13, color=INK); ax.set_ylabel(LABEL[yk], fontsize=13, color=INK)
        ax.tick_params(colors=INK)
        fig.text(0.11, 0.955, 'PLATE  —  Experience-weighted attraction on rock–paper–scissors', fontsize=16,
                 family='DejaVu Serif', color=INK)
        fx = eval(str(d['fixed']))
        fixtxt = ', '.join([f'α = {fx["alpha"]}' if 'alpha' in fx else '', f'ε_x = −ε_y = {fx["eps"]}' if 'eps' in fx else '']).strip(', ')
        fig.text(0.11, 0.933, f'Largest Lyapunov exponent over the ({xk}, {yk}) plane;  fixed {fixtxt};  zero-sum;  '
                 'starts x₀ = (.5,.3,.2), y₀ = (.2,.3,.5)', fontsize=11.5, color=INK)
        cap = (f'Q′ = (1−α) Q + β·A(ε) y,  x = softmax(Q), both players (Galla & Farmer 2013 learning rule; SAF 2002 payoffs). '
               f'{L.shape[1]}×{L.shape[0]} px, float64,\n{int(d["T0"])} transient + {int(d["T1"])} averaged steps, tangent-vector λ. '
               f'λ > {THR:g} (chaotic side) on {100*np.mean(L>0):.1f}% of pixels; max λ = {L.max()+THR:.3f}.\n'
               f'Pale-green interiors: strategies pinned within 10⁻⁶ of the simplex edge on {100*np.mean(d["xmin"]<1e-6):.0f}% of '
               'pixels (near-pure best-response cycling, strongly contracting).\nDeclared colour: Spectral split with the seam at '
               f'λ = {THR:g} (finite-time noise floor for quasi-periodic orbits), each side rank-normalised. Vertical cuts = coexisting attractors (fixed start).')
        fig.text(0.11, 0.075, cap, fontsize=9.5, color=INK, linespacing=1.55)
        fig.savefig(f'{GAL}/plate_{name}.png', facecolor=PAPER); plt.close(fig)
        print('rendered', name, flush=True)


if __name__ == '__main__':
    main()
