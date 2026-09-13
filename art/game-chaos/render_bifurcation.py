"""Wide single-ink bifurcation plate with a Spectral Lyapunov strip underneath (reads cache/bifurcation.npz)."""
import numpy as np
import matplotlib.pyplot as plt

from render_lib import GAL, PAPER, INK, spectral


def main():
    d = np.load('cache/bifurcation.npz')
    for ys in ('0.4168', '0.7'):
        X = d[f'x_{ys}']; s = d[f's_{ys}']; L = d[f'L_{ys}'].astype(float)
        W, Hh = 4800, 1500
        H = np.zeros((Hh, W))
        xi = np.clip(((s - s[0]) / (s[-1] - s[0]) * (W - 1)).astype(int), 0, W - 1)
        for k in range(X.shape[1]):
            yi = np.clip(((1 - X[:, k]) * (Hh - 1)).astype(int), 0, Hh - 1)
            np.add.at(H, (yi, xi), 1)
        # columns are 3000 samples on 4800 px: normalise by samples per column
        cols = np.bincount(xi, minlength=W).astype(float); cols[cols == 0] = 1
        H = H / cols[None] / X.shape[1] * 60
        cov = 1 - np.exp(-14.0 * H ** 0.8)
        ink = np.array([0.114, 0.114, 0.125]); paper = np.array([0.953, 0.933, 0.886])
        img = paper * (1 - cov[..., None]) + ink * cov[..., None]
        strip = spectral(L[None, :])
        fig = plt.figure(figsize=(24, 9.6), dpi=200, facecolor=PAPER)
        ax = fig.add_axes([0.05, 0.33, 0.92, 0.57]); ax.imshow(img, extent=[s[0], s[-1], 0, 1], aspect='auto', interpolation='lanczos')
        ax.set_ylabel('share of traffic on link 1  (x = y)', color=INK, fontsize=12); ax.tick_params(colors=INK, labelbottom=False)
        ax2 = fig.add_axes([0.05, 0.2, 0.92, 0.07]); ax2.imshow(strip, extent=[s[0], s[-1], 0, 1], aspect='auto', interpolation='nearest')
        ax2.set_yticks([]); ax2.set_xlabel(r'effective step size  $s=\eta(a+b)$', color=INK, fontsize=12); ax2.tick_params(colors=INK)
        ax2.text(s[0], 1.45, 'largest Lyapunov exponent (Spectral split at λ = 0: purple = periodic, red = chaotic)', fontsize=10, color=INK)
        fig.text(0.05, 0.94, f'Cascade: where two learners settle, as the step size grows  —  congestion game, y* = {ys}', fontsize=17,
                 family='DejaVu Serif', color=INK)
        fig.text(0.05, 0.06, f'3000 step sizes; for each, 600 iterates after 3000 transient steps of u′ = u − s(σ(u) − y*) drawn as ink density (declared: coverage = 1 − exp(−density)). '
                 f'λ > 0 on {100*np.mean(L>0):.0f}% of this line.\n' + ('This line passes through the shrimp field of the zoom sequence; ' if ys == '0.4168' else '') + 'periodic windows open and close through period-doubling cascades.',
                 fontsize=10.5, color=INK, linespacing=1.5)
        fig.savefig(f'{GAL}/bifurcation_ystar{ys}.png', facecolor=PAPER); plt.close(fig)
        print('bifurcation', ys)


if __name__ == '__main__':
    main()
