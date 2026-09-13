"""Shared analysis helpers: loading runs, Marchenko-Pastur law, power-law (alpha) fits, departure metrics.

No rendering here.  ESD convention (see train.py): W is oriented N x M with N >= M, Q = N/M,
X = W^T W / N, eigenvalues lambda_i = s_i^2 / N (M of them).
"""
import glob, json, os
from functools import lru_cache
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
GALLERY = os.path.join(HERE, 'gallery')
LAYERS = ['FC1', 'FC2', 'FC3']  # FC4 (10 x 1024) has only 10 eigenvalues: excluded from ESD work


@lru_cache(maxsize=32)
def load_run(name):
    z = np.load(os.path.join(CACHE, f'{name}.npz'))
    d = {k: z[k] for k in z.files}
    d['meta'] = json.loads(str(d['meta']))
    return d


def full_W(name, layer, step=None):
    """Return (steps, dict step->W) of saved full matrices, or one W if step given."""
    files = sorted(glob.glob(os.path.join(CACHE, f'{name}_W', 'step*.npz')))
    steps = [int(os.path.basename(f)[4:12]) for f in files]
    if step is None:
        return steps
    return np.load(os.path.join(CACHE, f'{name}_W', f'step{step:08d}.npz'))[layer]


def shape_NM(run, layer):
    N, M, _ = run['meta']['shapes'][layer]
    return N, M


# ------------------------------------------------------------------ Marchenko-Pastur
def mp_edges(sigma2, Q):
    return sigma2 * (1 - 1 / np.sqrt(Q)) ** 2, sigma2 * (1 + 1 / np.sqrt(Q)) ** 2


def mp_density(lam, sigma2, Q):
    """Density of eigenvalues of X = W^T W / N for iid entries of variance sigma2, Q = N/M >= 1."""
    lo, hi = mp_edges(sigma2, Q)
    lam = np.asarray(lam, float)
    out = np.zeros_like(lam)
    m = (lam > lo) & (lam < hi) & (lam > 0)
    out[m] = Q / (2 * np.pi * sigma2 * lam[m]) * np.sqrt((hi - lam[m]) * (lam[m] - lo))
    return out


def mp_density_log10(lam, sigma2, Q):
    """Density w.r.t. log10(lambda): rho(lambda) * lambda * ln 10."""
    return mp_density(lam, sigma2, Q) * np.asarray(lam) * np.log(10)


# ------------------------------------------------------------------ power-law fit (Clauset et al. 2009)
def fit_powerlaw(lam, min_tail=10, max_frac=0.95):
    """Continuous power law p(x) ~ x^-alpha for x >= xmin, xmin chosen by minimal KS distance.

    This is the same estimator WeightWatcher uses (powerlaw package, continuous MLE).
    Returns dict(alpha, xmin, n_tail, ks, alpha_se).
    """
    x = np.sort(np.asarray(lam, float)[np.asarray(lam) > 0])
    n = x.size
    best = None
    lo = int(n * (1 - max_frac))
    for i in range(lo, n - min_tail):
        xmin = x[i]
        t = x[i:]
        k = t.size
        L = np.log(t / xmin).sum()
        if L <= 0:
            continue
        a = 1 + k / L
        # KS between empirical CDF of tail and fitted CDF 1 - (x/xmin)^(1-a)
        F = 1 - (t / xmin) ** (1 - a)
        emp_hi = np.arange(1, k + 1) / k
        emp_lo = np.arange(0, k) / k
        D = max(np.abs(emp_hi - F).max(), np.abs(F - emp_lo).max())
        if best is None or D < best['ks']:
            best = dict(alpha=a, xmin=xmin, n_tail=k, ks=D, alpha_se=(a - 1) / np.sqrt(k))
    return best


def stable_rank(lam):
    return float(np.sum(lam) / np.max(lam))


def ks_two_sample_log(a, b):
    a = np.sort(np.log10(a[a > 0])); b = np.sort(np.log10(b[b > 0]))
    g = np.concatenate([a, b])
    Fa = np.searchsorted(a, g, 'right') / a.size
    Fb = np.searchsorted(b, g, 'right') / b.size
    return float(np.abs(Fa - Fb).max())


@lru_cache(maxsize=32)
def metrics(name, layer):
    """Per-checkpoint departure-from-randomness metrics for one run/layer."""
    r = load_run(name)
    lam = r[f'{layer}/lam']; sh = r[f'{layer}/lam_shuf']; var = r[f'{layer}/elem_var']
    N, M = shape_NM(r, layer); Q = N / M
    out = {k: [] for k in ['alpha', 'alpha_se', 'xmin', 'n_tail', 'ks', 'alpha_shuf', 'lmax_over_null',
                           'lmax_over_mp', 'n_out', 'srank', 'srank_shuf', 'ks_vs_null']}
    for t in range(lam.shape[0]):
        f = fit_powerlaw(lam[t]); fs = fit_powerlaw(sh[t])
        lp = mp_edges(var[t], Q)[1]
        out['alpha'].append(f['alpha']); out['alpha_se'].append(f['alpha_se']); out['xmin'].append(f['xmin'])
        out['n_tail'].append(f['n_tail']); out['ks'].append(f['ks']); out['alpha_shuf'].append(fs['alpha'])
        out['lmax_over_null'].append(lam[t].max() / sh[t].max())
        out['lmax_over_mp'].append(lam[t].max() / lp)
        out['n_out'].append(int((lam[t] > sh[t].max()).sum()))
        out['srank'].append(stable_rank(lam[t])); out['srank_shuf'].append(stable_rank(sh[t]))
        out['ks_vs_null'].append(ks_two_sample_log(lam[t], sh[t]))
    return {k: np.array(v) for k, v in out.items()}


def list_runs(pattern='mlp_bs*_s*'):
    fs = sorted(glob.glob(os.path.join(CACHE, pattern + '.npz')))
    return [os.path.basename(f)[:-4] for f in fs]


def log_kde(lam, grid, bw=0.04):
    """Gaussian KDE of log10(lambda) evaluated on grid (log10 units), normalised as a density in log10 lambda."""
    x = np.log10(lam[lam > 0])
    d = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(1) / (x.size * bw * np.sqrt(2 * np.pi))
