"""Core math for the double-descent post: closed forms + honest simulations (float64, numpy).

Linear model (Hastie, Montanari, Rosset & Tibshirani 2022):
    y = x^T beta + eps,  x ~ N(0, I_p),  ||beta||^2 = r2,  eps ~ N(0, sigma2),  n samples, gamma = p / n.
    ridge:  beta_hat = argmin (1/n)||y - X b||^2 + lam ||b||^2 = (X^T X / n + lam I)^-1 X^T y / n
    risk:   R = E_x[(x^T beta_hat - x^T beta)^2 | X] = ||beta_hat - beta||^2   (excess risk, noise floor excluded)

Marchenko-Pastur: the empirical spectral distribution of X^T X / n (p x p) converges to MP(gamma) with Stieltjes
transform m(z) = int dmu(s) / (s - z), the root of  gamma z m^2 + (z - 1 + gamma) m + 1 = 0.
"""
import numpy as np

# ----------------------------------------------------------------------------------------------------------------
# Marchenko-Pastur
# ----------------------------------------------------------------------------------------------------------------

def mp_edges(gamma):
    return (1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2


def mp_density(s, gamma):
    """Density of the continuous part of MP(gamma) (eigs of X^T X/n, p x p). For gamma > 1 there is also an atom
    of mass 1 - 1/gamma at 0."""
    a, b = mp_edges(gamma)
    s = np.asarray(s, float)
    out = np.zeros_like(s)
    m = (s > a) & (s < b)
    out[m] = np.sqrt((b - s[m]) * (s[m] - a)) / (2 * np.pi * gamma * s[m])
    return out


def mp_stieltjes_neg(lam, gamma):
    """m(-lam) = (1/p) tr (Sigma_hat + lam)^-1 in the limit, lam > 0.  Returns (m, dm/dz at z=-lam)."""
    lam = np.asarray(lam, float); gamma = np.asarray(gamma, float)
    b = 1 - gamma + lam
    m = (-b + np.sqrt(b * b + 4 * gamma * lam)) / (2 * gamma * lam)
    z = -lam
    # implicit differentiation of gamma z m^2 + (z - 1 + gamma) m + 1 = 0
    dm = -(gamma * m * m + m) / (2 * gamma * z * m + z - 1 + gamma)
    return m, dm


# ----------------------------------------------------------------------------------------------------------------
# Isotropic linear model: closed forms
# ----------------------------------------------------------------------------------------------------------------

def ridgeless_risk(gamma, r2=1.0, sigma2=1.0):
    """Hastie et al. Theorem 1 (isotropic): returns (risk, bias, variance). Infinite at gamma = 1."""
    gamma = np.asarray(gamma, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bias = np.where(gamma < 1, 0.0, r2 * (1 - 1 / gamma))
        var = np.where(gamma < 1, sigma2 * gamma / (1 - gamma), sigma2 / (gamma - 1))
        var = np.where(gamma == 1, np.inf, var)
    return bias + var, bias, var


def ridge_risk(gamma, lam, r2=1.0, sigma2=1.0):
    """Limiting ridge risk (isotropic), via MP Stieltjes transform:
        bias = r2 * lam^2 * m'(-lam),   variance = sigma2 * gamma * (m(-lam) - lam * m'(-lam)).
    Returns (risk, bias, variance). lam > 0."""
    m, dm = mp_stieltjes_neg(lam, gamma)
    bias = r2 * lam ** 2 * dm
    var = sigma2 * gamma * (m - lam * dm)
    return bias + var, bias, var


def optimal_lambda(gamma, r2=1.0, sigma2=1.0):
    """Bayes-optimal ridge for beta ~ N(0, r2/p I): lam* = sigma2 * gamma / r2."""
    return sigma2 * np.asarray(gamma, float) / r2


# ----------------------------------------------------------------------------------------------------------------
# General covariance (anisotropic) ridge: deterministic equivalent (Hastie et al. Thm 2 / Dobriban & Wager)
# ----------------------------------------------------------------------------------------------------------------

def general_ridge_risk(evals, beta_coef2, gamma, lam, sigma2=1.0, tol=1e-13, iters=10000):
    """Excess risk E_x (x^T(beta_hat - beta))^2 for x ~ N(0, Sigma), Sigma = diag(evals) (p = len(evals)),
    beta coordinates squared beta_coef2 in the eigenbasis, n = p / gamma, ridge lam >= 0 (lam = 0: ridgeless, gamma != 1).

    Uses the standard deterministic equivalent: kappa solves  kappa - lam = kappa * gamma * (1/p) sum_i e_i / (e_i + kappa).
    df2 = (1/n) sum_i e_i^2/(e_i+kappa)^2;  bias = kappa^2 sum_i b_i^2 e_i/(e_i+kappa)^2 / (1 - df2);
    variance = sigma2 * df2 / (1 - df2).  For lam=0, gamma<1: kappa=0 (bias 0, var = sigma2*gamma/(1-gamma))."""
    e = np.asarray(evals, float); b2 = np.asarray(beta_coef2, float)
    p = len(e); n = p / gamma
    if lam == 0 and gamma < 1:
        return sigma2 * gamma / (1 - gamma), 0.0, sigma2 * gamma / (1 - gamma)
    # solve n = sum e/(e+kappa) + n*lam/kappa  (i.e. kappa - lam = kappa/n sum e/(e+kappa)) by bisection in log kappa
    f = lambda k: k - lam - k / n * np.sum(e / (e + k))
    lo, hi = 1e-14, 1e6
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1 + tol:
            break
    k = np.sqrt(lo * hi)
    df2 = np.sum(e ** 2 / (e + k) ** 2) / n
    bias = k ** 2 * np.sum(b2 * e / (e + k) ** 2) / (1 - df2)
    var = sigma2 * df2 / (1 - df2)
    return bias + var, bias, var


# ----------------------------------------------------------------------------------------------------------------
# Simulation: ridge / ridgeless on Gaussian data, exact conditional bias & variance + one realized noise draw
# ----------------------------------------------------------------------------------------------------------------

def simulate_linear(n, p, lams, r2=1.0, sigma2=1.0, rng=None, cov_sqrt=None, beta=None):
    """One draw of X (n x p), beta (uniform on sphere radius sqrt(r2) unless given) and noise.
    lams: array of ridge values; lam = 0 means min-norm least squares (pseudoinverse).
    Returns dict with arrays over lams: risk (realized ||beta_hat-beta||_Sigma^2), bias, var (conditional on X,
    averaged over noise exactly), and the eigenvalues of X^T X / n (nonzero part)."""
    rng = np.random.default_rng(rng)
    X = rng.standard_normal((n, p))
    if cov_sqrt is not None:
        X = X * cov_sqrt  # diagonal covariance: cov_sqrt is a vector of sqrt eigenvalues
    if beta is None:
        beta = rng.standard_normal(p); beta *= np.sqrt(r2) / np.linalg.norm(beta)
    eps = rng.standard_normal(n) * np.sqrt(sigma2)
    y = X @ beta + eps
    Xs = X / np.sqrt(n)
    # thin SVD Xs = U S V^T via the smaller Gram matrix
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    S2 = S ** 2
    keep = S > S.max() * 1e-12
    U, S, S2, Vt = U[:, keep], S[keep], S2[keep], Vt[keep]
    bV = Vt @ beta                        # beta in the row-space basis
    uy = U.T @ (y / np.sqrt(n))
    sig = cov_sqrt ** 2 if cov_sqrt is not None else None
    out = dict(risk=[], bias=[], var=[], evals=S2)
    for lam in np.atleast_1d(lams):
        shrink = S / (S2 + lam)          # beta_hat = V diag(shrink) U^T y/sqrt(n)
        bhat = Vt.T @ (shrink * uy)
        err = bhat - beta
        filt = S2 / (S2 + lam)            # E[beta_hat | X] = V diag(filt) V^T beta
        bias_vec = Vt.T @ (filt * bV) - beta
        # noise part: beta_hat - E = V diag(shrink) U^T eps/sqrt(n) -> covariance sigma2/n V diag(shrink^2) V^T
        if sig is None:
            risk = err @ err
            bias = bias_vec @ bias_vec
            var = sigma2 / n * np.sum(shrink ** 2)
        else:
            risk = err @ (sig * err)
            bias = bias_vec @ (sig * bias_vec)
            var = sigma2 / n * np.sum(shrink ** 2 * np.einsum("ij,j,ij->i", Vt, sig, Vt))
        out["risk"].append(risk); out["bias"].append(bias); out["var"].append(var)
    for k in ("risk", "bias", "var"):
        out[k] = np.array(out[k])
    return out


# ----------------------------------------------------------------------------------------------------------------
# 1-D random-features regression (hero + widget)
# ----------------------------------------------------------------------------------------------------------------

def rff_features(x, W, B, scale):
    """Random Fourier features phi_j(x) = sqrt(2/P) cos(w_j x + b_j) (P = len(W)).  x: (m,), returns (m, P)."""
    return np.sqrt(2.0 / len(W)) * np.cos(np.outer(x, W) + B) * scale


def minnorm_fit(Phi, y, lam=0.0):
    """Min-norm least squares (lam=0) or ridge  argmin ||y - Phi a||^2 + lam ||a||^2.  Returns coefficients."""
    if lam == 0:
        return np.linalg.lstsq(Phi, y, rcond=None)[0]
    n, P = Phi.shape
    if P > n:
        return Phi.T @ np.linalg.solve(Phi @ Phi.T + lam * np.eye(n), y)
    return np.linalg.solve(Phi.T @ Phi + lam * np.eye(P), Phi.T @ y)


# ----------------------------------------------------------------------------------------------------------------
# Random-features regression: Mei & Montanari (2022) asymptotics
# ----------------------------------------------------------------------------------------------------------------
#   x ~ Unif(S^{d-1}(sqrt d)), theta_a ~ Unif(S^{d-1}(sqrt d)), f(x; a) = sum_a a_a sigma(<theta_a, x>/sqrt d)
#   a_hat = argmin (1/n) sum_j (y_j - f(x_j; a))^2 + (N lam / d) ||a||^2
#   psi1 = N/d, psi2 = n/d, zeta = mu1/mu_star, lam_bar = lam/mu_star^2
#   R_RF -> F1^2 B + (tau^2 + Fstar^2) V + Fstar^2       (test error excludes tau^2)

RELU_MU = dict(mu0=1 / np.sqrt(2 * np.pi), mu1=0.5, mustar=np.sqrt((np.pi - 2) / (4 * np.pi)))


def _mm_polys(chi, z, p1, p2):
    z2, z4, z6 = z ** 2, z ** 4, z ** 6
    E0 = (-chi ** 5 * z6 + 3 * chi ** 4 * z4 + (p1 * p2 - p2 - p1 + 1) * chi ** 3 * z6 - 2 * chi ** 3 * z4
          - 3 * chi ** 3 * z2 + (p1 + p2 - 3 * p1 * p2 + 1) * chi ** 2 * z4 + 2 * chi ** 2 * z2 + chi ** 2
          + 3 * p1 * p2 * chi * z2 - p1 * p2)
    E1 = p2 * chi ** 3 * z4 - p2 * chi ** 2 * z2 + p1 * p2 * chi * z2 - p1 * p2
    E2 = (chi ** 5 * z6 - 3 * chi ** 4 * z4 + (p1 - 1) * chi ** 3 * z6 + 2 * chi ** 3 * z4 + 3 * chi ** 3 * z2
          + (-p1 - 1) * chi ** 2 * z4 - 2 * chi ** 2 * z2 - chi ** 2)
    return E0, E1, E2


def mm_bias_var(psi1, psi2, lam_bar, zeta, iters=20000, tol=1e-13):
    """Mei-Montanari Definition 1: returns (B, V) for lam_bar > 0, via damped fixed point on xi = i sqrt(psi1 psi2 lam_bar).
    On the imaginary axis nu1 = i a, nu2 = i b with a, b > 0, so iterate on the real pair (a, b)."""
    u = np.sqrt(psi1 * psi2 * lam_bar)
    z2 = zeta ** 2
    a, b = psi1 / u, psi2 / u
    for it in range(iters):
        chi = -a * b                                   # nu1 nu2 = (ia)(ib) = -ab
        # nu1 = psi1 / (-xi - nu2 - z2 nu2/(1 - z2 nu1 nu2)); with xi = iu, nu2 = ib: denominator = -i(u + b + z2 b/(1 - z2 chi))
        a_new = psi1 / (u + b + z2 * b / (1 - z2 * chi))
        b_new = psi2 / (u + a + z2 * a / (1 - z2 * chi))
        da, db = abs(a_new - a), abs(b_new - b)
        a, b = 0.5 * a + 0.5 * a_new, 0.5 * b + 0.5 * b_new
        if max(da / a, db / b) < tol:
            break
    chi = -a * b
    E0, E1, E2 = _mm_polys(chi, zeta, psi1, psi2)
    return E1 / E0, E2 / E0


def mm_bias_var_ridgeless(psi1, psi2, zeta):
    """Mei-Montanari Theorem 3 (lambda -> 0 after d -> infinity). Infinite at psi1 = psi2."""
    psi = min(psi1, psi2)
    z2 = zeta ** 2
    chi = -(np.sqrt((psi * z2 - z2 - 1) ** 2 + 4 * z2 * psi) + (psi * z2 - z2 - 1)) / (2 * z2)
    E0, E1, E2 = _mm_polys(chi, zeta, psi1, psi2)
    return E1 / E0, E2 / E0


def mm_risk(psi1, psi2, lam, F1sq=1.0, tau2=0.0, Fstar2=0.0, mu=RELU_MU):
    """Asymptotic test error R_RF (excluding tau^2). lam is the paper's lambda (not lam_bar); lam = 0 -> ridgeless."""
    zeta = mu["mu1"] / mu["mustar"]
    if lam == 0:
        B, V = mm_bias_var_ridgeless(psi1, psi2, zeta)
    else:
        B, V = mm_bias_var(psi1, psi2, lam / mu["mustar"] ** 2, zeta)
    return F1sq * B + (tau2 + Fstar2) * V + Fstar2, F1sq * B, (tau2 + Fstar2) * V


def sphere(m, d, rng):
    g = rng.standard_normal((m, d))
    return g * (np.sqrt(d) / np.linalg.norm(g, axis=1, keepdims=True))


def simulate_rf(d, n, N, lams, F1sq=1.0, tau2=0.0, rng=None, n_test=4000, act=lambda t: np.maximum(t, 0)):
    """One draw of RF regression with linear target y = <beta, x> + noise, ||beta||^2 = F1sq.
    Returns test error E_x (f*(x) - f(x; a_hat))^2 for each lam (paper normalization)."""
    rng = np.random.default_rng(rng)
    beta = rng.standard_normal(d); beta *= np.sqrt(F1sq) / np.linalg.norm(beta)
    X = sphere(n, d, rng); Th = sphere(N, d, rng); Xt = sphere(n_test, d, rng)
    y = X @ beta + np.sqrt(tau2) * rng.standard_normal(n)
    Z = act(X @ Th.T / np.sqrt(d)); Zt = act(Xt @ Th.T / np.sqrt(d))
    ft = Xt @ beta
    out = []
    if N > n:   # kernel form: a = Z^T (Z Z^T + n c I)^-1 y
        K = Z @ Z.T
        ev, U = np.linalg.eigh(K)
        uy = U.T @ y
        ZtU = Zt @ (Z.T @ U)
        for lam in np.atleast_1d(lams):
            c = n * N * lam / d
            if c == 0:
                keep = ev > ev.max() * 1e-12
                coef = np.where(keep, uy / np.where(keep, ev, 1), 0.0)
            else:
                coef = uy / (ev + c)
            out.append(np.mean((ZtU @ coef - ft) ** 2))
    else:
        G = Z.T @ Z
        ev, U = np.linalg.eigh(G)
        uzy = U.T @ (Z.T @ y)
        ZtU = Zt @ U
        for lam in np.atleast_1d(lams):
            c = n * N * lam / d
            if c == 0:
                keep = ev > ev.max() * 1e-12
                coef = np.where(keep, uzy / np.where(keep, ev, 1), 0.0)
            else:
                coef = uzy / (ev + c)
            out.append(np.mean((ZtU @ coef - ft) ** 2))
    return np.array(out)


def simulate_linear_parts(n, p, lams, r2=1.0, rng=None):
    """Isotropic linear model, one draw of (X, beta, unit noise). For every lam (0 = min-norm) returns
        ss = ||e_s||^2 (conditional bias, exact),  sn = <e_s, e_n>,  nn = ||e_n||^2,  tr = E||e_n||^2 = (1/n) sum shrink^2
    where beta_hat - beta = e_s + sigma * e_n for noise eps = sigma * unit. Realized risk at any sigma:
        ss + 2 sigma sn + sigma^2 nn;  conditional (noise-averaged) risk: ss + sigma^2 tr.
    lams may be a 1-D array, or a callable(gamma) -> array (e.g. the optimal ridge).  Also returns min nonzero eig."""
    rng = np.random.default_rng(rng)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p); beta *= np.sqrt(r2) / np.linalg.norm(beta)
    eps = rng.standard_normal(n)
    if p >= n:
        ev, U = np.linalg.eigh(X @ X.T / n)             # s^2 and left singular vectors
        ev = np.clip(ev, 0, None); s = np.sqrt(ev)
        bV = (U.T @ (X @ beta)) / np.sqrt(n) / s         # V^T beta
        ue = U.T @ eps / np.sqrt(n)
        null_b2 = r2 - np.sum(bV ** 2)                   # part of beta outside row space
    else:
        ev, V = np.linalg.eigh(X.T @ X / n)
        ev = np.clip(ev, 0, None); s = np.sqrt(ev)
        bV = V.T @ beta
        ue = (V.T @ (X.T @ eps)) / n / s                  # U^T eps / sqrt(n), U = X V / (sqrt(n) s)
        null_b2 = 0.0
    lams = np.atleast_1d(lams(p / n) if callable(lams) else lams)
    out = np.zeros((len(lams), 4))
    for i, lam in enumerate(lams):
        filt = ev / (ev + lam) if lam > 0 else np.ones_like(ev)
        shrink = s / (ev + lam)
        es = (filt - 1) * bV
        en = shrink * ue
        out[i] = [np.sum(es ** 2) + null_b2, np.sum(es * en), np.sum(en ** 2), np.sum(shrink ** 2) / n]
    return out, ev.min()
