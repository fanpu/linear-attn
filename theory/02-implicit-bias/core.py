"""Core math for the implicit-bias post. Everything float64, numpy/scipy/cvxpy only.

Conventions
-----------
Binary data: rows x_n of X (n x d), labels y_n in {-1,+1}.  Z = y[:, None] * X  (so margins are Z @ w).
Logistic loss is a SUM over samples:  L(w) = sum_n log(1 + exp(-z_n . w)).
"""
import numpy as np
import cvxpy as cp
from scipy.special import expit
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# ----------------------------------------------------------------------------- data
def gaussian_separable(n=40, d=50, seed=0):
    """Random Gaussian inputs with random labels (separable w.p. 1 when n < d)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)) / np.sqrt(d)
    y = rng.choice([-1.0, 1.0], size=n)
    return X, y


def hero_2d(seed=2, phi_deg=62.0, gamma=0.9, a=2.5, b=-0.5, npc=24):
    """2D separable dataset (separator through the origin).  Two clouds whose centroid direction sits ~30 deg
    from the max-margin direction u = (cos phi, sin phi), which is pinned by two 'shy' support vectors at
    margin gamma.  All other points have margin >= 1.3 gamma (checked), so the Soudry asymptotics kick in cleanly."""
    rng = np.random.default_rng(seed)
    phi = np.deg2rad(phi_deg)
    u = np.array([np.cos(phi), np.sin(phi)])
    up = np.array([-u[1], u[0]])
    pos, neg = [], []
    while len(pos) < npc:
        p = rng.normal([2.3, 2.0], [0.75, 0.6])
        if p @ u >= 1.45 * gamma:
            pos.append(p)
    while len(neg) < npc:
        p = rng.normal([-2.2, -1.9], [0.7, 0.7])
        if -p @ u >= 1.45 * gamma:
            neg.append(p)
    X = np.vstack([pos, [gamma * u + a * up], neg, [-(gamma * u - b * up)]])
    y = np.r_[np.ones(npc + 1), -np.ones(npc + 1)]
    return X, y


# ----------------------------------------------------------------------------- max-margin solvers
def svm(X, y, norm="l2", solver=None):
    """min ||w||_norm  s.t.  y_n x_n.w >= 1 (no bias).  Returns (w, dual multipliers alpha)."""
    Z = y[:, None] * X
    w = cp.Variable(X.shape[1])
    obj = {"l2": 0.5 * cp.sum_squares(w), "l1": cp.norm1(w), "linf": cp.norm_inf(w)}[norm]
    cons = [Z @ w >= 1]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=solver or "CLARABEL")
    return np.asarray(w.value), np.asarray(cons[0].dual_value)


def margin(Z, w, norm="l2"):
    """Normalized margin min_n z_n.w / ||w||."""
    nrm = {"l2": np.linalg.norm(w, axis=-1), "l1": np.abs(w).sum(-1), "linf": np.abs(w).max(-1)}[norm]
    return (w @ Z.T).min(-1) / nrm


def angle(a, b):
    """Angle (radians) between vectors along the last axis; broadcasts."""
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.arccos(np.clip(c, -1, 1))


def unit_gap(a, b):
    """|| a/|a| - b/|b| ||, the quantity Soudry et al. bound."""
    return np.linalg.norm(a / np.linalg.norm(a, axis=-1, keepdims=True) - b / np.linalg.norm(b, axis=-1, keepdims=True), axis=-1)


# ----------------------------------------------------------------------------- logistic regression dynamics
def logistic_grad(Z, w):
    """Gradient of the summed logistic loss; w may be batched (..., d)."""
    s = expit(-(w @ Z.T))  # (..., n)
    return -(s @ Z)


def log_checkpoints(T, per_decade=40):
    ts = np.unique(np.round(np.logspace(0, np.log10(T), int(np.log10(T) * per_decade) + 1)).astype(np.int64))
    return ts


def run_gd(Z, eta, T, w0=None, mode="gd", per_decade=40):
    """Discrete GD variants on the summed logistic loss, recording w at log-spaced steps.

    mode: 'gd'        w <- w - eta * g
          'ngd'       w <- w - eta * g/||g||          (normalized GD, constant step)
          'ngd_sqrt'  w <- w - eta/sqrt(t+1) * g/||g||
          'sign'      w <- w - eta * sign(g)          (normalized L_inf steepest descent)
    """
    d = Z.shape[1]
    w = np.zeros(d) if w0 is None else w0.astype(np.float64).copy()
    ts = log_checkpoints(T, per_decade)
    out = np.empty((len(ts), d))
    k = 0
    for t in range(1, T + 1):
        g = logistic_grad(Z, w)
        if mode == "gd":
            w -= eta * g
        elif mode == "ngd":
            w -= eta * g / np.linalg.norm(g)
        elif mode == "ngd_sqrt":
            w -= eta / np.sqrt(t) * g / np.linalg.norm(g)
        elif mode == "sign":
            w -= eta * np.sign(g)
        else:
            raise ValueError(mode)
        if t == ts[k]:
            out[k] = w
            k += 1
    return ts, out


def flow_logtime(Z, s_end, w0=None, s_eval=None, rtol=1e-11, atol=1e-12):
    """Gradient flow dw/dt = -grad L integrated in log-time s = log t (t in (1, e^s_end]).

    dw/ds = t * sum_n z_n sigma(-z_n.w) = sum_n z_n exp(s - log(1+exp(z_n.w))), computed in log-domain,
    so t = 1e100 is no problem.  Starts at t=1 from w0 (default: the t=1 point of a short flow from 0).
    """
    d = Z.shape[1]
    if w0 is None:
        # integrate t in [0,1] from w=0 in linear time first
        sol0 = solve_ivp(lambda t, w: -logistic_grad(Z, w), (0, 1), np.zeros(d), rtol=rtol, atol=atol, method="DOP853")
        w0 = sol0.y[:, -1]

    def rhs(s, w):
        m = Z @ w
        return np.exp(s - np.logaddexp(0.0, m)) @ Z

    sol = solve_ivp(rhs, (0.0, s_end), w0, t_eval=s_eval, rtol=rtol, atol=atol, method="DOP853")
    return sol.t, sol.y.T


def soudry_wtilde(X, y, w_hat, alpha, eta=1.0, tol=1e-6):
    """Soudry et al. asymptotics w(t) ~ w_hat log t + w_tilde.  For support vectors S (alpha_n > tol):
    eta * exp(-z_n . w_tilde) = alpha_n.  Returns the minimum-norm w_tilde in span(Z_S) satisfying these,
    plus the SV index set.  (Components outside span(Z_S) are fixed by transients and are not predicted.)"""
    Z = y[:, None] * X
    S = np.where(alpha > tol)[0]
    rhs = np.log(eta / alpha[S])
    wt = np.linalg.lstsq(Z[S], rhs, rcond=None)[0]
    return wt, S


# ----------------------------------------------------------------------------- diagonal linear networks
def q_fun(z):
    """Woodworth et al. potential q(z) = 2 - sqrt(4+z^2) + z asinh(z/2)."""
    return 2 - np.sqrt(4 + z ** 2) + z * np.arcsinh(z / 2)


def Q_alpha(w, alpha):
    return alpha ** 2 * q_fun(w / alpha ** 2).sum(-1)


def q_alpha_min(X, y, alpha, nu0=None):
    """argmin Q_alpha(w) s.t. Xw = y.  Stationarity: asinh(w/(2 alpha^2)) = X^T nu, i.e.
    w = 2 alpha^2 sinh(X^T nu).  Solve the smooth convex dual  max_nu  nu.y - 2 alpha^2 sum cosh(X^T nu)
    with Newton's method (in log-safe form)."""
    n, d = X.shape
    nu = np.zeros(n) if nu0 is None else nu0.copy()
    a2 = alpha ** 2

    def f(nu):  # negative dual (convex)
        v = X.T @ nu
        return 2 * a2 * np.cosh(v).sum() - nu @ y

    for it in range(500):
        v = X.T @ nu
        w = 2 * a2 * np.sinh(v)
        g = X @ w - y
        if np.linalg.norm(g) < 1e-12 * max(1, np.linalg.norm(y)):
            break
        H = (X * (2 * a2 * np.cosh(v))) @ X.T
        step = np.linalg.solve(H + 1e-300 * np.eye(n), g)
        # backtracking on the dual objective
        f0, lam = f(nu), 1.0
        while lam > 1e-12:
            nn = nu - lam * step
            fn = f(nn)
            if np.isfinite(fn) and fn <= f0 - 1e-4 * lam * (g @ step):
                break
            lam *= 0.5
        nu = nn
    return 2 * a2 * np.sinh(X.T @ nu), nu


def min_l2_interp(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def basis_pursuit(X, y):
    w = cp.Variable(X.shape[1])
    cp.Problem(cp.Minimize(cp.norm1(w)), [X @ w == y]).solve(solver="CLARABEL")
    return np.asarray(w.value)


def diag_net_gd(X, y, alphas, steps=200000, c=0.25, tol=1e-10, record=None, depth=2, fixed_eta=None):
    """GD on the squared loss L = 1/(2n)||X w - y||^2 with w = u^D - v^D, u = v = alpha*1 at init,
    vectorized over a vector of alphas.  Uses a curvature-adaptive step eta_t = c / (local curvature bound)  (a time reparameterization of the same gradient flow; the flow's path does not depend on it).
    Returns final w (A x d) and optionally recorded w snapshots at loss-log-spaced points."""
    n, d = X.shape
    A = len(alphas)
    al = np.asarray(alphas, float)[:, None]
    u = np.repeat(al, d, 1).copy()
    v = u.copy()
    lam = np.linalg.eigvalsh(X.T @ X / n)[-1]
    D = depth
    snaps = []
    done = np.zeros(A, bool)
    for t in range(steps):
        w = u ** D - v ** D
        r = w @ X.T - y  # A x n
        loss = 0.5 * (r ** 2).mean(1)
        done = loss < tol
        if done.all():
            break
        g = r @ X / n  # dL/dw, A x d
        big = np.maximum(u, v)
        curv = (D * D * lam * (big ** (2 * D - 2)).max(1, keepdims=True) * 2
                + D * (D - 1) * (big ** max(D - 2, 0) * np.abs(g)).max(1, keepdims=True))
        eta = c / curv if fixed_eta is None else np.full_like(curv, fixed_eta)
        eta[done] = 0
        gu = D * u ** (D - 1) * g
        gv = -D * v ** (D - 1) * g
        u = u - eta * gu
        v = v - eta * gv
        if record is not None and t % record == 0:
            snaps.append(w.copy())
    return u ** D - v ** D, (np.array(snaps) if record is not None else None), t


def geometry_2d(seed=0, npc=22):
    """2D dataset whose L2, Linf and L1 max-margin directions all differ (~63, 45, 90 degrees).
    One tight support vector z1 = (0.4, 0.8) makes a long edge of the feasible polygon; every other point has margin
    >= 1.35 at all three reference solutions."""
    rng = np.random.default_rng(seed)
    refs = np.array([[0.5, 1.0], [1 / 1.2, 1 / 1.2], [0.0, 1.25]])
    Zs = [np.array([0.4, 0.8])]
    ys = [1.0]
    centers = {1.0: [1.35, 1.55], -1.0: [0.95, 1.85]}
    for lab in (1.0, -1.0):
        cnt = 0
        while cnt < npc:
            z = rng.normal(centers[lab], [0.45, 0.4])
            if (refs @ z).min() >= 1.35:
                Zs.append(z); ys.append(lab); cnt += 1
    Z = np.array(Zs)
    y = np.array(ys)
    return Z * y[:, None], y
