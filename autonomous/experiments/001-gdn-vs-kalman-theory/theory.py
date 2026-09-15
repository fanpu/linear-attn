"""Closed-form steady-state risk of the gated delta rule (NLMS form) on drifting regression, isotropic inputs.

Derivation (README section 3, P1). Let e_t = u_t - w_t be the prediction error of the decayed estimate.
    e_{t+1} = alpha (I - beta k k^T) e_t + (alpha - a) w_t + alpha beta sigma eps x/||x||^2 - sqrt(q/d) xi
with k = x/||x|| uniform on the sphere and independent of ||x||, e_t, w_t. Taking second moments, using
E[(k.e)^2] = ||e||^2/d and E[1/||x||^2] = 1/(d-2), gives a linear recursion in m = E||e||^2 and c = E[w.e]:
    c' = a alpha (1 - beta/d) c + a (alpha - a) - q
    m' = alpha^2 (1 - (2 beta - beta^2)/d) m + (alpha - a)^2 + 2 alpha (alpha - a)(1 - beta/d) c
         + alpha^2 beta^2 sigma^2/(d-2) + q
Its fixed point is the steady-state excess risk m (total risk m + sigma^2).
"""
import numpy as np


def gdn_excess_risk(alpha, beta, q, sigma2, d):
    a = np.sqrt(1 - q)
    c = (a * (alpha - a) - q) / (1 - a * alpha * (1 - beta / d))
    num = (alpha - a) ** 2 + 2 * alpha * (alpha - a) * (1 - beta / d) * c + alpha**2 * beta**2 * sigma2 / (d - 2) + q
    den = 1 - alpha**2 * (1 - (2 * beta - beta**2) / d)
    return num / den


def gdn_recursion(alpha, beta, q, sigma2, d, T, m0=1.0, c0=-1.0):
    """The same recursion iterated from u_0 = 0 (so e_0 = -w_0: m0 = 1, c0 = -1). Returns m_t for t < T."""
    a = np.sqrt(1 - q)
    m, c, out = m0, c0, np.empty(T)
    for t in range(T):
        out[t] = m
        m, c = (alpha**2 * (1 - (2 * beta - beta**2) / d) * m + (alpha - a) ** 2
                + 2 * alpha * (alpha - a) * (1 - beta / d) * c + alpha**2 * beta**2 * sigma2 / (d - 2) + q,
                a * alpha * (1 - beta / d) * c + a * (alpha - a) - q)
    return out


def optimal_gdn(q, sigma2, d, n=400):
    """Minimize the closed form over alpha in (0, 1], beta in (0, 2): coarse grid, then two zoomed refinements."""
    lo_a, hi_a, lo_b, hi_b = 0.0, 1.0, 1e-4, 2.0 - 1e-4
    for _ in range(3):
        A, B = np.meshgrid(np.linspace(max(lo_a, 1e-6), hi_a, n), np.linspace(lo_b, hi_b, n), indexing="ij")
        with np.errstate(divide="ignore", invalid="ignore"):
            R = gdn_excess_risk(A, B, q, sigma2, d)
        R = np.where(np.isfinite(R) & (R > 0), R, np.inf)
        i, j = np.unravel_index(np.argmin(R), R.shape)
        da, db = (hi_a - lo_a) * 4 / n, (hi_b - lo_b) * 4 / n
        best = (A[i, j], B[i, j], R[i, j])
        lo_a, hi_a = max(A[i, j] - da, 1e-6), min(A[i, j] + da, 1.0)
        lo_b, hi_b = max(B[i, j] - db, 1e-4), min(B[i, j] + db, 2.0 - 1e-4)
    return best
