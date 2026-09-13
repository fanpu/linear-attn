"""Is the deep-linear-network chaotic band the scalar band?  Transverse Lyapunov exponent test.

The scalar reduction (aligned singular vectors, balanced layers) puts mode 1 of the DLN on the map
u <- u - eta sigma_1^(4/3) (u^3 - 1) u^2.  We (i) run the full DLN from a state placed exactly on that
reduced attractor (and with a 1e-8 perturbation) and record survival, and (ii) propagate a tangent vector of
the full DLN GD map along the EXACT reduced orbit (the state is re-synthesised from u every step, so
round-off cannot leave the invariant set) restricted to the off-diagonal subspace (in the frame
where reduced states are diagonal); that subspace is invariant under the linearisation by the reflection
symmetries, so this is the transverse exponent for singular-vector rotations/mode mixing.  lambda_full > lambda_reduced means some direction outside the reduction (not the oscillating mode)
grows: the reduced attractor is transversally unstable in the matrix parameter space.
Output: cache/dln_transverse.npz
"""
import numpy as np
import torch
from common import dln_target, dln_grad, DT

torch.set_num_threads(4)
CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"


def synth(u, s, U, V):
    N = len(u)
    d = len(s)
    rho = s.clamp_min(0).pow(1 / 3).expand(N, d).clone()
    rho[:, 0] = float(s[0]) ** (1 / 3) * u
    D = torch.diag_embed(rho)
    return torch.stack([D @ V.T, D, U @ D], 1)


def main():
    M = dln_target()
    U, S, Vh = torch.linalg.svd(M)
    V = Vh.T
    etas = torch.tensor(np.linspace(0.030, 0.0668, 185), dtype=DT)
    N = len(etas)
    et = etas * float(S[0]) ** (4 / 3)
    u = torch.full((N,), 1.05, dtype=DT)
    for _ in range(20000):
        u = u - et * (u ** 3 - 1) * u ** 2
    g = torch.Generator().manual_seed(0)
    offd = 1 - torch.eye(5, dtype=DT)

    def to_rot(X):   # rotated frame where the reduced states are diagonal
        return torch.stack([X[:, 0] @ V, X[:, 1], U.T @ X[:, 2]], 1)

    def from_rot(X):
        return torch.stack([X[:, 0] @ V.T, X[:, 1], U @ X[:, 2]], 1)
    T = from_rot(torch.randn(N, 3, 5, 5, generator=g, dtype=DT) * offd)
    T /= T.flatten(1).norm(dim=1)[:, None, None, None]
    lam_full = torch.zeros(N, dtype=DT)
    lam_red = torch.zeros(N, dtype=DT)
    e4 = etas[:, None, None, None]
    gfun = lambda Wx: dln_grad(Wx, M)[0]
    steps = 4000
    for t in range(steps + 200):
        W = synth(u, S, U, V)
        _, jv = torch.func.jvp(gfun, (W,), (T,))
        T = T - e4 * jv
        T = from_rot(to_rot(T) * offd)   # stay in the off-diagonal (singular-vector-rotating) subspace
        nv = T.flatten(1).norm(dim=1)
        if t >= 200:
            lam_full += torch.log(nv)
            lam_red += torch.log((1 - et * (5 * u ** 4 - 2 * u)).abs())
        T = T / nv[:, None, None, None]
        u = u - et * (u ** 3 - 1) * u ** 2
    lam_full /= steps
    lam_red /= steps
    # survival of the full DLN started on the reduced attractor
    W = synth(u, S, U, V) + 1e-8 * torch.randn(N, 3, 5, 5, generator=g, dtype=DT)
    alive = torch.ones(N, dtype=torch.bool)
    for _ in range(20000):
        W = W - e4 * dln_grad(W, M)[0]
        bad = ~torch.isfinite(W).flatten(1).all(1) | (W.flatten(1).abs().amax(1) > 1e4)
        alive &= ~bad
        W[bad] = 0
    np.savez(f"{CACHE}/dln_transverse.npz", etas=etas.numpy(), eta_tilde=et.numpy(), lam_full=lam_full.numpy(),
             lam_reduced=lam_red.numpy(), alive=alive.numpy())
    for i in range(0, N, 8):
        print(f"eta={etas[i]:.4f} (scalar {et[i]:.3f})  lambda_reduced={lam_red[i]:+.3f}  lambda_transverse(offdiag)={lam_full[i]:+.3f}  survives={bool(alive[i])}")


if __name__ == "__main__":
    main()
