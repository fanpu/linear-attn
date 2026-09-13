"""Problem definitions: model, 2D slice of initialisation space, and outcome classifiers.

Each problem returns (prob, center, u, v) and classify(theta_final, status) -> dict of int arrays:
  raw    identity of the solution reached, in the network's own coordinates
  canon  identity after quotienting the symmetry group (permutations / signs)
Negative codes: -1 not converged by T, -2 diverged.
"""
import numpy as np
import torch
from engine import MLP, ScalarFact, DEV, DT

NC, DIV = -1, -2


def _fill_status(code, status):
    code = code.astype(np.int32)
    code[status == 1] = NC
    code[status == 2] = DIV
    return code


# ------------------------------------------------------------------ depth-3 scalar factorisation
# f(x,y,z) = 1/4 (xyz - 1)^2 ; zero set has 4 components, labelled by sign pattern.
# Slice: the plane x + y + z = sqrt(3) s, with in-plane axes u, v (so S3 acts as the dihedral D3).
FACT3_U = np.array([1., -1., 0.]) / np.sqrt(2)
FACT3_V = np.array([1., 1., -2.]) / np.sqrt(6)
FACT3_N = np.ones(3) / np.sqrt(3)
FACT3_NAMES = {0: '+++', 1: '+--', 2: '-+-', 3: '--+'}


def fact3(s=1.0):
    return ScalarFact(3), s * np.sqrt(3) * FACT3_N, FACT3_U, FACT3_V


def fact3_classify(theta, status):
    sg = theta > 0
    # which coordinate is the lone positive one (or all positive)
    raw = np.where(sg.all(1), 0, np.where(sg[:, 0], 1, np.where(sg[:, 1], 2, 3)))
    canon = np.where(raw == 0, 0, 1)                 # modulo permutations of (x,y,z)
    full = np.zeros_like(raw)                        # modulo permutations and paired sign flips
    return dict(raw=_fill_status(raw, status), canon=_fill_status(canon, status),
                full=_fill_status(full, status))


# ------------------------------------------------------------------ XOR 2-2-1 tanh network
XOR_X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], float)
XOR_T = np.array([-1, 1, 1, -1], float)


def xor(seed=0, H=2, scale=1.0):
    prob = MLP(XOR_X, XOR_T, H)
    rng = np.random.default_rng(seed)
    c0 = rng.normal(size=prob.D) * scale
    u = rng.normal(size=prob.D); u /= np.linalg.norm(u)
    v = rng.normal(size=prob.D); v -= u * (u @ v); v /= np.linalg.norm(v)
    return prob, c0, u, v


def unit_codes(prob, theta, chunk=1 << 20):
    """4-bit code per hidden unit: which of the 4 XOR inputs put the unit on its + side."""
    out = []
    for s in range(0, theta.shape[0], chunk):
        th = torch.as_tensor(theta[s:s + chunk], dtype=DT, device=DEV)
        z = prob.pre(th)                              # (P,N,H)
        w = torch.tensor([1, 2, 4, 8], device=DEV)
        out.append(((z > 0).long() * w[None, :, None]).sum(1).cpu().numpy())
    return np.concatenate(out)                         # (P,H) in 0..15


def xor_classify(prob, theta, status):
    H = prob.H
    codes = unit_codes(prob, theta)
    base = 16
    raw = np.zeros(len(theta), np.int64)
    for h in range(H):
        raw = raw * base + codes[:, h]
    # sign flip of a tanh unit (w,b,a) -> (-w,-b,-a) leaves the network function unchanged and
    # maps code c -> 15 - c ; permutation reorders units. canonical = sorted sign-normalised codes.
    cn = np.sort(np.minimum(codes, 15 - codes), 1)
    canon = np.zeros(len(theta), np.int64)
    for h in range(H):
        canon = canon * base + cn[:, h]
    return dict(raw=_fill_status(raw, status), canon=_fill_status(canon, status))


PROBLEMS = {'fact3': fact3, 'xor': xor}
