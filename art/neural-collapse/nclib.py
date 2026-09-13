"""Shared analysis helpers for the neural-collapse renders (CPU only, reads cache/)."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_run(tag):
    """Return meta, metrics (list of dicts sorted by epoch), and ckpt list [(epoch, path)]."""
    d = os.path.join(HERE, "cache", tag)
    meta = json.load(open(os.path.join(d, "meta.json")))
    mets = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    mets.sort(key=lambda m: m["epoch"])
    C = len(meta["classes"])
    ntr_full = 5000 * C
    ck = []
    for p in glob.glob(os.path.join(d, "ep*.npz")):
        ck.append((float(int(os.path.basename(p)[2:6])), p))
    for p in glob.glob(os.path.join(d, "it*.npz")):
        it = int(os.path.basename(p)[2:6])
        ck.append((it * meta["bs"] / ntr_full, p))
    ck.sort()
    return meta, mets, ck


def fourier_etf(C):
    """Orthonormal discrete-Fourier basis of 1-perp, shape C x (C-1), and the plane list.

    Row j of the returned matrix is the ideal simplex-ETF vertex j (norm sqrt((C-1)/C))
    written in Fourier coordinates. planes[k] = column indices of frequency k.
    """
    j = np.arange(C)
    cols, planes = [], {}
    for k in range(1, (C - 1) // 2 + 1):
        planes[k] = [len(cols), len(cols) + 1]
        cols += [np.sqrt(2 / C) * np.cos(2 * np.pi * k * j / C), np.sqrt(2 / C) * np.sin(2 * np.pi * k * j / C)]
    if C % 2 == 0:
        planes[C // 2] = [len(cols)]
        cols.append(np.cos(np.pi * j) / np.sqrt(C))
    F = np.stack(cols, 1)
    return F, planes


class Aligner:
    """Isometric map from feature space onto the ideal-ETF Fourier frame.

    Centred train class means M (C x d) -> orthonormal basis U of their span (d x (C-1)),
    then orthogonal Procrustes R so that (M U R)/s best matches the ideal vertices F.
    Coordinates of a feature h are ((h - muG) U R) / s  -- a rotation + one global scale
    (no shear), so non-ETF shapes stay non-ETF.
    """

    def __init__(self, mu, muG, allow_reflection=True, ref=None):
        C = mu.shape[0]
        self.C = C
        F, self.planes = fourier_etf(C)
        M = (mu - muG).astype(np.float64)
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        U = Vt[: C - 1].T
        A = M @ U
        target = F if ref is None else ref
        P, _, Qt = np.linalg.svd(A.T @ target)
        R = P @ Qt
        if not allow_reflection and np.linalg.det(R) < 0:
            P[:, -1] *= -1
            R = P @ Qt
        self.proj = U @ R
        self.muG = muG.astype(np.float64)
        self.scale = np.linalg.norm(A) / np.linalg.norm(F)
        self.coords_means = (M @ self.proj) / self.scale
        self.F = F
        self.sv = S
        # relative Procrustes residual: 0 for a perfect (scaled, rotated) simplex ETF
        self.residual = np.linalg.norm(self.coords_means - F) / np.linalg.norm(F)

    def __call__(self, H):
        return ((H.astype(np.float64) - self.muG) @ self.proj) / self.scale


def star_order(C, k):
    """Vertex order that draws the {C/k} star polygon edges j -> j+1 in class-index order."""
    return np.arange(C + 1) % C


def ckpt_arrays(path, keys=None):
    z = np.load(path)
    return {k: z[k] for k in (keys or z.files)}


def labels(meta):
    return np.array(meta["y_sub_train"]), np.array(meta["y_sub_test"])


def pick_ckpts(ck, epochs):
    """Nearest available checkpoint to each requested epoch."""
    E = np.array([e for e, _ in ck])
    return [ck[int(np.argmin(np.abs(E - e)))] for e in epochs]
