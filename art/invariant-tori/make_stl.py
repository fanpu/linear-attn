"""STL: one regular eps = 0.5 orbit as a tube (r3d.tube_mesh), declared "a torus woven from its own orbit".

Rule (declared): orbit kam K from the regular set, dt = 0.01 samples, time window [0, T]; the model is scaled so its
longest side is 80 mm and the tube radius is 0.6 mm.  T is the longest window (step 10) for which no two tube
pieces farther apart than pi * 2.2 r along the curve come within 2.2 r of each other, so the tube never intersects
itself and the mesh is a single closed, non-self-intersecting surface.
python make_stl.py [kam] [--preview]
"""
import argparse
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

import render_lib as L
from render_lib import r3d

RMM, SIZE_MM = 0.6, 80.0


def self_avoiding(P, r):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    pairs = cKDTree(P).query_pairs(2.2 * r, output_type='ndarray')
    if len(pairs) == 0:
        return True, np.inf
    far = pairs[np.abs(s[pairs[:, 0]] - s[pairs[:, 1]]) > np.pi * 2.2 * r]
    return len(far) == 0, (np.linalg.norm(P[far[:, 0]] - P[far[:, 1]], axis=1).min() if len(far) else np.inf)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('kam', type=int, nargs='?', default=201)
    ap.add_argument('--preview', action='store_true'); a = ap.parse_args()
    d = np.load(os.path.join(L.CACHE, 'orbits_eps05_fine.npz')); i = list(d['kam_index']).index(a.kam)
    X = np.load(os.path.join(L.CACHE, 'stereo_eps05_fine.npz'))['X'][i]
    best = None
    for T in range(20, 1001, 10):
        P = X[:T * 100 + 1:2]
        ext = np.ptp(P, 0).max(); r = RMM / SIZE_MM * ext
        ok, _ = self_avoiding(P, r)
        if not ok:
            break
        best = (T, P, r, ext)
    T, P, r, ext = best
    Pr, _ = r3d.sample_polyline(torch.tensor(P, dtype=torch.float64), 0.5 * r)
    Pr = Pr.numpy()
    verts, faces = r3d.tube_mesh(Pr, r, n_sides=16, cap=True)
    vmm = r3d.scale_to_mm(verts, SIZE_MM)
    wt = r3d.is_watertight(faces); vol = r3d.mesh_volume(vmm, faces)
    out = os.path.join(L.GAL, f'torus_woven_kam{a.kam}.stl')
    r3d.write_stl(out, vmm, faces, header=f'invariant-tori kam{a.kam} eps0.5 H2.8 t<={T}')
    print(f'kam {a.kam}: T={T}, chart extent {ext:.3f}, r={r:.4f} chart = {RMM} mm, length {np.linalg.norm(np.diff(P, axis=0), axis=1).sum():.1f}'
          f' chart, {len(faces)} faces, watertight {wt}, volume {vol:.1f} mm^3, size {np.ptp(vmm, 0).round(1)} mm, '
          f'{os.path.getsize(out) / 1e6:.1f} MB')
    if a.preview:
        S = 1200
        c = 0.5 * (P.min(0) + P.max(0)); w, v = np.linalg.eigh(np.cov((P - P.mean(0)).T))
        e = 0.8 * v[:, 0] + 0.6 * v[:, 1]; e /= np.linalg.norm(e)
        cam = r3d.Camera(tuple(c + 40 * e), tuple(c), up=tuple(v[:, 2]), width=S, height=S, ortho_height=1.1 * ext)
        pts = torch.tensor(Pr, dtype=torch.float32)
        hit = r3d.splat_spheres(pts, r, cam); m = hit['mask']
        lum = 0.35 + 0.65 * r3d.lambert(hit['normal'][m], tuple(0.6 * v[:, 2] + 0.5 * e + 0.4 * v[:, 1]), ambient=0.0)
        img = torch.ones(S, S, 3) * torch.tensor(L.PAPER * 0.85, dtype=torch.float32)
        img[m] = torch.tensor([0.93, 0.91, 0.87]) * lum[:, None]
        r3d.save_png(os.path.join(L.GAL, f'torus_woven_kam{a.kam}_preview.png'), img)
