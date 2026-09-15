"""Exact meridional slice of the 12 eps = 0 tori (companion to the plaster/glow renders).

Plane: through c containing the doughnut axis a and the wedge direction b (render_lib.tori_frame), i.e. normal
n = a x b.  The 12 orbits are re-integrated at every RK4 step (dt = 0.01) to T = 1e5 (same starts, same steps as
orbits_eps0.npz) and every sign change of (X - c).n is located by linear interpolation between consecutive steps.
Output cache/slice_eps0.npz: in-plane coordinates (u = (X-c).b, v = (X-c).a) and the torus index.
python compute_slice.py
"""
import json
import os
import time

import numpy as np

import chart as C
import compute_orbits as CO
import render_lib as L

if __name__ == '__main__':
    t0 = time.time()
    c, a, b = L.tori_frame()
    n = np.cross(a, b)
    ch = C.Chart(np.array(json.load(open(os.path.join(L.CACHE, 'pole.json')))['pole']), 2.8)
    s0 = np.load(os.path.join(L.CACHE, 'orbits_eps0.npz'))['s0']
    U, V, I, chord = [], [], [], 0.0
    for k in range(len(s0)):
        r = CO.run(s0[k:k + 1], 0.0, T=100000.0, dt=0.01, maxsec=10)
        X = ch.forward_logits(r['L'][0]) - c
        chord = max(chord, float(np.linalg.norm(np.diff(X, axis=0), axis=1).max()))
        d = X @ n
        j = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
        w = d[j] / (d[j] - d[j + 1])
        P = X[j] + w[:, None] * (X[j + 1] - X[j])
        U.append(P @ b); V.append(P @ a); I.append(np.full(len(j), k))
    U, V, I = map(np.concatenate, (U, V, I))
    np.savez(os.path.join(L.CACHE, 'slice_eps0.npz'), u=U, v=V, torus=I, c=c, a=a, b=b, max_chord=chord, T=100000.0)
    print(f'slice: {len(U)} crossings, max step chord {chord:.4f}, {time.time() - t0:.0f}s', flush=True)
