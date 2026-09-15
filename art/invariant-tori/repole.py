"""M2 ruling 1 (controller, 2026-09-15): the first pole sat inside the corner island at 11.85 deg from
regular orbit kam 38 (chart scale up to 47x).  Re-choose the pole without kam 38, then replace kam 38
by another eps = 0.5 island orbit whose minimum angle to the new pole is >= 20 deg.

  1. pole over every stored orbit except kam 38            -> cache/pole_no38.json
  2. candidates: game-chaos kam_eps0.50 orbits with lambda <= 2.5e-3 (T = 4e4) and filled section area in
     (0.003, 0.05) (same filter as the first 8), not already chosen; integrate T = 2e4 like the others
  3. keep candidates >= 20 deg from the new pole; pick the one farthest (1 - IoU of filled section masks)
     from the 7 kept orbits.
The first pole is kept as cache/pole_first.json for the second-pole plate.
python repole.py   (then set REGULAR_KAM05 in compute_orbits.py and rerun eps05 + compute_chart.py all)
"""
import json
import os

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_dilation

import chart as C
import compute_chart as CC
import compute_orbits as CO
import replicator_c as rc

CACHE = CC.CACHE


def masks(sec, ns, idx, R=160):
    out = {}
    for o in idx:
        s = sec[o, :ns[o]]
        H, _, _ = np.histogram2d(s[:, 0], s[:, 4], bins=R, range=[[0, .8], [0, .8]])
        out[o] = binary_fill_holes(binary_dilation(H > 0, iterations=1))
    return out


if __name__ == '__main__':
    if not os.path.exists(os.path.join(CACHE, 'pole_no38.json')):
        CC.do_pole(exclude_kam=[38], out_name='pole_no38.json')
    P = json.load(open(os.path.join(CACHE, 'pole_no38.json'))); p = np.array(P['pole'])
    k = np.load(CO.GC_KAM05); sec, ns, lam = k['sec'], k['nsec'], k['lyap']
    reg = np.where(lam <= 2.5e-3)[0]
    M = masks(sec, ns, reg)
    kept = [o for o in CO.REGULAR_KAM05 if o != 38]
    cand = [o for o in reg if 0.003 < M[o].mean() < 0.05 and o not in CO.REGULAR_KAM05]
    r = rc.integrate(rc.probs_to_logits(k['x0'][cand], k['y0'][cand]), 0.5, -0.5, h=0.01, T=20000.0,
                     every=100, traj_every=10, ntraj=200000)
    q = C.to_sphere(C.logits_to_u(r['traj']))
    ang = np.degrees(np.arccos(np.clip(q @ p, -1, 1))).min(1)
    ok = [i for i in range(len(cand)) if ang[i] >= 20.0 and r['lyap'][i] <= 5e-3]
    def iou_dist(a, b):
        return 1 - (M[a] & M[b]).sum() / (M[a] | M[b]).sum()
    score = [min(iou_dist(cand[i], o) for o in kept) for i in ok]
    order = np.argsort(score)[::-1]
    for j in order[:8]:
        i = ok[j]
        print(f'kam {cand[i]}: min angle {ang[i]:.2f} deg, lambda {r["lyap"][i]:.2e}, area {M[cand[i]].mean():.4f}, '
              f'IoU-distance to kept {score[j]:.3f}')
    best = cand[ok[order[0]]]
    print('new pole min angle', P['min_angle_deg'], 'scale', P['chart_scale_min'], P['chart_scale_max'])
    print('choice', best)
    json.dump(dict(replacement=int(best), candidates=len(cand), eligible=len(ok)),
              open(os.path.join(CACHE, 'repole_choice.json'), 'w'))
