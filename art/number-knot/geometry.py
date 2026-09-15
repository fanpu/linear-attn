"""M2 geometry (CPU, numpy): every 3-D bead position for the Number Knot renders, measured and null.
Reads cache/hs + fits.py; writes cache/geom_M2.npz and cache/geom_M2.json. Renders read only these.

Charts (all declared, see README captions):
- Helix: orthonormal frame from the K&T fit at (template 1, layer 1, 0-999, T=100).
  e1 = fitted cos direction, e2 = fitted sin direction orthogonalised against e1, e3 = fitted linear
  direction orthogonalised against both. Bead = orthogonal projection of the measured (PCA-100) state
  onto (e1, e2, e3), in residual-stream units. Null: identical pipeline with shuffled labels.
- Knot: theta, rho = polar coords of a bead in its T=100 plane; phi, r = polar coords in its T=10 plane
  (same cell, same construction). Declared torus composition:
  ((rho + s r cos phi) cos theta, (rho + s r cos phi) sin theta, s r sin phi), s = MINOR_SCALE (declared).
- Towers: per layer, the M1 mean-difference plane (days, months) or the T=100 fitted plane (numbers), rotated
  (orthogonal Procrustes, reflection allowed) so the class means best match the canonical calendar angles
  2πk/K (numbers: the fitted frame already fixes the phase), divided by the layer's RMS bead radius
  (declared per-layer normalisation: the residual norm grows with depth), height = layer index (declared).
"""
from __future__ import annotations

import json

import numpy as np

import common as C
import fits as F

TEMPLATE, LAYER = 1, 1
SEED = 7
MINOR_SCALE = 0.35  # declared: measured minor radii (median 1.04) ~ major (1.28) would self-intersect


def frame(Y, a, T):
    """Orthonormal (e1 cos, e2 sin, e3 linear) frame in PCA space from the K&T fit; returns frame (3, k), fit R²."""
    B = F.basis(a, T)
    coef, *_ = np.linalg.lstsq(B, Y - Y.mean(0), rcond=None)       # (4, k): const, lin, cos, sin
    c_lin, c_cos, c_sin = coef[1], coef[2], coef[3]
    e1 = c_cos / np.linalg.norm(c_cos)
    e2 = c_sin - (c_sin @ e1) * e1; e2 /= np.linalg.norm(e2)
    e3 = c_lin - (c_lin @ e1) * e1 - (c_lin @ e2) * e2; e3 /= np.linalg.norm(e3)
    return np.stack([e1, e2, e3]), float(np.degrees(np.arccos(abs(c_cos @ c_sin) / np.linalg.norm(c_cos) / np.linalg.norm(c_sin))))


def project(X, a, T):
    """Measured beads in the fit frame, plus the fitted curve C B(a') on a fine grid a' (declared 'fit' line)."""
    Y, _, _, ev = F.pca(X, 100)
    E, ang = frame(Y, a, T)
    P = (Y - Y.mean(0)) @ E.T                                        # (N, 3)
    B = F.basis(a, T)
    coef, *_ = np.linalg.lstsq(B, Y - Y.mean(0), rcond=None)
    af = np.linspace(0, len(a) - 1, 20 * len(a))
    wf = 2 * np.pi * af / T
    Bf = np.stack([np.ones_like(af), af / max(np.max(a), 1.0), np.cos(wf), np.sin(wf)], 1)
    project.fit = (Bf @ coef) @ E.T                                  # (20N, 3) fitted curve in the same frame
    return P, dict(pca_var=float(ev.sum()), cos_sin_angle_deg=ang,
                   dR2=F.r2_fit(Y, F.basis(a, T)) - F.r2_fit(Y, F.basis(a, None)))


def procrustes_to_circle(Pm, K):
    """Orthogonal 2x2 Q (reflection allowed) minimising ||Pm Q - canon * s||."""
    canon = np.stack([np.cos(2 * np.pi * np.arange(K) / K), np.sin(2 * np.pi * np.arange(K) / K)], 1)
    U, _, Vt = np.linalg.svd(Pm.T @ canon)
    return U @ Vt


def main():
    rng = np.random.default_rng(SEED)
    out, meta = {}, {}
    hs = C.CACHE / "hs"
    X = np.load(hs / f"olmo_numbers_t{TEMPLATE}.npy").astype(np.float64)   # (1000, 17, 2048)
    N, L1, _ = X.shape
    a = np.arange(N)
    perm = rng.permutation(N)
    a_sh = np.empty(N); a_sh[perm] = a
    labels = {"measured": a, "null": a_sh}

    # helix + knot at the chosen cell
    for kind, lab in labels.items():
        P100, m100 = project(X[:, LAYER], lab, 100)
        out[f"helixfit_{kind}"] = project.fit
        P10, m10 = project(X[:, LAYER], lab, 10)
        F10 = project.fit
        out[f"helix_{kind}"] = P100
        out[f"plane10_{kind}"] = P10
        meta[f"helix_{kind}"] = m100; meta[f"plane10_{kind}"] = m10
        rho, th = np.hypot(P100[:, 0], P100[:, 1]), np.arctan2(P100[:, 1], P100[:, 0])
        r, ph = np.hypot(P10[:, 0], P10[:, 1]), np.arctan2(P10[:, 1], P10[:, 0])
        s = MINOR_SCALE
        out[f"knot_{kind}"] = np.stack([(rho + s * r * np.cos(ph)) * np.cos(th),
                                        (rho + s * r * np.cos(ph)) * np.sin(th), s * r * np.sin(ph)], 1)
        Fh = out[f"helixfit_{kind}"]
        rf, tf = np.hypot(Fh[:, 0], Fh[:, 1]), np.arctan2(Fh[:, 1], Fh[:, 0])
        r1, p1 = np.hypot(F10[:, 0], F10[:, 1]), np.arctan2(F10[:, 1], F10[:, 0])
        out[f"knotfit_{kind}"] = np.stack([(rf + s * r1 * np.cos(p1)) * np.cos(tf),
                                           (rf + s * r1 * np.cos(p1)) * np.sin(tf), s * r1 * np.sin(p1)], 1)
        meta[f"knot_{kind}"] = dict(rho_median=float(np.median(rho)), r_median=float(np.median(r)),
                                    minor_scale=s, z_lin_range=float(np.ptp(P100[:, 2])))
        # angular error of the measured angles vs the label angles (in-sample)
        for T, ang in ((100, th), (10, ph)):
            err = np.angle(np.exp(1j * (ang - 2 * np.pi * lab / T)))
            meta[f"knot_{kind}"][f"median_abs_angle_err_T{T}_deg"] = float(np.degrees(np.median(np.abs(err))))

    # helix layer sweep (film): per layer, beads + fitted curve in that layer's fit frame; both panels divided by the
    # measured layer's RMS in-plane bead radius (declared; the residual norm grows with depth)
    sweep_scale = []
    for l in range(L1):
        Pm, _ = project(X[:, l], a, 100); Fm = project.fit
        Pn, _ = project(X[:, l], a_sh, 100); Fn = project.fit
        cm, cn = Pm.mean(0), Pn.mean(0)
        sc = np.sqrt((Pm[:, :2] ** 2).sum(1).mean())
        sweep_scale.append(float(sc))
        for kind, P, Fc, c in (("measured", Pm, Fm, cm), ("null", Pn, Fn, cn)):
            out.setdefault(f"sweep_{kind}", np.zeros((L1, N, 3), np.float32))[l] = (P - c) / sc
            out.setdefault(f"sweepfit_{kind}", np.zeros((L1, len(Fc), 3), np.float32))[l] = (Fc - c) / sc
    meta["sweep_scale"] = sweep_scale

    # numbers tower: T=100 plane at every layer, template 1
    for kind, lab in labels.items():
        tw = np.zeros((L1, N, 2)); info = []
        for l in range(L1):
            P, m = project(X[:, l], lab, 100)
            scale = np.sqrt((P[:, :2] ** 2).sum(1).mean())
            tw[l] = P[:, :2] / scale
            info.append(dict(layer=l, rms=float(scale), **m))
        out[f"tower_numbers_{kind}"] = tw
        meta[f"tower_numbers_{kind}"] = info

    # calendar towers (Qwen3)
    for name, T, W in [("days", C.DAY_TEMPLATES, C.DAYS), ("months", C.MONTH_TEMPLATES, C.MONTHS)]:
        arr = np.stack([np.load(hs / f"qwen_{name}_t{k}.npy") for k in range(len(T))]).astype(np.float64)
        nt, K, L1q, D = arr.shape
        lab = np.tile(np.arange(K), nt); tmpl = np.repeat(np.arange(nt), K)
        H = arr.reshape(nt * K, L1q, D)
        pperm = rng.permutation(nt * K)
        for kind, lb in (("measured", lab), ("null", lab[pperm])):
            tw = np.zeros((L1q, nt * K, 2)); means = np.zeros((L1q, K, 2)); info = []
            for l in range(L1q):
                s = F.calendar_scores(H[:, l], lb, tmpl, K)
                Pm = np.stack([s["P"][lb == k].mean(0) for k in range(K)])
                c = Pm.mean(0)
                Q = procrustes_to_circle(Pm - c, K)
                P = (s["P"] - c) @ Q
                scale = np.sqrt((P ** 2).sum(1).mean())
                tw[l] = P / scale
                means[l] = ((Pm - c) @ Q) / scale
                info.append(dict(layer=l, r2_in=s["r2_in"], r2_ho=s["r2_ho"], cyclic=s["cyclic"], rms=float(scale),
                                 plane_var=s["plane_var"]))
            out[f"tower_{name}_{kind}"] = tw
            out[f"tower_{name}_{kind}_means"] = means       # means of the labels used by the pipeline
            meta[f"tower_{name}_{kind}"] = info
        out[f"tower_{name}_labels"] = lab                  # true labels (colour)
        out[f"tower_{name}_null_perm"] = pperm
    out["a"] = a; out["a_null"] = a_sh
    np.savez_compressed(C.CACHE / "geom_M2.npz", **out)
    (C.CACHE / "geom_M2.json").write_text(json.dumps(dict(template=TEMPLATE, layer=LAYER, seed=SEED, **meta), indent=1))
    h = meta
    print(json.dumps({k: h[k] for k in h if k.startswith(("helix", "plane10", "knot"))}, indent=1))
    for k in ("tower_numbers_measured", "tower_days_measured", "tower_months_measured"):
        print(k, [round(d["rms"], 1) for d in h[k]])


if __name__ == "__main__":
    main()
