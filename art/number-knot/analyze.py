"""CPU analysis for Number Knot M1: helix fits + Fourier + nulls (OLMo-2 numbers), circle fits +
nulls (Qwen3 days, months). Reads cache/hs (or cache/tiny with --tiny); writes
cache/fits_numbers.npz, cache/fits_calendar.npz, cache/tables_M1.md, cache/summary_M1.json.

  OMP_NUM_THREADS=4 python analyze.py [--tiny] [--perms 200]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import common as C
import fits as F

KEYS = ["lin", 2, 5, 10, 100]           # column order of the r2 arrays
COORD_T = [5, 10, 100]                  # periods whose decoded helix coordinates are stored


def load_set(d, name, n_tmpl):
    return [np.load(d / f"{name}_t{k}.npy").astype(np.float32) for k in range(n_tmpl)]


def numbers(d, n_perm, out, log):
    meta = json.loads((d / "meta.json").read_text())
    nt = len(C.NUMBER_TEMPLATES)
    sets = {"numbers": load_set(d, "olmo_numbers", nt), "random": load_set(d, "olmo_random", nt)}
    N, L1, _ = sets["numbers"][0].shape
    ranges = [r for r in (100, 1000) if r <= N] or [N]
    rng = np.random.default_rng(1)
    res = {}
    for R in ranges:
        a = np.arange(R)
        perms = np.array([rng.permutation(R) for _ in range(n_perm)])
        fperms = perms[:50]
        r2 = np.zeros((2, nt, L1, len(KEYS)))
        null = np.zeros((2, nt, L1, len(KEYS), n_perm), dtype=np.float32)
        evr = np.zeros((2, nt, L1))
        spec = np.zeros((2, nt, L1, 2, R // 2 + 1), dtype=np.float32)       # raw, detrended
        spec_null99 = np.zeros((nt, L1, 2, R // 2 + 1), dtype=np.float32)   # numbers, shuffled a
        coords = np.zeros((nt, L1, len(COORD_T), R, 3), dtype=np.float32)
        extra_bases = {"poly3": F.basis_poly3(a), "mod10": F.basis_mod(a, 10)}
        if R >= 1000:
            extra_bases["mod100"] = F.basis_mod(a, 100)
        r2x = np.zeros((2, nt, L1, len(extra_bases)))
        r2x_null = np.zeros((2, nt, L1, len(extra_bases), 2))                   # mean, 99th pct
        for s, (sname, arrs) in enumerate(sets.items()):
            for k in range(nt):
                t0 = time.time()
                for l in range(L1):
                    Y, _, _, ev = F.pca(arrs[k][:R, l], 100)
                    evr[s, k, l] = ev.sum()
                    rr, nn = F.helix_table(Y, a, perms)
                    r2[s, k, l] = [rr[c] for c in KEYS]
                    null[s, k, l] = [nn[c] for c in KEYS]
                    for j, (xn, B) in enumerate(extra_bases.items()):
                        r2x[s, k, l, j] = F.r2_fit(Y, B)
                        nx = F.r2_perm(Y, B, perms[:50])
                        r2x_null[s, k, l, j] = nx.mean(), np.quantile(nx, 0.99)
                    for j, det in enumerate((False, True)):
                        spec[s, k, l, j] = F.fourier_power(Y, a, det)
                        if sname == "numbers":
                            ps = np.stack([F.fourier_power(Y[p], a, det) for p in fperms])
                            spec_null99[k, l, j] = np.quantile(ps, 0.99, axis=0)
                    if sname == "numbers" and R >= 20:
                        for j, T in enumerate(COORD_T):
                            coords[k, l, j] = F.helix_coords(Y, a, T)[0]
                log(f"range {R} {sname} t{k}: {time.time() - t0:.1f}s")
        res[R] = dict(r2=r2, null=null, evr=evr, spec=spec, spec_null99=spec_null99, coords=coords,
                      r2x=r2x, r2x_null=r2x_null, extra_keys=np.array(list(extra_bases)))
        np.savez_compressed(out / f"fits_numbers_{R}.npz", keys=np.array([str(c) for c in KEYS]),
                            coord_T=np.array(COORD_T), perms=perms, **res[R])
    return res, meta


def calendar(d, n_perm, out, log):
    rng = np.random.default_rng(2)
    res = {}
    for name, T, W in [("qwen_days", C.DAY_TEMPLATES, C.DAYS), ("qwen_months", C.MONTH_TEMPLATES, C.MONTHS)]:
        arr = np.stack(load_set(d, name, len(T)))           # (nt, K, L+1, D)
        nt, K, L1, D = arr.shape
        lab = np.tile(np.arange(K), nt)
        tmpl = np.repeat(np.arange(nt), K)
        H_all = arr.reshape(nt * K, L1, D).astype(np.float64)
        order_perms = [rng.permutation(K) for _ in range(n_perm)]
        point_perms = [rng.permutation(nt * K) for _ in range(n_perm)]
        cols = ["r2_in", "r2_ho", "cyclic", "plane_var"]
        obs = np.zeros((L1, 4))
        n_order = np.zeros((L1, 2, n_perm))
        n_point = np.zeros((L1, 2, n_perm))
        proj_means = np.zeros((L1, K, 2))
        proj = np.zeros((L1, nt * K, 2))
        t0 = time.time()
        for l in range(L1):
            H = H_all[:, l]
            s = F.calendar_scores(H, lab, tmpl, K)
            obs[l] = [s[c] for c in cols]
            proj_means[l], proj[l] = s["proj_means"], s["P"]
            for i in range(n_perm):
                so = F.calendar_scores(H, lab, tmpl, K, order=order_perms[i])
                n_order[l, :, i] = so["r2_in"], so["r2_ho"]
                sp = F.calendar_scores(H, lab[point_perms[i]], tmpl, K)
                n_point[l, :, i] = sp["r2_in"], sp["r2_ho"]
        # exact order null for days: all 6!/2 = 360 cyclic orders distinct up to rotation and reflection
        # (the free 2x2 A makes rotated/reflected orders score identically to the true order)
        exact = np.zeros((0,))
        if K == 7:
            import itertools
            classes = [(0,) + p for p in itertools.permutations(range(1, K)) if p[0] < p[-1]]
            exact = np.zeros((L1, len(classes)))
            for l in range(L1):
                for i, seq in enumerate(classes):
                    order = np.empty(K, int); order[list(seq)] = np.arange(K)
                    exact[l, i] = F.calendar_scores(H_all[:, l], lab, tmpl, K, order=order)["r2_ho"]
            assert classes[0] == tuple(range(K))
        log(f"{name}: {time.time() - t0:.1f}s")
        res[name] = dict(obs=obs, n_order=n_order, n_point=n_point, proj_means=proj_means, proj=proj, exact_order_r2ho=exact,
                         labels=lab, templates=tmpl, words=np.array(W))
        np.savez_compressed(out / f"fits_{name}.npz", cols=np.array(cols), **res[name])
    return res


def q99(x, axis=-1):
    return np.quantile(x, 0.99, axis=axis)


def number_tables(res, lines, summary):
    Ts = [2, 5, 10, 100]
    for R, r in res.items():
        r2, null = r["r2"], r["null"]
        xk = list(r["extra_keys"])
        xd = r["r2x"] - r2[..., :1]                                # ΔR² of extra bases over linear
        xd_null = r["r2x_null"] - null[..., 0, :].mean(-1)[..., None, None]
        lin = r2[..., :1]
        dr2 = r2[..., 1:] - lin                                   # (2, nt, L1, 4)
        dnull = null[..., 1:, :] - null[..., :1, :]
        fw = dnull.max(axis=2)                                    # max over layers per perm: (2, nt, 4, P)
        fw99 = q99(fw)                                            # (2, nt, 4)
        cell99 = q99(dnull)                                       # (2, nt, L1, 4)
        r2null99 = q99(null[..., 1:, :])
        nt, L1 = r2.shape[1], r2.shape[2]
        summary[f"numbers_{R}"] = {}
        for k in range(nt):
            lines.append(f"\n### OLMo-2 numbers 0–{R - 1}, template {k}: `{C.NUMBER_TEMPLATES[k]}`\n")
            lines.append("ΔR²_T = R²([1,a,cos,sin]) − R²([1,a]) on the top-100 PCs. Null columns are 99th percentiles"
                         f" over {null.shape[-1]} label shuffles: `sh` = numbers with shuffled a, `rt` = 1000 random"
                         " single tokens with random labels (same templates, same pipeline). PCA% = variance of the"
                         " layer's cloud captured by the 100 PCs.\n")
            hdr = "| layer | PCA% | R²_lin (sh99) | " + " | ".join(f"T={T}: R² / ΔR² (sh99, rt99)" for T in Ts) + " |"
            lines.append(hdr)
            lines.append("|" + "---|" * (3 + len(Ts)))
            for l in range(L1):
                row = f"| {l} | {100 * r['evr'][0, k, l]:.0f} | {r2[0, k, l, 0]:.3f} ({q99(null[0, k, l, 0]):.3f}) | "
                row += " | ".join(
                    f"{r2[0, k, l, j + 1]:.3f} / **{dr2[0, k, l, j]:.3f}** ({cell99[0, k, l, j]:.3f}, {cell99[1, k, l, j]:.3f})"
                    for j in range(len(Ts)))
                lines.append(row + " |")
            lines.append("| family-wise 99% (max over layers) | | | " + " | ".join(
                f"ΔR² sh {fw99[0, k, j]:.3f}, rt {fw99[1, k, j]:.3f}" for j in range(len(Ts))) + " |")
            lines.append(f"\n*Circle vs. residue-class geometry, 0–{R - 1}, `{C.NUMBER_TEMPLATES[k]}`.* ΔR²_mod m ="
                         " R²([1, a, one-hot(a mod m)]) − R²_lin (in brackets: shuffle mean, 99%). share_T = ΔR²_T /"
                         " ΔR²_mod: an isotropic set of residue clusters (no circle) gives share 2/(m−1) per circle"
                         " (T=10, 5: 0.22; T=2: 0.11 for m=10; T=100: 0.02 for m=100); a pure circle gives 1."
                         " poly3 = ΔR²([1, a, a², a³]), a non-periodic comparator with the circle's column count.\n")
            cols = ["poly3", "mod10"] + (["mod100"] if "mod100" in xk else [])
            lines.append("| layer | ΔR² poly3 | ΔR² mod10 | share T=10 | share T=5 | share T=2 |"
                         + (" ΔR² mod100 | share T=100 |" if "mod100" in xk else " ΔR² T=100 |"))
            lines.append("|" + "---|" * (6 + (2 if "mod100" in xk else 1)))
            for l in range(L1):
                m10 = xd[0, k, l, xk.index("mod10")]
                row = (f"| {l} | {xd[0, k, l, xk.index('poly3')]:.3f} ({xd_null[0, k, l, xk.index('poly3'), 1]:.3f}) | "
                       f"{m10:.3f} ({xd_null[0, k, l, xk.index('mod10'), 0]:.3f}, {xd_null[0, k, l, xk.index('mod10'), 1]:.3f}) | "
                       + " | ".join(f"{dr2[0, k, l, Ts.index(T)] / m10:.2f}" for T in (10, 5, 2)) + " |")
                if "mod100" in xk:
                    m100 = xd[0, k, l, xk.index("mod100")]
                    row += (f" {m100:.3f} ({xd_null[0, k, l, xk.index('mod100'), 0]:.3f}, {xd_null[0, k, l, xk.index('mod100'), 1]:.3f}) |"
                            f" {dr2[0, k, l, 3] / m100:.3f} |")
                else:
                    row += f" {dr2[0, k, l, 3]:.3f} |"
                lines.append(row)
            best = {}
            for j, T in enumerate(Ts):
                l = int(np.argmax(dr2[0, k, :, j]))
                thr = max(fw99[0, k, j], fw99[1, k, j])
                best[T] = dict(layer=l, dR2=float(dr2[0, k, l, j]), R2=float(r2[0, k, l, j + 1]),
                               fw99_shuffle=float(fw99[0, k, j]), fw99_random=float(fw99[1, k, j]),
                               ratio=float(dr2[0, k, l, j] / thr), R2_null99_rt=float(r2null99[1, k, l, j]),
                               clear=bool(dr2[0, k, l, j] >= 2 * thr and dr2[0, k, l, j] >= 0.01))
            summary[f"numbers_{R}"][k] = best
            # Fourier at the best T=10 layer
            l = best[10]["layer"]
            sp, sn = r["spec"][0, k, l, 1], r["spec_null99"][k, l, 1]
            top = np.argsort(sp)[::-1][:6]
            lines.append(f"\nFourier (detrended, layer {l}): top bins " + ", ".join(
                f"T={R / f:.1f} ({sp[f]:.3f}, null99 {sn[f]:.3f})" for f in top if f > 0))
            lines.append("; harmonics of 10 (bins 1–5 × N/10): " + ", ".join(
                f"T={10 / m:.2g}: {sp[m * R // 10]:.4f}" for m in range(1, 6) if m * R // 10 < len(sp)))
            lines.append("; at K&T periods: " + ", ".join(
                f"T={T}: {sp[R // T]:.4f} vs null99 {sn[R // T]:.4f}" for T in Ts if R // T < len(sp)) + "\n")


def calendar_tables(res, lines, summary):
    for name, r in res.items():
        obs, no, npnt = r["obs"], r["n_order"], r["n_point"]
        K = len(r["words"])
        lines.append(f"\n### Qwen3-0.6B {name.split('_')[1]} (K={K}, {r['templates'].max() + 1} templates)\n")
        lines.append("Supervised mean-difference plane (top-2 PCs of the class means). R²_in: circle "
                     "regression P = c + A[cos 2πk/K, sin 2πk/K] on all points; R²_ho: plane + circle fitted on "
                     "even templates, scored on odd (and swapped). Nulls: 99th pct over shuffles of the class "
                     "order (`ord`) and of all point labels (`pts`), through the identical pipeline. "
                     "cyclic: the class means' angular order equals the calendar order.\n")
        lines.append("| layer | plane var% | R²_in (ord99, pts99) | R²_ho (ord99, pts99) | cyclic |")
        lines.append("|---|---|---|---|---|")
        L1 = obs.shape[0]
        for l in range(L1):
            lines.append(f"| {l} | {100 * obs[l, 3]:.0f} | {obs[l, 0]:.3f} ({q99(no[l, 0]):.3f}, {q99(npnt[l, 0]):.3f}) "
                         f"| **{obs[l, 1]:.3f}** ({q99(no[l, 1]):.3f}, {q99(npnt[l, 1]):.3f}) | {'yes' if obs[l, 2] else 'no'} |")
        fw_o = q99(no[:, 1].max(0)); fw_p = q99(npnt[:, 1].max(0))
        lines.append(f"| family-wise 99% (max over layers) | | | ord {fw_o:.3f}, pts {fw_p:.3f} | |")
        margin = obs[:, 1] - np.maximum(q99(no[:, 1], -1), q99(npnt[:, 1], -1))
        l = int(np.argmax(margin))
        if r["exact_order_r2ho"].size:
            ex = r["exact_order_r2ho"]                       # column 0 is the true order
            rank = (ex[:, 1:] >= ex[:, :1]).sum(1) + 1
            lines.append(f"\nExact order null (all {ex.shape[1]} cyclic orders up to rotation/reflection), rank of the true "
                         "order by held-out R² per layer: " + ", ".join(f"L{i}:{rk}" for i, rk in enumerate(rank)))
            summary[name + "_exact_rank"] = rank.tolist()
        summary[name] = dict(best_layer=l, r2_ho=float(obs[l, 1]), r2_in=float(obs[l, 0]),
                             cyclic=bool(obs[l, 2]), fw99_order=float(fw_o), fw99_points=float(fw_p),
                             n_cyclic_layers=int(obs[:, 2].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()
    d = C.CACHE / ("tiny" if args.tiny else "hs")
    out = C.CACHE / ("tiny" if args.tiny else "")
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    nres, meta = numbers(d, args.perms, out, log)
    cres = calendar(d, args.perms, out, log)
    lines, summary = ["# Number Knot M1 tables (generated by analyze.py)\n"], {}
    number_tables(nres, lines, summary)
    calendar_tables(cres, lines, summary)
    summary["wall_seconds"] = time.time() - t0
    (out / "tables_M1.md").write_text("\n".join(lines) + "\n")
    (out / "summary_M1.json").write_text(json.dumps(summary, indent=1))
    log("done")


if __name__ == "__main__":
    main()
