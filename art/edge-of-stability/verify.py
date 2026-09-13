"""Verification numbers for the README (no GPU). usage: python verify.py <cache.npz> [ts]

Per step size:
  t_edge          first step with lambda_1 >= 2/eta
  hover           median and 5-95% range of (lambda_1 - 2/eta)/(2/eta) after t_edge + 200
  cold_vs_warm    max |warm - cold| / cold over the periodic cold-start checks (sharpness audit)
  resid           median Rayleigh-Ritz residual ||H u1 - lambda_1 u1|| / lambda_1
  overlap         median |<u1(t), u1(t-E)>| and fraction of refreshes with overlap < 0.95
  loss            loss at t_edge, final loss, fraction of EoS steps where loss went up
  crossings       sign changes of (-1)^t x_t while the amplitude is above the floor, and the share
                  that coincide (within 2 steps) with an eigenvector overlap < 0.95 (identity swap,
                  a coordinate artefact rather than a phase slip)
  burst period    peak of the autocorrelation of log amplitude after t_edge (repeating structure)
"""
import json
import sys

from eos_common import *  # noqa: F401,F403


def per_model(d, m, ts=40, amin=1e-5):
    inv = float(d["invs"][m])
    loss = d["loss"][:, m].astype(float)
    alive = np.isfinite(loss)
    out = {"inv": inv, "diverged_at": int(d["diverged_at"][m])}
    if alive.sum() < 100 or d["diverged_at"][m] >= 0:
        return out
    lam = ffill(d["evals"][:, m].astype(float))
    E = int(d["meta"]["eig_every"])
    below = np.flatnonzero(lam[:, 0] < inv)
    first_below = int(below[0]) if len(below) else 0  # init sharpness 88 can start above the edge
    out["t_first_below"] = first_below
    above = np.flatnonzero((lam[:, 0] >= inv) & (np.arange(len(lam)) > first_below))
    if not len(above):
        out["t_edge"] = None
        out["lam1_final"] = float(lam[-1, 0])
        return out
    te = int(above[0])
    out["t_edge"] = te
    h = (lam[te + 200:, 0] - inv) / inv
    if len(h):
        out["hover_median"] = float(np.median(h))
        out["hover_p5_p95"] = [float(np.percentile(h, 5)), float(np.percentile(h, 95))]
        out["lam2_hover_median"] = float(np.median((lam[te + 200:, 1] - inv) / inv))
        out["lam3_hover_median"] = float(np.median((lam[te + 200:, 2] - inv) / inv))
    ct, cv = d["checks_t"], d["checks_v"]
    if len(ct):
        warm = np.array([lam[t, 0] for t in ct])
        cold = cv[:, m, 0]
        out["cold_vs_warm_max_rel"] = float(np.max(np.abs(warm - cold) / cold))
        out["cold_checks"] = [[int(a), float(b), float(c)] for a, b, c in zip(ct, warm, cold)]
    fr = d["eig_fresh"]
    r = d["resid"][fr, m, 0] / d["evals"][fr, m, 0]
    out["resid_rel_median"] = float(np.nanmedian(r))
    out["resid_rel_p95"] = float(np.nanpercentile(r, 95))
    ov = d["u1_overlap_prev"][:, m]
    ovf = ov[np.isfinite(ov)]
    out["overlap_median"] = float(np.median(ovf))
    out["overlap_frac_lt_0.95"] = float(np.mean(ovf < 0.95))
    out["loss_at_edge"] = float(loss[te])
    out["loss_final"] = float(loss[alive][-1])
    dl = np.diff(loss[te:])
    out["frac_steps_loss_up_after_edge"] = float(np.mean(dl > 0))
    x = d["x"][:, m].astype(float)
    x[:ts] = np.nan
    T = len(x)
    amp = np.sqrt(0.5 * (x ** 2 + np.r_[x[1:], np.nan] ** 2))
    xd = x * (-1.0) ** np.arange(T)
    ok = np.isfinite(xd) & (amp > amin)
    sc = np.flatnonzero(ok[:-1] & ok[1:] & (np.sign(xd[:-1]) != np.sign(xd[1:])))
    bad = np.flatnonzero(np.nan_to_num(ov, nan=1) < 0.95)
    swap = np.array([np.any(np.abs(bad - s) <= max(2, E)) for s in sc]) if len(sc) else np.zeros(0, bool)
    out["crossings"] = int(len(sc))
    out["crossings_at_eigvec_swap_frac"] = float(swap.mean()) if len(sc) else None
    out["crossings_per_100_steps_after_edge"] = float(100 * np.sum(sc > te) / max(1, T - te))
    # amplitude dips at crossings: amplitude at crossing relative to local (+-20 step) max
    rel = [amp[s] / np.nanmax(amp[max(0, s - 20):s + 20]) for s in sc if np.isfinite(amp[s])]
    out["crossing_amp_rel_median"] = float(np.median(rel)) if rel else None
    # the same audit in the windowed-PCA coordinate (swap-free)
    c = braid(d, m, kind="pca", ts=ts)
    scp, ampp = crossings(c, ts)
    swp = np.array([np.any(np.abs(bad - s) <= max(2, E)) for s in scp]) if len(scp) else np.zeros(0, bool)
    out["pca_crossings"] = int(len(scp))
    out["pca_crossings_after_edge"] = int(np.sum(scp > te + 100))
    out["pca_crossing_steps"] = [int(v) for v in scp[:50]]
    out["pca_crossings_at_eigvec_swap_frac"] = float(swp.mean()) if len(scp) else None
    _, coss, cents = pca_braid(d, m)
    out["pca_cos_u1_median"] = float(np.median(coss))
    out["pca_cos_u1_median_after_edge"] = float(np.median(coss[cents > te + 100])) if np.any(cents > te + 100) else None
    # null for the swap-coincidence audit: share of ALL post-edge steps that lie within 2 steps of a swap
    badm = np.zeros(T, bool)
    badm[bad] = True
    near = np.convolve(badm.astype(float), np.ones(5), "same") > 0
    post = np.arange(T) > te + 100
    out["swap_window_base_rate_after_edge"] = float(near[post].mean())
    scu = sc[sc > te + 100]
    out["crossings_after_edge"] = int(len(scu))
    out["crossings_after_edge_at_swap_frac"] = float(near[scu].mean()) if len(scu) else None
    scq = scp[scp > te + 100]
    out["pca_crossings_after_edge_at_swap_frac"] = float(near[scq].mean()) if len(scq) else None
    relp = [ampp[s] / np.nanmax(ampp[max(0, s - 20):s + 20]) for s in scq if np.isfinite(ampp[s])]
    out["pca_crossing_amp_rel_median"] = float(np.median(relp)) if relp else None
    if "pca" == COORD:
        amp = ampp
    la = np.log(amp[te + 50:])
    la = la[np.isfinite(la)]
    if len(la) > 400:
        la = la - la.mean()
        ac = np.correlate(la, la, "full")[len(la) - 1:]
        ac /= ac[0]
        lag = np.arange(len(ac))
        z = np.flatnonzero(ac < 0)
        sel = (lag >= (z[0] if len(z) else 10)) & (lag <= 1500)
        if sel.any():
            pk = lag[sel][np.argmax(ac[sel])]
            out["burst_autocorr_peak_lag"] = int(pk)
            out["burst_autocorr_peak_val"] = float(ac[pk])
    # burst spacing: local maxima of the (7-step max-filtered) amplitude that are the largest within +-15 steps
    from scipy.signal import find_peaks
    env = envelope(np.nan_to_num(amp), 3)[te + 100:]
    pk, _ = find_peaks(env, distance=15, prominence=0.3 * np.median(env))
    if len(pk) > 3:
        dpk = np.diff(pk)
        out["burst_spacing_median"] = float(np.median(dpk))
        out["burst_spacing_p25_p75"] = [float(np.percentile(dpk, 25)), float(np.percentile(dpk, 75))]
        out["bursts"] = int(len(pk))
    return out


if __name__ == "__main__":
    f = sys.argv[1]
    d = load(f)
    res = {"file": f, "steps": int(d["steps_done"]), "meta": d["meta"],
           "models": [per_model(d, m) for m in range(len(d["invs"]))]}
    path = os.path.join(CACHE, f.replace(".npz", "_verify.json"))
    json.dump(res, open(path, "w"), indent=1)
    keys = ["inv", "t_edge", "crossings_after_edge", "crossings_after_edge_at_swap_frac",
            "swap_window_base_rate_after_edge", "crossing_amp_rel_median", "pca_crossings_after_edge_at_swap_frac",
            "pca_crossing_amp_rel_median", "pca_cos_u1_median_after_edge", "pca_crossings", "pca_crossings_after_edge", "pca_cos_u1_median", "burst_autocorr_peak_lag", "burst_spacing_median", "bursts"]
    for mm in res["models"]:
        print(" ".join(f"{k}={mm[k]:.3g}" if isinstance(mm.get(k), float) else f"{k}={mm.get(k)}"
                       for k in keys if k in mm))
    print("wrote", path)
