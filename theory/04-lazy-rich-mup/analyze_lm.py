"""Collect cache/lm/*.json into tidy tables and estimate the optimal learning rate of each curve."""
import glob, json, math, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(pattern="*"):
    rows = []
    for p in glob.glob(f"{HERE}/cache/lm/{pattern}.json"):
        r = json.load(open(p))
        rows.append(r)
    return rows


def optimum(lrs, losses):
    """Minimum of a parabola through the best grid point and its two neighbours, in log2(lr).
    Returns (log2 lr*, loss*, at_edge). Diverged / missing runs are NaN and ignored."""
    x = np.log2(np.asarray(lrs, float)); y = np.asarray(losses, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan"), True
    i = int(np.argmin(y))
    if i == 0 or i == len(x) - 1:
        return float(x[i]), float(y[i]), True
    xs, ys = x[i - 1:i + 2], y[i - 1:i + 2]
    a, b, c = np.polyfit(xs, ys, 2)
    if a <= 0:
        return float(x[i]), float(y[i]), False
    xm = float(np.clip(-b / (2 * a), xs[0], xs[-1]))
    return xm, float(np.polyval([a, b, c], xm)), False


def curves(rows, key_fn, steps=None):
    """Group runs into curves: key -> sorted (lr, val, run)."""
    out = {}
    for r in rows:
        if steps is not None and r["steps"] != steps:
            continue
        out.setdefault(key_fn(r), []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r["lr"])
    return out


def summary():
    rows = load()
    width = curves([r for r in rows if r["L"] == 4 and not r["depth_mup"] and r["steps"] == 1000],
                   lambda r: (r["param"], r["d"]))
    # SP and muP coincide at the base width d=128: reuse the muP runs for SP
    if ("mup", 128) in width:
        width[("sp", 128)] = width[("mup", 128)]
    res = {"width": {}}
    for (param, d), rs in sorted(width.items()):
        lrs = [r["lr"] for r in rs]; vals = [r["val"] if not r["diverged"] else float("nan") for r in rs]
        xm, ym, edge = optimum(lrs, vals)
        res["width"][f"{param}_{d}"] = dict(param=param, d=d, lrs=lrs, val=vals, opt_log2lr=xm, opt_val=ym, edge=edge,
                                            diverged=[r["diverged"] for r in rs])
        print(f"{param:4s} d={d:5d}  best log2 lr {xm:6.2f} (lr {2 ** xm:.2e}) loss {ym:.3f} {'EDGE' if edge else ''}  "
              + " ".join("  nan " if not math.isfinite(v) else f"{v:.3f}" for v in vals))
    return res


if __name__ == "__main__":
    summary()
