"""Export compact widget data files (widgets/data_*.js) from cache/.

    .venv/bin/python 04-lazy-rich-mup/export_widgets.py toy
    .venv/bin/python 04-lazy-rich-mup/export_widgets.py mup
"""
import base64, glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = f"{HERE}/cache"
WID = f"{HERE}/widgets"


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def write(name, var, obj):
    path = f"{WID}/data_{name}.js"
    with open(path, "w") as f:
        f.write(f"window.{var} = {json.dumps(obj, separators=(',', ':'))};\n")
    print(path, f"{os.path.getsize(path) / 1e6:.2f} MB")


def toy():
    files = sorted(glob.glob(f"{C}/toy/widget_a*.npz"), key=lambda p: float(p.split("_a")[-1][:-4]))
    runs, alphas = [], []
    for p in files:
        d = np.load(p)
        F, m = d["w"].shape[:2]
        th0 = np.concatenate([d["w"][0].ravel(), d["b"][0], d["a"][0]])
        move = [float(np.linalg.norm(np.concatenate([d["w"][k].ravel(), d["b"][k], d["a"][k]]) - th0) / np.linalg.norm(th0))
                for k in range(F)]
        scale, q = [], np.zeros((F, m, 4), dtype=np.int16)
        for k in range(F):
            s = [float(np.abs(d["w"][k]).max() / 32000), float(np.abs(d["b"][k]).max() / 32000),
                 float(np.abs(d["a"][k]).max() / 32000)]
            q[k, :, :2] = np.round(d["w"][k] / s[0]); q[k, :, 2] = np.round(d["b"][k] / s[1]); q[k, :, 3] = np.round(d["a"][k] / s[2])
            scale.append(s)
        alphas.append(float(d["alpha"]))
        runs.append(dict(p=b64(q), scale=scale, loss=[round(float(v), 5) for v in d["loss"]],
                         acc=[round(float(v), 4) for v in d["acc"]], move=[float(f"{v:.4g}") for v in move]))
        steps = [int(s) for s in d["step"]]
        x, y, R = d["x"].astype(np.float32), d["y"].astype(np.float32), float(d["R"])
    write("toy", "TOY_DATA", dict(alphas=[a if a != int(a) else int(a) for a in alphas], m=int(m), R=R, steps=steps,
                                  x=b64(x), y=b64(y), runs=runs))


def mup():
    sys.path.insert(0, HERE)
    from analyze_lm import load, optimum
    rows = load()

    def curve(rs, size):
        rs = sorted(rs, key=lambda r: r["lr"])
        lrs = [r["lr"] for r in rs]
        val = [None if r["diverged"] else round(r["val"], 4) for r in rs]
        xm, ym, edge = optimum(lrs, [np.nan if v is None else v for v in val])
        hist = []
        for r in rs:
            st, tr = np.array(r["hist"]["step"]), np.array(r["hist"]["train"])
            k = len(st) // 2 * 2
            if k < 2:
                hist.append(None); continue
            hist.append([st[:k:2].tolist(), np.round(tr[:k].reshape(-1, 2).mean(1), 3).tolist()])
        return dict(size=size, lrs=lrs, val=val, opt=None if not np.isfinite(xm) else round(xm, 4),
                    optval=None if not np.isfinite(ym) else round(ym, 4), edge=bool(edge), hist=hist)

    out = {"width": {"sp": [], "mup": []}, "depth": {"plain": [], "dmup": []}, "steps": 1000,
           "log2lr": [-12, -5], "ylim": [2.6, 4.4], "yticks": [2.8, 3.2, 3.6, 4.0, 4.4]}
    for param in ["sp", "mup"]:
        for d in [128, 256, 512, 1024]:
            p = "mup" if d == 128 else param
            rs = [r for r in rows if r["param"] == p and r["d"] == d and r["L"] == 4 and not r["depth_mup"] and r["steps"] == 1000]
            if rs:
                out["width"][param].append(curve(rs, d))
    for key, dm in [("plain", False), ("dmup", True)]:
        for L in [2, 4, 8, 16]:
            rs = [r for r in rows if r["param"] == "mup" and r["d"] == 128 and r["L"] == L and r["steps"] == 1000
                  and (r["depth_mup"] == dm or L == 4) and not (L == 4 and r["depth_mup"])]
            if L == 4:
                rs = [r for r in rs if np.log2(r["lr"]) >= -11]
            if rs:
                out["depth"][key].append(curve(rs, L))
    write("mup", "MUP_DATA", out)


if __name__ == "__main__":
    {"toy": toy, "mup": mup}[sys.argv[1]]()
