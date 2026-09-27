"""Follow-up (2026-09-26 late): THREE SHADOWS, FASHION-MNIST. Do the MNIST triptych's three mechanisms
(SGD -> pixel variance, Adam -> pixel support, signum -> pixel rarity) predict the Fashion-MNIST shadows?

Sources: cache/followup_fashion_{sgd,adam,signum}/ (followup_fashion_imp.py; jobs 974-976), seeds 0,1.
MNIST reference numbers are recomputed with the same metrics from followup_triptych.SETS (jobs 664, 728, 873, 880).

Metrics (seed-mean shadow = surviving first-layer weights per pixel at round R):
  r_std     r with pixel std over the 55k training images            (the variance prediction)
  r_lit     r with "ever lit"                                         (the support prediction; undefined on Fashion,
                                                                       where every pixel is lit in >= 10 images)
  r_rar     r with rarity = log10(55000 / images lighting the pixel), over lit pixels   (the rarity prediction)
  cv_lit    std/mean of the shadow over lit pixels (flatness; the support prediction on Fashion is a flat field, cv -> 0)
  dec       mean kept in the rarest / commonest tenth of lit pixels (by lit count), and their ratio
  disp0     round-0 mean |W_T - W_0| per pixel after dense training, and its r with std and rarity
Outputs: gallery/followup_fashion_triptych_plate.png (followup_triptych.py's plate, same treatment),
         gallery/followup_fashion_predictors.png (the three predicted maps, same dots), cache/followup_fashion_triptych.json

    python followup_fashion_triptych.py [--round 15]
"""
import argparse, json
import numpy as np
from PIL import Image, ImageDraw
from common import *
from imp import load_dataset
import followup_triptych as FT

FSETS = [
    ("sgd", "SGD", [("followup_fashion_sgd", "fashion_sgd_lr0.1", s) for s in (0, 1)]),
    ("adam", "ADAM", [("followup_fashion_adam", "fashion_adam_eps1e-8", s) for s in (0, 1)]),
    ("signum", "SIGNUM", [("followup_fashion_signum", "fashion_signum_lr1e-4", s) for s in (0, 1)]),
]
READINGS = {"sgd": "a soft core: the pixels that vary most (as on MNIST)",
            "adam": "not a plateau: a frame, like signum's",
            "signum": "a rim: the pixels least often lit (as on MNIST)"}  # written after the numbers (README)
DEFAULT_READING = {"sgd": "SGD", "adam": "Adam", "signum": "signum"}


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def pixstats(name):
    x = load_dataset(name)[0][:55000]
    cnt = (x > 0).sum(0); lit = cnt > 0
    rar = np.where(lit, np.log10(55000 / np.maximum(cnt, 1)), 0.0)
    o = np.argsort(cnt[lit], kind="stable"); k = len(o) // 10
    idx_lit = np.where(lit)[0]
    return dict(cnt=cnt, std=x.std(0), lit=lit, rar=rar, rare=idx_lit[o[:k]], common=idx_lit[o[-k:]],
                rare_max=int(cnt[idx_lit[o[k - 1]]]), common_min=int(cnt[idx_lit[o[-k]]]))


def disp0(cond, cfg, seed):
    z0 = np.load(f"{CACHE}/{cond}/round_00.npz")
    m = [i for i in range(len(z0["modes"])) if int(z0["seeds"][i]) == seed and (cfg is None or z0["cfg"][i] == cfg)
         and z0["modes"][i] == "imp"][0]
    return np.abs(z0["W0"][m].astype(np.float64) - z0["init_W0"][m]).mean(1)


def metrics(sets, P, R):
    out = {}
    L = P["lit"]
    for key, lab, seeds in sets:
        sh, acc, acc0, dp = [], [], [], []
        for cond, cfg, s in seeds:
            M, ac, ac0 = FT.load_seed(cond, cfg, s, R)
            sh.append(M.sum(1).astype(float)); acc.append(ac); acc0.append(ac0); dp.append(disp0(cond, cfg, s))
        sh = np.stack(sh); s = sh.mean(0); d = np.mean(dp, 0)
        rare, com = float(s[P["rare"]].mean()), float(s[P["common"]].mean())
        out[key] = dict(label=lab, n=len(sh), shadow=s.tolist(), acc=float(np.mean(acc)), acc0=float(np.mean(acc0)),
                        r_std=corr(s, P["std"]), r_lit=corr(s, P["lit"].astype(float)),
                        r_rar=corr(s[L], P["rar"][L]), cv_lit=float(s[L].std() / s[L].mean()),
                        rare=rare, common=com, rare_common=rare / com, never_lit=float(s[~L].mean()) if (~L).any() else None,
                        r_seeds=float(np.mean([corr(sh[i], sh[j]) for i in range(len(sh)) for j in range(i + 1, len(sh))])),
                        disp_r_std=corr(d, P["std"]), disp_r_rar=corr(d[L], P["rar"][L]),
                        disp_rare_common=float(d[P["rare"]].mean() / d[P["common"]].mean()),
                        holes=[int(x.sum()) for x in sh])
    keys = [k for k, *_ in sets]
    out["_cross"] = {f"{p}~{q}": corr(np.array(out[p]["shadow"]), np.array(out[q]["shadow"]))
                     for i, p in enumerate(keys) for q in keys[i + 1:]}
    return out


def table(name, m, P):
    print(f"\n{name}: rarest tenth of lit pixels lit in <= {P['rare_max']} images, commonest tenth in >= {P['common_min']}")
    print(f"{'opt':7s} n  r(std) r(lit) r(rar)  cv_lit  rare  common  ratio | disp0: r(std) r(rar) ratio | acc  dense  r_seeds")
    for k, o in m.items():
        if k.startswith("_"):
            continue
        print(f"{k:7s} {o['n']}  {o['r_std']:6.2f} {o['r_lit']:6.2f} {o['r_rar']:6.2f}  {o['cv_lit']:6.2f} {o['rare']:5.1f} {o['common']:6.1f} {o['rare_common']:6.2f}"
              f" |        {o['disp_r_std']:6.2f} {o['disp_r_rar']:6.2f} {o['disp_rare_common']:5.2f} | {100*o['acc']:.1f} {100*o['acc0']:.1f}  {o['r_seeds']:.2f}")
    print("cross:", {k: round(v, 2) for k, v in m["_cross"].items()})


def plate(m, P, R, cell, out):
    keys = ["sgd", "adam", "signum"]
    panel = 28 * cell; fs = panel // 60
    margin = panel // 7; gap = panel // 8; head = fs * 7; foot = fs * 11
    W = 2 * margin + 3 * panel + 2 * gap; H = margin + head + panel + foot + margin // 2
    im = Image.new("RGB", (W, H), CREAM); d = ImageDraw.Draw(im)
    text(d, (margin, margin), "THREE SHADOWS OF THE SAME DATA: FASHION-MNIST", int(fs * 1.8))
    text(d, (margin, margin + int(fs * 2.6)),
         f"LeNet-300-100 lottery tickets on Fashion-MNIST, pixels in [0,1], round {R} of iterative magnitude pruning: "
         f"{m['sgd']['holes'][0]:,} input weights ({100 * 0.8 ** R:.1f}%) survive in every ticket. Only the optimiser differs.", fs, MUTED)
    for k, key in enumerate(keys):
        o = m[key]
        x = margin + k * (panel + gap); y = margin + head
        im.paste(dots(np.array(o["shadow"]), cell), (x, y))
        text(d, (x, y + panel + fs), o["label"], int(fs * 1.5))
        text(d, (x + panel, y + panel + int(fs * 1.35)), READINGS.get(key, DEFAULT_READING[key]), fs, INK, anchor="ra")
        text(d, (x, y + panel + int(fs * 3.2)),
             f"rarest tenth of pixels (lit in ≤{P['rare_max']:,} images) keeps {o['rare']:.1f}; commonest (≥{P['common_min']:,}) keeps {o['common']:.1f}",
             fs, MUTED)
        text(d, (x, y + panel + int(fs * 4.8)),
             f"r(pixel std) {o['r_std']:.2f} · r(rarity) {o['r_rar']:.2f} · flatness cv {o['cv_lit']:.2f} · {o['n']} seeds", fs, MUTED)
        acc, acc0 = 100 * o["acc"], 100 * o["acc0"]
        worse = acc < acc0 - 0.5
        text(d, (x, y + panel + int(fs * 6.4)),
             f"ticket test {acc:.1f}% (dense {acc0:.1f}%)" + ("  — a worse ticket" if worse else ""), fs, INK if worse else MUTED)
    text(d, (margin, H - margin // 2 - int(fs * 1.2)),
         "dot area = seed-mean connections kept / max in panel.  Every Fashion pixel is lit in at least 10 of 55,000 images, so 'ever lit' is "
         "constant; rarity = log10(55,000 / images lighting the pixel).  signum = sign of an EMA momentum (β 0.9), lr 1e-4; SGD lr 0.1; Adam lr 1.2e-3.",
         int(fs * 0.9), MUTED)
    im.save(out); print(out, im.size)


def predictors(P, cell, out):
    """The three mechanisms' predicted shadows for Fashion, same dot treatment (value / max in panel)."""
    maps = [("VARIANCE (SGD's rule)", "pixel std over 55k images", P["std"]),
            ("SUPPORT (Adam's rule)", "ever lit: all 784 pixels, a flat field", P["lit"].astype(float)),
            ("RARITY (signum's rule)", "log10(55,000 / images lighting it)", P["rar"])]
    panel = 28 * cell; fs = panel // 60
    margin = panel // 7; gap = panel // 8; head = fs * 7; foot = fs * 5
    W = 2 * margin + 3 * panel + 2 * gap; H = margin + head + panel + foot + margin // 2
    im = Image.new("RGB", (W, H), CREAM); d = ImageDraw.Draw(im)
    text(d, (margin, margin), "WHAT EACH MECHANISM PREDICTS FOR FASHION-MNIST", int(fs * 1.8))
    text(d, (margin, margin + int(fs * 2.6)), "Pixel statistics of the 55,000 training images, drawn in the plate's dots. "
         "No network: these are the maps the MNIST mechanisms say the shadows should follow.", fs, MUTED)
    for k, (lab, sub, v) in enumerate(maps):
        x = margin + k * (panel + gap); y = margin + head
        im.paste(dots(v, cell), (x, y))
        text(d, (x, y + panel + fs), lab, int(fs * 1.5))
        text(d, (x, y + panel + int(fs * 3.2)), sub, fs, MUTED)
    im.save(out); print(out, im.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=15)
    ap.add_argument("--cell", type=int, default=64)
    a = ap.parse_args()
    PF, PM = pixstats("fashion"), pixstats("mnist")
    mf = metrics(FSETS, PF, a.round)
    mm = metrics([(k, lab, seeds) for k, lab, _, seeds in FT.SETS], PM, 15)
    table(f"FASHION round {a.round}", mf, PF); table("MNIST round 15 (reference, same metrics)", mm, PM)
    # cross-dataset: does the Fashion shadow match its own dataset's predictor better than the MNIST predictor?
    xd = {k: dict(r_fashion_vs_mnist_shadow=corr(np.array(mf[k]["shadow"]), np.array(mm[k]["shadow"])),
                  r_with_mnist_std=corr(np.array(mf[k]["shadow"]), PM["std"])) for k in ("sgd", "adam", "signum")}
    print("cross-dataset:", {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in xd.items()})
    strip = lambda m: {k: {kk: vv for kk, vv in v.items() if kk != "shadow"} if not k.startswith("_") else v for k, v in m.items()}
    json.dump(dict(round=a.round, fashion=strip(mf), mnist=strip(mm), cross_dataset=xd,
                   fashion_rare_max=PF["rare_max"], fashion_common_min=PF["common_min"],
                   mnist_rare_max=PM["rare_max"], mnist_common_min=PM["common_min"]),
              open(f"{CACHE}/followup_fashion_triptych.json", "w"), indent=1)
    plate(mf, PF, a.round, a.cell, f"{GALLERY}/followup_fashion_triptych_plate.png")
    predictors(PF, a.cell, f"{GALLERY}/followup_fashion_predictors.png")


if __name__ == "__main__":
    main()
