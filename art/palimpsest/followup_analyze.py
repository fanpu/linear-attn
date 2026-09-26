"""Follow-up (2026-09-26 evening): summarise cache/followup/*.npz (adam_restart.py).

Per run: A-specific ghost L(t) = g(A) - mean g(D1..D4), averaged over the 8-192 c/img bands;
  scraped   = min of L over steps 1..30 (how far A was erased)
  resurface = max over t after that trough of L(t) - trough (0 for a monotone fade)
  blank     = number of steps <=300 where the output's spatial std < 0.05 (B's std is 0.25-0.3)
  dead      = fraction of units active on no pixel, per ReLU layer (max over phase 2; at step 0)
  revived   = units dead at some step <= 150 and alive at step 1000 (layers 3+4, fraction of 512)
Writes cache/followup/summary.json and prints a table.
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "cache", "followup")


def summarise(path):
    z = np.load(path)
    st = z["steps"]; g = z["ghost"]
    names = [str(x) for x in z["template_names"]]
    ix = {k: i for i, k in enumerate(names)}
    L = g[:, ix["A"]] - g[:, [ix[f"D{i}"] for i in range(1, 5)]].mean(1)
    Lc = g[:, ix["C"]]
    Lm = L[:, 3:].mean(1); Lcm = Lc[:, 3:].mean(1)
    def res(curve):
        w = (st >= 1) & (st <= 30)
        i0 = np.where(w)[0][np.argmin(curve[w])]
        after = curve[i0:]
        j = int(np.argmax(after))
        return float(curve[i0]), int(st[i0]), float(after[j] - curve[i0]), int(st[i0 + j]), float(after[j])
    tr, ttr, amp, tpk, pk = res(Lm)
    trc, _, ampc, tpkc, _ = res(Lcm)
    act = z["act"].astype(np.float32)  # (T, 4, 256)
    dead = (act == 0)
    early = (st <= 150)
    rev = (dead[early][:, 2:].any(0) & ~dead[-1, 2:]).mean()
    fstd = z["fstd"]
    blank = int((fstd[st <= 300] < 0.05).sum())
    return dict(name=os.path.basename(path)[:-4], psnr1=float(z["psnr1"]), step1_L=float(Lm[st == 1][0]),
                trough=tr, trough_step=ttr, resurface=amp, peak_step=tpk, peak=pk,
                resurface_C=ampc, peak_step_C=tpkc,
                blank=blank, dead0=[round(float(dead[0, k].mean()), 3) for k in range(4)],
                deadmax=[round(float(dead[:, k].mean(1).max()), 3) for k in range(4)], revived34=float(rev),
                psnrB_300=float(z["psnrB"][st == 300][0]), psnrB_final=float(z["psnrB"][-1]),
                finest_peak=float(L[:, 6][st >= 10].max()))


def main():
    rows = [summarise(p) for p in sorted(glob.glob(os.path.join(D, "*B_s*_*.npz")))]
    json.dump(rows, open(os.path.join(D, "summary.json"), "w"), indent=1)
    print(f"{'run':22s} {'L@1':>6s} {'trough':>7s} {'@':>4s} {'resurf':>7s} {'@':>4s} {'resC':>6s} {'blank':>5s} "
          f"{'dead L3/L4 max':>15s} {'revived':>7s} {'B@300':>6s} {'B@1k':>6s}")
    for r in rows:
        print(f"{r['name']:22s} {r['step1_L']:6.3f} {r['trough']:7.3f} {r['trough_step']:4d} {r['resurface']:7.3f} {r['peak_step']:4d} "
              f"{r['resurface_C']:6.3f} {r['blank']:5d} {r['deadmax'][2]:7.3f}/{r['deadmax'][3]:.3f} {r['revived34']:7.3f} "
              f"{r['psnrB_300']:6.1f} {r['psnrB_final']:6.1f}")


if __name__ == "__main__":
    main()
