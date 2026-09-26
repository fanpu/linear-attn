"""Follow-up (2026-09-26 evening) plate: the blank page is a restarted optimiser's, and the layer
beneath it still holds the under-text.

Rows (all from followup_probe.py, one CPU trajectory each, Fourier features sigma 32, width 256):
  1 restarted Adam (fresh moments, the original protocol): the page (network output)
  2 restarted Adam: the layer beneath (best linear readout of page A from the last hidden layer)
  3 Adam with its state carried over from phase 1: the page
  4 carried state: the layer beneath
  5 control, the network that held page C, restarted Adam: the layer beneath (readout of A)
Every panel: top-left quarter, one declared exposure (ink density 0.08-0.45 -> 0-1), the same as
resurface_ff32_w256.png; nearest-neighbour enlargement.
"""
import json, os, sys
import numpy as np
from look import normal_bl, up
from plates import frame

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "cache", "followup")
LO, HI = 0.08, 0.45


def expo(x):
    return normal_bl(np.clip((np.asarray(x, np.float32) - LO) / (HI - LO), 0, 1))


def main(steps=(0, 3, 20, 60, 80, 150), k=2, out=None):
    rows = [("A", "fresh", "out", "1 restarted · page"),
            ("A", "fresh", "recon_A", "2 restarted · layer beneath"),
            ("A", "carry", "out", "3 carried · page"),
            ("A", "carry", "recon_A", "4 carried · layer beneath"),
            ("C", "fresh", "recon_A", "5 held C · layer beneath")]
    pan, lab = [], []
    for first, var, key, name in rows:
        z = np.load(os.path.join(F, f"probe_{first}B_s0_{var}.npz"))
        js = {r["step"]: r for r in json.load(open(os.path.join(F, f"probe_{first}B_s0_{var}.json")))}
        st = list(z["steps"])
        for s in steps:
            i = st.index(s)
            img = z[key][i].astype(np.float32)[:128, :128]
            pan.append(up(expo(img), k))
            r = js[s]
            if key == "out":
                lab.append(f"{name} · step {s} · page std {r['fstd']:.3f}")
            else:
                lab.append(f"{name} · step {s} · R²(A) {r['r2_A']:.2f}")
    title = "Restart the optimiser, and the under-text comes back"
    fr, ca = js_get("A", "fresh"), js_get("A", "carry")
    cap = ("Fourier features (σ 32, width 256) that held page A (Lucretius), then trained on page B with Adam. Top-left quarter, steps of the second training.\n"
           "Rows 1–2: Adam restarted with zeroed moments, as in the original run (and as fine-tuning usually does). "
           "Rows 3–4: Adam's moment estimates carried over from the first training.\n"
           "Row 5: control, the same network after holding page C (Iliad), restarted.  'Page' is the network's output.  'Layer beneath' is the best linear readout\n"
           "of page A from the 256 units of the last hidden layer (ridge, fitted to A at each step): what the layer can still express, not what the network shows.\n"
           f"Restarted: at step 5 the page is blank (std {fr[5]['fstd']:.3f}), yet the readout still recovers A at R² {fr[5]['r2_A']:.2f}, with {fr[5]['dead']*100:.0f}% of last-layer units dead.\n"
           "The readout has cancelled the letters, not the layer; they come back to the page around steps 60–100.\n"
           f"Carried state: larger first steps, not smaller (v still holds phase 1's tiny gradients; parameter displacement 1.7× after one step, 2.5× after five).\n"
           f"{ca[20]['dead']*100:.0f}% of last-layer units are dead by step 20, R²(A) falls to {ca[20]['r2_A']:.2f}, and A never returns. Control: R²(A) ≤ 0.006 throughout.\n"
           "Declared exposure: ink density 0.08–0.45 stretched to the full range, identically for all panels.")
    img = frame(pan, len(steps), title, cap, labels=lab, label_size=13)
    out = out or os.path.join(HERE, "gallery", "followup_restart_ff32_w256.png")
    img.save(out)
    print(out, img.size)


def js_get(first, var):
    return {r["step"]: r for r in json.load(open(os.path.join(F, f"probe_{first}B_s0_{var}.json")))}


if __name__ == "__main__":
    main()
