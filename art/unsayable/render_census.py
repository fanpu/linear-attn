"""Plate: census with its null.  Top: every candidate the indicator flags (top 2%, minus partial-UTF-8
and unreachable).  Bottom: an equal number of ordinary tokens drawn at random from the rest of the
vocabulary.  Same treatment for both: a token is printed in ink if the model cannot say it
(p_max < 0.01 across the three repetition prompts) and as a pale ghost if it can.
Order within each panel (declared): by the model's indicator, most under-trained first."""
import argparse, json, math, os
import look as L
import typeset as ts
from render_register import NICE, CACHE


def panel(d, rows, x0, y0, w, cols, cell, title, sub):
    L.smallcaps(d, (x0, y0), title, 40, L.INK, track=0.14)
    k = sum(r["verified"] for r in rows)
    txt = sub.format(k=k, n=len(rows), pct=100 * k / max(1, len(rows)), )
    f = ts.font(ts.F["text_i"], 40)
    d.text((x0 + w, y0), txt, font=f, fill=L.INK, anchor="rs")
    y = y0 + 36
    d.line([(x0, y), (x0 + w, y)], fill=L.INK, width=3)
    y += 18
    for j, r in enumerate(rows):
        cx = x0 + (j % cols) * cell
        cy = y + (j // cols) * cell
        s = r["str"]
        size = cell * 0.62
        wv = L.vis_width(s, size)
        if wv > cell * 0.9:
            size *= cell * 0.9 / wv
        size = max(size, 9)
        wv = L.vis_width(s, size)
        col = L.INK if r["verified"] else L.PALE
        mk = L.GREY if r["verified"] else (222, 214, 198)
        L.draw_visible(d, (cx + (cell - wv) / 2, cy + cell * 0.72), s, size, col, marker_fill=mk)
    return y + math.ceil(len(rows) / cols) * cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = json.load(open(os.path.join(CACHE, f"{a.model}_table.json")))
    cand = sorted([r for r in R if r["set"] == "cand"], key=lambda r: r["ind"])
    rand = sorted([r for r in R if r["set"] == "rand"], key=lambda r: r["ind"])
    n = len(cand)
    M = 200
    w = L.W - 2 * M
    # choose a cell so both panels fit on the sheet
    avail = L.H - 560 - 420 - 2 * 60
    cols = 20
    while True:
        cell = w / cols
        if 2 * math.ceil(n / cols) * cell <= avail:
            break
        cols += 1
    cell = w / cols
    im, d = L.sheet()
    L.smallcaps(d, (M, 300), "UNSAYABLE", 96, L.INK, track=0.16)
    ts.draw(d, (M, 390), f"census and control — {NICE[a.model]}", 56, [(ts.F["text_i"], 0)], L.INK)
    f = ts.font(ts.F["text_i"], 38)
    for k, t in enumerate(["ink: the model cannot say it (p < 0.01 in all three repetition prompts)",
                           "ghost: it can"]):
        d.text((L.W - M, 300 + 52 * k), t, font=f, fill=L.GREY, anchor="rs")
    y = 560
    y = panel(d, cand, M, y, w, cols, cell, "FLAGGED BY THE INDICATOR",
              "the most under-trained 2% by weights alone: {k:,} of {n:,} unsayable ({pct:.1f}%)")
    y += 110
    if rand:
        y = panel(d, rand, M, y, w, cols, cell, "CONTROL: ORDINARY TOKENS AT RANDOM",
                  "same number, drawn from the other 98%: {k:,} of {n:,} unsayable ({pct:.1f}%)")
    notes = ["Each token is set to fit its cell (declared); grey marks are transcription: ␣ space, ↵ newline.",
             "Order in both panels: indicator value, most under-trained first. Token ids are fed directly; greedy decoding; test after Land & Bartolo (2024)."]
    for k, t in enumerate(notes):
        ts.draw(d, (M, L.H - 250 + 46 * k), t, 32, [(ts.F["text_i"], 0)] + ts.SERIF_CHAIN, L.GREY)
    im.save(a.out, optimize=True)
    print("wrote", a.out, cols, "cols", round(cell), "px cells")


if __name__ == "__main__":
    main()
