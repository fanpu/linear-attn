"""Plate: concordance.  Tokens every Qwen3 size fails to say, each followed by what each size said
when asked to repeat it (chat template, greedy).  Same tokenizer, four models, four failures.

Selection (declared): verified (p_max < 0.01) in all listed models, ordered by the first model's
indicator (most under-trained first), at most --cap per category, then --n rows."""
import argparse, json, os
from collections import Counter
import look as L
import typeset as ts
from render_register import NICE, CACHE, load, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-0.6b,qwen3-1.7b,qwen3-4b,qwen3-8b")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--field", default="chat")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--order", default="ind", choices=["ind", "varied"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if os.path.exists(os.path.join(CACHE, f"{m}_table.json"))]
    T = {m: load(m) for m in models}
    ok = set.intersection(*[{i for i, r in T[m].items() if r["verified"]} for m in models])
    R0 = T[models[0]]
    rows = sorted((R0[i] for i in ok), key=lambda r: r["ind"])
    if a.order == "varied":
        # declared alternative ordering: rows where the four replies differ most come first
        def nvar(r):
            return len({T[m][r["id"]][a.field].strip() for m in models})
        rows = sorted(rows, key=lambda r: (-nvar(r), r["ind"]))
    sel, cc = [], Counter()
    for r in rows:
        if cc[r["cat"]] < a.cap:
            sel.append(r); cc[r["cat"]] += 1
    sel = sel[a.skip:a.skip + a.n]

    im, d = L.sheet()
    M = 200
    right = L.W - M
    L.smallcaps(d, (M, 320), "UNSAYABLE", 110, L.INK, track=0.16)
    ts.draw(d, (M, 420), "one tokenizer, four models, the same silence", 60, [(ts.F["text_i"], 0)], L.INK)
    f = ts.font(ts.F["text_i"], 38)
    for k, t in enumerate([f"{len(ok):,} tokens are unsayable (p < 0.01) in all {len(models)} Qwen3 sizes;",
                           "each was asked “Please repeat the string '‹token›'.”",
                           "their answers, in madder"]):
        d.text((right, 320 + 52 * k), t, font=f, fill=L.GREY, anchor="rs")
    y0 = 560
    d.line([(M, y0), (right, y0)], fill=L.INK, width=4)
    cA, cB = M, M + 330
    cw = (right - (cB + 620)) / len(models)
    cols = [cB + 620 + k * cw for k in range(len(models))]
    L.smallcaps(d, (cA, y0 + 52), "NO.", 28, L.GREY)
    L.smallcaps(d, (cB, y0 + 52), "THE TOKEN", 28, L.GREY)
    for x, m in zip(cols, models):
        L.smallcaps(d, (x, y0 + 52), NICE[m].upper(), 28, L.GREY, track=0.1)
    y = y0 + 80
    d.line([(M, y), (right, y)], fill=L.GREY, width=1)
    rowh = (L.H - 400 - y) / a.n
    for r in sel:
        base = y + rowh * 0.66
        ts.draw(d, (cA, y + rowh * 0.40), f"№ {r['id']}", 32, [(ts.F["mono"], 0)], L.INK)
        hx = " ".join(f"{c:02X}" for c in bytes.fromhex(r["hex"]))
        ts.draw(d, (cA, y + rowh * 0.40 + 40), hx if len(hx) <= 18 else hx[:17] + "…", 24, [(ts.F["mono"], 0)], L.GREY)
        sz = fit(r["str"], rowh * 0.66, 540)
        L.draw_visible(d, (cB, base), r["str"], sz, L.INK)
        for x, m in zip(cols, models):
            rep = T[m][r["id"]][a.field]
            rep = rep.strip("\n") if rep.strip() else rep
            if rep == "":
                rep = "∅"
            s2 = fit(rep, rowh * 0.46, cw - 40)
            if s2 < 22:
                L.draw_visible(d, (x, base), rep, 22, L.MADDER, marker_fill=(200, 150, 140), max_w=cw - 40)
            else:
                L.draw_visible(d, (x, base), rep, s2, L.MADDER, marker_fill=(200, 150, 140))
        y += rowh
        d.line([(M, y), (right, y)], fill=L.PALE, width=1)
    d.line([(M, y), (right, y)], fill=L.INK, width=4)
    notes = ["Tokens fed as ids; chat template, thinking off, greedy decoding, up to 40 tokens; replies set to fit their column (declared), cut with … below 22 px.",
             (f"Order: rows whose four replies differ most first, then {NICE[models[0]]}'s indicator; at most {a.cap} per category. \u2205 = empty reply." if a.order == "varied" else f"Order: {NICE[models[0]]} indicator, most under-trained first; at most {a.cap} per category. \u2205 = empty reply."),
             "Grey marks are transcription: ␣ space, ↵ newline, ⟨0x85⟩ a byte that is not UTF-8 on its own. A boxed word is one of the model's own control tokens."]
    for k, t in enumerate(notes):
        ts.draw(d, (M, y + 70 + 44 * k), t, 31, [(ts.F["text_i"], 0)] + ts.SERIF_CHAIN, L.GREY)
    im.save(a.out, optimize=True)
    print("wrote", a.out, len(sel), "rows of", len(ok))


if __name__ == "__main__":
    main()
