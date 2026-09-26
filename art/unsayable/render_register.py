"""Plate: the register.  One row per verified-unsayable token: id, bytes, indicator; the token at
display size in ink; what the model said when asked to repeat it, in madder.

Selection (declared): tokens verified (p_max < 0.01) in every model listed in --models, ordered by
the --reply model's indicator (most under-trained first), at most --cap per category."""
import argparse, json, os, textwrap
from collections import Counter
import look as L
import typeset as ts

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
NICE = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B",
        "qwen3-8b": "Qwen3-8B", "olmo2-1b": "OLMo-2-0425-1B"}


def load(tag):
    return {r["id"]: r for r in json.load(open(os.path.join(CACHE, f"{tag}_table.json"))) if r["set"] in ("cand", "union", "altc", "rand")}


def fmt_p(p):
    if p < 1e-4:
        return f"{p:.0e}".replace("e-0", "e-")
    return f"{p:.4f}"


def fit(s, size, maxw, chain=None):
    w = L.vis_width(s, size, chain)
    return size if w <= maxw else size * maxw / w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-0.6b,qwen3-1.7b,qwen3-4b,qwen3-8b")
    ap.add_argument("--reply", default="qwen3-0.6b")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--skip", type=int, default=0, help="skip the first k selected (for a second plate)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Unsayable")
    ap.add_argument("--plain", action="store_true", help="show the plain-completion transcript as the main reply")
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if os.path.exists(os.path.join(CACHE, f"{m}_table.json"))]
    T = {m: load(m) for m in models}
    R = T[a.reply]
    ok = set.intersection(*[{i for i, r in T[m].items() if r["verified"]} for m in models])
    rows = sorted((R[i] for i in ok), key=lambda r: r["ind"])
    sel, cc = [], Counter()
    for r in rows:
        if cc[r["cat"]] < a.cap:
            sel.append(r); cc[r["cat"]] += 1
    sel = sel[a.skip:a.skip + a.n]

    meta = json.load(open(os.path.join(CACHE, f"{a.reply}_tokens.json")))
    tied = meta["tied"]
    ilab = "C" if tied else "|e|"
    ides = (f"C = cosine distance of the token's output row to the mean of the {meta['n_ref']} rows beyond the tokenizer's vocabulary"
            if tied else "|e| = norm of the token's input-embedding row (untied embeddings)")
    im, d = L.sheet()
    M = 230
    # --- header
    L.smallcaps(d, (M, 330), a.title.upper(), 118, L.INK, track=0.16)
    ts.draw(d, (M, 430), "words a model owns and cannot say", 64, [(ts.F["text_i"], 0)], L.INK)
    right = L.W - M
    hdr = [f"{NICE[a.reply]}, asked to repeat each token, replied in madder",
           f"every token here verified unsayable (p < 0.01) in {len(models)} of {len(models)} models:",
           " · ".join(NICE[m] for m in models)]
    for k, t in enumerate(hdr):
        f = ts.font(ts.F["text_i"] if k < 2 else ts.F["text"], 40)
        d.text((right, 330 + 56 * k), t, font=f, fill=L.GREY, anchor="rs")
    y0 = 560
    d.line([(M, y0), (right, y0)], fill=L.INK, width=4)
    cA, cB, cC = M, M + 560, M + 1640
    for x, t in [(cA, "NO. · BYTES · INDICATOR"), (cB, "THE TOKEN"),
                 (cC, "WHAT IT SAID  (PLAIN COMPLETION)" if a.plain else "WHAT IT SAID  (CHAT)  ·  PLAIN COMPLETION BELOW")]:
        L.smallcaps(d, (x, y0 + 52), t, 28, L.GREY, track=0.12)
    y = y0 + 80
    d.line([(M, y), (right, y)], fill=L.GREY, width=1)
    rowh = (L.H - 420 - y) / a.n
    mono = ts.MONO_CHAIN
    for r in sel:
        base = y + rowh * 0.62
        # column A: apparatus
        ts.draw(d, (cA, y + 62), f"№ {r['id']}", 38, [(ts.F["mono"], 0)], L.INK)
        b = bytes.fromhex(r["hex"])
        hx = " ".join(f"{c:02X}" for c in b)
        hx = textwrap.wrap(hx, 30)
        for k, line in enumerate(hx[:2]):
            ts.draw(d, (cA, y + 108 + 36 * k), line + ("…" if k == 1 and len(hx) > 2 else ""), 27, [(ts.F["mono"], 0)], L.GREY)
        nm = L.uname(r["str"]) or f"{len(r['str'].strip())} chars · {r['cat']}"
        nm = textwrap.wrap(nm, 34)
        yy = y + 108 + 36 * min(2, len(hx)) + 4
        for k, line in enumerate(nm[:2]):
            ts.draw(d, (cA, yy + 32 * k), line.upper() if L.uname(r["str"]) else line, 24, [(ts.F["serif"], 0)], L.GREY)
        yy += 32 * min(2, len(nm)) + 6
        ts.draw(d, (cA, yy), f"{ilab} {r['ind']:.3f}   p {fmt_p(r['p_max'])}", 26, [(ts.F["mono"], 0)], L.GREY)
        # column B: the token
        sz = fit(r["str"], rowh * 0.62, cC - cB - 80)
        L.draw_visible(d, (cB, base), r["str"], sz, L.INK)
        # column C: what it said
        main = r["raw"] if a.plain else r["chat"]
        main = main.strip("\n") if main.strip() else main
        if main == "":
            main = "∅"
        sz2 = fit(main, rowh * 0.42, right - cC)
        L.draw_visible(d, (cC, y + rowh * 0.50), main, max(sz2, 30), L.MADDER, marker_fill=(200, 150, 140),
                       max_w=right - cC)
        if not a.plain and "raw" in r:
            L.draw_visible(d, (cC, y + rowh * 0.50 + 60), r["raw"], 30, L.MADDER,
                           marker_fill=(200, 150, 140), max_w=right - cC)
        y += rowh
        d.line([(M, y), (right, y)], fill=L.PALE, width=1)
    d.line([(M, y), (right, y)], fill=L.INK, width=4)
    notes = [(f"Tokens are fed as token ids, never as text. Chat: \u201cPlease repeat the string '\u2039token\u203a'.\u201d under the {NICE[a.reply]} chat template, thinking off, greedy."
              if not a.plain else f"Tokens are fed as token ids, never as text. {NICE[a.reply]} is a base model with no chat template: the request is a bare completion."),
             "Plain: the same request as a bare completion (“User: …\\nAssistant:”), greedy, 40 tokens, cut here to one line.",
             ides + " (Land & Bartolo 2024); p = the highest probability the model gives the token across three repetition prompts.",
             "Grey marks are transcription, not text: ␣ space, ↵ newline, … cut; a boxed word is a control token. Order: most under-trained first, at most %d per category." % a.cap]
    for k, t in enumerate(notes):
        ts.draw(d, (M, y + 70 + 44 * k), t, 31, [(ts.F["text_i"], 0)] + ts.SERIF_CHAIN, L.GREY)
    im.save(a.out, optimize=True)
    print("wrote", a.out, len(sel), "rows")


if __name__ == "__main__":
    main()
