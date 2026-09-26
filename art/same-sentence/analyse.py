"""The Same Sentence -- statistics. Reads cache/article1.json, cache/tokens.json.
Writes cache/table.json (one row per text) and cache/summary.json, prints the report numbers."""
import json, os, math, collections
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "cache/article1.json")))
R = json.load(open(os.path.join(HERE, "cache/tokens.json")))
TOKS = list(R)
BYTE_EXACT = [k for k in TOKS if k != "XLM-R"]


def plate_set(texts):
    """Declared selection: every stage>=4 text whose BCP-47 primary subtag is a two-letter
    ISO 639-1 code; one text per (primary subtag, script); ties: the file named exactly by
    its ISO 639-3 code, else the shortest file key, else the alphabetically last (= most recent where keys are years).
    One hand override: Greek uses the modern monotonic text, not the polytonic one."""
    groups = collections.defaultdict(list)
    for t in texts:
        p = t["bcp47"].split("-")[0]
        if len(p) == 2:
            groups[(p, t["script"])].append(t)
    out = []
    for g in groups.values():
        g.sort(key=lambda t: (t["f"] != t["iso3"], len(t["f"])))
        best = [t for t in g if (t["f"] != t["iso3"], len(t["f"])) == (g[0]["f"] != g[0]["iso3"], len(g[0]["f"]))]
        pick = sorted(best, key=lambda t: t["f"])[-1]
        if pick["f"] == "ell_polytonic":
            pick = next(t for t in g if t["f"] == "ell_monotonic")
        out.append(pick)
    return out


def main():
    eng = next(t for t in T if t["f"] == "eng")
    ne = {k: R[k]["rows"]["eng"]["n"] for k in TOKS}
    rows = []
    for t in T:
        r = dict(f=t["f"], name=t["name"], script=t["script"], bcp47=t["bcp47"], dir=t["dir"],
                 chars=t["chars"], bytes=t["bytes"], char_ratio=t["chars"] / eng["chars"],
                 byte_ratio=t["bytes"] / eng["bytes"])
        for k in TOKS:
            rr = R[k]["rows"][t["f"]]
            r[k] = rr["n"]; r[k + "_tax"] = rr["n"] / ne[k]; r[k + "_rt"] = rr["rt"]
            r[k + "_unk"] = rr["unk"]; r[k + "_raw"] = rr["n_raw"]
        rows.append(r)
    ps = {t["f"] for t in plate_set(T)}
    for r in rows:
        r["plate"] = r["f"] in ps
    json.dump(rows, open(os.path.join(HERE, "cache/table.json"), "w"), ensure_ascii=False, indent=0)

    S = {}
    print(f"English: {eng['chars']} chars, {eng['bytes']} bytes; tokens:", ne)
    for subset, rs in [("all", rows), ("plate", [r for r in rows if r["plate"]])]:
        print(f"\n=== {subset}: {len(rs)} texts ===")
        S[subset] = {"n": len(rs)}
        br = np.array([r["byte_ratio"] for r in rs]); cr = np.array([r["char_ratio"] for r in rs])
        print(f"  char ratio  median {np.median(cr):.2f}  p5 {np.percentile(cr,5):.2f} p95 {np.percentile(cr,95):.2f} max {cr.max():.2f}")
        print(f"  byte ratio  median {np.median(br):.2f}  p5 {np.percentile(br,5):.2f} p95 {np.percentile(br,95):.2f} max {br.max():.2f}")
        for k in TOKS:
            tax = np.array([r[k + "_tax"] for r in rs])
            n = np.array([r[k] for r in rs]); b = np.array([r["bytes"] for r in rs])
            tpb = n / b; tpb_e = ne[k] / eng["bytes"]
            comp = tpb / tpb_e  # tokens-per-byte premium relative to English
            lt, lb, lc = np.log(tax), np.log(br), np.log(comp)
            r2 = np.corrcoef(lt, lb)[0, 1] ** 2
            # log tax = log byte_ratio + log comp ; variance share
            vshare_b = np.cov(lt, lb)[0, 1] / lt.var(ddof=1)
            i_max, i_min = int(tax.argmax()), int(tax.argmin())
            S[subset][k] = dict(median=float(np.median(tax)), p5=float(np.percentile(tax, 5)),
                                p95=float(np.percentile(tax, 95)), max=float(tax.max()), min=float(tax.min()),
                                argmax=rs[i_max]["name"], argmin=rs[i_min]["name"], nmax=int(n.max()), nmin=int(n.min()),
                                frac_gt2=float((tax > 2).mean()), frac_gt5=float((tax > 5).mean()),
                                r2_bytes=float(r2), var_share_bytes=float(vshare_b),
                                comp_median=float(np.median(comp)), comp_max=float(comp.max()),
                                unk_texts=int(sum(r[k + "_unk"] > 0 for r in rs)))
            s = S[subset][k]
            print(f"  {k:9s} tax median {s['median']:.2f} [p5 {s['p5']:.2f}, p95 {s['p95']:.2f}] "
                  f"min {s['min']:.2f} ({s['argmin']}) max {s['max']:.2f} ({s['argmax']}, {s['nmax']} tok) "
                  f">2x {s['frac_gt2']:.0%} >5x {s['frac_gt5']:.0%} | R2(log tax~log bytes) {s['r2_bytes']:.2f} "
                  f"byte-share {s['var_share_bytes']:.2f} | tok/byte premium median {s['comp_median']:.2f} max {s['comp_max']:.2f}"
                  + (f" | texts with <unk> {s['unk_texts']}" if s['unk_texts'] else ""))
        # rank changes
        M = np.array([[r[k] for k in TOKS] for r in rs])
        rho = spearmanr(M).correlation
        S[subset]["spearman"] = {f"{a}|{b}": float(rho[i, j]) for i, a in enumerate(TOKS) for j, b in enumerate(TOKS) if i < j}
        print("  spearman GPT-2 vs Qwen3 %.3f, GPT-2 vs BLOOM %.3f, Qwen3 vs XLM-R %.3f, Llama-2 vs Qwen3 %.3f, bytes vs Qwen3 %.3f" % (
            S[subset]["spearman"]["GPT-2|Qwen3"], S[subset]["spearman"]["GPT-2|BLOOM"],
            S[subset]["spearman"]["Qwen3|XLM-R"], S[subset]["spearman"]["Llama-2|Qwen3"],
            spearmanr([r["bytes"] for r in rs], [r["Qwen3"] for r in rs]).correlation))
    # who gains from the bigger vocabulary: GPT-2 -> Qwen3 by script (plate set)
    print("\n=== GPT-2 tokens / Qwen3 tokens, by script (plate set) ===")
    by = collections.defaultdict(list)
    for r in rows:
        if r["plate"]:
            by[r["script"]].append(r["GPT-2"] / r["Qwen3"])
    g = sorted(((np.median(v), s, len(v)) for s, v in by.items()), reverse=True)
    for m, s, n in g:
        print(f"  {s}: {m:.2f} (n={n})")
    S["gain_by_script"] = {s: dict(median=float(m), n=n) for m, s, n in g}
    # NFC effect
    print("\n=== raw (non-NFC) vs NFC token counts, 20 texts ===")
    for t in T:
        if not t["nfc_same"]:
            d = {k: R[k]["rows"][t["f"]]["n_raw"] - R[k]["rows"][t["f"]]["n"] for k in ["GPT-2", "OLMo-2", "BLOOM", "Qwen3"]}
            print("  ", t["name"], d)
    json.dump(S, open(os.path.join(HERE, "cache/summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
