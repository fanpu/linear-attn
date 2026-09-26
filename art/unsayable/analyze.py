"""Counts, null controls, categories and cross-model overlap.  CPU only.

Writes cache/summary.json and cache/<model>_table.json (one row per tested token).
"""
import json, math, os, re, sys, unicodedata
from collections import Counter
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokens import MODELS, snap, CACHE

THR = 0.01          # paper's verification threshold on max target probability
COMMON = 5          # "common" = >= 5 occurrences in the local mixed corpus (declared)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def script_of(ch):
    try:
        nm = unicodedata.name(ch)
    except ValueError:
        return "UNNAMED"
    w = nm.split()
    if w[0] == "CJK" or nm.startswith("IDEOGRAPHIC"):
        return "HAN"
    if w[0] in ("HANGUL",):
        return "HANGUL"
    if w[0] in ("HIRAGANA", "KATAKANA"):
        return "KANA"
    if w[0] in ("ARABIC", "HEBREW", "THAI", "CYRILLIC", "GREEK", "LATIN", "DEVANAGARI", "BENGALI",
                "TAMIL", "GEORGIAN", "ARMENIAN", "COPTIC", "GLAGOLITIC", "CANADIAN", "ETHIOPIC",
                "MONGOLIAN", "TIBETAN", "KHMER", "LAO", "MYANMAR", "SINHALA", "GUJARATI", "TELUGU",
                "KANNADA", "MALAYALAM", "GURMUKHI", "ORIYA", "SYRIAC", "THAANA", "CHEROKEE", "RUNIC",
                "OGHAM", "BOPOMOFO", "YI", "TIFINAGH", "VAI", "BAMUM", "JAVANESE", "BALINESE",
                "LIMBU", "BUGINESE", "BRAILLE"):
        return w[0]
    if w[0] == "DIGIT":
        return "DIGIT"
    return "SYMBOL"


def category(s):
    """Declared heuristic categoriser (first matching rule wins)."""
    t = s.strip()
    if not t:
        return "whitespace"
    if all(ord(c) < 128 for c in t):
        if re.search(r"[_(){}\[\];=<>$@#\\/]|^[.]\w|[a-z][A-Z]", t) or re.fullmatch(r"[A-Z][a-z]+[A-Z]\w*", t):
            return "code identifier"
        if re.fullmatch(r"[A-Za-z]+", t):
            return "ASCII word"
        if re.fullmatch(r"[0-9]+", t):
            return "digits"
        return "ASCII punctuation"
    scripts = Counter(script_of(c) for c in t if not c.isspace())
    sc = scripts.most_common(1)[0][0]
    if sc == "LATIN":
        return "Latin with diacritics"
    if sc == "SYMBOL":
        if any(unicodedata.category(c) == "So" for c in t):
            return "symbol / emoji"
        return "punctuation / mark"
    return {"HAN": "Han ideograph", "HANGUL": "Hangul syllable", "KANA": "Kana"}.get(sc, sc.title() + " script")


def bytes_text(ids, meta):
    """Decode generated ids via their exact bytes; undecodable bytes become \u27e8 0xE2 \u27e9 (declared)."""
    toks = meta["tokens"]
    # special (control) tokens are wrapped in U+E000 ... U+E001 so the renderer can mark them
    b = b"".join((("\ue000".encode() + bytes.fromhex(toks[i]["hex"]) + "\ue001".encode())
                  if toks[i]["kind"] == "special" else bytes.fromhex(toks[i]["hex"]))
                 if i < len(toks) else b"" for i in ids)
    out, i = [], 0
    while i < len(b):
        for L in (1, 2, 3, 4):
            try:
                out.append(b[i:i + L].decode("utf-8")); i += L; break
            except UnicodeDecodeError:
                continue
        else:
            out.append("\u27e80x%02X\u27e9" % b[i]); i += 1
    return "".join(out)


def said(tokstr, reply):
    t = tokstr.strip()
    return bool(t) and t in reply.replace("\ue000", "").replace("\ue001", "")


def load_model(tag):
    meta = json.load(open(os.path.join(CACHE, f"{tag}_tokens.json")))
    ind = np.load(os.path.join(CACHE, f"{tag}_ind.npz"))
    tok = AutoTokenizer.from_pretrained(snap(MODELS[tag]))
    rows = {}
    for s in ["cand", "rand", "union", "altc"]:
        p = os.path.join(CACHE, f"{tag}_{s}_verify.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            b = bytes.fromhex(meta["tokens"][r["id"]]["hex"])
            ts = b.decode("utf-8")
            out = {"id": r["id"], "set": s, "hex": b.hex(), "str": ts, "p_max": r["p_max"],
                   "p": {n: max(r[n]["p"]) for n in ["p1", "p2", "p3"]},
                   "ind": float(ind["primary"][r["id"]]), "rank": int(ind["rank"][r["id"]]),
                   "count": int(ind["counts"][r["id"]]), "cat": category(ts)}
            for n in ["raw", "chat"]:
                if n in r:
                    txt = bytes_text(r[n]["gen"], meta)
                    txt = re.split(r"\ue000<\|im_end\|>\ue001|\ue000<\|endoftext\|>\ue001|\nUser:", txt)[0]
                    out[n] = txt
                    out["said_" + n] = said(ts, txt)
            out["p1_greedy"] = tok.decode(r["p1"]["gen"])
            out["verified"] = r["p_max"] < THR
            rows[(s, r["id"])] = out
    return meta, ind, list(rows.values())


def main():
    summ = {"threshold": THR, "common_min_count": COMMON, "models": {}}
    verified_sets, cand_sets = {}, {}
    for tag in MODELS:
        if not os.path.exists(os.path.join(CACHE, f"{tag}_tokens.json")):
            continue
        meta, ind, rows = load_model(tag)
        if not rows:
            continue
        json.dump(rows, open(os.path.join(CACHE, f"{tag}_table.json"), "w"), ensure_ascii=False)
        m = {"primary": meta["primary"], "V": meta["V"], "kinds": meta["kinds"], "ntop": meta["ntop"],
             "n_cand": meta["n_cand"], "n_ref_rows": meta["n_ref"]}
        for s in ["cand", "rand"]:
            R = [r for r in rows if r["set"] == s]
            if not R:
                continue
            k = sum(r["verified"] for r in R)
            d = {"n": len(R), "verified": k, "rate_ci": wilson(k, len(R))}
            for n in ["raw", "chat"]:
                if n in R[0]:
                    ks = sum(not r["said_" + n] for r in R)
                    d["failed_to_say_" + n] = ks
                    d["failed_to_say_" + n + "_ci"] = wilson(ks, len(R))
            d["categories"] = Counter(r["cat"] for r in R if (r["verified"] or s == "rand")).most_common()
            if s == "cand":
                C = [r for r in R if r["count"] >= COMMON]
                kc = sum(r["verified"] for r in C)
                d["common_in_cand"] = {"n": len(C), "verified": kc, "rate_ci": wilson(kc, len(C)),
                                       "examples": [(r["str"], r["count"], round(r["p_max"], 4)) for r in
                                                    sorted(C, key=lambda r: -r["count"])[:25]]}
                verified_sets[tag] = {r["id"] for r in R if r["verified"]}
                cand_sets[tag] = {r["id"] for r in R}
                # verified-by-category rates
                cc = Counter(r["cat"] for r in R); cv = Counter(r["cat"] for r in R if r["verified"])
                d["cat_rates"] = {c: [cv[c], cc[c]] for c in cc}
            m[s] = d
        # alternative indicator for untied models: top 2% by C(E_out, u_ref)
        if os.path.exists(os.path.join(CACHE, f"{tag}_altc_verify.jsonl")):
            kinds = [t["kind"] for t in meta["tokens"]]
            real = [i for i in range(len(kinds)) if kinds[i] != "special"]
            order = sorted(real, key=lambda i: float(ind["C"][i]))[:int(ind["top"].shape[0])]
            ids = [i for i in order if kinds[i] == "ok"]
            st = {r["id"]: r["verified"] for r in rows}
            k = sum(st.get(i, False) for i in ids)
            m["altc"] = {"n": len(ids), "tested": sum(i in st for i in ids), "verified": k, "rate_ci": wilson(k, len(ids)),
                         "unreachable_in_top": sum(kinds[i] == "unreachable" for i in order)}
            print(tag, "altc", m["altc"])
        summ["models"][tag] = m
        print(tag, json.dumps({s: {k: v for k, v in m[s].items() if k in ("n", "verified", "rate_ci", "failed_to_say_raw", "failed_to_say_chat")} for s in ["cand", "rand"] if s in m}))
    # overlap across Qwen3 sizes
    q = [t for t in verified_sets if t.startswith("qwen3")]
    ov = {}
    for a in q:
        for b in q:
            if a < b:
                A, B = verified_sets[a], verified_sets[b]
                both_tested = cand_sets[a] & cand_sets[b]
                ov[f"{a}|{b}"] = {"|A|": len(A), "|B|": len(B), "inter": len(A & B), "jaccard": len(A & B) / max(1, len(A | B)),
                                  "cand_jaccard": len(cand_sets[a] & cand_sets[b]) / max(1, len(cand_sets[a] | cand_sets[b]))}
    # equal-footing overlap: every Qwen3 model tested on the union U of all Qwen3 candidate sets
    U = set().union(*[set(int(i) for i in np.load(os.path.join(CACHE, f"{t}_ind.npz"))["cand"]) for t in q]) if q else set()
    status = {}
    for t in q:
        st = {}
        for s_ in ["cand", "rand", "union"]:
            p_ = os.path.join(CACHE, f"{t}_{s_}_verify.jsonl")
            if os.path.exists(p_):
                for line in open(p_):
                    r = json.loads(line); st[r["id"]] = r["p_max"] < THR
        status[t] = st
    if q and all(len(U & set(status[t])) == len(U) for t in q):
        VU = {t: {i for i in U if status[t][i]} for t in q}
        ov["union"] = {"n_union": len(U), "verified": {t: len(VU[t]) for t in q}}
        for a_ in q:
            for b_ in q:
                if a_ < b_:
                    A, B = VU[a_], VU[b_]
                    ov["union"][f"{a_}|{b_}"] = {"inter": len(A & B), "jaccard": round(len(A & B) / max(1, len(A | B)), 3)}
        k_of = Counter(sum(i in VU[t] for t in q) for i in U)
        ov["union"]["unsayable_in_k_models"] = dict(sorted(k_of.items()))
        ov["union"]["all_models"] = len(set.intersection(*VU.values()))
        json.dump(sorted(set.intersection(*VU.values())), open(os.path.join(CACHE, "unsayable_all_qwen.json"), "w"))
    if q:
        allv = set.intersection(*[verified_sets[t] for t in q])
        ov["all_qwen_verified_intersection"] = len(allv)
        small = [t for t in q if t != "qwen3-8b"]
        if len(small) > 1:
            ov["tied_0.6_1.7_4_intersection"] = len(set.intersection(*[verified_sets[t] for t in small]))
    summ["overlap"] = ov
    json.dump(summ, open(os.path.join(CACHE, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(ov, indent=1))


if __name__ == "__main__":
    main()
