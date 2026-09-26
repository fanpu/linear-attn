"""Tokenizer analysis + weight-based under-trained-token indicators (CPU only).

Follows Land & Bartolo (2024), "Fishing for Magikarp", §2:
  * reference untrained rows t_ref = embedding rows above the tokenizer's vocabulary size;
  * tied embeddings  -> primary indicator C(E_out, u_ref) (cosine distance to the mean ref row);
  * untied embeddings -> primary indicator L2(E_in) (norm of the input embedding row);
  * also computed: C(E~_out, u~_ref) with the first principal component removed (Appendix A);
  * candidates = the most-likely-under-trained 2% of the vocabulary by the primary indicator,
    minus partial-UTF-8 and unreachable tokens (and special tokens).

Writes cache/<tag>_tokens.json and cache/<tag>_ind.npz.
"""
import argparse, glob, json, os, re, sys, unicodedata
import numpy as np, torch
from safetensors import safe_open
from transformers import AutoTokenizer

HUB = os.path.expanduser("~/.cache/huggingface/hub")
MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B", "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B", "qwen3-8b": "Qwen/Qwen3-8B",
    "olmo2-1b": "allenai/OLMo-2-0425-1B",
}
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def snap(repo):
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots")
    return os.path.join(d, sorted(os.listdir(d))[0])


def load_rows(path, key):
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, "pt") as st:
            if key in st.keys():
                return st.get_tensor(key).float()
    return None


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_DEC = {v: k for k, v in bytes_to_unicode().items()}


def token_bytes(tok, i, special):
    if i in special:
        return special[i].encode()
    s = tok.convert_ids_to_tokens(i)
    return bytes(BYTE_DEC[c] for c in s)


def cosdist(A, x):
    return 1 - (A @ x) / (A.norm(dim=1) * x.norm() + 1e-12)


def corpus_counts(tok, max_chars=40_000_000):
    """Token counts on a local mixed corpus (declared proxy for 'common'):
    OpenR1-Math text (English + LaTeX), Python sources (code), /usr/share/dict word lists
    in 8 languages (each word with a leading space)."""
    texts = [open(os.path.join(CACHE, "corpus_math.txt"), encoding="utf-8").read()]
    n = len(texts[0])
    for f in sorted(glob.glob("/usr/lib/python3.12/**/*.py", recursive=True)):
        try:
            t = open(f, encoding="utf-8").read()
        except Exception:
            continue
        texts.append(t); n += len(t)
        if n > max_chars * 0.85: break
    for d in ["american-english", "british-english", "french", "ngerman", "spanish", "italian", "portuguese", "brazilian"]:
        p = f"/usr/share/dict/{d}"
        if os.path.exists(p):
            w = open(p, encoding="utf-8", errors="ignore").read().split("\n")
            texts.append(" " + " ".join(w))
    cnt = np.zeros(len(tok), dtype=np.int64)
    for t in texts:
        for j in range(0, len(t), 200_000):
            ids = tok(t[j:j + 200_000], add_special_tokens=False)["input_ids"]
            np.add.at(cnt, np.asarray(ids), 1)
    return cnt, sum(len(t) for t in texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frac", type=float, default=0.02)
    args = ap.parse_args()
    tag = args.model
    path = snap(MODELS[tag])
    cfg = json.load(open(os.path.join(path, "config.json")))
    tok = AutoTokenizer.from_pretrained(path)
    V = len(tok)
    special = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    E_in = load_rows(path, "model.embed_tokens.weight")
    tied = cfg.get("tie_word_embeddings", False)
    E_out = E_in if tied else load_rows(path, "lm_head.weight")
    Vm = E_in.shape[0]
    ref = np.arange(V, Vm)
    print(tag, "tokenizer", V, "matrix", Vm, "ref rows", len(ref), "tied", tied, flush=True)

    # --- tokenizer analysis
    recs = []
    for i in range(V):
        b = token_bytes(tok, i, special)
        try:
            s = b.decode("utf-8"); partial = False
        except UnicodeDecodeError:
            s = None; partial = True
        if i in special:
            kind = "special"
        elif partial:
            kind = "partial_utf8"
        else:
            enc = tok(s, add_special_tokens=False)["input_ids"]
            kind = "ok" if enc == [i] else "unreachable"
        recs.append({"id": i, "hex": b.hex(), "kind": kind})
    kinds = np.array([r["kind"] for r in recs])
    print({k: int((kinds == k).sum()) for k in set(kinds)}, flush=True)

    # --- indicators
    u = E_out[ref].mean(0)
    C = cosdist(E_out, u).numpy()
    Ec = E_out - E_out.mean(0, keepdim=True)
    U1 = torch.linalg.svd(Ec[:V], full_matrices=False)[2][0]   # 1st PC over real vocab rows
    Et = E_out - (E_out @ U1)[:, None] * U1[None]
    Ct = cosdist(Et, Et[ref].mean(0)).numpy()
    L2in = E_in.norm(dim=1).numpy()
    L2out_ref = (E_out - u).norm(dim=1).numpy()
    primary = C if tied else L2in                 # low = more under-trained, for both
    pname = "C(E_out,u_ref)" if tied else "L2(E_in)"

    real = np.array([k != "special" for k in kinds])
    order = np.argsort(primary[:V], kind="stable")
    order = order[real[order]]
    ntop = int(round(args.frac * real.sum()))
    top = order[:ntop]
    cand = np.array([i for i in top if kinds[i] == "ok"])
    rank = np.empty(V, dtype=np.int64); rank[order] = np.arange(len(order)); rank[~real] = -1
    print(f"top {args.frac:.0%}: {ntop}; excl partial {sum(kinds[top]=='partial_utf8')}, "
          f"unreachable {sum(kinds[top]=='unreachable')} -> {len(cand)} candidates", flush=True)

    # --- null: equal-size random sample of ordinary tokens outside the candidate top-2%
    rng = np.random.default_rng(0)
    pool = np.array([i for i in range(V) if kinds[i] == "ok" and rank[i] >= ntop])
    rand = np.sort(rng.choice(pool, size=len(cand), replace=False))

    cnt, nchar = corpus_counts(tok)
    np.savez(os.path.join(CACHE, f"{tag}_ind.npz"), C=C, Ct=Ct, L2in=L2in, L2out_ref=L2out_ref,
             primary=primary, rank=rank, cand=cand, rand=rand, top=top, ref=ref, counts=cnt)
    meta = {"model": tag, "repo": MODELS[tag], "V": V, "Vm": Vm, "tied": tied, "primary": pname,
            "n_ref": len(ref), "ntop": ntop, "n_cand": len(cand), "n_rand": len(rand),
            "kinds": {k: int((kinds == k).sum()) for k in set(kinds)},
            "corpus_chars": nchar, "tokens": recs}
    json.dump(meta, open(os.path.join(CACHE, f"{tag}_tokens.json"), "w"))
    # quick peek
    for i in cand[:40]:
        print(i, round(float(primary[i]), 4), repr(bytes.fromhex(recs[i]["hex"]).decode()), cnt[i])


if __name__ == "__main__":
    main()
