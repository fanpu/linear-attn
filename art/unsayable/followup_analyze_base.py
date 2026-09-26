"""Follow-up (2026-09-26 evening): chat vs -Base on the same tokens.  CPU only.

For Qwen3-0.6B and 1.7B, the -Base checkpoints were run (followup_verify.py) on the union of all
Qwen3 candidate sets (3,033 tokens, which contains each model's own candidate set) and on the chat
model's own random-ordinary-token control.  This compares:
  * the unsayable set (p_max < 0.01, the paper's test) chat vs base, on the chat model's candidates,
    on the union, and on the random control;
  * the replies to "Please repeat the string '<X>'." (chat template and plain completion), in
    particular the chat model's characteristic failure reply.
Writes cache/followup_base_summary.json and cache/followup_<tag>-base_table.json.
"""
import json, os, re, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokens import CACHE
from analyze import bytes_text, said, category, THR, wilson

SPLIT = r"<\|im_end\|>|<\|endoftext\|>|\nUser:"


def base_rows(tag, meta, ind):
    rows = {}
    for s in ["union", "rand"]:
        p = os.path.join(CACHE, f"followup_{tag}-base_{s}_verify.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            ts_ = bytes.fromhex(meta["tokens"][r["id"]]["hex"]).decode("utf-8")
            out = {"id": r["id"], "set": s, "str": ts_, "p_max": r["p_max"], "verified": r["p_max"] < THR,
                   "ind": float(ind["primary"][r["id"]]), "cat": category(ts_)}
            for n in ["raw", "chat"]:
                txt = bytes_text(r[n]["gen"], meta)
                full = txt
                txt = re.split(SPLIT, txt)[0]
                out[n] = txt
                out[n + "_full"] = full
                out["said_" + n] = said(ts_, txt)
            rows[r["id"]] = out
    return rows


def main():
    summ = {}
    for tag in ["qwen3-0.6b", "qwen3-1.7b"]:
        meta = json.load(open(os.path.join(CACHE, f"{tag}_tokens.json")))
        ind = np.load(os.path.join(CACHE, f"{tag}_ind.npz"))
        cand = [int(i) for i in ind["cand"]]
        rand = [int(i) for i in ind["rand"]]
        chat = {}
        for r in json.load(open(os.path.join(CACHE, f"{tag}_table.json"))):
            chat.setdefault(r["id"], r)
        base = base_rows(tag, meta, ind)
        if not base:
            continue
        json.dump(list(base.values()), open(os.path.join(CACHE, f"followup_{tag}-base_table.json"), "w"), ensure_ascii=False)
        m = {}
        for name, ids in [("cand", cand), ("rand", rand), ("union", sorted(set(base) - set(rand)))]:
            ids = [i for i in ids if i in base and i in chat]
            vc = {i for i in ids if chat[i]["verified"]}
            vb = {i for i in ids if base[i]["verified"]}
            m[name] = {"n": len(ids), "chat_unsayable": len(vc), "base_unsayable": len(vb),
                       "chat_ci": wilson(len(vc), len(ids)), "base_ci": wilson(len(vb), len(ids)),
                       "both": len(vc & vb), "chat_only": len(vc - vb), "base_only": len(vb - vc),
                       "jaccard": len(vc & vb) / max(1, len(vc | vb)),
                       "base_failed_to_say_chat": sum(not base[i]["said_chat"] for i in ids),
                       "base_failed_to_say_raw": sum(not base[i]["said_raw"] for i in ids),
                       "chat_failed_to_say_chat": sum(not chat[i]["said_chat"] for i in ids),
                       "chat_failed_to_say_raw": sum(not chat[i]["said_raw"] for i in ids)}
            if name == "cand":
                both = sorted(vc & vb)
                for field in ["chat", "raw"]:
                    cc = Counter(chat[i][field] for i in vc)
                    cb = Counter(base[i][field] for i in vb)
                    top_c, n_c = cc.most_common(1)[0]
                    m[name][f"{field}_reply_chat_top"] = cc.most_common(8)
                    m[name][f"{field}_reply_base_top"] = cb.most_common(8)
                    m[name][f"{field}_chat_top_reply_rate_in_base"] = [sum(base[i][field] == top_c for i in vb), len(vb)]
                    m[name][f"{field}_chat_top_reply_rate_in_chat"] = [n_c, len(vc)]
                    m[name][f"{field}_distinct_replies"] = [len(cc), len(cb)]
                    # same token, same reply?
                    m[name][f"{field}_same_reply_on_both"] = sum(chat[i][field] == base[i][field] for i in both)
                    # random control: does the base model give its own top reply to ordinary tokens?
                    tb, _ = cb.most_common(1)[0]
                    m[name][f"{field}_base_top_reply_on_rand"] = [sum(base[i][field] == tb for i in rand if i in base), len(rand)]
                    m[name][f"{field}_chat_top_reply_on_rand_base"] = [sum(base[i][field] == top_c for i in rand if i in base), len(rand)]
                m[name]["base_chat_continues_past_turn"] = sum("<|im_end|>" not in base[i]["chat_full"] for i in vb)
                m[name]["examples_both"] = [(chat[i]["str"], chat[i]["chat"], base[i]["chat"], base[i]["raw"]) for i in
                                            sorted(both, key=lambda i: chat[i]["ind"])[:30]]
                m[name]["cat_base_only"] = Counter(base[i]["cat"] for i in vb - vc).most_common(6)
                m[name]["cat_chat_only"] = Counter(chat[i]["cat"] for i in vc - vb).most_common(6)
            m[name + "_rows"] = None
        summ[tag] = {k: v for k, v in m.items() if v is not None}
        print("=" * 30, tag)
        for name in ["cand", "union", "rand"]:
            x = m[name]
            print(f"{name:6s} n={x['n']}  unsayable chat {x['chat_unsayable']}  base {x['base_unsayable']}  both {x['both']} "
                  f"jaccard {x['jaccard']:.2f}  | fail-to-say chat-tmpl: chat {x['chat_failed_to_say_chat']} base {x['base_failed_to_say_chat']}"
                  f"  raw: chat {x['chat_failed_to_say_raw']} base {x['base_failed_to_say_raw']}")
        x = m["cand"]
        for field in ["chat", "raw"]:
            print(f"-- {field} replies among unsayable: chat top {x[f'{field}_reply_chat_top'][:4]}")
            print(f"   base top {x[f'{field}_reply_base_top'][:5]}")
            print(f"   chat's top reply given by base: {x[f'{field}_chat_top_reply_rate_in_base']}; distinct replies chat/base "
                  f"{x[f'{field}_distinct_replies']}; same reply on shared unsayable: {x[f'{field}_same_reply_on_both']}; "
                  f"base top reply on base rand: {x[f'{field}_base_top_reply_on_rand']}")
        print("base chat-template replies that never close the turn:", x["base_chat_continues_past_turn"])
        for e in x["examples_both"][:12]:
            print("   ", [s[:50] for s in e])
    json.dump(summ, open(os.path.join(CACHE, "followup_base_summary.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
