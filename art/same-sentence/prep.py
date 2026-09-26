"""The Same Sentence -- data.

Article 1 of the UDHR in every stage>=4 translation of the UDHR-in-XML corpus
(eric-muller/udhr, formerly Unicode's "UDHR in Unicode"), tokenised by eight tokenizers.
Per text and tokenizer: token count, per-token byte spans (exact, from the token bytes),
exact round-trip check. Writes cache/article1.json and cache/tokens.json.

CPU only.
"""
import glob, json, os, re, unicodedata
import xml.etree.ElementTree as ET
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
UD = os.path.join(HERE, "cache/udhr-main/data/udhr")
NS = "{http://efele.net/udhr}"
HUB = os.path.expanduser("~/.cache/huggingface/hub")


def snap(repo, fn="tokenizer.json"):
    g = glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*/{fn}")
    assert g, repo
    return g[0]


# name, path, family, vocab note
TOKS = [
    ("GPT-2", os.path.join(HERE, "cache/tok/openai-community__gpt2/tokenizer.json"), "bytebpe"),
    ("GPT-NeoX", snap("fla-hub/rwkv7-168M-pile"), "bytebpe"),
    ("Llama-2", snap("fla-hub/gla-1.3B-100B"), "sp_bytefallback"),
    ("OLMo-2", snap("allenai/OLMo-2-0425-1B"), "bytebpe"),
    ("Qwen2.5", snap("Qwen/Qwen2.5-7B-Instruct"), "bytebpe"),
    ("Qwen3", snap("Qwen/Qwen3-0.6B-Base"), "bytebpe"),
    ("BLOOM", os.path.join(HERE, "cache/tok/bigscience__bloom-560m/tokenizer.json"), "bytebpe"),
    ("XLM-R", os.path.join(HERE, "cache/tok/FacebookAI__xlm-roberta-base/tokenizer.json"), "unigram"),
]


def bytes_to_unicode():  # GPT-2's byte<->printable-unicode table
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


U2B = bytes_to_unicode()


def load_texts():
    idx = ET.parse(os.path.join(UD, "index.xml")).getroot()
    out = []
    for e in idx:
        a = e.attrib
        if a.get("stage") not in ("4", "5"):
            continue
        fn = os.path.join(UD, f"udhr_{a['f']}.xml")
        if not os.path.exists(fn):
            continue
        root = ET.parse(fn).getroot()
        art = [x for x in root.iter(NS + "article") if x.attrib.get("number") == "1"]
        if not art:
            continue
        paras = ["".join(p.itertext()) for p in art[0].iter(NS + "para")]
        # declared: whitespace runs (XML line breaks / indentation) collapsed to one space;
        # multiple paragraphs joined by one space; then NFC (two of the tokenizers apply NFC
        # themselves; 20 of the 527 source texts are not in NFC -- raw form kept as text_raw).
        raw = " ".join(re.sub(r"\s+", " ", p).strip() for p in paras if p.strip())
        txt = unicodedata.normalize("NFC", raw)
        if len(txt) < 20:
            continue
        out.append(dict(f=a["f"], iso3=a["iso639-3"], script=a["iso15924"], bcp47=a["bcp47"],
                        dir=a["dir"], stage=int(a["stage"]), name=a["n"], text=txt, text_raw=raw,
                        chars=len(txt), bytes=len(txt.encode("utf-8")),
                        nfc_same=raw == txt))
    return out


def token_bytes(tok, fam, enc, text):
    """Exact byte span per token (list of (start, end) into text.encode('utf-8'))."""
    tb = text.encode("utf-8")
    if fam == "bytebpe":
        pieces = [bytes(U2B[c] for c in t) for t in enc.tokens]
        assert b"".join(pieces) == tb, "byte-level pieces do not concatenate to the text"
    elif fam == "sp_bytefallback":
        pieces = []
        for t in enc.tokens:
            m = re.fullmatch(r"<0x([0-9A-F]{2})>", t)
            pieces.append(bytes([int(m.group(1), 16)]) if m else t.replace("▁", " ").encode("utf-8"))
        cat = b"".join(pieces)
        # SentencePiece dummy prefix: a leading space that is not in the text
        if cat[:1] == b" " and tb[:1] != b" ":
            pieces[0] = pieces[0][1:]
            cat = cat[1:]
        assert cat == tb, "SP pieces do not concatenate to the text"
    else:  # unigram with normaliser: use tokenizer-reported char offsets (approximate)
        cb = [len(text[:i].encode("utf-8")) for i in range(len(text) + 1)]
        spans = [(cb[s], cb[e]) for s, e in enc.offsets]
        return spans
    spans, p = [], 0
    for pc in pieces:
        spans.append((p, p + len(pc))); p += len(pc)
    return spans


def main():
    texts = load_texts()
    print(len(texts), "stage>=4 texts with Article 1")
    json.dump(texts, open(os.path.join(HERE, "cache/article1.json"), "w"), ensure_ascii=False, indent=0)
    res = {}
    for name, path, fam in TOKS:
        tok = Tokenizer.from_file(path)
        vs = tok.get_vocab_size(with_added_tokens=False)
        unk_id = None
        if fam == "unigram":
            unk_id = tok.token_to_id("<unk>")
        rows = {}
        nrt = 0
        for t in texts:
            enc = tok.encode(t["text"], add_special_tokens=False)
            dec = tok.decode(enc.ids, skip_special_tokens=False)
            rt = dec == t["text"]
            nrt += (not rt)
            spans = token_bytes(tok, fam, enc, t["text"])
            rows[t["f"]] = dict(n=len(enc.ids), rt=rt, spans=spans,
                                unk=sum(i == unk_id for i in enc.ids) if unk_id is not None else 0,
                                bytefb=sum(bool(re.fullmatch(r"<0x[0-9A-F]{2}>", s)) for s in enc.tokens),
                                n_raw=len(tok.encode(t["text_raw"], add_special_tokens=False).ids))
        res[name] = dict(vocab=vs, family=fam, path=path, rows=rows)
        print(f"{name:9s} vocab={vs:7d} round-trip failures={nrt}")
    json.dump(res, open(os.path.join(HERE, "cache/tokens.json"), "w"))


if __name__ == "__main__":
    main()
