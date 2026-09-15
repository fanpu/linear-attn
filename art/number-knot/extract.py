"""Forward passes for Number Knot M1 (one queued GPU job).

Records the residual stream at the target token (last position) after the embedding and after
every decoder block (forward hooks on the blocks, so the last entry is BEFORE the final norm).
Stored float16 in cache/hs/<name>.npy, one file per (model, set, template): shape (N, L+1, D).
A rerun skips files that already exist (checkpoint per chunk).

  python extract.py --device cuda            # full run (via art/_shared/gpu1.sh)
  python extract.py --device cpu --tiny      # 20 numbers, 2 blocks, into cache/tiny/
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common as C


def load(name, device, dtype, n_blocks=None):
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, dtype=dtype)
    if n_blocks is not None:
        model.model.layers = model.model.layers[:n_blocks]
        model.config.num_hidden_layers = n_blocks
    return tok, model.to(device).eval()


@torch.no_grad()
def residuals(model, prefix, targets, device, batch):
    """(N, L+1, D) float16 residual stream at the last position of prefix + [target]."""
    grabbed = []

    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        grabbed.append(h[:, -1].float().cpu())

    hs = [model.model.embed_tokens.register_forward_hook(hook)]
    hs += [blk.register_forward_hook(hook) for blk in model.model.layers]
    out = []
    try:
        for s in range(0, len(targets), batch):
            tg = targets[s:s + batch]
            ids = torch.tensor([prefix + [t] for t in tg], device=device)
            grabbed.clear()
            model(input_ids=ids, use_cache=False)
            assert len(grabbed) == len(model.model.layers) + 1
            out.append(torch.stack(grabbed, 1))
    finally:
        for h in hs:
            h.remove()
    x = torch.cat(out).numpy()
    amax = float(np.abs(x).max())
    assert np.isfinite(x).all() and amax < 6e4, amax
    return x.astype(np.float16), amax


def run_set(model, tok, name, prefixes, targets, outdir, device, batch, log):
    for k, pref in enumerate(prefixes):
        f = outdir / f"{name}_t{k}.npy"
        if f.exists():
            print(f"skip {f.name}", flush=True)
            continue
        t0 = time.time()
        x, amax = residuals(model, pref, targets, device, batch)
        np.save(f.with_suffix(".tmp.npy"), x)
        f.with_suffix(".tmp.npy").rename(f)
        dt = time.time() - t0
        log[f.name] = dict(shape=list(x.shape), prefix=tok.convert_ids_to_tokens(pref), absmax=amax, seconds=dt)
        print(f"{f.name} {x.shape} absmax={amax:.1f} {dt:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.10)
    dtype = torch.float32
    n_num = 20 if args.tiny else C.N_NUMBERS
    n_blocks = 2 if args.tiny else None
    outdir = (C.CACHE / "tiny") if args.tiny else (C.CACHE / "hs")
    outdir.mkdir(parents=True, exist_ok=True)
    meta_f = outdir / "meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    meta.update(dtype="float32 compute, float16 storage", device=str(dev),
                torch=torch.__version__, date=time.strftime("%F"))
    log = meta.setdefault("files", {})
    t_all = time.time()

    tok, model = load(C.OLMO, dev, dtype, n_blocks)
    prefixes, num_ids = C.build_number_prompts(tok, tok.bos_token_id, n_num)
    rnd_ids, n_cand = C.random_token_ids(tok, n_num, seed=0, exclude=set(num_ids))
    meta["olmo"] = dict(model=C.OLMO, templates=C.NUMBER_TEMPLATES, bos_id=tok.bos_token_id,
                        number_ids=num_ids, random_ids=rnd_ids,
                        random_tokens=tok.convert_ids_to_tokens(rnd_ids), random_candidates=n_cand,
                        n_layers_plus_emb=len(model.model.layers) + 1)
    run_set(model, tok, "olmo_numbers", prefixes, num_ids, outdir, dev, args.batch, log)
    run_set(model, tok, "olmo_random", prefixes, rnd_ids, outdir, dev, args.batch, log)
    meta_f.write_text(json.dumps(meta, indent=1))
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    tok, model = load(C.QWEN, dev, dtype, n_blocks)
    for name, T, W in [("qwen_days", C.DAY_TEMPLATES, C.DAYS), ("qwen_months", C.MONTH_TEMPLATES, C.MONTHS)]:
        prefixes, tg = C.word_prompts(tok, T, W)
        meta[name] = dict(model=C.QWEN, templates=T, words=W, target_ids=tg,
                          n_layers_plus_emb=len(model.model.layers) + 1)
        run_set(model, tok, name, prefixes, tg, outdir, dev, args.batch, log)
    meta["wall_seconds_this_run"] = time.time() - t_all
    meta_f.write_text(json.dumps(meta, indent=1))
    print("done", time.time() - t_all, flush=True)


if __name__ == "__main__":
    main()
