"""workload.py - the scripted phases whose sensor traces Pulse renders. Run through record.py.

Phases (wall-clock timestamps of every boundary go to --phases-out):
  idle | matmul burst (8192^2 bf16) | idle | device copy 2 GiB | idle | Qwen3-0.6B prefill B=64 x 256 |
  idle | Qwen3 decode B=64 | idle | Qwen3 decode B=1 | idle | square wave 1 s on / 1 s off | 0.25 s | idle
With --soak S: idle 60 s, matmul for S seconds, idle 300 s (thermal time constants).
"""
import argparse, json, os, sys, time
import torch

DECODE_MAP = "/home/fzeng/ml/research/art/decode-map"
sys.path.insert(0, DECODE_MAP)


class Phases:
    def __init__(self, path):
        self.path, self.rows = path, []

    def run(self, name, fn, seconds, **meta):
        torch.cuda.synchronize(); t0 = time.time(); n = 0
        while time.time() - t0 < seconds:
            fn(); n += 1
        torch.cuda.synchronize(); t1 = time.time()
        self.rows.append(dict(name=name, t0=t0, t1=t1, iters=n, **meta))
        json.dump(self.rows, open(self.path, "w"), indent=1)
        print(f"{name:18s} {t1 - t0:6.1f}s  {n} iters", flush=True)

    def idle(self, seconds):
        self.run("idle", lambda: time.sleep(0.05), seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases-out", required=True)
    ap.add_argument("--soak", type=int, default=0)
    ap.add_argument("--short", action="store_true", help="smoke test: 5 s phases")
    a = ap.parse_args()
    P = Phases(a.phases_out)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    A = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16); Bm = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    mm = lambda: (A @ Bm, torch.cuda.synchronize())
    if a.soak:
        P.idle(60); P.run("matmul_soak", mm, a.soak, flop_per_iter=2 * 8192 ** 3); P.idle(300)
        return
    s = 5 if a.short else 60
    g = 5 if a.short else 45
    src = torch.empty(1 << 31, dtype=torch.uint8, device=dev); dst = torch.empty_like(src)
    cp = lambda: (dst.copy_(src), torch.cuda.synchronize())
    from qwen import Qwen3
    m = Qwen3()
    ids64 = torch.randint(0, 150000, (64, 256), device=dev); m.alloc(64, 512 + 256)
    with torch.inference_mode():
        P.idle(s)
        P.run("matmul_burst", mm, s, flop_per_iter=2 * 8192 ** 3)
        P.idle(g)
        P.run("copy_2GiB", cp, s, bytes_per_iter=2 * (1 << 31))
        P.idle(g)
        P.run("prefill_B64x256", lambda: (m.forward(ids64, 0), torch.cuda.synchronize()), s, tokens_per_iter=64 * 256)
        P.idle(g)
        state = {"t": 256}
        one = torch.randint(0, 150000, (64, 1), device=dev)

        def dec64():
            m.forward(one, state["t"]); state["t"] += 1
            if state["t"] >= 512 + 250:
                state["t"] = 256
            torch.cuda.synchronize()
        P.run("decode_B64", dec64, s, tokens_per_iter=64)
        P.idle(g)
        m.alloc(1, 512 + 256); m.forward(ids64[:1], 0); state["t"] = 256
        one1 = one[:1]

        def dec1():
            m.forward(one1, state["t"]); state["t"] += 1
            if state["t"] >= 512 + 250:
                state["t"] = 256
            torch.cuda.synchronize()
        P.run("decode_B1", dec1, s, tokens_per_iter=1)
        P.idle(g)
        for period in (2.0, 0.5):
            def sq(period=period):
                t = time.time()
                while time.time() - t < period / 2:
                    mm()
                time.sleep(period / 2)
            P.run(f"square_{period:g}s", sq, s, period_s=period)
        P.idle(2 * g if not a.short else 5)


if __name__ == "__main__":
    main()
