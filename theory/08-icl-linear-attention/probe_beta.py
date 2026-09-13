"""Probe the write strengths beta_t that trained (Gated) DeltaNet layers use on context vs query tokens."""
import json, pathlib, torch
from seqmodels import ICLModel
HERE = pathlib.Path(__file__).parent
torch.manual_seed(0)
X = torch.randn(512, 40, 10); w = torch.randn(512, 10); y = torch.einsum("bnd,bd->bn", X, w)
out = {}
for f in sorted((HERE / "cache").glob("seq_*delta_L*_s0.0_seed0.pt")):
    ck = torch.load(f, map_location="cpu"); a = ck["args"]
    m = ICLModel(a["kind"], a["d"], a["width"], a["heads"], a["layers"]); m.load_state_dict(ck["state"]); m.eval()
    rec = []
    for li, b in enumerate(m.blocks):
        mix = b["mix"]
        def hook(mod, inp, outp, li=li):
            x = inp[0]
            beta = torch.sigmoid(mod.b(x))                        # [B, T, H]
            ent = {"layer": li, "beta_query": beta[:, 0::2].mean((0, 1)).tolist(), "beta_context": beta[:, 1::2].mean((0, 1)).tolist()}
            if mod.kind == "gdelta":
                la = -mod.A_log.exp() * torch.nn.functional.softplus(mod.a(x) + mod.dt_bias)
                ent["alpha_query"] = la[:, 0::2].exp().mean((0, 1)).tolist(); ent["alpha_context"] = la[:, 1::2].exp().mean((0, 1)).tolist()
            rec.append(ent)
        mix.register_forward_hook(hook)
    with torch.no_grad():
        m(X, y)
    out[f.stem] = rec
    print(f.stem, json.dumps(rec))
json.dump(out, open(HERE / "cache" / "beta_probe.json", "w"), indent=1)
