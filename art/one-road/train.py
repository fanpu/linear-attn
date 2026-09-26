"""Train one network and record its predictive distribution on the fixed probe sets.

Output cache/runs/<task>/<name>.npz:
  steps           : checkpoint steps (0 = untrained init), log-spaced
  lp_tr, lp_te    : log-softmax on the train / test probe sets, shape (K, 1000, C)
  also the full-test accuracy at the end, and init/final weights in cache/weights/.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
import pasar_job
from models import build

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--arch", required=True)
ap.add_argument("--opt", default="sgd")          # sgd | adam | adamw
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--labels", default="true")      # true | shuf
ap.add_argument("--epochs", type=int, default=40)   # images
ap.add_argument("--steps", type=int, default=20000)  # modadd (full batch)
ap.add_argument("--K", type=int, default=90)
ap.add_argument("--bs", type=int, default=125)  # divides 10,000: fixed shapes for the CUDA graph
a = ap.parse_args()

pasar_job.apply_memory_limit()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed); np.random.seed(a.seed)

d = np.load(os.path.join(ROOT, "cache", f"data_{a.task}.npz"))
C = int(d["C"])
img = a.task != "modadd"
cast = (lambda z: torch.tensor(z, dtype=torch.float32, device=dev)) if img else (lambda z: torch.tensor(z, device=dev))
xtr, xte = cast(d["xtr"]), cast(d["xte"])
ytr = torch.tensor(d["ytr_shuf"] if a.labels == "shuf" else d["ytr"], device=dev)
ptr, pte = torch.tensor(d["ptr"], device=dev), torch.tensor(d["pte"], device=dev)
yte = torch.tensor(d["yte"], device=dev)

model = build(a.task, a.arch, C, tuple(xtr.shape[1:]) if img else None).to(dev)
init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
lr = a.lr if a.lr is not None else {"sgd": 0.02, "adam": 1e-3, "adamw": 1e-3}[a.opt]
if a.opt == "sgd":
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=a.wd)
elif a.opt == "adam":
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=a.wd, capturable=dev == "cuda")
else:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=a.wd, betas=(0.9, 0.98), capturable=dev == "cuda")

N = len(ytr)
spe = (N + a.bs - 1) // a.bs if img else 1
T = a.epochs * spe if img else a.steps
ck = np.unique(np.concatenate([[0], np.round(np.geomspace(1, T, a.K)).astype(int)]))
ckset = set(ck.tolist())


@torch.no_grad()
def probe():
    model.eval()
    outs = []
    for X in (xtr[ptr], xte[pte]):
        lp = torch.cat([F.log_softmax(model(X[i:i + 500]).double(), -1) for i in range(0, len(X), 500)])
        outs.append(lp.float().cpu().numpy())
    model.train()
    return outs


@torch.no_grad()
def full_test_acc():
    model.eval()
    acc = sum((model(xte[i:i + 1000]).argmax(-1) == yte[i:i + 1000]).sum().item() for i in range(0, len(yte), 1000))
    model.train()
    return acc / len(yte)


LTR, LTE, t0 = [], [], time.time()
g = torch.Generator(device=dev); g.manual_seed(1000 + a.seed)
bs = a.bs if img else N
static_x = xtr[:bs].clone(); static_y = ytr[:bs].clone()


def train_step():
    loss = F.cross_entropy(model(static_x), static_y)
    loss.backward()
    if a.arch == "gru":
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss


def load_batch(step):
    global order
    if not img:
        return
    if step % spe == 0:
        order = torch.randperm(N, device=dev, generator=g)
    idx = order[(step % spe) * bs:(step % spe + 1) * bs]
    static_x.copy_(xtr[idx]); static_y.copy_(ytr[idx])


# eager warm-up steps on a side stream, then the whole train step is captured as one CUDA graph
# (same arithmetic, far fewer kernel launches on a shared GPU)
use_graph = dev == "cuda"
WARM = 3
graph = None
side = torch.cuda.Stream() if use_graph else None
step = 0
model.train()
while True:
    if step in ckset:
        if use_graph: torch.cuda.synchronize()
        lt, le = probe(); LTR.append(lt); LTE.append(le)
        pasar_job.progress(step, T)
        if not np.isfinite(lt).all():
            print("non-finite predictions at step", step); break
    if step >= T:
        break
    load_batch(step)
    if not use_graph:
        opt.zero_grad(set_to_none=True); train_step()
    elif step < WARM:
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            opt.zero_grad(set_to_none=True); train_step()
        torch.cuda.current_stream().wait_stream(side)
    else:
        if graph is None:
            graph = torch.cuda.CUDAGraph()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph):
                train_step()
        graph.replay()
    step += 1

name = f"{a.arch}_{a.opt}_{a.labels}_s{a.seed}"
od = os.path.join(ROOT, "cache", "runs", a.task); os.makedirs(od, exist_ok=True)
wd_ = os.path.join(ROOT, "cache", "weights", a.task); os.makedirs(wd_, exist_ok=True)
LTR, LTE = np.stack(LTR), np.stack(LTE)
dt16 = np.float16 if a.task == "modadd" else np.float32
ytr_p = ytr[ptr].cpu().numpy(); yte_p = yte[pte].cpu().numpy()
tr_loss = -np.take_along_axis(LTR, ytr_p[None, :, None], 2).mean((1, 2))
te_loss = -np.take_along_axis(LTE, yte_p[None, :, None], 2).mean((1, 2))
tr_acc = (LTR.argmax(-1) == ytr_p).mean(1); te_acc = (LTE.argmax(-1) == yte_p).mean(1)
meta = dict(vars(a), lr=lr, T=T, name=name, wall=time.time() - t0, full_test_acc=full_test_acc(),
            final_probe_train_loss=float(tr_loss[-1]), final_probe_train_acc=float(tr_acc[-1]),
            final_probe_test_acc=float(te_acc[-1]))
np.savez(os.path.join(od, name + ".npz"), steps=ck[:len(LTR)], lp_tr=LTR.astype(dt16), lp_te=LTE.astype(dt16),
         tr_loss=tr_loss, te_loss=te_loss, tr_acc=tr_acc, te_acc=te_acc, meta=json.dumps(meta))
torch.save({"init": init_state, "final": {k: v.cpu() for k, v in model.state_dict().items()}},
           os.path.join(wd_, name + ".pt"))
print(json.dumps(meta))
