"""Benchmark + correctness check for the per-frame fresh-minibatch measurement (kind B) used by the
boiling film.  Compares the grouped-conv evaluator in dl_common.BatchedNet (used for B/B') with a
'stacked-direction' evaluator (FreshNet below): every net sees its own 16 images, so one
ordinary conv with weights [w0; dx; dy] over all P*16 images gives conv(h,w0), conv(h,dx),
conv(h,dy), combined per image as conv(h,w0) + a_p conv(h,dx) + b_p conv(h,dy).  Exact algebra;
only floating-point summation order differs."""
import time, json, torch
import torch.nn.functional as F
import pasar_job
from dl_common import BatchedNet, load_cifar, load_trained, random_direction
from dl_common import EPS

class FreshNet:
    def __init__(self, net, dx, dy, device="cuda"):
        sd = {k: v.detach().to(device).float() for k, v in net.state_dict().items()}
        self.nl = len(net.convs)
        self.Wcat, self.cout = [], []
        for i in range(self.nl):
            k = f"convs.{i}.weight"
            self.Wcat.append(torch.cat([sd[k], dx[k].to(device), dy[k].to(device)], 0).contiguous())
            self.cout.append(sd[k].shape[0])
        self.bn = [(sd[f"bns.{i}.weight"], sd[f"bns.{i}.bias"]) for i in range(self.nl)]
        self.fc = (sd["fc.weight"], dx["fc.weight"].to(device), dy["fc.weight"].to(device), sd["fc.bias"])

    def forward(self, x, a, b):
        """x: (P, n, 3, 32, 32) -- n private images per net; a, b: (P,). BN train mode over each
        net's n images.  Returns logits (n, P, 10) (same layout as BatchedNet.forward)."""
        P, n = x.shape[:2]
        h = x.reshape(P * n, *x.shape[2:])
        ai = a.repeat_interleave(n).view(-1, 1, 1, 1); bi = b.repeat_interleave(n).view(-1, 1, 1, 1)
        for i in range(self.nl):
            C = self.cout[i]
            o = F.conv2d(h, self.Wcat[i], padding=1)
            h = o[:, :C] + ai * o[:, C:2 * C] + bi * o[:, 2 * C:]
            N, _, H, W = h.shape
            hv = h.view(P, n, C, H, W)
            mu = hv.mean((1, 3, 4), keepdim=True)
            var = hv.var((1, 3, 4), keepdim=True, unbiased=False)
            g_, b_ = self.bn[i]
            h = ((hv - mu) * torch.rsqrt(var + EPS)).view(N, C, H, W) * g_.view(1, -1, 1, 1) + b_.view(1, -1, 1, 1)
            h = F.relu(h)
            if i < self.nl - 1:
                h = F.max_pool2d(h, 2)
        f = h.mean((2, 3)).view(P, n, -1)
        w0, u, v, fb = self.fc
        Wp = w0[None] + a.view(-1, 1, 1) * u[None] + b.view(-1, 1, 1) * v[None]     # (P, 10, C)
        return torch.einsum("pnc,pkc->npk", f, Wp) + fb


pasar_job.apply_memory_limit()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
net = load_trained(dev)
dx, dy = random_direction(net, 1), random_direction(net, 2)
bnet = BatchedNet(net, dx, dy, dev); fnet = FreshNet(net, dx, dy, dev)
xtr, ytr = load_cifar(True, dev)

# correctness
P = 64
g = torch.Generator(device=dev).manual_seed(0)
a = torch.linspace(-1, 1, P, device=dev); b = torch.full_like(a, 0.37)
idx = torch.randint(0, 50000, (P, 16), device=dev, generator=g)
with torch.no_grad():
    l1 = bnet.forward(xtr[idx], a, b, bn_train=True, shared_input=False)        # (16, P, 10)
    l2 = fnet.forward(xtr[idx], a, b)                                            # (16, P, 10)
    y = ytr[idx].t()
    L1 = F.cross_entropy(l1.permute(0, 2, 1), y, reduction="none").double().mean(0)
    L2 = F.cross_entropy(l2.permute(0, 2, 1), y, reduction="none").double().mean(0)
err = float((L1.log10() - L2.log10()).abs().max())
print(json.dumps(dict(check="fresh vs grouped", max_abs_log10_diff=err, logit_maxdiff=float((l1 - l2).abs().max()))), flush=True)
pasar_job.progress(1, 3)
for name, fn in (("grouped", lambda x, a, b: bnet.forward(x, a, b, bn_train=True, shared_input=False)),
                 ("stacked", lambda x, a, b: fnet.forward(x, a, b))):
    for P in ((256, 1024) if name == "grouped" else (1024, 2048)):
        a = torch.linspace(-1, 1, P, device=dev); b = torch.full_like(a, 0.3)
        with torch.no_grad():
            idx = torch.randint(0, 50000, (P, 16), device=dev)
            fn(xtr[idx], a, b); torch.cuda.synchronize(); t = time.time(); R = 3
            for _ in range(R):
                idx = torch.randint(0, 50000, (P, 16), device=dev)
                fn(xtr[idx], a, b)
            torch.cuda.synchronize(); dt = (time.time() - t) / R
        print(json.dumps(dict(impl=name, P=P, pts_per_s=P / dt, s_per_256sq=65536 / (P / dt),
                              maxmem_GB=torch.cuda.max_memory_allocated() / 1e9)), flush=True)
pasar_job.progress(3, 3)
