"""Every architecture in the atlas. Default PyTorch initialisation throughout."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_ARCHS = ["logreg", "mlp_256", "mlp_2048", "mlp_deep", "cnn", "resnet", "vit", "gru"]
MODADD_ARCHS = ["logreg", "mlp", "tf1", "tf2"]


class MLP(nn.Module):
    def __init__(self, d_in, width, depth, C):
        super().__init__()
        layers, d = [nn.Flatten()], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]; d = width
        layers.append(nn.Linear(d, C))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, ch, hw, C):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
        self.h = nn.Sequential(nn.Linear(64 * (hw // 4) ** 2, 128), nn.ReLU(), nn.Linear(128, C))

    def forward(self, x):
        return self.h(self.f(x))


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = (nn.Sequential() if stride == 1 and cin == cout else
                   nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(o)) + self.sc(x))


class ResNet8(nn.Module):
    def __init__(self, ch, C):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(ch, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.layers = nn.Sequential(Block(16, 16, 1), Block(16, 32, 2), Block(32, 64, 2))
        self.fc = nn.Linear(64, C)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1).flatten(1))


class ViT(nn.Module):
    def __init__(self, ch, hw, C, patch=4, d=64, depth=4, heads=4):
        super().__init__()
        self.patch = nn.Conv2d(ch, d, patch, patch)
        n = (hw // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d); self.fc = nn.Linear(d, C)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.fc(self.ln(self.enc(t)[:, 0]))


class GRUrows(nn.Module):
    def __init__(self, ch, hw, C, h=128):
        super().__init__()
        self.rnn = nn.GRU(ch * hw, h, batch_first=True); self.fc = nn.Linear(h, C)

    def forward(self, x):
        seq = x.permute(0, 2, 1, 3).flatten(2)  # rows as time steps
        return self.fc(self.rnn(seq)[0][:, -1])


# ---------------- modular addition -----------------
class ModLogreg(nn.Module):
    def __init__(self, p):
        super().__init__(); self.ea = nn.Embedding(p, p); self.eb = nn.Embedding(p, p)
        nn.init.normal_(self.ea.weight, 0, 1 / math.sqrt(2 * p)); nn.init.normal_(self.eb.weight, 0, 1 / math.sqrt(2 * p))

    def forward(self, x):
        return self.ea(x[:, 0]) + self.eb(x[:, 1])


class ModMLP(nn.Module):
    def __init__(self, p, d=128, h=512):
        super().__init__(); self.e = nn.Embedding(p, d)
        self.net = nn.Sequential(nn.Linear(2 * d, h), nn.ReLU(), nn.Linear(h, p))

    def forward(self, x):
        return self.net(self.e(x).flatten(1))


class ModTF(nn.Module):
    """Nanda et al.-style transformer: no LayerNorm, ReLU MLP, learned positions, read at '='."""

    def __init__(self, p, n_layers=1, d=128, heads=4, d_mlp=512):
        super().__init__()
        self.p, self.d, self.h = p, d, heads
        self.emb = nn.Parameter(torch.randn(p + 1, d) / math.sqrt(d))
        self.pos = nn.Parameter(torch.randn(3, d) / math.sqrt(d))
        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d, bias=False) for _ in range(n_layers)])
        self.o = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_layers)])
        self.m1 = nn.ModuleList([nn.Linear(d, d_mlp) for _ in range(n_layers)])
        self.m2 = nn.ModuleList([nn.Linear(d_mlp, d) for _ in range(n_layers)])
        self.unemb = nn.Linear(d, p, bias=False)

    def forward(self, x):
        B = len(x)
        tok = torch.cat([x, torch.full((B, 1), self.p, device=x.device, dtype=x.dtype)], 1)
        r = F.embedding(tok, self.emb) + self.pos  # (plain indexing has a very slow backward on CUDA)
        dh = self.d // self.h
        mask = torch.triu(torch.ones(3, 3, dtype=torch.bool, device=x.device), 1)
        for qkv, o, m1, m2 in zip(self.qkv, self.o, self.m1, self.m2):
            q, k, v = qkv(r).view(B, 3, 3, self.h, dh).permute(2, 0, 3, 1, 4)
            a = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            a = a.masked_fill(mask, float("-inf")).softmax(-1)
            r = r + o((a @ v).transpose(1, 2).reshape(B, 3, self.d))
            r = r + m2(F.relu(m1(r)))
        return self.unemb(r[:, -1])


def build(task, arch, C, shape=None):
    if task == "modadd":
        return {"logreg": lambda: ModLogreg(C), "mlp": lambda: ModMLP(C),
                "tf1": lambda: ModTF(C, 1), "tf2": lambda: ModTF(C, 2)}[arch]()
    ch, hw = shape[0], shape[1]
    D = ch * hw * hw
    return {"logreg": lambda: MLP(D, 0, 0, C), "mlp_256": lambda: MLP(D, 256, 1, C),
            "mlp_2048": lambda: MLP(D, 2048, 1, C), "mlp_deep": lambda: MLP(D, 512, 4, C),
            "cnn": lambda: CNN(ch, hw, C), "resnet": lambda: ResNet8(ch, C),
            "vit": lambda: ViT(ch, hw, C), "gru": lambda: GRUrows(ch, hw, C)}[arch]()
