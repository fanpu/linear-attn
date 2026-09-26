"""Build the four task files used by every run (CPU only).

Each task file cache/data_<task>.npz holds
  xtr, ytr   : the 10,000-example training subset every run trains on
  xte, yte   : the full test set (images) / the held-out pairs (modadd)
  ptr, pte   : indices of the fixed probe sets (1,000 train examples, 1,000 test examples)
  ytr_shuf   : a fixed random relabelling of ytr (the memorisation null)
Images are stored as float16 normalised with the training-subset mean/std per channel.
"""
import gzip, os, pickle, struct
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "data")
OUT = os.path.join(ROOT, "cache")
NTR, NPROBE = 10_000, 1_000


def idx(path):
    with open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        nd = magic & 0xFF
        shape = struct.unpack(">" + "I" * nd, f.read(4 * nd))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def mnistlike(name):
    d = os.path.join(DATA, name, "raw")
    xtr = idx(os.path.join(d, "train-images-idx3-ubyte"))[:, None].astype(np.float32) / 255
    ytr = idx(os.path.join(d, "train-labels-idx1-ubyte")).astype(np.int64)
    xte = idx(os.path.join(d, "t10k-images-idx3-ubyte"))[:, None].astype(np.float32) / 255
    yte = idx(os.path.join(d, "t10k-labels-idx1-ubyte")).astype(np.int64)
    return xtr, ytr, xte, yte


def cifar():
    d = os.path.join(DATA, "cifar-10-batches-py")
    xs, ys = [], []
    for i in range(1, 6):
        b = pickle.load(open(os.path.join(d, f"data_batch_{i}"), "rb"), encoding="bytes")
        xs.append(b[b"data"]); ys += b[b"labels"]
    b = pickle.load(open(os.path.join(d, "test_batch"), "rb"), encoding="bytes")
    f = lambda a: a.reshape(-1, 3, 32, 32).astype(np.float32) / 255
    return f(np.concatenate(xs)), np.array(ys), f(b[b"data"]), np.array(b[b"labels"])


def save(task, xtr, ytr, xte, yte, rng):
    sel = rng.permutation(len(ytr))[:NTR]
    xtr, ytr = xtr[sel], ytr[sel]
    m = xtr.mean(axis=(0, 2, 3), keepdims=True); s = xtr.std(axis=(0, 2, 3), keepdims=True)
    xtr = ((xtr - m) / s).astype(np.float16); xte = ((xte - m) / s).astype(np.float16)
    ptr = np.sort(rng.permutation(NTR)[:NPROBE]); pte = np.sort(rng.permutation(len(yte))[:NPROBE])
    np.savez(os.path.join(OUT, f"data_{task}.npz"), xtr=xtr, ytr=ytr, xte=xte, yte=yte,
             ptr=ptr, pte=pte, ytr_shuf=rng.permutation(ytr), C=10)
    print(task, xtr.shape, xte.shape, np.bincount(ytr[ptr]))


def modadd(rng, p=97, frac=0.4):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    x = np.stack([a.ravel(), b.ravel()], 1).astype(np.int64); y = (x[:, 0] + x[:, 1]) % p
    perm = rng.permutation(p * p); ntr = int(frac * p * p)
    tr, te = perm[:ntr], perm[ntr:]
    ptr = np.sort(rng.permutation(ntr)[:NPROBE]); pte = np.sort(rng.permutation(len(te))[:NPROBE])
    np.savez(os.path.join(OUT, "data_modadd.npz"), xtr=x[tr], ytr=y[tr], xte=x[te], yte=y[te],
             ptr=ptr, pte=pte, ytr_shuf=rng.permutation(y[tr]), C=p)
    print("modadd", ntr, len(te))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    save("mnist", *mnistlike("MNIST"), np.random.default_rng(0))
    save("fashion", *mnistlike("FashionMNIST"), np.random.default_rng(1))
    save("cifar", *cifar(), np.random.default_rng(2))
    modadd(np.random.default_rng(3))
