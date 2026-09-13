"""Load the CNN sweep histories (from checkpoints, falling back to logs) onto a common (width, epoch) grid."""
import pathlib, re, collections
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def load(tag="main", n=10000, noise=20, use_logs=True):
    d = collections.defaultdict(dict)
    if use_logs:
        for f in sorted((HERE / "cache" / "cnn" / "logs").glob(f"{tag}_n{n}_*.log")):
            for l in f.read_text().splitlines():
                m = re.match(r"\[k=\s*(\d+)\] ep\s+(\d+) train_err ([\d.]+) test_err ([\d.]+) test_loss ([\d.]+) memorized ([\d.]+)", l)
                if m:
                    d[int(m[1])][int(m[2])] = dict(train_err=float(m[3]), test_err=float(m[4]), test_loss=float(m[5]), fit_noisy=float(m[6]))
    else:
        import torch
        for f in (HERE / "cache" / "cnn" / tag).glob(f"n{n}_p{noise}_k*_s0.pt"):
            st = torch.load(f, map_location="cpu", weights_only=False)["state"]
            for h in st["hist"]:
                d[st["k"]][h["epoch"]] = h
    ks = sorted(d)
    common = sorted(set.intersection(*[set(d[k]) for k in ks]))
    E = np.array(common)
    get = lambda key: np.array([[d[k][e][key] for e in common] for k in ks])
    return dict(k=np.array(ks), epoch=E, train_err=get("train_err"), test_err=get("test_err"),
                test_loss=get("test_loss"), fit_noisy=get("fit_noisy"))


def smooth_epochs(A, w=2, epochs=None, sigma_dec=0.07):
    """Smooth each width's history in training time. With `epochs`, a Gaussian kernel in log10(epoch) of width
    sigma_dec decades (causal-free, symmetric); otherwise a running mean over +-w neighbouring evaluations."""
    out = np.empty_like(A, dtype=float)
    if epochs is not None:
        le = np.log10(np.asarray(epochs, float))
        W = np.exp(-0.5 * ((le[:, None] - le[None, :]) / sigma_dec) ** 2)
        W /= W.sum(1, keepdims=True)
        return A @ W.T
    for j in range(A.shape[1]):
        lo, hi = max(0, j - w), min(A.shape[1], j + w + 1)
        out[:, j] = A[:, lo:hi].mean(1)
    return out


def crossing(xs, ys, level, log_x=True):
    """First x where the (decreasing) curve ys crosses below `level`, log-linear interpolation. nan if never."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    below = np.where(ys <= level)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return xs[0]
    x0, x1 = (np.log(xs[i - 1]), np.log(xs[i])) if log_x else (xs[i - 1], xs[i])
    t = (ys[i - 1] - level) / (ys[i - 1] - ys[i])
    v = x0 + t * (x1 - x0)
    return float(np.exp(v)) if log_x else float(v)


def peak_location(xs, ys, log_x=True):
    """Argmax with a parabolic refinement in log x over the 3 points around the max."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    i = int(np.argmax(ys))
    if i == 0 or i == len(xs) - 1:
        return xs[i], i
    lx = np.log(xs[i - 1:i + 2]) if log_x else xs[i - 1:i + 2]
    c = np.polyfit(lx, ys[i - 1:i + 2], 2)
    if c[0] >= 0:
        return xs[i], i
    v = -c[1] / (2 * c[0])
    v = min(max(v, lx[0]), lx[2])
    return (float(np.exp(v)) if log_x else float(v)), i
