"""Offline dataset loaders. They only READ data/; run scripts/fetch_data.py once to create it.  [AI]

The data directory is <course root>/data, or $TDYN_DATA if set.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return Path(os.environ.get("TDYN_DATA", ROOT / "data"))


def _load(name: str):
    path = data_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/fetch_data.py` once (needs network); "
            "nothing in the course downloads at run time.")
    return np.load(path)


def load_mnist(n_train: int | None = None, n_test: int | None = None, *, flatten: bool = True,
               scale: bool = True, seed: int = 0):
    """MNIST as numpy: (x_train, y_train, x_test, y_test).

    flatten -> x of shape (n, 784); scale -> float32 in [0, 1] (else uint8).
    n_train / n_test pick a fixed random subset (numpy seed `seed`), sorted by index.
    """
    d = _load("mnist.npz")
    xs = []
    for split, n in (("train", n_train), ("test", n_test)):
        x, y = d[f"x_{split}"], d[f"y_{split}"].astype(np.int64)
        if n is not None and n < len(y):
            idx = np.sort(np.random.default_rng(seed).choice(len(y), size=n, replace=False))
            x, y = x[idx], y[idx]
        if flatten:
            x = x.reshape(len(x), -1)
        if scale:
            x = x.astype(np.float32) / 255.0
        xs += [x, y]
    return tuple(xs)


def load_cifar10_5k(n: int | None = None, *, flatten: bool = True, scale: bool = True,
                    classes: list[int] | None = None):
    """The fixed 5000-example CIFAR-10 training subset (500 per class, seed 0): (x, y).

    flatten -> (n, 3072) in HWC order; scale -> float32 in [0, 1]. `classes` keeps only those
    labels (relabelled 0..k-1 in the given order); n keeps the first n examples after that.
    """
    d = _load("cifar10_5k.npz")
    x, y = d["x"], d["y"].astype(np.int64)
    if classes is not None:
        keep = np.isin(y, classes)
        x, y = x[keep], y[keep]
        remap = {c: i for i, c in enumerate(classes)}
        y = np.array([remap[int(c)] for c in y], dtype=np.int64)
    if n is not None:
        x, y = x[:n], y[:n]
    if flatten:
        x = x.reshape(len(x), -1)
    if scale:
        x = x.astype(np.float32) / 255.0
    return x, y
