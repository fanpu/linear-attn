"""Download the course datasets into data/ once, so everything else runs offline.  [AI]

    uv run python scripts/fetch_data.py            # writes data/mnist.npz, data/cifar10_5k.npz
    uv run python scripts/fetch_data.py --force    # re-download even if the files exist

data/mnist.npz        x_train (60000, 28, 28) uint8, y_train (60000,) uint8,
                      x_test (10000, 28, 28) uint8,  y_test (10000,) uint8
data/cifar10_5k.npz   x (5000, 32, 32, 3) uint8 (HWC, RGB), y (5000,) uint8,
                      idx (5000,) int64 = positions in the concatenated CIFAR-10 training
                      batches 1-5; 500 examples per class drawn with numpy seed 0;
                      classes (10,) str

Nothing in tdyn/ downloads at run time: tdyn.data only reads these two files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import pickle
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
MNIST_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"  # published on the CIFAR-10 page
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def _get(url: str) -> bytes:
    print(f"  GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tdyn-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _parse_idx(raw: bytes) -> np.ndarray:
    """IDX format: magic (2 zero bytes, dtype code 0x08 = uint8, ndim), big-endian dims, data."""
    assert raw[0] == 0 and raw[1] == 0 and raw[2] == 0x08, "not an unsigned-byte IDX file"
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims).copy()


def fetch_mnist(out: Path) -> None:
    arrays = {}
    for key, fname in MNIST_FILES.items():
        last_err = None
        for mirror in MNIST_MIRRORS:
            try:
                arrays[key] = _parse_idx(gzip.decompress(_get(mirror + fname)))
                break
            except Exception as e:  # try the next mirror
                last_err = e
        else:
            raise RuntimeError(f"could not download {fname}: {last_err}")
    assert arrays["x_train"].shape == (60000, 28, 28) and arrays["x_test"].shape == (10000, 28, 28)
    assert arrays["y_train"].shape == (60000,) and arrays["y_test"].shape == (10000,)
    np.savez_compressed(out, **arrays)


def fetch_cifar5k(out: Path, n_per_class: int = 500, seed: int = 0) -> None:
    raw = _get(CIFAR_URL)
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != CIFAR_MD5:
        raise RuntimeError(f"CIFAR-10 tarball md5 {md5} != {CIFAR_MD5}")
    xs, ys = [], []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for i in range(1, 6):
            f = tar.extractfile(f"cifar-10-batches-py/data_batch_{i}")
            d = pickle.load(f, encoding="bytes")
            xs.append(np.asarray(d[b"data"], dtype=np.uint8))
            ys.append(np.asarray(d[b"labels"], dtype=np.uint8))
    x = np.concatenate(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NCHW -> NHWC
    y = np.concatenate(ys)
    assert x.shape == (50000, 32, 32, 3)
    rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(y == c), size=n_per_class, replace=False) for c in range(10)
    ]))
    np.savez_compressed(out, x=np.ascontiguousarray(x[idx]), y=y[idx], idx=idx.astype(np.int64),
                        classes=np.array(CIFAR_CLASSES))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = p.parse_args(argv)
    DATA.mkdir(exist_ok=True)
    for name, fn in [("mnist.npz", fetch_mnist), ("cifar10_5k.npz", fetch_cifar5k)]:
        path = DATA / name
        if path.exists() and not args.force:
            print(f"{path} exists, skipping (use --force to refetch)")
            continue
        print(f"fetching {name}")
        with tempfile.NamedTemporaryFile(dir=DATA, suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            fn(tmp_path)
            tmp_path.replace(path)
            path.chmod(0o644)
        finally:
            tmp_path.unlink(missing_ok=True)
        print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
