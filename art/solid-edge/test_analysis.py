"""OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest -q test_analysis.py"""
import numpy as np

from se_analysis import boundary3d, box_counts3d, dimension3d, edges3d


def test_plane_is_two():
    L = np.zeros((64, 64, 64), np.uint8); L[:, :, 29:] = 1
    B = boundary3d(L)
    assert B.sum() == 2 * 64 * 64
    assert edges3d(L).sum() == 63 * 63
    f, s, c, _ = dimension3d(L)
    assert abs(f['D'] - 2) < 0.02


def test_sphere_is_two():
    z, y, x = np.mgrid[:64, :64, :64]
    L = ((x - 31.7) ** 2 + (y - 32.2) ** 2 + (z - 30.9) ** 2 < 24 ** 2).astype(np.uint8)
    f, *_ = dimension3d(L, 2, 16)
    assert 1.8 < f['D'] < 2.2


def test_noise_is_three():
    L = (np.random.default_rng(0).random((64, 64, 64)) < 0.5).astype(np.uint8)
    f, *_ = dimension3d(L)
    assert f['D'] > 2.9


def test_box_counts_sizes():
    s, c = box_counts3d(np.ones((64, 64, 64), bool))
    assert list(s) == [1, 2, 4, 8, 16] and list(c) == [64 ** 3, 32 ** 3, 16 ** 3, 8 ** 3, 4 ** 3]
