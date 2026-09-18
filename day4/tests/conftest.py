import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config):
    # after fla/tests/conftest.py: GPU-only tests are marked so the CPU subset
    # runs anywhere (`pytest -m "not gpu"`). Added Day 4.
    config.addinivalue_line("markers", "gpu: needs a CUDA device and fla's Triton kernels")
