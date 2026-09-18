"""Tests for the state-size counter, against the accounting table of the handout. [AI harness]"""
import pytest

from .adapters import run_state_elements


@pytest.mark.parametrize("mixer,d,H,L,T,expect", [
    ("attn", 64, 1, 2, 512, 131072),      # 2 layers x 2 x 512 x 64 (heads do not enter)
    ("attn", 128, 2, 2, 512, 262144),
    ("linattn", 64, 2, 2, 512, 4096),     # 2 layers x 2 heads x 32 x 32
    ("linattn", 128, 2, 2, 512, 16384),
    ("deltanet", 256, 2, 2, 512, 65536),
    ("deltanet", 512, 2, 2, 512, 262144), # equals attention's KV cache at d = 128, T = 512
    ("gdn", 128, 2, 2, 512, 16384),
    ("linattn", 128, 2, 2, 4096, 16384),  # independent of T
    ("attn", 128, 2, 1, 4096, 1048576),
])
def test_state_elements(mixer, d, H, L, T, expect):
    assert run_state_elements(mixer, d, H, L, T) == expect
