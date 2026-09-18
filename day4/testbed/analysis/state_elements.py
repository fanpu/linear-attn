"""State size of a mixer at generation time, in numbers held. [you]

after zoology/model.py (`_compute_state_size`, which sums each mixer's
`state_size(sequence_length)` over layers). Used as the x-axis of the
recall-versus-state-size plot in `experiments/day4_mqar/plot.py`.
"""
from __future__ import annotations


def state_elements(mixer: str, d_model: int, num_heads: int, n_layers: int, seq_len: int) -> int:
    """Total numbers a model must hold to continue generating after `seq_len` tokens.

    Args:
        mixer: `"attn"`, `"linattn"`, `"deltanet"`, or `"gdn"`.
        d_model: model width.
        num_heads: heads per mixer; the head dimension is `d_model // num_heads`
            for both keys and values.
        n_layers: number of mixer layers.
        seq_len: number of tokens seen so far (matters only for attention).

    Returns:
        For attention, the KV cache: per layer, a key and a value of size
        `d_model` for each of the `seq_len` tokens. For the three linear
        mixers, per layer one `head_dim x head_dim` matrix per head. Summed
        over layers. Counts of numbers, not bytes.
    """
    raise NotImplementedError
