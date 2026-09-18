"""Learning-rate schedule.  [you]
# Interface after karpathy/nanoGPT train.py (get_lr): warmup, then cosine to a floor.
"""

from math import cos, pi


def lr_at(
    step: int,
    total_steps: int,
    peak_lr: float = 6e-4,
    warmup_frac: float = 0.03,
    final_frac: float = 0.1,
) -> float:
    """Learning rate used for the update that consumes batch `step` (0-indexed).

    Let S = total_steps and W = max(1, round(warmup_frac * S)).
      Warmup, step < W:      lr = peak_lr * (step + 1) / W, so step W - 1 is at peak_lr.
      Decay,  step >= W:     p = (step - W) / (S - 1 - W) runs from 0 to 1, and
                             lr = peak_lr * [final_frac + (1 - final_frac) * 0.5 * (1 + cos(pi * p))],
                             which is peak_lr at p = 0 and final_frac * peak_lr at p = 1.
    Must be exact at the endpoints, non-decreasing on [0, W), non-increasing on
    [W, S), and cosine-shaped (the test checks p = 0.25 and p = 0.5).
    """
    warmup_steps = max(1, round(warmup_frac * total_steps))
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    p = (step - warmup_steps) / (total_steps - warmup_steps - 1)
    lr = peak_lr * (final_frac + (1 - final_frac) * 0.5 * (1 + cos(pi * p)))

    return lr


if __name__ == "__main__":
    # plot
    import matplotlib.pyplot as plt

    x = range(10000)
    y = [lr_at(s, 10000) for s in range(10000)]
    plt.plot(x, y)
    plt.savefig("lr_sample.png")
