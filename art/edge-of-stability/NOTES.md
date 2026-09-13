# edge-of-stability NOTES (handoff)

## State
- eos_train.py patched: float32 default, reverse-over-reverse HVPs sharing one graph, k=3 subspace
  iteration (tiny eigh on CPU: GPU eigh of 3x3 took 0.19 s on GB10), --eig-every, periodic saves.
- float64 is ~17x slower on GB10 (HVP 1.38 s vs 0.066 s), so float32 is used.
- Init sharpness (seed 0, n=5000) = 88.0. 2/eta=20 diverges at step 31.
- Scouts running: cache/scout_inv{60,100,150,250}.npz (8000 steps, eig every 5), logs/scout_*.log

## Paper setup (checked in arXiv:2103.00065 text)
first 5000 CIFAR-10 train, per-channel standardise with full-CIFAR stats, 3072-200-200-10 tanh,
PyTorch default init, MSE 0.5*sum_classes averaged over examples; Fig 3 uses eta=2/600.

## Next
pick 4 LRs from scouts -> main runs eig-every 1; eta sweep; render library; pieces.
