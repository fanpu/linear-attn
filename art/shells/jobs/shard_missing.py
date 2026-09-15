"""Print how many slabs of shard K/N of a volume are still missing (same centre-first order as volume.py)."""
import os, sys
sdir, n, K, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
order = sorted(range(n), key=lambda k: (abs(k - n // 2), k))
print(sum(1 for pos, k in enumerate(order) if pos % N == K and not os.path.exists(os.path.join(sdir, f"slab{k:02d}.npz"))))
