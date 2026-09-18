"""Toy: why the additive rule loses associations and the delta rule keeps the latest one.

Writes n random unit keys with random values into a d x d state, one at a
time, with each rule, then queries every key and reports how far the
retrieved vector is from the stored value. Pure numpy, two-line recurrences,
d = 16. Run it, then change n (the number of pairs written).

    python examples/retrieval_toy.py
"""
import numpy as np

rng = np.random.default_rng(0)
d, n = 16, 12
K = rng.normal(size=(n, d))
K /= np.linalg.norm(K, axis=1, keepdims=True)     # unit-norm keys, not orthogonal
V = rng.normal(size=(n, d))

S_add = np.zeros((d, d))
S_delta = np.zeros((d, d))
for t in range(n):
    S_add += np.outer(K[t], V[t])                              # additive: S += k v^T
    S_delta += np.outer(K[t], V[t] - S_delta.T @ K[t])         # delta:    S += k (v - S^T k)^T, beta = 1

err_add = np.linalg.norm(S_add.T @ K.T - V.T, axis=0) / np.linalg.norm(V, axis=1)
err_delta = np.linalg.norm(S_delta.T @ K.T - V.T, axis=0) / np.linalg.norm(V, axis=1)
overlap = np.abs(K @ K.T - np.eye(n)).max()

print(f"d = {d}, {n} pairs written; largest |k_i . k_j| between distinct keys = {overlap:.2f}")
print("relative retrieval error per key (0 = exact), keys in the order written:")
print("  additive:", np.array2string(err_add, precision=2))
print("  delta:   ", np.array2string(err_delta, precision=2, suppress_small=True))
print(f"mean error: additive {err_add.mean():.2f}, delta {err_delta.mean():.2f}; "
      f"last key: additive {err_add[-1]:.2f}, delta {err_delta[-1]:.2e}")
