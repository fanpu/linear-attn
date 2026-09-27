"""Follow-up (2026-09-26 late): do the three optimiser shadows generalise to Fashion-MNIST?

Reuses adam_dial.py unchanged (same stacked elementwise optimiser, LeNet-300-100, raw [0,1] pixels, batch 60,
20k steps per round, layer-wise IMP 20/20/10%, rewind to init, rounds 0..15), with only the dataset swapped to
Fashion-MNIST and three new one-configuration families, one pasar job each:
  fashion_sgd     SGD lr 0.1            (MNIST: job 728, via imp.py/torch SGD; adam_dial's sgd kind is identical)
  fashion_adam    Adam lr 1.2e-3 e 1e-8 (MNIST: job 664)
  fashion_signum  signum lr 1e-4 b 0.9  (MNIST: jobs 873 + 880)
Output: cache/followup_fashion_<opt>/round_XX.npz in adam_dial's format.

    python followup_fashion_imp.py --family fashion_sgd --seeds 0,1
"""
import imp as IMP
import adam_dial as AD

AD.FAMILIES["fashion_sgd"] = [("fashion_sgd_lr0.1", "sgd", 0.1, 0.0, 0.0, 0.0, 0.0)]
AD.FAMILIES["fashion_adam"] = [("fashion_adam_eps1e-8", "adam", 1.2e-3, 0.9, 0.999, 1e-8, 0.0)]
AD.FAMILIES["fashion_signum"] = [("fashion_signum_lr1e-4", "signum", 1e-4, 0.9, 0.0, 0.0, 0.0)]
AD.load_dataset = lambda name: IMP.load_dataset("fashion")  # adam_dial hard-codes "mnist"; this is the only change

if __name__ == "__main__":
    AD.main()
