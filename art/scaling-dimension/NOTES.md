# scaling-dimension NOTES (living handoff)

## State (2026-09-13 14:35)
- `idlib.py` TwoNN/MLE/N(r) (tested by test_idlib.py). `ts_train.py` batched teacher/student sweep (per-(fam,d) shared pools, saves after each width).
- RUNNING: main sweep `cache/main/ts_main.npz` (log `cache/logs/ts_main.log`), fams relu0,relub; d=2,3,4,5,6,8,10,12; 3 seeds; widths 6,16,45,4,11,32,90,8,23,64; 60k steps, batch 1024, Adam 3e-3 -> cosine to 3e-6 over last 60%. ~4 min/width on GPU slot.
- QUEUED: real-data CNN sweep `cache/real/real.json` (log `cache/logs/real.log`), cifar10,fmnist,mnist, widths c=2..24, 12 epochs, early stopping; pixel ID too.
- `ts_analyze.py <npz> --id_widths 16,45,90` -> `<npz>_analysis.{json,npz}` (alpha fits, student final-hidden IDs, N(r) curves).

## Doc-claim check (GPT-2 d>=90)
Sharma&Kaplan JMLR sec 3.3 (refs/jmlr.txt ~l.1157): alpha=0.076 -> 4/alpha~53; last-token activations, 10k vectors, all layers: ID roughly constant across layers **except first layer significantly smaller (50-80)**; "we conclude that since d>90, d >= 4/alpha ~ 53". Single contiguous passage gives ID~7. So doc's "d>=90 across layers" is right for all but the first layer; inequality not equality.

## Next
analyse -> renderers (diptych, fan, agreement plate, zoom film) in plotter/dark/riso/Spectral -> README.
