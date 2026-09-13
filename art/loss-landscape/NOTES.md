# NOTES: loss-landscape (handoff)

- The orchestrator queued 4 trainings via gpu_run.sh (40 epochs each; the schedule is shortened from Li et al.'s 300 and scaled proportionally by train.py):
  `train.py --depth {20,56} [--noshort] --epochs 40`. Logs are logs/train_resnet{20,20noshort,56,56noshort}.log; checkpoints go to cache/ckpt/.
- Old killed 60-epoch checkpoints were moved to cache/old_ckpt/ (partial runs, not to be used as final).
- Next: once trainings finish, compute landscapes with landscape.py (see TASKS.md loss-landscape section), render cartographic styles, then write the README.
