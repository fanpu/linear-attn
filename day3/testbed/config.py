"""Run configuration: one TOML file per run.  [AI-owned]
# after pytorch/torchtitan torchtitan/config_manager.py: a run's flags live in its
# TOML file, grouped into [job], [model], [optimizer], [lr_scheduler],
# [training], [validation], [checkpoint], [metrics]. The section and key names
# are torchtitan's where a counterpart exists (local_batch_size, seq_len,
# max_norm, warmup_*, log_freq, dump_folder); where torchtitan counts in steps
# we count in fractions of the run, because today's three sizes have different
# step counts and one fraction gives them the same schedule shape.
"""
import hashlib
import os
import tomllib

REQUIRED = {
    "job": ["dump_folder"],
    "model": ["mixer", "size", "vocab_size"],
    "optimizer": ["name", "lr", "betas", "weight_decay"],
    "lr_scheduler": ["warmup_frac", "decay_type", "min_lr_factor"],
    "training": ["local_batch_size", "seq_len", "budget_tokens", "steps", "max_norm", "dataset_path"],
    "validation": ["every_frac", "windows", "final_windows"],
    "checkpoint": ["every_frac"],
    "metrics": ["log_freq"],
}


def load_config(path: str) -> dict:
    """Parse a run's TOML file and check that every required key is present.
    Missing keys raise here, at launch, rather than as a KeyError mid-run."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for section, keys in REQUIRED.items():
        if section not in cfg:
            raise KeyError(f"{path}: missing [{section}] section")
        for k in keys:
            if k not in cfg[section]:
                raise KeyError(f"{path}: missing key {section}.{k}")
    assert cfg["optimizer"]["name"] == "AdamW", "only AdamW is implemented"
    assert cfg["lr_scheduler"]["decay_type"] == "cosine", "only cosine decay is implemented"
    return cfg


def source_hash(root: str = None) -> str:
    """SHA-1 of every .py file under testbed/ and scripts/, so that env.json pins
    the code a run used even without a git checkout.
    # after KellerJordan/modded-nanogpt train_gpt.py, which logs its own source."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha1()
    for sub in ("testbed", "scripts"):
        for dirpath, _, files in sorted(os.walk(os.path.join(root, sub))):
            for name in sorted(files):
                if name.endswith(".py"):
                    p = os.path.join(dirpath, name)
                    h.update(p.encode()); h.update(open(p, "rb").read())
    return h.hexdigest()[:12]
