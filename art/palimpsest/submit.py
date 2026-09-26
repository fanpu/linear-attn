"""Submit the sweep to pasar: one job per configuration. Skips runs whose output already exists.

    python submit.py main [--dry]
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)

# learning rates chosen from the lr pilot (cache/pilot/lr_*.json); see README
# (jobs 768-773, 400 steps on page A at 256 px, width 256): Adam FF 3e-4 was the smoothest of
# {3e-4,1e-3,3e-3}; SIREN Adam 1e-4 is Sitzmann's default. SGD+momentum 0.9 barely moves any of
# these nets in 400 steps; the only settings that learned at all were SIREN lr 1e-2 and FF8 lr 1e-1.
LR = {
    ("relu", "adam"): 1e-3, ("ff8", "adam"): 3e-4, ("ff32", "adam"): 3e-4, ("siren", "adam"): 1e-4,
    ("ff8", "sgd"): 1e-1, ("siren", "sgd"): 1e-2,
}
ARCH = {"relu": ["--arch", "relu"], "ff8": ["--arch", "ff", "--sigma", "8"],
        "ff32": ["--arch", "ff", "--sigma", "32"], "siren": ["--arch", "siren"]}


def name_of(arch, width, opt, first, seed, res):
    tag = {"relu": "relu", "ff8": "ff8", "ff32": "ff32", "siren": "siren30"}[arch]
    return f"{tag}_w{width}_{opt}_{first}B_s{seed}_n{res}"


def configs(which):
    C = []
    F = ["A", "C", "none"]
    if which == "main":
        for arch in ["siren", "ff32", "ff8"]:
            for width in [256, 64, 512]:
                for first in F:
                    C.append(dict(arch=arch, width=width, opt="adam", first=first, seed=0, res=256))
        for first in F:
            C.append(dict(arch="relu", width=256, opt="adam", first=first, seed=0, res=256))
    if which == "sgd":
        for arch in ["siren", "ff8"]:
            for first in F:
                C.append(dict(arch=arch, width=256, opt="sgd", first=first, seed=0, res=256, steps1=4000, steps2=4000))
    if which == "seeds":
        for arch in ["siren", "ff32", "ff8"]:
            for first in ["A", "none"]:
                C.append(dict(arch=arch, width=256, opt="adam", first=first, seed=1, res=256))
    if which == "hero":
        for first in ["A", "none"]:
            C.append(dict(arch="siren", width=256, opt="adam", first=first, seed=0, res=512, nsnap=160))
    return C


def est_minutes(c):
    return {64: 15, 256: 40, 512: 80}.get(c["width"], 40) * (2 if c["res"] == 512 else 1)


def main():
    which = sys.argv[1]
    dry = "--dry" in sys.argv
    steps = dict(steps1=int(os.environ.get("STEPS1", 2000)), steps2=int(os.environ.get("STEPS2", 2000)))
    for c in configs(which):
        name = name_of(c["arch"], c["width"], c["opt"], c["first"], c["seed"], c["res"])
        if os.path.exists(os.path.join(HERE, "cache", "runs", name + ".npz")):
            continue
        cmd = [".venv/bin/python", "palimpsest/train.py", *ARCH[c["arch"]], "--width", str(c["width"]),
               "--opt", c["opt"], "--lr", str(LR[(c["arch"], c["opt"])]), "--first", c["first"],
               "--seed", str(c["seed"]), "--res", str(c["res"]), "--steps1", str(c.get("steps1", steps["steps1"])),
               "--steps2", str(c.get("steps2", steps["steps2"])), "--nsnap", str(c.get("nsnap", 80)), "--name", name]
        mem = "3G" if c["res"] < 512 else "5G"
        sub = ["pasar", "submit", "--json", "--time", f"{est_minutes(c)}m", "--mem", mem,
               "--name", "pal-" + name, "--tag", "art-palimpsest", "--by", "art-palimpsest",
               "--note", "Palimpsest: does a coordinate net retrained on page B keep page A's fine detail?",
               "--cwd", ART, "--", "env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", *cmd]
        if dry:
            print(" ".join(cmd))
            continue
        r = subprocess.run(sub, capture_output=True, text=True)
        try:
            j = json.loads(r.stdout)
            print(j["id"], j["name"], j["state"])
        except Exception:
            print("FAILED", name, r.stdout, r.stderr)


if __name__ == "__main__":
    main()
