#!/usr/bin/env python
"""Train a DeepReach value function from one of the paper's configs.

Each file in configs/train/ is the exact config DeepReach saved for a network used in the
paper (W&B fields removed).  This launcher turns it into a `run_experiment.py` command for
the DeepReach fork (external/deepreach) and runs it, writing the run to runs/<name>/.

Usage (DeepReach / torch environment):
  python scripts/train/train.py air3d_ea_6d_k3            # Air3D, estimation-aware 6D (kappa = 3)
  python scripts/train/train.py pquad_plain_6d            # planar quadrotor, plain 6D
  python scripts/train/train.py pquad_ea_9d_k3            # planar quadrotor, estimation-aware 9D (kappa = 3)
  python scripts/train/train.py pquad_ea_9d_k2            # planar quadrotor, estimation-aware 9D (kappa = 2)
  python scripts/train/train.py dubins_ea_9d_k3           # Dubins car, estimation-aware 9D (kappa = 3)
  python scripts/train/train.py air3d_ea_6d_k3 --dry-run  # print the command only
  python scripts/train/train.py air3d_ea_6d_k3 -- --use_wandb --wandb_project P ...  # extra DeepReach args

To use a trained run in the experiment scripts, pass its checkpoint, e.g.
  --ckpt runs/air3d_ea_6d_k3/training/checkpoints/model_final.pth
or copy runs/<name> to checkpoints/<name>.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "src"))
from ea_reach.paths import DEEPREACH_ROOT, REPO_ROOT, add_deepreach_to_path  # noqa: E402

CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "train")

# Keys that are set by this launcher or are not command-line options of run_experiment.py.
SKIP = {"config_filepath", "mode", "experiments_dir", "use_wandb", "refinement_schedule_list"}
# Options declared with type=bool: argparse would turn the string "False" into True, so they
# are passed only when True ("True" -> True); a False value falls back to the parser default.
TYPE_BOOL = {"use_lbfgs", "adj_rel_grads"}


def store_true_flags(run_experiment):
    """Names of the store_true options of run_experiment.py (passed as bare flags)."""
    src = open(run_experiment).read()
    flags = set()
    for block in src.split("p.add_argument(")[1:]:
        head = block.split(")")[0]
        m = re.search(r'"--(\w+)"', head)
        if m and "store_true" in head:
            flags.add(m.group(1))
    return flags


def read_config(path):
    cfg = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        cfg[k] = v
    return cfg


def config_to_args(cfg, flags):
    args = []
    for k, v in cfg.items():
        if k in SKIP or k.startswith("wandb_") or v == "None":
            continue
        if v in ("True", "False"):
            if v == "False":
                continue                         # store_true default / type=bool default
            args += [f"--{k}"] if k in flags else [f"--{k}", "True"]
            continue
        args += [f"--{k}", v]
    return args


def main():
    names = sorted(f[:-4] for f in os.listdir(CONFIG_DIR) if f.endswith(".txt"))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("name", choices=names, help="config in configs/train/")
    ap.add_argument("--experiments-dir", default=os.path.join(REPO_ROOT, "runs"))
    ap.add_argument("--experiment-name", default=None, help="default: the config name")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

    add_deepreach_to_path()
    run_experiment = os.path.join(DEEPREACH_ROOT, "run_experiment.py")
    cfg = read_config(os.path.join(CONFIG_DIR, f"{a.name}.txt"))
    cfg["experiment_name"] = a.experiment_name or a.name

    cmd = [sys.executable, run_experiment, "--mode", "train",
           "--experiments_dir", os.path.abspath(a.experiments_dir)]
    cmd += config_to_args(cfg, store_true_flags(run_experiment)) + extra
    print("cd", shlex.quote(DEEPREACH_ROOT), "&&", " ".join(shlex.quote(c) for c in cmd))
    if a.dry_run:
        return
    # run_experiment.py imports `dynamics`, `utils`, ... relative to its own directory
    sys.exit(subprocess.call(cmd, cwd=DEEPREACH_ROOT))


if __name__ == "__main__":
    main()
