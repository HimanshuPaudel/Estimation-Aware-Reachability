"""Repository paths and DeepReach import setup.

The DeepReach fork lives in the git submodule ``external/deepreach``.  Its modules
(``dynamics``, ``utils``, ``experiments``) are imported as top-level packages, so its
directory has to be on ``sys.path``.  Call :func:`add_deepreach_to_path` before
``from dynamics import dynamics``.

Set the environment variable ``DEEPREACH_ROOT`` to use a different DeepReach checkout.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEEPREACH_ROOT = os.path.abspath(os.environ.get(
    "DEEPREACH_ROOT", os.path.join(REPO_ROOT, "external", "deepreach")))

CHECKPOINTS = os.path.join(REPO_ROOT, "checkpoints")
DATA = os.path.join(REPO_ROOT, "data")
FIGURES = os.path.join(REPO_ROOT, "figures")
OUTPUTS = os.path.join(REPO_ROOT, "outputs")


def repo_path(*parts):
    """Absolute path inside the repository."""
    return os.path.join(REPO_ROOT, *parts)


def checkpoint(run, name="model_final.pth"):
    """Path of a DeepReach checkpoint in the run layout <run>/training/checkpoints/<name>."""
    return os.path.join(CHECKPOINTS, run, "training", "checkpoints", name)


def add_deepreach_to_path():
    """Put the DeepReach fork on sys.path; return its directory."""
    if not os.path.isdir(os.path.join(DEEPREACH_ROOT, "dynamics")):
        raise FileNotFoundError(
            f"DeepReach not found at '{DEEPREACH_ROOT}'. Run "
            "`git submodule update --init --recursive`, or set DEEPREACH_ROOT.")
    if DEEPREACH_ROOT not in sys.path:
        sys.path.insert(0, DEEPREACH_ROOT)
    return DEEPREACH_ROOT
