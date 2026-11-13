"""Estimation-aware reachability: shared code for the paper experiments.

Modules
  paths       repository paths and DeepReach (git submodule) import setup
  belief_cbf  belief control barrier function baseline (Vahs et al., arXiv:2309.06499)
  air3d_sim   Air3D + CT-EKF simulator, DeepReach value-function wrapper, animations
"""
from .paths import REPO_ROOT, DEEPREACH_ROOT, add_deepreach_to_path, checkpoint, repo_path  # noqa: F401
