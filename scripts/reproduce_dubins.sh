#!/usr/bin/env bash
# Dubins car with CT-EKF and a position-dependent beacon measurement (Sec. 6.3, App. D):
# nominal grid BRT, then the safety-filter table and the trajectory figure.
#
#   PY_HJ=<python of the hj_reachability env> PY_TORCH=<python of the DeepReach env> \
#       bash scripts/reproduce_dubins.sh
#
# Uses the shipped network checkpoints/dubins_ea_9d_k3. To retrain it:
#   $PY_TORCH scripts/train/train.py dubins_ea_9d_k3
set -euo pipefail
cd "$(dirname "$0")/.."
PY_HJ=${PY_HJ:-python}
PY_TORCH=${PY_TORCH:-python}

$PY_HJ scripts/grid/dubins_nominal_grid.py                                     # data/dubins_nominal_*
$PY_TORCH scripts/experiments/dubins_safety_filter.py --n_rollouts 2000 "$@"   # outputs/dubins_safety_filter/
