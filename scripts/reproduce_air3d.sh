#!/usr/bin/env bash
# Air3D with CT-EKF (Sec. 6.1, App. B): grid solutions, safety-filter table, BRT figures.
#
#   PY_HJ=<python of the hj_reachability env> PY_TORCH=<python of the DeepReach env> \
#       bash scripts/reproduce_air3d.sh
#
# Uses the shipped network checkpoints/air3d_ea_6d_k3. To retrain it first:
#   $PY_TORCH scripts/train/train.py air3d_ea_6d_k3
set -euo pipefail
cd "$(dirname "$0")/.."
PY_HJ=${PY_HJ:-python}
PY_TORCH=${PY_TORCH:-python}

# 1) grid HJ solutions (hj_reachability)
$PY_HJ scripts/grid/air3d_nominal_grid.py                                  # data/air3d_nominal_xy3_*  (safety filter)
$PY_HJ scripts/grid/air3d_nominal_grid.py --xy-max 1 --shape 201 201 200 \
    --out-prefix data/air3d_nominal_xy1                                    # data/air3d_nominal_xy1_*  (overlay figure)
$PY_HJ scripts/grid/air3d_ea_4d_grid.py                                    # data/air3d_ea4d_*         (grid validation)

# 2) safety-filter rollouts: 15 positions x 500 seeds = 7500 rollouts per row
$PY_TORCH scripts/experiments/air3d_safety_filter.py                       # outputs/air3d_safety_filter/

# 3) figures
$PY_TORCH scripts/plotting/air3d_brt_deepreach.py                          # learned BRT vs sigma (Fig. brt_sigma)
$PY_TORCH scripts/plotting/air3d_grid_vs_deepreach.py                      # learned vs grid BRT + IoU table (App. B)
$PY_TORCH scripts/plotting/air3d_grid_vs_nominal.py                        # nominal vs estimation-aware grid BRT (App. B)
$PY_HJ    scripts/plotting/air3d_grid_4d.py                                # 4D grid BRT panels
