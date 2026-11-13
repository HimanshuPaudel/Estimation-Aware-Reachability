#!/usr/bin/env bash
# Planar quadrotor with H-infinity filter (Sec. 6.2, App. C): rollout table for kappa = 3 and 2.
#
#   PY_TORCH=<python of the DeepReach env> bash scripts/reproduce_quadrotor.sh
#
# Uses the shipped networks checkpoints/pquad_{plain_6d,ea_9d_k3,ea_9d_k2}. To retrain:
#   $PY_TORCH scripts/train/train.py pquad_plain_6d     (and pquad_ea_9d_k3, pquad_ea_9d_k2)
set -euo pipefail
cd "$(dirname "$0")/.."
PY_TORCH=${PY_TORCH:-python}

# kappa = 3 (also the sample-rollout BRT figure)
$PY_TORCH scripts/experiments/quadrotor_safety_filter.py --n_rollouts 2000 \
    --est_ckpt checkpoints/pquad_ea_9d_k3/training/checkpoints/model_final.pth \
    --out outputs/quadrotor_safety_filter/kappa_3

# kappa = 2
$PY_TORCH scripts/experiments/quadrotor_safety_filter.py --n_rollouts 2000 \
    --est_ckpt checkpoints/pquad_ea_9d_k2/training/checkpoints/model_final.pth \
    --out outputs/quadrotor_safety_filter/kappa_2
