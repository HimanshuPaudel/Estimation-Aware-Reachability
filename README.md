# Estimation-Aware Reachability

Code for the paper **Estimation Aware Reachability** (Himanshu Paudel, Javier Borquez).

Hamilton–Jacobi (HJ) reachability assumes the state is known exactly, but a deployed
controller acts on a noisy state estimate. We solve HJ reachability over the estimator's
*belief* (estimate and covariance of a CT-EKF or an extended H∞ filter), so the evolution of
the uncertainty becomes part of the safety problem. The measurement innovation is a bounded
adversary whose set scales with the innovation covariance, and the unsafe set is enlarged by
a covariance-dependent margin with confidence multiplier κ. When the value function
certifies safety and the estimator is consistent, the true state is safe with probability
at least 1 − δ_x(κ).

Only the Hamiltonian and the target set change, so the value functions are learned with
[DeepReach](https://github.com/smlbansal/deepreach) without changes to its architecture, loss,
or training procedure (the fork in `external/deepreach` adds the augmented dynamics), plus MPC supervision
([DeepReach MPC](https://arxiv.org/abs/2505.03830) / [MADR](https://arxiv.org/abs/2510.18845)).
They are validated against grid solutions from
[hj_reachability](https://github.com/StanfordASL/hj_reachability) and used as
least-restrictive safety filters.

Case studies:

| System | Estimator | State (learned) | Paper |
|---|---|---|---|
| Air3D pursuit–evasion | CT-EKF | 6D `(x̂, ŷ, ψ, Σxx, Σyy, Σxy)` | Sec. 6.1, App. B |
| Planar quadrotor | extended H∞ filter | 9D `(x̂, ŷ, θ, vx, vy, ω, Pxx, Pyy, Pxy)` | Sec. 6.2, App. C |
| Dubins car, beacon with position-dependent noise | CT-EKF (heading not measured) | 9D `(x̂, ŷ, θ̂, Σxx, Σyy, Σθθ, Σxy, Σxθ, Σyθ)` | Sec. 6.3, App. D |

Training settings for all networks are in App. A.

## Repository layout

```
.
├── external/deepreach/        git submodule: DeepReach fork with the estimation-aware dynamics
│                              (EstimationAwareAir3D, EstimationAwarePlanarQuadrotor,
│                              EstimationAwareDubins3D in dynamics/dynamics.py)
├── src/ea_reach/              shared Python package
│   ├── paths.py               repo paths, puts the DeepReach submodule on sys.path
│   ├── air3d_sim.py           Air3D + CT-EKF simulator, value-network wrapper
│   └── belief_cbf.py          belief CBF baseline (Vahs et al., RA-L 2023)
├── scripts/
│   ├── grid/                  hj_reachability grid solutions (nominal + 4D estimation-aware)
│   ├── train/train.py         trains a network from configs/train/<name>.txt
│   ├── experiments/           safety-filter rollouts (paper tables)
│   ├── plotting/              BRT figures and grid-vs-learned comparison
│   └── reproduce_*.sh         end-to-end pipelines per system
├── configs/train/             DeepReach training configs of the paper networks
├── checkpoints/               trained networks used in the paper (DeepReach run layout)
├── results/                   reference outputs behind the paper tables
├── data/                      grid solutions (generated, git-ignored)
├── figures/, outputs/         generated figures / experiment outputs (git-ignored)
├── requirements.txt           DeepReach / torch environment
└── requirements-hj.txt        hj_reachability / JAX environment
```

## Setup

```bash
git clone --recursive https://github.com/Himanshu069/Estimation-Aware-Reachability.git
cd Estimation-Aware-Reachability
```

The grid solver (JAX) and DeepReach (PyTorch) are installed in **two separate
environments**, since their numpy/CUDA requirements conflict. Python 3.12 was used.

```bash
# DeepReach / torch: training, safety-filter experiments, most plots
python -m venv .venv-torch && source .venv-torch/bin/activate
pip install -r requirements.txt 
#Install the appropriate PyTorch package for your system
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
deactivate

# hj_reachability / JAX: grid solutions
python -m venv .venv-hj && source .venv-hj/bin/activate
pip install -r requirements-hj.txt
deactivate
```

The scripts find `src/` and the DeepReach submodule by themselves, so `pip install -e .`
is optional and every script can be run from any directory. Set `DEEPREACH_ROOT` to use a
different DeepReach checkout.

## Reproducing the paper

The trained networks are in `checkpoints/`, so training is optional. Each pipeline script
takes the two interpreters as `PY_HJ` and `PY_TORCH`:

```bash
export PY_HJ=.venv-hj/bin/python PY_TORCH=.venv-torch/bin/python
bash scripts/reproduce_air3d.sh        # grids, Air3D safety-filter table, BRT figures
bash scripts/reproduce_quadrotor.sh    # quadrotor table, kappa = 3 and kappa = 2
bash scripts/reproduce_dubins.sh       # nominal Dubins grid, Dubins table and trajectory figure
```

Every table and figure of the paper produced by code, with the script that makes it and the
file it writes (`figures/` and `outputs/` are created on first use):

| Paper item | Script | Output | Env |
|---|---|---|---|
| Table `hyper`: training settings (App. A) | `configs/train/*.txt`, `scripts/train/train.py` | `runs/<name>/` | torch |
| Fig. `brt_sigma`: learned Air3D BRT vs. σ (Sec. 6.1) | `scripts/plotting/air3d_brt_deepreach.py` | `figures/air3d_brt_deepreach/brt_deepreach_compare_t_1.00_psi_3.14.pdf` | torch |
| Table `rollout_stats_offset03`: Air3D safety filter (Sec. 6.1) | `scripts/experiments/air3d_safety_filter.py` | `outputs/air3d_safety_filter/` | torch |
| Table `grid_vs_dr`, Fig. `grid_vs_dr`: learned vs. grid BRT (App. B) | `scripts/grid/air3d_ea_4d_grid.py` → `scripts/plotting/air3d_grid_vs_deepreach.py` | `figures/air3d_grid_vs_deepreach/` (`paper_overlay_4d_vs_deepreach_psi_pi_tau_1.00.pdf`, `metrics.csv`, `volume_metrics.csv`) | hj → torch |
| Fig. `nominal_vs_estaware_0p001` and the 13.6× / 37× BRT area ratios (Sec. 6.1, App. B) | `scripts/grid/air3d_nominal_grid.py --xy-max 1 --shape 201 201 200 --out-prefix data/air3d_nominal_xy1` → `scripts/plotting/air3d_grid_vs_nominal.py` | `figures/air3d_grid_vs_nominal/paper_overlay_vanilla_tau_1.00.pdf`; ratios = `expans` column of the printout | hj (grid), either (plot) |
| Table `quad_kappa`: quadrotor, κ = 2 and 3 (Sec. 6.2) | `scripts/experiments/quadrotor_safety_filter.py --n_rollouts 2000`, once per `--est_ckpt` | `outputs/quadrotor_safety_filter/kappa_{3,2}/table.tex` | torch |
| Fig. `quad_brt_slice` and the 3.65× area ratio at engagement (Sec. 6.2) | same script, κ = 3 run | `brt_slices_engagement.pdf`; ratio in the log | torch |
| Table `dubins`, Fig. `dubins_traj` (Sec. 6.3) | `scripts/grid/dubins_nominal_grid.py` → `scripts/experiments/dubins_safety_filter.py --n_rollouts 2000` | `outputs/dubins_safety_filter/` (`table.md`, `trajectories.pdf`) | hj → torch |

Fig. `motivation` (Sec. 4) is a schematic and has no script. `scripts/plotting/air3d_grid_4d.py`
plots more slices of the 4D grid solution; this figure is not in the paper.

The experiment scripts default to the paper's scenarios (initial-state ranges, goals,
thresholds, noise levels); only the number of rollouts is passed explicitly where shown.

Reference outputs of the runs reported in the paper are in `results/`
(`results/air3d/table.md`, `results/quadrotor/kappa_{3,2}/table.tex`,
`results/dubins/table.md`, ...). In these files and in the code, "vanilla" and "plain" mean
the paper's nominal BRT. The rollouts use common random numbers with fixed seeds, so with the
shipped checkpoints the tables should be reproduced up to GPU floating-point differences
(the Dubins table reproduces exactly).

Every script has `--help`; most also have a settings block at the top of the file.

## Baselines

Every experiment script runs the baselines of the paper next to the estimation-aware filter,
on the same initial states and pre-drawn noise:

| Baseline | Implementation |
|---|---|
| No filter | `none` method of each experiment script |
| Nominal / plain BRT, queried at the certainty-equivalent state | Air3D, Dubins: grid solutions from `scripts/grid/` (`GridVanilla`, `GridNominal`); quadrotor: plain 6D network `checkpoints/pquad_plain_6d` |
| Nominal BRT + constant margin κ√λ̄ (steady-state and largest covariance) | `InflatedVanilla` (Air3D), `InflatedOracle` (quadrotor), `Inflated` (Dubins) |
| Belief CBF ([Vahs et al., RA-L 2023](https://doi.org/10.1109/LRA.2023.3330662), [arXiv](https://arxiv.org/abs/2309.06499)) | `src/ea_reach/belief_cbf.py` (`BeliefCBF`, belief dynamics `Air3DBelief`, `QuadBelief`, `DubinsBelief`); the module docstring lists the adaptations to our setting |

A single method can be run with `--filters` (Air3D) or `--methods` (Dubins); `python -m
ea_reach.belief_cbf` runs the belief-CBF self-tests.

## Trained networks

| Checkpoint | Dynamics class | Used for |
|---|---|---|
| `checkpoints/air3d_ea_6d_k3` | `EstimationAwareAir3D` (κ = 3) | Air3D |
| `checkpoints/pquad_plain_6d` | `PlanarQuadrotor` | quadrotor, plain BRT baselines |
| `checkpoints/pquad_ea_9d_k3` | `EstimationAwarePlanarQuadrotor` (κ = 3) | quadrotor |
| `checkpoints/pquad_ea_9d_k2` | `EstimationAwarePlanarQuadrotor` (κ = 2, xy ∈ [-2.5, 2.5]) | quadrotor |
| `checkpoints/dubins_ea_9d_k3` | `EstimationAwareDubins3D` (κ = 3) | Dubins car |

Each folder follows the DeepReach run layout (`config_*.txt` +
`training/checkpoints/model_final.pth`); the scripts read the dynamics parameters from the
config next to the checkpoint and check them against their own settings.

## Training

```bash
$PY_TORCH scripts/train/train.py air3d_ea_6d_k3            # -> runs/air3d_ea_6d_k3/
$PY_TORCH scripts/train/train.py pquad_ea_9d_k3 --dry-run  # print the DeepReach command only
$PY_TORCH scripts/train/train.py pquad_plain_6d -- --use_wandb --wandb_project <project>
```

`configs/train/<name>.txt` is the config DeepReach saved for each network (settings in
App. A, Table `hyper`). The launcher turns it into a `run_experiment.py` command for the
submodule.

To use a new run, pass its checkpoint with `--ckpt` / `--est_ckpt`, or copy the run folder to
`checkpoints/<name>`.

## Citation

```bibtex
...
```

## Acknowledgements

Built on [DeepReach](https://github.com/smlbansal/deepreach) and its MPC-guided extensions
([DeepReach MPC](https://arxiv.org/abs/2505.03830), [MADR](https://arxiv.org/abs/2510.18845)),
and on [hj_reachability](https://github.com/StanfordASL/hj_reachability). The belief-CBF baseline implements the belief-space CBF formulation proposed in ([arXiv:2309.06499](https://arxiv.org/abs/2309.06499)).
