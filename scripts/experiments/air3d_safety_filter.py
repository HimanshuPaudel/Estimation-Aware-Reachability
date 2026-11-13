"""
Comparison experiment: No filter vs vanilla Air3D vs estimation-aware Air3D.

Setup (matches the notebook sketch)
  * N_POS initial positions placed at a radial OFFSET from the BRT boundary (V = 0 at t_query),
    found by bisection along rays from the evader.  Default 15 positions.
  * N_SEEDS noise seeds per position (default 500)  ->  N_POS * N_SEEDS rollouts
    per table row (default 7500), each simulated for T_sim = 5 s.
  * Initial covariance Sigma0 in CONFIG['sigmas'] (default {0.001, 0.025, 0.05}).
  * Filters: none / vanilla Air3D (ignores Sigma) / vanilla Air3D with an inflated target
    (constant margin kappa*sqrt(sigma), at the steady-state and the maximum covariance) /
    estimation-aware / belief CBF (belief_cbf.py).
    Inflating the target by a constant c is exact as a threshold shift: V_c = V - c, so the
    inflated filters reuse the vanilla grid with V - c (same gradient, same optimal control).
  * Pursuers (one block of rows each):
      pure_pursuit : saturated pure pursuit on the TRUE evader position
      worst_case   : HJ-optimal disturbance  uP = -uP_max * sign(dV/dpsi)  of the vanilla grid
                     value at the TRUE relative state (the pursuer knows the true state), at every
                     step (pure pursuit only where dV/dpsi is exactly 0).
  * Same evader, same pursuer, same Gaussian noise (process Q on the pursuer,
    measurement R, estimator with the Riccati covariance) for every filter:
    noise is pre-drawn once and shared (common random numbers), so differences
    between rows come from the filter only.

Columns
  Collision rate         fraction of rollouts where the TRUE distance <= Rc within T_sim
  Intervention rate      fraction of rollouts in which the filter engaged at least once
                         (fraction of time steps engaged is saved in the CSV)
  Mean time to capture   mean capture time over the rollouts that were captured
  First intervention t   mean time of the first engagement over rollouts that engaged
  Sims alive             rollouts not captured by T_sim (incl. ones that left the domain),
                         shown as  alive / total
  Min V est / Min V std  minimum over time of the value function evaluated at the TRUE relative
                         state (estimation-aware net with the filter's current Sigma, and the
                         vanilla grid).  Shown as mean over rollouts and fraction of rollouts
                         with min V < 0 (true state entered that BRT).  Median is in the CSV/JSON.

Inputs: the estimation-aware DeepReach network (checkpoints/air3d_ea_6d_k3) and the nominal
Air3D grid solution from scripts/grid/air3d_nominal_grid.py (data/air3d_nominal_xy3_*).

Usage (DeepReach / torch environment, from any directory):
  python scripts/experiments/air3d_safety_filter.py                 # uses CONFIG (paper setting)
  python scripts/experiments/air3d_safety_filter.py --offset 0.1    # override the radial offset
"""
import os
import sys
import csv
import json
import time
import argparse
import itertools
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "src"))
from ea_reach.paths import checkpoint, repo_path, OUTPUTS
import ea_reach.air3d_sim as sfa
from ea_reach.air3d_sim import (CFG, ValueModel, est_kwargs, std_kwargs,
                                     check_plant, est_dsdt, rk4, wrap)  # std_kwargs no longer needed
from ea_reach.belief_cbf import Air3DBelief, BeliefCBF, theorem3_margin

FILTERS = [("none", "No filter"), ("std", "Vanilla Air3D"),
           ("std_ss", "Vanilla Air3D + margin (steady state)"),
           ("std_max", "Vanilla Air3D + margin (max)"),
           ("est", "Estimation-aware Air3D"), ("bcbf", "Belief CBF")]
PURSUERS = [("pure_pursuit", "Pure pursuit"), ("worst_case", "Worst-case (HJ)")]

# =========================================================================== #
# CONFIG  -- edit these values, then run the file
# =========================================================================== #
CONFIG = dict(
    ckpt=checkpoint("air3d_ea_6d_k3"),                                   # estimation-aware DeepReach net (kappa=3)
    vanilla_values=repo_path("data", "air3d_nominal_xy3_values_all.npy"),   # grid HJ vanilla Air3D on [-3,3]^2, (nt, nx, ny, npsi)
    vanilla_coords=repo_path("data", "air3d_nominal_xy3_grid_coords.npy"),  # [x, y, psi]
    grid_T=1.0,                   # saved grid times are linspace(0, grid_T, nt)
    sigmas=[0.001, 0.025, 0.05],   # initial covariance Sigma0 rows
    n_pos=15,                     # initial positions near the BRT boundary
    n_seeds=500,                  # noise seeds per position  (rollouts per row = n_pos * n_seeds)
    psi0=np.pi / 2,               # initial relative heading [rad]
    boundary="est",               # "est": positions on the estimation-aware V=0 at each row's Sigma0
                                  # "std": positions on the vanilla Air3D V=0 (same for every Sigma0)
    offset=0.3,                   # radial shift of start points from V=0 [m] (+ = outside the BRT)
    threshold=0.05,               # filter engages when V <= threshold
    hysteresis=0.02,              # filter disengages when V > threshold + hysteresis
    seed=0,                       # base seed for the shared noise
    out=os.path.join(OUTPUTS, "air3d_safety_filter"),
    # Belief CBF baseline (belief_cbf.py).  delta is set from CFG["kappa"] (same sigma multiplier).
    bcbf_pole=2.0,                # ECBF gains zeta = (pole^2, 2*pole); sweep with --bcbf-pole
    bcbf_eps=None,                # Thm. 3 margin for Pr[leave C_b] <= eps; None -> margin 0
    bcbf_robust=True,             # worst case over the pursuer turn rate |uP| <= uP_max
    # Inflated vanilla baselines: constant margin kappa * sqrt(sigma) on the vanilla target.
    #   ss  : steady-state isotropic covariance sqrt(Q R)              (= 0.05 here)
    #   max : largest lambda_max(Sigma) in the training domain, sigma_hi + sigma_xy_max (= 0.075)
    margin_sigmas=dict(std_ss=CFG["sigma_ss"], std_max=CFG["sigma_hi"] + CFG["sigma_xy_max"]),
    # Pursuers to run (one block of rows each); see PURSUERS.  The paper reports worst_case only;
    # add "pure_pursuit" (or pass --pursuers pure_pursuit worst_case) for the extra block.
    pursuers=["worst_case"],
)


# --------------------------------------------------------------------------- #
# Vanilla Air3D from the grid HJ solution
# --------------------------------------------------------------------------- #
class GridVanilla:
    """Vanilla Air3D value function from the grid HJ solution (x, y, psi).

    V and its gradient are linearly interpolated (periodic in psi, states clamped to
    the grid).  The evader maximises V, so the HJ-optimal turn rate is
        uE* = uE_max * sign( y*V_x - x*V_y - V_psi ).
    Same interface as ValueModel: .n, .rng, .value_batch(t, Z), .query(t, Z)."""

    def __init__(self, values_path, coords_path, T, uE_max):
        from scipy.interpolate import RegularGridInterpolator
        vals = np.load(values_path)                                  # (nt, nx, ny, npsi)
        xs, ys, psis = [np.asarray(c, float) for c in np.load(coords_path, allow_pickle=True)]
        if abs((psis[-1] - psis[0]) - 2 * np.pi) < 1e-6:             # endpoint duplicated -> drop it
            psis, vals = psis[:-1], vals[..., :-1]
        self.nt, self.T, self.uE_max = vals.shape[0], T, uE_max
        self.times = np.linspace(0.0, T, self.nt)
        self.psi0 = psis[0]
        dpsi = 2 * np.pi / len(psis)
        pe = np.r_[psis[0] - dpsi, psis, psis[-1] + dpsi]            # periodic padding
        self.n = 3
        self.rng = np.array([[xs[0], xs[-1]], [ys[0], ys[-1]], [-np.pi, np.pi]])
        self.itp = []
        for k in range(self.nt):
            V = np.concatenate([vals[k][..., -1:], vals[k], vals[k][..., :1]], axis=2)
            gx, gy, gp = np.gradient(V, xs, ys, pe)
            self.itp.append([RegularGridInterpolator((xs, ys, pe), f, bounds_error=False, fill_value=None)
                             for f in (V, gx, gy, gp)])

    def _frame(self, t):
        return int(np.argmin(np.abs(self.times - t)))

    def _prep(self, Z):
        Z = np.array(Z, float)[:, :3]
        Z[:, 0] = np.clip(Z[:, 0], *self.rng[0])
        Z[:, 1] = np.clip(Z[:, 1], *self.rng[1])
        Z[:, 2] = self.psi0 + ((Z[:, 2] - self.psi0) % (2 * np.pi))   # wrap into [psi0, psi0 + 2pi)
        return Z

    def value_batch(self, t, Z):
        return self.itp[self._frame(t)][0](self._prep(Z))

    def query(self, t, Z):
        Zp = self._prep(Z)
        V, gx, gy, gp = [f(Zp) for f in self.itp[self._frame(t)]]
        u = self.uE_max * np.sign(Zp[:, 1] * gx - Zp[:, 0] * gy - gp)
        return V, u

    def pursuer_terms(self, t, Z):
        """V and dV/dpsi.  uP enters only psi_dot = uP - uE, so the pursuer (minimising V)
        picks uP* = -uP_max * sign(dV/dpsi)."""
        Zp = self._prep(Z)
        itp = self.itp[self._frame(t)]
        return itp[0](Zp), itp[3](Zp)


class InflatedVanilla:
    """Vanilla Air3D with the target inflated by a constant margin c.
    For the avoid problem with cost min_tau l, replacing l by l - c gives exactly V - c,
    so this is the vanilla grid shifted by c (same gradient, same optimal control)."""

    def __init__(self, base, margin):
        self.base, self.margin = base, float(margin)
        self.n, self.rng = base.n, base.rng

    def value_batch(self, t, Z):
        return self.base.value_batch(t, Z) - self.margin

    def query(self, t, Z):
        V, u = self.base.query(t, Z)
        return V - self.margin, u


def sanity_check_grid_control(grid, est, c=CFG, n=2000, seed=1):
    """Compare the grid-derived turn rate with the estimation-aware net's optimal turn rate at
    the smallest Sigma (they should nearly agree) and print the V difference.  If the agreement is
    ~0 the sign convention is flipped (agreement ~1 - x means flip it)."""
    rng = np.random.default_rng(seed)
    r, a = rng.uniform(c["Rc"], 1.5, n), rng.uniform(-np.pi, np.pi, n)
    Z = np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(-np.pi, np.pi, n)]
    Vg, ug = grid.query(c["t_query"], Z)
    s = c["sigma_lo"]
    Ze = np.c_[Z, np.full(n, s), np.full(n, s), np.zeros(n)]
    Ve, ue = query_batch(est, c["t_query"], Ze)
    band = np.abs(Vg) < 0.3                                       # states where the control matters
    agree = float(np.mean(np.sign(ug[band]) == np.sign(ue[band]))) if band.any() else float("nan")
    print(f"grid-vs-network check (Sigma={s:g}): sign(u) agreement {agree:.2f} on {int(band.sum())} states, "
          f"median |V_grid - V_net| = {np.median(np.abs(Vg - Ve)):.3f}")
    return agree


def assert_matches_training(ckpt, kw):
    """kappa enters boundary_fn (added by io_to_value) and the ranges set the input normalisation,
    so the dynamics kwargs must equal the checkpoint's training config."""
    exp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))))
    cfgs = sorted(f for f in os.listdir(exp) if f.startswith("config_")) if os.path.isdir(exp) else []
    if not cfgs:
        print(f"WARNING: no training config next to {ckpt}; cannot check kappa / xy_max")
        return
    train = dict(l.strip().split(" = ", 1) for l in open(os.path.join(exp, cfgs[-1])) if " = " in l)
    bad = {k: (v, train[k]) for k, v in kw.items() if k in train and abs(float(train[k]) - float(v)) > 1e-9}
    assert not bad, f"CFG differs from training config {cfgs[-1]} (here, trained): {bad} -> edit CFG"
    print(f"est model kwargs match training config {cfgs[-1]} (kappa={kw['kappa']}, xy_max={kw['xy_max']})")


# --------------------------------------------------------------------------- #
# Initial positions on the BRT boundary
# --------------------------------------------------------------------------- #
def boundary_positions(model, with_sigma, sigma, psi0, n_pos, t, c=CFG,
                       offset=0.0, n_rays=180, n_bisect=30):
    """Return n_pos points (x, y) in the evader body frame where V(t, x, y, psi0[, sigma]) ~= 0,
    shifted radially by `offset`.
    One ray per angle from the evader; bisect between the capture circle (V <= 0) and the
    domain edge (V > 0).  Rays with no sign change are discarded, the rest are subsampled evenly."""
    ang = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    r_lo = np.full(n_rays, c["Rc"])
    r_hi = np.full(n_rays, float(np.min(model.rng[:2, 1])) - 1e-3)

    def V_at(r):
        Z = np.c_[r * np.cos(ang), r * np.sin(ang), np.full(n_rays, psi0)]
        if with_sigma:
            Z = np.c_[Z, np.full(n_rays, sigma), np.full(n_rays, sigma), np.zeros(n_rays)]
        return model.value_batch(t, Z)

    ok = (V_at(r_lo) <= 0) & (V_at(r_hi) > 0)
    for _ in range(n_bisect):
        mid = 0.5 * (r_lo + r_hi)
        neg = V_at(mid) <= 0
        r_lo = np.where(neg, mid, r_lo)
        r_hi = np.where(neg, r_hi, mid)
    r_b = 0.5 * (r_lo + r_hi) + offset
    valid = np.flatnonzero(ok)
    if len(valid) < n_pos:
        raise RuntimeError(f"Only {len(valid)} rays cross V=0 (need {n_pos}) for sigma={sigma}, "
                           f"psi0={psi0:.2f}. Try another --psi0 or a smaller --n-pos.")
    pick = valid[np.round(np.linspace(0, len(valid) - 1, n_pos)).astype(int)]
    return np.c_[r_b[pick] * np.cos(ang[pick]), r_b[pick] * np.sin(ang[pick])]


# --------------------------------------------------------------------------- #
# Batched value / optimal-control query
# --------------------------------------------------------------------------- #
def query_batch(vm, t, Z, chunk=4096):
    """V and HJ-optimal evader turn rate for a batch of states (rows of Z)."""
    if isinstance(vm, (GridVanilla, InflatedVanilla)):
        return vm.query(t, Z)
    Z = np.array(Z, dtype=float)[:, :vm.n]
    Z[:, 2] = wrap(Z[:, 2])
    Zc = np.clip(Z, vm.rng[:, 0], vm.rng[:, 1])
    Vs, Us = [], []
    for s in range(0, len(Zc), chunk):
        zc = Zc[s:s + chunk]
        coord = torch.tensor(np.c_[np.full(len(zc), t), zc], dtype=torch.float32, device=sfa.DEVICE)
        inp = vm.dyn.coord_to_input(coord)
        out = vm.model({"coords": inp})
        mo = out["model_out"].squeeze(-1)
        V = vm.dyn.io_to_value(out["model_in"], mo)
        dv = vm.dyn.io_to_dv(out["model_in"], mo)
        dV = dv[:, 1:].detach().cpu().double()
        u = vm.dyn.optimal_control(torch.tensor(zc, dtype=torch.float64), dV)
        Vs.append(V.detach().cpu().numpy().reshape(-1))
        Us.append(u.detach().cpu().numpy().reshape(len(zc), -1)[:, 0])
    return np.concatenate(Vs), np.concatenate(Us)


# --------------------------------------------------------------------------- #
# Batched simulation (same physics as ea_reach.air3d_sim.simulate)
# --------------------------------------------------------------------------- #
def make_noise(n, n_steps, seed):
    rng = np.random.default_rng(seed)
    return dict(init=rng.standard_normal((n, 2)),
                meas=rng.standard_normal((n_steps, n, 2)).astype(np.float32),
                proc=rng.standard_normal((n_steps, n, 2)).astype(np.float32))


def simulate_batch(p0, psi0, sigma0, filt, model, noise, a, c=CFG, models_all=None,
                   pursuer="pure_pursuit", grid=None):
    N, dt = len(p0), c["dt"]
    T = int(round(c["T_sim"] / dt))
    thE, E = np.full(N, np.pi / 2), np.zeros((N, 2))
    cE, sE = np.cos(thE), np.sin(thE)
    P = E + np.c_[cE * p0[:, 0] - sE * p0[:, 1], sE * p0[:, 0] + cE * p0[:, 1]]
    thP = thE + psi0

    z = np.zeros((6, N))                       # [x_hat, y_hat, psi, Sxx, Syy, Sxy]
    z[:2] = (p0 + np.sqrt(sigma0) * noise["init"]).T
    z[2], z[3], z[4], z[5] = wrap(psi0), sigma0, sigma0, 0.0

    done = np.zeros(N, bool)
    captured = np.zeros(N, bool)
    left = np.zeros(N, bool)
    t_cap = np.full(N, np.nan)
    first_t = np.full(N, np.nan)
    engaged = np.zeros(N, bool)
    qp_infeasible = np.zeros(N, bool)          # belief CBF only: QP had no solution at some step
    act_steps = np.zeros(N)
    live_steps = np.zeros(N)
    wc_steps = np.zeros(N)                     # steps in which the pursuer used the HJ disturbance
    alive_steps = np.zeros(N)
    min_V_est = np.full(N, np.inf)             # min over time of V at the TRUE state
    min_V_std = np.full(N, np.inf)

    for k in range(T):
        live = ~done
        if not live.any():
            break
        dx = P - E
        cs, sn = np.cos(thE), np.sin(thE)
        p = np.c_[cs * dx[:, 0] + sn * dx[:, 1], -sn * dx[:, 0] + cs * dx[:, 1]]  # true rel. pos (body frame)

        cap_now = live & (np.hypot(p[:, 0], p[:, 1]) <= c["Rc"])
        captured |= cap_now
        t_cap[cap_now] = k * dt
        left_now = live & ~cap_now & ((np.abs(z[0]) > c["xy_max"]) | (np.abs(z[1]) > c["xy_max"]))
        left |= left_now
        done |= cap_now | left_now
        live = ~done
        if not live.any():
            break

        z[2] = wrap(thP - thE)
        alive_steps[live] += 1

        # min-V metric: value at the TRUE relative state (filter's current Sigma for the est net)
        if models_all is not None:
            im = np.flatnonzero(live)
            Zt = np.c_[p[im], z[2, im], z[3:6, im].T]
            Ve_, _ = query_batch(models_all["est"], c["t_query"], Zt)
            Vs_, _ = query_batch(models_all["std"], c["t_query"], Zt)
            min_V_est[im] = np.minimum(min_V_est[im], Ve_)
            min_V_std[im] = np.minimum(min_V_std[im], Vs_)

        u_nom = np.clip(c["heading_gain"] * wrap(np.pi / 2 - thE), -c["uE_max"], c["uE_max"])
        uE = u_nom.copy()
        if filt == "bcbf":
            # belief CBF: QP around the nominal command at every step (no threshold / hysteresis)
            idx = np.flatnonzero(live)
            u_b, _, e, feas = model.filter(z.T[idx], u_nom[idx])
            uE[idx] = u_b[:, 0]
            qp_infeasible[idx] |= ~feas
            new = idx[e & np.isnan(first_t[idx])]
            first_t[new] = k * dt
            act_steps[idx] += e
            live_steps[idx] += 1
        elif filt != "none":
            idx = np.flatnonzero(live)
            V, u_opt = query_batch(model, c["t_query"], z.T[idx])
            e = np.where(V <= a.threshold, True, np.where(V > a.threshold + a.hysteresis, False, engaged[idx]))
            engaged[idx] = e
            uE[idx] = np.where(e, u_opt, u_nom[idx])
            new = idx[e & np.isnan(first_t[idx])]
            first_t[new] = k * dt
            act_steps[idx] += e
            live_steps[idx] += 1

        err = wrap(np.arctan2(E[:, 1] - P[:, 1], E[:, 0] - P[:, 0]) - thP)
        uP = c["uP_max"] * np.clip(c["pursuer_gain"] * err, -1.0, 1.0)
        if pursuer == "worst_case":
            # HJ-optimal disturbance of the vanilla value at the TRUE relative state
            im = np.flatnonzero(live)
            Vw, gpsi = grid.pursuer_terms(c["t_query"], np.c_[p[im], z[2, im]])
            wc = np.abs(gpsi) > 1e-6
            uP[im[wc]] = -c["uP_max"] * np.sign(gpsi[wc])
            wc_steps[im] += wc

        # estimator: Riccati propagation + innovation on the noisy measurement
        zn = rk4(lambda s: est_dsdt(s, uE, uP, c), z, dt)
        meas = p + np.sqrt(c["R"] / dt) * noise["meas"][k].astype(float)
        i0, i1 = meas[:, 0] - z[0], meas[:, 1] - z[1]
        zn[0] += dt * (z[3] * i0 + z[5] * i1) / c["R"]
        zn[1] += dt * (z[5] * i0 + z[4] * i1) / c["R"]
        zn[3] = np.clip(zn[3], c["sigma_lo"], c["sigma_hi"])
        zn[4] = np.clip(zn[4], c["sigma_lo"], c["sigma_hi"])
        lim = np.sqrt(zn[3] * zn[4])
        zn[5] = np.clip(zn[5], -lim, lim)

        # true world motion
        hE, hP = thE + 0.5 * dt * uE, thP + 0.5 * dt * uP
        En = E + dt * c["vE"] * np.c_[np.cos(hE), np.sin(hE)]
        Pn = (P + dt * c["vP"] * np.c_[np.cos(hP), np.sin(hP)]
              + np.sqrt(c["Q"] * dt) * noise["proc"][k].astype(float))
        thEn, thPn = thE + dt * uE, thP + dt * uP

        E[live], P[live], thE[live], thP[live] = En[live], Pn[live], thEn[live], thPn[live]
        z[:, live] = zn[:, live]

    return dict(captured=captured, left=left, t_cap=t_cap, first_t=first_t,
                intervened=~np.isnan(first_t),
                frac_time_active=np.divide(act_steps, live_steps, out=np.zeros(N), where=live_steps > 0),
                min_V_est=min_V_est, min_V_std=min_V_std, qp_infeasible=qp_infeasible,
                frac_time_wc=np.divide(wc_steps, alive_steps, out=np.zeros(N), where=alive_steps > 0))


# --------------------------------------------------------------------------- #
# Belief CBF baseline
# --------------------------------------------------------------------------- #
def make_bcbf(a, c=CFG):
    """Belief CBF on the same belief z = [x_hat, y_hat, psi, Sxx, Syy, Sxy] as the HJ filter."""
    sys_b = Air3DBelief(vP=c["vP"], vE=c["vE"], uE_max=c["uE_max"], uP_max=c["uP_max"], Q=c["Q"], R=c["R"])
    check_bcbf_dynamics(sys_b, c)
    probe = BeliefCBF(sys_b, R_c=c["Rc"], kappa=c["kappa"])
    margin = 0.0
    if a.bcbf_eps is not None:
        margin = theorem3_margin((c["sigma_lo"], c["sigma_hi"]), c["R"], c["dt"], probe.delta, a.bcbf_eps)
    cbf = BeliefCBF(sys_b, R_c=c["Rc"], kappa=c["kappa"], pole=a.bcbf_pole, margin=margin,
                    robust=a.bcbf_robust)
    print(f"belief CBF: delta={cbf.delta:.4f} (kappa={c['kappa']}), pole={a.bcbf_pole}, "
          f"margin gamma={margin:.4f}, robust={a.bcbf_robust}")
    return cbf


def check_bcbf_dynamics(sys_b, c=CFG, n=200, seed=2):
    """The belief CBF's F + G uE + D uP must equal the estimator drift used in simulation (est_dsdt)."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        z = np.array([*rng.uniform(-2, 2, 2), rng.uniform(-3, 3),
                      *rng.uniform(0.001, 0.05, 2), rng.uniform(-0.02, 0.02)])
        uE, uP = rng.uniform(-4, 4), rng.uniform(-3, 3)
        t = torch.tensor(z[None], dtype=torch.float64)
        f = (sys_b.F(t) + sys_b.G(t)[..., 0] * uE + sys_b.D(t)[..., 0] * uP)[0].numpy()
        assert np.allclose(f, est_dsdt(z, uE, uP, c), atol=1e-10), "belief CBF dynamics != est_dsdt"


# --------------------------------------------------------------------------- #
# Aggregation / output
# --------------------------------------------------------------------------- #
def wilson(k, n, zc=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + zc ** 2 / n
    m = (ph + zc ** 2 / (2 * n)) / d
    h = zc * np.sqrt(ph * (1 - ph) / n + zc ** 2 / (4 * n ** 2)) / d
    return (m - h, m + h)


MINV_KEYS = ("min_V_est", "min_V_std")
MINV_STATS = ("mean", "median", "frac_below0")


def summarize(r, filt):
    n = len(r["captured"])
    nc = int(r["captured"].sum())
    lo, hi = wilson(nc, n)
    out = dict(
        n=n, n_captured=nc, collision_rate=nc / n, collision_ci95=[lo, hi],
        mean_time_to_capture=float(np.nanmean(r["t_cap"])) if nc else None,
        n_alive=int(n - nc), n_left_domain=int(r["left"].sum()),
        qp_infeasible_rate=float(r["qp_infeasible"].mean()) if filt == "bcbf" else None,
        frac_time_wc_pursuer=float(r["frac_time_wc"].mean()),
    )
    if filt == "none":
        out.update(intervention_rate=None, frac_time_active=None,
                   first_intervention_mean=None, first_intervention_std=None)
    else:
        ni = int(r["intervened"].sum())
        out.update(intervention_rate=ni / n, frac_time_active=float(r["frac_time_active"].mean()),
                   first_intervention_mean=float(np.nanmean(r["first_t"])) if ni else None,
                   first_intervention_std=float(np.nanstd(r["first_t"])) if ni else None)
    for key in MINV_KEYS:
        m = r[key][np.isfinite(r[key])]
        out[f"{key}_mean"] = float(np.mean(m))
        out[f"{key}_median"] = float(np.median(m))
        out[f"{key}_frac_below0"] = float(np.mean(m < 0))
    return out


def fmt(v, spec, none="N/A"):
    return none if v is None else format(v, spec)


def render_table(rows, n_total):
    head = ["Pursuer", "Filter", "Sigma0", "Collision rate", "Intervention rate",
            "Mean time to capture (s)", "First intervention t (s)", f"Sims alive (/{n_total})",
            "Min V est (mean)", "Min V est (<0)", "Min V std (mean)", "Min V std (<0)"]
    body = []
    for r in rows:
        s = r["stats"]
        body.append([r["pursuer_label"], r["filter_label"], f"{r['sigma0']:g}",
                     f"{100 * s['collision_rate']:.1f}%",
                     "N/A" if s["intervention_rate"] is None else f"{100 * s['intervention_rate']:.1f}%",
                     fmt(s["mean_time_to_capture"], ".2f", "-"),
                     fmt(s["first_intervention_mean"], ".2f", "N/A" if s["intervention_rate"] is None else "-"),
                     f"{s['n_alive']}",
                     f"{s['min_V_est_mean']:.3f}", f"{100 * s['min_V_est_frac_below0']:.1f}%",
                     f"{s['min_V_std_mean']:.3f}", f"{100 * s['min_V_std_frac_below0']:.1f}%"])
    return head, body


def to_markdown(head, body):
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def to_plain(head, body):
    w = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(head)]
    line = lambda r: "  ".join(x.ljust(w[i]) for i, x in enumerate(r))
    out = [line(head), "  ".join("-" * x for x in w)]
    prev = None
    for r in body:
        if prev is not None and r[:2] != prev:
            out.append("")
        out.append(line(r))
        prev = r[:2]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=float, default=None, help="override CONFIG['offset'] [m]")
    ap.add_argument("--out", type=str, default=None, help="override output directory")
    ap.add_argument("--bcbf-pole", type=float, default=None, help="override CONFIG['bcbf_pole']")
    ap.add_argument("--bcbf-eps", type=float, default=None, help="Thm. 3 eps (adds the margin gamma)")
    ap.add_argument("--filters", nargs="+", default=None, choices=[k for k, _ in FILTERS],
                    help="run only these filters (e.g. --filters bcbf for a gain sweep)")
    ap.add_argument("--pursuers", nargs="+", default=None, choices=[k for k, _ in PURSUERS],
                    help="run only these pursuers (default: CONFIG['pursuers'])")
    ap.add_argument("--n-seeds", type=int, default=None, help="override CONFIG['n_seeds']")
    ap.add_argument("--n-pos", type=int, default=None, help="override CONFIG['n_pos']")
    cli = ap.parse_args()

    cfg = dict(CONFIG)
    if cli.offset is not None:
        cfg["offset"] = cli.offset
        cfg["out"] = f"{CONFIG['out']}_off{cli.offset:g}"
    if cli.out is not None:
        cfg["out"] = cli.out
    if cli.bcbf_pole is not None:
        cfg["bcbf_pole"] = cli.bcbf_pole
    if cli.bcbf_eps is not None:
        cfg["bcbf_eps"] = cli.bcbf_eps
    if cli.pursuers is not None:
        cfg["pursuers"] = cli.pursuers
    if cli.n_seeds is not None:
        cfg["n_seeds"] = cli.n_seeds
    if cli.n_pos is not None:
        cfg["n_pos"] = cli.n_pos
    a = SimpleNamespace(**cfg)
    assert a.boundary in ("est", "std"), "CONFIG['boundary'] must be 'est' or 'std'"
    os.makedirs(a.out, exist_ok=True)

    assert_matches_training(a.ckpt, est_kwargs())
    est = ValueModel(a.ckpt, "EstimationAwareAir3D", est_kwargs())
    check_plant(est.dyn)
    std = GridVanilla(a.vanilla_values, a.vanilla_coords, a.grid_T, CFG["uE_max"])
    sanity_check_grid_control(std, est)
    bcbf = make_bcbf(a)
    models = {"est": est, "std": std, "bcbf": bcbf}
    for key, sig in a.margin_sigmas.items():
        margin = CFG["kappa"] * np.sqrt(sig)
        models[key] = InflatedVanilla(std, margin)
        print(f"{key}: vanilla target inflated by kappa*sqrt({sig:g}) = {margin:.4f} "
              f"(filter engages when V_std <= {a.threshold + margin:.4f})")
    filters = [f for f in FILTERS if cli.filters is None or f[0] in cli.filters]
    pursuers = [p for p in PURSUERS if p[0] in a.pursuers]

    N = a.n_pos * a.n_seeds
    T = int(round(CFG["T_sim"] / CFG["dt"]))
    noise = make_noise(N, T, a.seed)       # shared by every row -> paired comparison
    print(f"{a.n_pos} positions x {a.n_seeds} seeds = {N} rollouts per row, {CFG['T_sim']:g}s horizon, "
          f"offset={a.offset:g} m, device={sfa.DEVICE}")

    rows, positions = [], {}
    for (pkey, plabel), (fkey, flabel), s0 in itertools.product(pursuers, filters, a.sigmas):
        if a.boundary == "est":
            pos = boundary_positions(est, True, s0, a.psi0, a.n_pos, CFG["t_query"], CFG, a.offset)
        else:
            pos = boundary_positions(std, False, None, a.psi0, a.n_pos, CFG["t_query"], CFG, a.offset)
        positions[f"{s0:g}"] = pos.tolist()
        p0 = np.repeat(pos, a.n_seeds, axis=0)
        t0 = time.time()
        r = simulate_batch(p0, a.psi0, s0, fkey, models.get(fkey), noise, a, models_all=models,
                           pursuer=pkey, grid=std)
        st = summarize(r, fkey)
        rows.append(dict(pursuer=pkey, pursuer_label=plabel, filter=fkey, filter_label=flabel,
                         margin=getattr(models.get(fkey), "margin", None), sigma0=s0, stats=st))
        print(f"  [{plabel}] {flabel:38s} Sigma0={s0:<6g} collision {100 * st['collision_rate']:5.1f}%  "
              f"alive {st['n_alive']}/{N}  minV_est {st['min_V_est_mean']:.3f}  "
              + (f"QP infeasible {100 * st['qp_infeasible_rate']:.1f}%  " if fkey == "bcbf" else "")
              + (f"HJ pursuer active {100 * st['frac_time_wc_pursuer']:.0f}% of steps  "
                 if pkey == "worst_case" else "")
              + f"({time.time() - t0:.0f}s)", flush=True)

    head, body = render_table(rows, N)
    print("\n" + to_plain(head, body))
    print(f"\nAcross {N} rollouts per row ({CFG['T_sim']:g} s horizon), offset = {a.offset:g} m.")

    with open(os.path.join(a.out, "table.md"), "w") as f:
        f.write(to_markdown(head, body)
                + f"\n\nAcross {N} rollouts per row ({CFG['T_sim']:g} s horizon), offset = {a.offset:g} m.\n")
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pursuer", "filter", "margin", "sigma0", "n", "n_captured", "collision_rate", "ci95_lo", "ci95_hi",
                    "intervention_rate", "frac_time_active", "mean_time_to_capture",
                    "first_intervention_mean", "first_intervention_std", "n_alive", "n_left_domain",
                    "qp_infeasible_rate", "frac_time_wc_pursuer"]
                   + [f"{k}_{s}" for k in MINV_KEYS for s in MINV_STATS])
        for r in rows:
            s = r["stats"]
            w.writerow([r["pursuer_label"], r["filter_label"], r["margin"], r["sigma0"], s["n"], s["n_captured"], s["collision_rate"],
                        *s["collision_ci95"], s["intervention_rate"], s["frac_time_active"],
                        s["mean_time_to_capture"], s["first_intervention_mean"],
                        s["first_intervention_std"], s["n_alive"], s["n_left_domain"],
                        s["qp_infeasible_rate"], s["frac_time_wc_pursuer"]]
                       + [s[f"{k}_{st}"] for k in MINV_KEYS for st in MINV_STATS])
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(dict(args=vars(a), cfg={k: v for k, v in CFG.items()}, rows=rows,
                       initial_positions=positions), f, indent=2, default=float)
    print("saved", os.path.join(a.out, "table.md / results.csv / results.json"))


if __name__ == "__main__":
    main()