#!/usr/bin/env python
"""
Safety-filter comparison for the 9D estimation-aware Dubins car (EstimationAwareDubins3D).

True system + CT-EKF (position-only beacon measurements, noise r(p) = r0 + r1 ||p - b||^2) with a
goal-seeking heading controller that drives from near the beacon (2, 0) through an obstacle at the
origin to the goal (-2, 0), so the position uncertainty grows along the way.  Every method runs on
the SAME initial states and the SAME pre-drawn noise (common random numbers), so differences between
rows come from the filter alone.

Methods
  none     nominal controller only
  nom      least-restrictive filter from the nominal 3D Dubins BRT (grid, scripts/grid/dubins_nominal_grid.py),
           queried at the certainty-equivalent state (x_hat, y_hat, th_hat)
  nom_ss   nominal BRT with the target inflated by kappa*sqrt(lambda_ss), lambda_ss = lambda_max of the
           steady-state EKF position covariance at the obstacle (CARE with r(0))
  nom_max  same with lambda_max of the largest covariance in the training box (S_hi + Sxy_max)
           Inflating the target by a constant m is exact as a threshold shift for the avoid problem:
           V_m = V - m (same gradient, same optimal control).
  est      filter from the 9D estimation-aware value function (DeepReach), queried at the belief z
           (skipped if --est_ckpt is missing)
  bcbf     belief CBF (belief_cbf.DubinsBelief), QP on omega at every step
A value-function filter applies its HJ-optimal turn rate whenever V <= delta (+ optional hysteresis).

Metrics (true state): collision rate (||p|| <= Rc) with 95% Wilson CI, min clearance (mean / worst),
intervention rate, first-intervention time and distance, active fraction, effort int |w - w_nom| dt,
goal reached, left domain, QP infeasible (belief CBF; rollouts with any infeasible step and fraction of
steps).  Infeasibility is expected when the car heads straight at the obstacle centre: there
d(h')/d(theta) = 0, so omega has no first-order effect on h''.

Inputs: the nominal Dubins grid from scripts/grid/dubins_nominal_grid.py (data/dubins_nominal_*)
and the 9D estimation-aware DeepReach network (default checkpoints/dubins_ea_9d_k3; the 'est'
row is skipped if it is missing).

Usage (DeepReach / torch environment, from any directory):
  python scripts/experiments/dubins_safety_filter.py --n_rollouts 2000          # paper table (Sec. 6.3)
  python scripts/experiments/dubins_safety_filter.py --est_ckpt <run>/training/checkpoints/model_final.pth
  python scripts/experiments/dubins_safety_filter.py --methods bcbf --bcbf_pole 4 --out outputs/dubins_pole4   # CBF gain sweep
The 9D dynamics parameters are read from the checkpoint's training config (config_*.txt) when
available, otherwise from the class defaults (+ --dyn_kwargs).
"""
import argparse
import csv
import inspect
import json
import math
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_continuous_are

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "src"))
from ea_reach.paths import add_deepreach_to_path, checkpoint, repo_path, OUTPUTS   # noqa: E402

DEEPREACH_DIR = add_deepreach_to_path()
from dynamics import dynamics as D          # noqa: E402
from utils import modules                   # noqa: E402
from ea_reach.belief_cbf import DubinsBelief, BeliefCBF, theorem3_margin   # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METHODS = [("none", "No filter"), ("nom", "Nominal BRT"),
           ("nom_ss", "Nominal BRT + margin (steady state)"), ("nom_max", "Nominal BRT + margin (max)"),
           ("est", "Estimation-aware BRT"), ("bcbf", "Belief CBF")]
LAB = dict(METHODS)
COL = {"none": "0.45", "nom": "k", "nom_ss": "tab:blue", "nom_max": "tab:cyan",
       "est": "tab:orange", "bcbf": "tab:purple"}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# =============================================================================================== #
# Value functions
# =============================================================================================== #
class GridNominal:
    """Nominal Dubins3D value from the hj_reachability grid; linear interpolation (periodic in theta).
    The car maximises V and omega enters only theta' -> omega* = omega_max sign(dV/dtheta)."""

    def __init__(self, values_path, coords_path, T, t_query, omega_max):
        vals = np.load(values_path)
        xs, ys, ths = [np.asarray(c, float) for c in np.load(coords_path, allow_pickle=True)]
        nt = vals.shape[0]
        k = int(np.argmin(np.abs(np.linspace(0.0, T, nt) - t_query)))
        V = vals[k]
        dth = ths[1] - ths[0]
        te = np.r_[ths[0] - dth, ths, ths[-1] + dth]
        V = np.concatenate([V[..., -1:], V, V[..., :1]], axis=2)
        gth = np.gradient(V, te, axis=2)
        self.lo = np.array([xs[0], ys[0]]); self.hi = np.array([xs[-1], ys[-1]])
        self.th0, self.omega_max = ths[0], omega_max
        self.itp_V = RegularGridInterpolator((xs, ys, te), V, bounds_error=False, fill_value=None)
        self.itp_g = RegularGridInterpolator((xs, ys, te), gth, bounds_error=False, fill_value=None)
        print(f"nominal grid: {vals.shape}, frame t={k * T / (nt - 1):.2f}")

    def _prep(self, z):
        Z = np.array(z[:, :3], float)
        Z[:, :2] = np.clip(Z[:, :2], self.lo, self.hi)
        Z[:, 2] = self.th0 + ((Z[:, 2] - self.th0) % (2 * np.pi))
        return Z

    def query(self, z):
        Z = self._prep(z)
        return self.itp_V(Z), self.omega_max * np.sign(self.itp_g(Z))


class Inflated:
    """Target inflated by a constant margin m: V_m = V - m (exact for the avoid problem)."""

    def __init__(self, base, margin):
        self.base, self.margin = base, float(margin)

    def query(self, z):
        V, u = self.base.query(z)
        return V - self.margin, u


class EstOracle:
    """9D estimation-aware DeepReach value function and its optimal turn rate."""

    def __init__(self, dyn, ckpt, net_cfg, t_eval):
        self.dyn, self.t = dyn, t_eval
        net = modules.SingleBVPNet(in_features=dyn.input_dim, out_features=1, type=net_cfg["model"],
                                   mode=net_cfg["model_mode"], final_layer_factor=1.,
                                   hidden_features=net_cfg["num_nl"], num_hidden_layers=net_cfg["num_hl"],
                                   periodic_transform_fn=dyn.periodic_transform_fn)
        sd = torch.load(ckpt, map_location=DEV)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        net.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
        self.net = net.to(DEV).eval()

    def query(self, z, chunk=8192):
        Vs, Us = [], []
        for i in range(0, len(z), chunk):
            zz = np.array(z[i:i + chunk], float)
            zz[:, 2] = wrap(zz[:, 2])
            zt = self.dyn.clip_state(torch.as_tensor(zz, dtype=torch.float32, device=DEV))
            tt = torch.full((len(zt), 1), self.t, device=DEV)
            inp = self.dyn.coord_to_input(torch.cat([tt, zt], -1))
            with torch.enable_grad():
                out = self.net({"coords": inp})
                x, y = out["model_in"], out["model_out"].squeeze(-1)
                V = self.dyn.io_to_value(x, y)
                dv = self.dyn.io_to_dv(x, y)
            u = self.dyn.optimal_control(zt, dv[..., 1:].detach())
            Vs.append(V.detach().cpu().numpy()); Us.append(u.detach().cpu().numpy()[:, 0])
        return np.concatenate(Vs), np.concatenate(Us)


# =============================================================================================== #
# Training config / dynamics
# =============================================================================================== #
def read_train_config(ckpt):
    """Parse config_*.txt of the experiment that produced ckpt (…/<exp>/training/checkpoints/x.pth)."""
    if not ckpt:
        return {}
    exp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))))
    cfgs = sorted(f for f in os.listdir(exp) if f.startswith("config_") and f.endswith(".txt")) \
        if os.path.isdir(exp) else []
    out = {}
    if cfgs:
        for line in open(os.path.join(exp, cfgs[-1])):
            if " = " in line:
                k, v = line.split(" = ", 1)
                out[k.strip()] = v.strip()
        print(f"training config: {os.path.join(exp, cfgs[-1])}")
    return out


def make_dynamics(train_cfg, overrides):
    """EstimationAwareDubins3D with the training-config values (so V and the CBF/plant agree)."""
    kw = {}
    for name, p in inspect.signature(D.EstimationAwareDubins3D).parameters.items():
        if name in train_cfg and p.default is not inspect.Parameter.empty and name != "set_mode":
            try:
                kw[name] = type(p.default)(train_cfg[name])
            except ValueError:
                pass
    kw.update(overrides)
    dyn = D.EstimationAwareDubins3D(set_mode="avoid", **kw)
    print("dynamics: " + ", ".join(f"{k}={getattr(dyn, a)}" for k, a in
                                   [("v", "v"), ("omega_max", "omega_max"), ("kappa", "kappa"), ("Rc", "Rc"),
                                    ("beacon_x", "bx"), ("beacon_y", "by"), ("r0", "r0"), ("r1", "r1"),
                                    ("qx", "qx"), ("qy", "qy"), ("qt", "qt")]))
    return dyn


def r_of(dyn, p):
    return dyn.r0 + dyn.r1 * ((p[..., 0] - dyn.bx) ** 2 + (p[..., 1] - dyn.by) ** 2)


def steady_state_cov(dyn, p, th=0.0):
    """Steady-state EKF covariance for straight motion at heading th with the noise frozen at r(p):
    A S + S A^T + Q - S C^T C S / r = 0  (CARE of the dual problem)."""
    A = np.array([[0, 0, -dyn.v * math.sin(th)], [0, 0, dyn.v * math.cos(th)], [0, 0, 0]])
    C = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    Q = np.diag([dyn.qx, dyn.qy, dyn.qt])
    return solve_continuous_are(A.T, C.T, Q, float(r_of(dyn, np.asarray(p, float))) * np.eye(2))


def lam_max_pos(S):
    return float(np.linalg.eigvalsh(S[:2, :2]).max())


def to_z(S):
    return np.array([S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[0, 2], S[1, 2]])


# =============================================================================================== #
# Belief CBF
# =============================================================================================== #
def make_bcbf(dyn, a):
    sys_b = DubinsBelief(dyn.v, dyn.omega_max, dyn.qx, dyn.qy, dyn.qt, dyn.r0, dyn.r1, (dyn.bx, dyn.by))
    # the CBF's belief drift must equal the 9D dynamics with zero innovation
    rng = np.random.default_rng(3)
    lo, hi = [np.array(v) for v in zip(*dyn.state_test_range())]
    z = rng.uniform(lo, hi, (300, 9))
    w = rng.uniform(-dyn.omega_max, dyn.omega_max, 300)
    zt = torch.tensor(z, dtype=torch.float64)
    f_cbf = (sys_b.F(zt) + sys_b.G(zt)[..., 0] * torch.tensor(w)[:, None]).numpy()
    f_dyn = dyn.dsdt(zt, torch.tensor(w)[:, None], torch.zeros(300, 2, dtype=torch.float64)).numpy()
    assert np.allclose(f_cbf, f_dyn, atol=1e-10), "DubinsBelief != EstimationAwareDubins3D.dsdt (e = 0)"
    probe = BeliefCBF(sys_b, R_c=dyn.Rc, kappa=dyn.kappa)
    margin = 0.0
    if a.bcbf_eps is not None:   # Thm. 3 with the largest noise level in the domain (conservative)
        margin = theorem3_margin((1e-3, 0.18), dyn.r_max, a.dt, probe.delta, a.bcbf_eps)
    cbf = BeliefCBF(sys_b, R_c=dyn.Rc, kappa=dyn.kappa, pole=a.bcbf_pole, margin=margin, robust=False)
    print(f"belief CBF: delta={cbf.delta:.4f} (kappa={dyn.kappa}), pole={a.bcbf_pole}, margin={margin:.4f}")
    return cbf


# =============================================================================================== #
# Simulation
# =============================================================================================== #
def make_init(dyn, a, rng):
    N = a.n_rollouts
    x0 = np.c_[rng.uniform(*a.x0_range, N), rng.uniform(*a.y0_range, N), rng.uniform(*a.th0_range, N)]
    z0 = np.zeros((N, 9))
    for i in range(N):
        S = steady_state_cov(dyn, x0[i, :2], x0[i, 2]) if a.P0_mode == "ss" else a.P0 * np.eye(3)
        z0[i, :3] = x0[i] + np.linalg.cholesky(S + 1e-12 * np.eye(3)) @ rng.standard_normal(3)
        z0[i, 3:] = to_z(S)
    return dict(x0=x0, z0=z0)


def make_noise(a, N, rng):
    T = int(round(a.T_sim / a.dt))
    return dict(w=rng.standard_normal((T, N, 3)), n=rng.standard_normal((T, N, 2)))


def nominal(z, a):
    """Heading controller toward the goal, on the estimate."""
    g = np.array(a.goal)
    err = wrap(np.arctan2(g[1] - z[:, 1], g[0] - z[:, 0]) - z[:, 2])
    return np.clip(a.k_heading * err, -a.omega_max, a.omega_max)


def simulate(method, model, init, noise, a, dyn):
    N, dt = len(init["x0"]), a.dt
    T = noise["w"].shape[0]
    x, z = init["x0"].copy(), init["z0"].copy()
    Q = np.array([dyn.qx, dyn.qy, dyn.qt]) * a.noise_scale
    g = np.array(a.goal)

    done = np.zeros(N, bool)
    collided, left, reached = np.zeros(N, bool), np.zeros(N, bool), np.zeros(N, bool)
    t_end = np.full(N, np.nan)
    goal_err = np.full(N, np.nan)
    first_t, first_d = np.full(N, np.nan), np.full(N, np.nan)
    engaged = np.zeros(N, bool)
    infeas = np.zeros(N, bool)
    infeas_n = np.zeros(N)
    act, live_n = np.zeros(N), np.zeros(N)
    effort = np.zeros(N)
    min_clear = np.full(N, np.inf)
    traj = np.full((T + 1, N, 2), np.nan)
    traj_hat = np.full((T + 1, N, 2), np.nan)

    for k in range(T + 1):
        live = ~done
        traj[k, live], traj_hat[k, live] = x[live, :2], z[live, :2]
        dist = np.hypot(x[:, 0], x[:, 1])
        min_clear[live] = np.minimum(min_clear[live], dist[live] - dyn.Rc)
        col = live & (dist <= dyn.Rc)
        lft = live & ~col & ((np.abs(x[:, 0]) > dyn.xy_max) | (np.abs(x[:, 1]) > dyn.xy_max))
        p_goal = z[:, :2] if a.goal_on_estimate else x[:, :2]     # the robot can only check its estimate
        rch = live & ~col & ~lft & (np.linalg.norm(p_goal - g, axis=1) < a.goal_tol)
        goal_err[rch] = np.linalg.norm(x[rch, :2] - g, axis=1)    # true distance to the goal on arrival
        collided |= col; left |= lft; reached |= rch
        t_end[col | lft | rch] = k * dt
        done |= col | lft | rch
        live = ~done
        if k == T or not live.any():
            break
        idx = np.flatnonzero(live)

        w_nom = nominal(z, a)
        w = w_nom.copy()
        if method == "bcbf":
            u, _, e, feas = model.filter(z[idx], w_nom[idx])
            w[idx] = u[:, 0]
            infeas[idx] |= ~feas
            infeas_n[idx] += ~feas
        elif method != "none":
            V, u_opt = model.query(z[idx])
            e = np.where(V <= a.delta, True, np.where(V > a.delta + a.hysteresis, False, engaged[idx]))
            engaged[idx] = e
            w[idx] = np.where(e, u_opt, w_nom[idx])
        if method != "none":
            new = idx[e & np.isnan(first_t[idx])]
            first_t[new], first_d[new] = k * dt, dist[new]
            act[idx] += e
            live_n[idx] += 1
        w = np.clip(w, -a.omega_max, a.omega_max)
        effort[idx] += np.abs(w[idx] - w_nom[idx]) * dt

        # CT-EKF (Euler): drift and Riccati from the 9D model with e = 0, plus K * innovation increment
        zt = torch.tensor(z[idx], dtype=torch.float64)
        drift = dyn.dsdt(zt, torch.tensor(w[idx, None]), torch.zeros(len(idx), 2, dtype=torch.float64)).numpy()
        zi = z[idx]
        r_hat = r_of(dyn, zi[:, :2])
        r_true = a.noise_scale * r_of(dyn, x[idx, :2])
        inn = (x[idx, :2] - zi[:, :2]) * dt + np.sqrt(r_true * dt)[:, None] * noise["n"][k, idx]
        K = np.stack([np.c_[zi[:, 3], zi[:, 6]], np.c_[zi[:, 6], zi[:, 4]], np.c_[zi[:, 7], zi[:, 8]]], 1) \
            / r_hat[:, None, None]                                           # (n, 3, 2) = Sigma C^T / r
        zn = zi + dt * drift
        zn[:, :3] += np.einsum("nij,nj->ni", K, inn)
        zn[:, 2] = wrap(zn[:, 2])
        zn[:, 3:6] = np.maximum(zn[:, 3:6], 1e-6)
        # true Dubins car
        xi = x[idx]
        xn = xi + dt * np.c_[dyn.v * np.cos(xi[:, 2]), dyn.v * np.sin(xi[:, 2]), w[idx]] \
            + np.sqrt(Q * dt) * noise["w"][k, idx]
        xn[:, 2] = wrap(xn[:, 2])
        x[idx], z[idx] = xn, zn

    return dict(collided=collided, left=left, reached=reached, t_end=t_end, goal_err=goal_err, first_t=first_t,
                first_d=first_d, intervened=~np.isnan(first_t),
                frac_active=np.divide(act, live_n, out=np.zeros(N), where=live_n > 0),
                effort=effort, min_clear=min_clear, qp_infeasible=infeas,
                qp_infeasible_steps=np.divide(infeas_n, live_n, out=np.zeros(N), where=live_n > 0),
                traj=traj, traj_hat=traj_hat)


# =============================================================================================== #
# Output
# =============================================================================================== #
def wilson(k, n, zc=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + zc ** 2 / n
    cen = (p + zc ** 2 / (2 * n)) / den
    h = zc * math.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n ** 2)) / den
    return max(0.0, cen - h), min(1.0, cen + h)


def summarize(r, m):
    n, k = len(r["collided"]), int(r["collided"].sum())
    lo, hi = wilson(k, n)
    eng = r["intervened"]
    nan = float("nan")
    return dict(n=n, collisions=k, collision_rate=k / n, ci_lo=lo, ci_hi=hi,
                min_clear_mean=float(r["min_clear"].mean()), min_clear_worst=float(r["min_clear"].min()),
                intervention_rate=float(eng.mean()) if m != "none" else nan,
                first_t_mean=float(np.nanmean(r["first_t"])) if eng.any() else nan,
                first_dist_mean=float(np.nanmean(r["first_d"])) if eng.any() else nan,
                frac_active=float(r["frac_active"].mean()) if m != "none" else nan,
                effort_mean=float(r["effort"].mean()),
                goal_reached=float(r["reached"].mean()), left_domain=float(r["left"].mean()),
                goal_err_mean=float(np.nanmean(r["goal_err"])) if r["reached"].any() else nan,
                qp_infeasible=float(r["qp_infeasible"].mean()) if m == "bcbf" else nan,
                qp_infeasible_steps=float(r["qp_infeasible_steps"].mean()) if m == "bcbf" else nan)


COLS = [("Collision rate", "collision_rate", "{:.3f}"), ("95% CI", None, None),
        ("Mean min. clearance [m]", "min_clear_mean", "{:.3f}"),
        ("Worst min. clearance [m]", "min_clear_worst", "{:.3f}"),
        ("Intervention rate", "intervention_rate", "{:.3f}"), ("First intervention t [s]", "first_t_mean", "{:.2f}"),
        ("First intervention $\\|p\\|$ [m]", "first_dist_mean", "{:.2f}"),
        ("Active fraction", "frac_active", "{:.3f}"), ("Effort $\\int|\\omega-\\omega_{nom}|dt$", "effort_mean", "{:.2f}"),
        ("Goal reached", "goal_reached", "{:.3f}"), ("True goal error [m]", "goal_err_mean", "{:.3f}"), ("Left domain", "left_domain", "{:.3f}"),
        ("QP infeasible (rollouts)", "qp_infeasible", "{:.3f}"),
        ("QP infeasible (steps)", "qp_infeasible_steps", "{:.3f}")]


def cell(s, key, fmt):
    if key is None:
        return f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
    v = s[key]
    return "--" if (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def write_tables(rows, out, note):
    ms = list(rows)
    md = ["| Metric | " + " | ".join(LAB[m] for m in ms) + " |", "|---" * (len(ms) + 1) + "|"]
    tex = ["\\begin{tabular}{l" + "c" * len(ms) + "}", "\\toprule",
           "Metric & " + " & ".join(LAB[m] for m in ms) + " \\\\", "\\midrule"]
    for name, key, fmt in COLS:
        cells = [cell(rows[m], key, fmt) for m in ms]
        md.append(f"| {name.replace('$', '')} | " + " | ".join(cells) + " |")
        tex.append(name + " & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "table.md"), "w").write("\n".join(md) + f"\n\n{note}\n")
    open(os.path.join(out, "table.tex"), "w").write("\n".join(tex) + "\n")
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = list(next(iter(rows.values())))
        w.writerow(["method"] + keys)
        for m in ms:
            w.writerow([LAB[m]] + [rows[m][k] for k in keys])
    print("\n" + "\n".join(md) + f"\n\n{note}")


def fig_trajectories(H, dyn, a, path, n_plot):
    ms = list(H)
    ncol = min(3, len(ms))
    nrow = math.ceil(len(ms) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.7 * nrow), squeeze=False)
    for ax in axes.flat[len(ms):]:
        ax.axis("off")
    for ax, m in zip(axes.flat, ms):
        r = H[m]
        ax.add_patch(plt.Circle((0, 0), dyn.Rc, color="0.75"))
        for i in range(min(n_plot, r["traj"].shape[1])):
            c = "tab:red" if r["collided"][i] else COL[m]
            ax.plot(r["traj"][:, i, 0], r["traj"][:, i, 1], color=c, lw=0.8, alpha=0.7)
        ax.plot(dyn.bx, dyn.by, "k*", ms=12, mfc="gold", label="beacon")
        ax.plot(*a.goal, "gx", ms=9, mew=2.5, label="goal")
        ax.set(xlim=(-dyn.xy_max, dyn.xy_max), ylim=(-dyn.xy_max, dyn.xy_max), aspect="equal",
               title=f"{LAB[m]}\ncollisions {100 * r['collided'].mean():.1f}%")
        ax.title.set_fontsize(11)
    axes[0, 0].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =============================================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est_ckpt", default=checkpoint("dubins_ea_9d_k3"))
    ap.add_argument("--grid_values", default=repo_path("data", "dubins_nominal_values_all.npy"))
    ap.add_argument("--grid_coords", default=repo_path("data", "dubins_nominal_grid_coords.npy"))
    ap.add_argument("--grid_T", type=float, default=1.0, help="horizon of the saved grid (times linspace(0,T,nt))")
    ap.add_argument("--t_query", type=float, default=None, help="value-function time (default: training tMax)")
    ap.add_argument("--dyn_kwargs", default="{}", help="override EstimationAwareDubins3D args (JSON)")
    ap.add_argument("--methods", nargs="+", default=None, choices=[m for m, _ in METHODS])
    ap.add_argument("--out", default=os.path.join(OUTPUTS, "dubins_safety_filter"))
    ap.add_argument("--n_rollouts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--T_sim", type=float, default=8.0)
    ap.add_argument("--delta", type=float, default=0.05, help="filter engages when V <= delta")
    ap.add_argument("--hysteresis", type=float, default=0.0, help="disengage when V > delta + hysteresis")
    ap.add_argument("--goal", type=float, nargs=2, default=[-2.0, 0.0])
    ap.add_argument("--goal_tol", type=float, default=0.3)
    ap.add_argument("--goal_on_estimate", action=argparse.BooleanOptionalAction, default=True,
                    help="declare the goal reached when the ESTIMATE is within goal_tol (true error is reported)")
    ap.add_argument("--x0_range", type=float, nargs=2, default=[2.2, 2.6])
    ap.add_argument("--y0_range", type=float, nargs=2, default=[-0.5, 0.5])
    ap.add_argument("--th0_range", type=float, nargs=2, default=[3.0416, 3.2416], help="pi -/+ 0.1")
    ap.add_argument("--k_heading", type=float, default=2.0)
    ap.add_argument("--noise_scale", type=float, default=1.0, help="true noise = scale x filter's q, r")
    ap.add_argument("--P0_mode", choices=["ss", "fixed"], default="ss",
                    help="ss: steady-state EKF covariance at the start state; fixed: --P0 * I")
    ap.add_argument("--P0", type=float, default=0.02)
    ap.add_argument("--bcbf_pole", type=float, default=2.0)
    ap.add_argument("--bcbf_eps", type=float, default=None, help="Thm. 3 eps -> margin gamma (default 0)")
    ap.add_argument("--n_plot", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    have_est = bool(a.est_ckpt) and os.path.isfile(a.est_ckpt)
    train = read_train_config(a.est_ckpt) if have_est else {}
    dyn = make_dynamics(train, json.loads(a.dyn_kwargs))
    a.omega_max = dyn.omega_max
    t_query = a.t_query if a.t_query is not None else float(train.get("tMax", 1.0))

    nom = GridNominal(a.grid_values, a.grid_coords, a.grid_T, t_query, dyn.omega_max)
    S_ss = steady_state_cov(dyn, (0.0, 0.0))
    lam_ss = lam_max_pos(S_ss)
    lo, hi = [np.array(v) for v in zip(*dyn.state_test_range())]
    lam_max = float(hi[3] + hi[6])            # S_hi + Sxy_max: largest lambda_max(Sigma_pp) in the box
    models = {"nom": nom,
              "nom_ss": Inflated(nom, dyn.kappa * math.sqrt(lam_ss)),
              "nom_max": Inflated(nom, dyn.kappa * math.sqrt(lam_max)),
              "bcbf": make_bcbf(dyn, a)}
    print(f"steady state at the obstacle (r={r_of(dyn, np.zeros(2)):.3f}): lambda_max={lam_ss:.4f} -> "
          f"margin {models['nom_ss'].margin:.3f};  box max lambda_max={lam_max:.3f} -> margin "
          f"{models['nom_max'].margin:.3f}  (kappa={dyn.kappa})")
    if have_est:
        net_cfg = dict(model=train.get("model", "sine"), model_mode=train.get("model_mode", "mlp"),
                       num_nl=int(train.get("num_nl", 512)), num_hl=int(train.get("num_hl", 3)))
        models["est"] = EstOracle(dyn, a.est_ckpt, net_cfg, t_query)
    else:
        print(f"WARNING: estimation-aware checkpoint not found ('{a.est_ckpt}'); skipping 'est'.")

    methods = [m for m, _ in METHODS if (a.methods is None or m in a.methods) and (m == "none" or m in models)]
    init = make_init(dyn, a, np.random.default_rng(a.seed))
    noise = make_noise(a, a.n_rollouts, np.random.default_rng(a.seed + 1))
    lam0 = np.array([np.linalg.eigvalsh(np.array([[z[3], z[6]], [z[6], z[4]]])).max() for z in init["z0"]])
    print(f"{a.n_rollouts} rollouts, {a.T_sim:g}s, dt={a.dt}; initial lambda_max(Sigma_pp) "
          f"mean {lam0.mean():.4f}; device {DEV}")

    H, rows = {}, {}
    for m in methods:
        t0 = time.time()
        H[m] = simulate(m, models.get(m), init, noise, a, dyn)
        rows[m] = summarize(H[m], m)
        s = rows[m]
        print(f"  {LAB[m]:38s} collision {100 * s['collision_rate']:5.1f}%  goal {100 * s['goal_reached']:5.1f}%  "
              f"left {100 * s['left_domain']:4.1f}%  effort {s['effort_mean']:.2f}"
              + (f"  QP infeasible {100 * s['qp_infeasible']:.1f}% of rollouts, "
                 f"{100 * s['qp_infeasible_steps']:.1f}% of steps" if m == "bcbf" else "")
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    note = (f"{a.n_rollouts} rollouts per method, {a.T_sim:g} s, kappa={dyn.kappa}, delta={a.delta}, "
            f"margins: steady state {models['nom_ss'].margin:.3f}, max {models['nom_max'].margin:.3f}.")
    write_tables(rows, a.out, note)
    fig_trajectories(H, dyn, a, os.path.join(a.out, "trajectories.pdf"), a.n_plot)
    np.savez(os.path.join(a.out, "rollouts.npz"),
             **{f"{m}_{k}": v for m, r in H.items() for k, v in r.items() if k not in ("traj", "traj_hat")}, **init)
    json.dump(dict(args=vars(a), rows=rows, lam_ss=lam_ss, lam_max=lam_max, kappa=dyn.kappa),
              open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=float)
    print("saved", a.out)


if __name__ == "__main__":
    main()
