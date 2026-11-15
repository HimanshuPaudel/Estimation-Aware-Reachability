#!/usr/bin/env python
"""
Rollout experiments: estimation-aware BRT (9D, H-inf filter) vs plain BRT (6D)
for the planar quadrotor, used as a least-restrictive safety filter.

Pipeline
  1. Sample N initial states (same for every method, same noise / wind realisation).
  2. Roll out three controllers on the SAME true system + SAME H-inf filter:
        none  : nominal goal-seeking controller only
        plain : nominal + filter from the 6D BRT, evaluated at [xhat, yhat, th, vx, vy, w]
                (ignores estimation uncertainty -> certainty-equivalent)
        plain_ss / plain_max :
                plain BRT with the target inflated by a constant kappa*sqrt(P): P = P_inf (steady state
                of the Riccati ODE) or P_hi + Pxy_max (largest lambda_max(P) in the training box).
                For the avoid problem, inflating l by m gives exactly V - m, so these reuse the plain
                net with V - m (same gradient, same optimal control).
        est   : nominal + filter from the 9D BRT, evaluated at [xhat, ..., Pxx, Pyy, Pxy]
        bcbf  : belief CBF (belief_cbf.py) on the commanded acceleration
  3. Log the first step at which the filter engages (V <= delta).
  4. Slice both BRTs at the state where the filter first engages (shared by both methods,
     since trajectories are identical until the first engagement) and plot.
  5. Aggregate statistics over all rollouts (+ optional noise-level sweep).

Wind (true system): default --wind hj --wind_scale 1 = the disturbance model both value functions
were trained against, |d_i| <= dist_max, chosen adversarially (HJ-optimal for the plain value
function at the true state, at every step).

Outputs (all in --out):
  single_rollout.pdf, brt_slices_engagement.pdf, multi_rollout_summary.pdf,
  noise_sweep.pdf (if several --noise_scales), table.tex, rollouts_main.npz, summary.json
  value_function_6d_vs_9d.pdf   : value functions 6D vs 9D, side by side
  brt_filled.pdf                : filled BRTs, plain background
  brt_boundary.pdf              : BRT boundaries only, plain background

Colour scheme matches the Air3D safety-filter figures:
  standard / plain model  -> black, dashed
  estimation-aware model  -> coloured (EST_COLOR), solid

Usage (DeepReach / torch environment, from any directory):
  python scripts/experiments/quadrotor_safety_filter.py --n_rollouts 2000              # kappa = 3
  python scripts/experiments/quadrotor_safety_filter.py --n_rollouts 2000 \\
      --est_ckpt checkpoints/pquad_ea_9d_k2/training/checkpoints/model_final.pth      # kappa = 2
Default checkpoint paths are set in EDIT 0 below (CLI flags override them).
"""
import argparse, inspect, json, math, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ EDIT 0: checkpoint paths ========================
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "src"))
from ea_reach.paths import add_deepreach_to_path, checkpoint, OUTPUTS   # noqa: E402

# DeepReach fork (git submodule external/deepreach, or $DEEPREACH_ROOT)
DEEPREACH_DIR = add_deepreach_to_path()
# Checkpoint paths: absolute, or relative to the cwd or to DEEPREACH_DIR.
PLAIN_CKPT = checkpoint("pquad_plain_6d")
EST_CKPT = checkpoint("pquad_ea_9d_k3")      # kappa = 3; pquad_ea_9d_k2 for kappa = 2
OUT_DIR = os.path.join(OUTPUTS, "quadrotor_safety_filter")
# ==============================================================================


def resolve_path(p):
    """Use p as given if it exists, otherwise try it relative to DEEPREACH_DIR."""
    p = os.path.expanduser(p)
    if os.path.exists(p):
        return p
    alt = os.path.join(DEEPREACH_DIR, p)
    return alt if os.path.exists(alt) else p


# ============================ EDIT 1: repo imports ============================
from dynamics import dynamics as D      # PlanarQuadrotor, EstimationAwarePlanarQuadrotor
from utils import modules               # SingleBVPNet
# ==============================================================================

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- paper colour scheme (same as the Air3D comparison figure) ----
STD_COL = "k"            # standard / plain model: black, dashed
EST_COL = "tab:orange"   # estimation-aware model: coloured, solid  (change here to recolour every figure)
VCMAP = "RdBu"           # colormap for the value-function heatmaps
BCBF_COL = "tab:purple"  # belief CBF baseline
COL = {"none": "0.45", "plain": STD_COL, "plain_ss": "tab:blue", "plain_max": "tab:cyan",
       "est": EST_COL, "bcbf": BCBF_COL}
LS = {"none": ":", "plain": "--", "plain_ss": "--", "plain_max": "--", "est": "-", "bcbf": "-."}
LAB = {"none": "nominal only", "plain": "filter, plain BRT",
       "plain_ss": "filter, plain BRT + margin (steady state)", "plain_max": "filter, plain BRT + margin (max)",
       "est": "filter, estimation-aware BRT", "bcbf": "belief CBF"}
METHODS = ("none", "plain", "plain_ss", "plain_max", "est", "bcbf")


# ------------------------------------------------------------------ model I/O
def load_net(dyn, ckpt, a):
    # ========================= EDIT 2: match your training args =========================
    net = modules.SingleBVPNet(in_features=dyn.input_dim, out_features=1, type=a.model,
                               mode=a.model_mode, final_layer_factor=1.,
                               hidden_features=a.num_nl, num_hidden_layers=a.num_hl,
                               periodic_transform_fn=dyn.periodic_transform_fn)
    # ====================================================================================
    sd = torch.load(ckpt, map_location=DEV)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd)
    return net.to(DEV).eval()


class ValueOracle:
    """V(t_eval, z), dV/dz and the optimal (controller) action from a trained DeepReach net."""

    def __init__(self, dyn, net, t_eval):
        self.dyn, self.net, self.t = dyn, net, t_eval

    def _coords(self, z):
        z = self.dyn.clip_state(torch.as_tensor(z, dtype=torch.float32, device=DEV))
        t = torch.full((z.shape[0], 1), self.t, device=DEV)
        return z, self.dyn.coord_to_input(torch.cat([t, z], dim=-1))

    def query(self, z):
        z, inp = self._coords(z)
        with torch.enable_grad():
            out = self.net({"coords": inp})
            x, y = out["model_in"], out["model_out"].squeeze(-1)
            V = self.dyn.io_to_value(x, y)
            dv = self.dyn.io_to_dv(x, y)
        dvds = dv[..., 1:].detach()
        u = self.dyn.optimal_control(z, dvds)
        return V.detach().cpu().numpy(), u.detach().cpu().numpy()

    def value_grad(self, z):
        """V and dV/dz (real units) at z."""
        z, inp = self._coords(z)
        with torch.enable_grad():
            out = self.net({"coords": inp})
            x, y = out["model_in"], out["model_out"].squeeze(-1)
            V = self.dyn.io_to_value(x, y)
            dv = self.dyn.io_to_dv(x, y)
        return V.detach().cpu().numpy(), dv[..., 1:].detach().cpu().numpy()

    @torch.no_grad()
    def value(self, z, chunk=20000):
        out = []
        for i in range(0, len(z), chunk):
            _, inp = self._coords(z[i:i + chunk])
            o = self.net({"coords": inp})
            out.append(self.dyn.io_to_value(o["model_in"], o["model_out"].squeeze(-1)).cpu().numpy())
        return np.concatenate(out)


class InflatedPlain:
    """Plain BRT with the target inflated by a constant margin m: V_m = V - m (exact for avoid)."""

    def __init__(self, base, margin):
        self.base, self.margin = base, float(margin)
        self.dyn, self.t = base.dyn, base.t

    def query(self, z):
        V, u = self.base.query(z)
        return V - self.margin, u

    def value(self, z, chunk=20000):
        return self.base.value(z, chunk) - self.margin


# ------------------------------------------------------------ nominal control
def nominal(xh, s, a, ph):
    """Goal-seeking PD (uses the estimate xhat, as a real controller would)."""
    return acc_to_thrust(nominal_acc(xh, s, a), s, a, ph)


def nominal_acc(xh, s, a):
    """Outer loop: desired acceleration toward the goal (this is what the belief CBF filters)."""
    goal = np.array(a.goal)
    vdes = np.clip(a.kp * (goal - xh), -a.v_nom, a.v_nom)
    return np.clip(a.kv * (vdes - s[:, 3:5]), -a.a_max, a.a_max)


def acc_to_thrust(acc, s, a, ph):
    """Inner loop: desired acceleration -> attitude setpoint -> rotor thrusts (unchanged from nominal)."""
    ay_tot = ph["g"] + acc[:, 1]
    th_des = np.clip(-np.arctan2(acc[:, 0], ay_tot), -0.6, 0.6)
    T = ph["m"] * ay_tot / np.maximum(np.cos(s[:, 2]), 0.5)
    alpha = a.kth * (th_des - s[:, 2]) - a.kom * s[:, 5]
    dU = ph["I"] * alpha / ph["arm"]
    return np.clip(np.stack([(T + dU) / 2, (T - dU) / 2], 1), 0, ph["umax"])


# ------------------------------------------------------------------ simulator
def simulate(method, orc, init, noise, a, ph, wind_orc=None):
    """Batched Euler-Maruyama rollout of true system + H-inf filter (+ optional BRT filter)."""
    N, T, dt = init["s0"].shape[0], a.steps, a.dt
    s, xh, P = init["s0"].copy(), init["xh0"].copy(), init["P0"].copy()   # P=[Pxx,Pyy,Pxy]
    c = 1 / ph["r"] - 1 / ph["gamma"] ** 2
    qt, rt = ph["q"] * noise["scale"], ph["r"] * noise["scale"]           # true noise intensities
    H = dict(s=np.zeros((T + 1, N, 6)), xh=np.zeros((T + 1, N, 2)), P=np.zeros((T + 1, N, 3)),
             V=np.full((T, N), np.nan), eng=np.zeros((T, N), bool), du=np.zeros((T, N)),
             infeas=np.zeros((T, N), bool), wind_hj=np.zeros((T, N), bool))
    for k in range(T + 1):
        H["s"][k], H["xh"][k], H["P"][k] = s, xh, P
        if k == T:
            break
        u_nom = nominal(xh, s, a, ph)
        u = u_nom
        if method == "bcbf":
            # belief CBF filters the desired acceleration; same inner loop converts it to thrusts.
            # H["V"] stores the barrier h_b, "eng" = the QP modified the reference acceleration.
            b = np.concatenate([xh, s[:, 3:5], P], 1)        # [xhat, yhat, vx, vy, Pxx, Pyy, Pxy]
            acc, h, eng, feas = orc.filter(b, nominal_acc(xh, s, a))
            u = acc_to_thrust(acc, s, a, ph)
            H["V"][k], H["eng"][k], H["infeas"][k] = h, eng, ~feas
        elif method != "none":
            z = np.concatenate([xh, s[:, 2:6]] + ([P] if method == "est" else []), 1)
            V, u_opt = orc.query(z)
            eng = V <= a.delta
            u = np.where(eng[:, None], u_opt, u_nom)
            H["V"][k], H["eng"][k] = V, eng
        u = np.clip(u, 0, ph["umax"])
        H["du"][k] = np.linalg.norm(u - u_nom, axis=1)

        if a.wind == "hj":
            # adversarial wind at the trained bound: HJ-optimal disturbance of the plain (6D) value
            # function at the TRUE state, d = -dmax sign(dV/dv), at every step
            _, gw = wind_orc.value_grad(s)
            d = -a.wind_scale * ph["dmax"] * np.sign(gw[:, 3:5])
            H["wind_hj"][k] = True
        elif a.wind == "corner":
            d = noise["wind_const"]
        elif a.wind == "toward":                       # pushes toward obstacle (box-bounded)
            d = -a.wind_scale * ph["dmax"] * np.sign(s[:, 0:2])
        else:
            d = np.zeros((N, 2))
        Tn, th = u.sum(1), s[:, 2]
        acc = np.stack([-Tn * np.sin(th) / ph["m"], Tn * np.cos(th) / ph["m"] - ph["g"]], 1) + d
        alpha = ph["arm"] * (u[:, 0] - u[:, 1]) / ph["I"]

        # filter: xhat' = v + (1/r) P e,  e*dt = (p - xhat) dt + sqrt(r dt) * xi
        inn = (s[:, 0:2] - xh) * dt + np.sqrt(rt * dt) * noise["n"][k]
        Pxx, Pyy, Pxy = P[:, 0], P[:, 1], P[:, 2]
        dxh = dt * s[:, 3:5] + np.stack([Pxx * inn[:, 0] + Pxy * inn[:, 1],
                                         Pxy * inn[:, 0] + Pyy * inn[:, 1]], 1) / ph["r"]
        dP = np.stack([ph["q"] - c * (Pxx ** 2 + Pxy ** 2),
                       ph["q"] - c * (Pxy ** 2 + Pyy ** 2),
                       -c * Pxy * (Pxx + Pyy)], 1)

        s = np.concatenate([s[:, 0:2] + dt * s[:, 3:5] + np.sqrt(qt * dt) * noise["w"][k],
                            (th + dt * s[:, 5])[:, None],
                            s[:, 3:5] + dt * acc,
                            (s[:, 5] + dt * alpha)[:, None]], 1)
        xh = xh + dxh
        P = P + dt * dP
    return H


# -------------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    if n == 0:
        return 0., 0.
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0., cen - h), min(1., cen + h)


def metrics(H, a, ph):
    T, N = a.steps, H["s"].shape[1]
    dist = np.linalg.norm(H["s"][:, :, 0:2], axis=2)                      # true ||p||
    clear = dist - ph["R"]
    eng_any = H["eng"].any(0)
    k_eng = np.where(eng_any, H["eng"].argmax(0), T)
    idx = np.arange(N)
    goal_err = np.linalg.norm(H["s"][-1, :, 0:2] - np.array(a.goal), axis=1)
    return dict(min_clear=clear.min(0), collided=clear.min(0) < 0, engaged=eng_any, k_eng=k_eng,
                t_eng=k_eng * a.dt,
                dist_true_eng=np.where(eng_any, dist[np.minimum(k_eng, T), idx], np.nan),
                dist_est_eng=np.where(eng_any, np.linalg.norm(H["xh"][np.minimum(k_eng, T), idx], axis=1), np.nan),
                frac_active=H["eng"].mean(0), effort=H["du"].sum(0) * a.dt,
                goal_err=goal_err, reached=goal_err < a.goal_tol, qp_infeasible=H["infeas"].any(0))


def summarize(M):
    rows = {}
    for m, x in M.items():
        n = len(x["collided"])
        k = int(x["collided"].sum())
        lo, hi = wilson(k, n)
        rows[m] = dict(collision_rate=k / n, ci_lo=lo, ci_hi=hi,
                       min_clear_mean=float(x["min_clear"].mean()), min_clear_min=float(x["min_clear"].min()),
                       engaged_rate=float(x["engaged"].mean()),
                       eng_dist_true_mean=float(np.nanmean(x["dist_true_eng"])) if x["engaged"].any() else float("nan"),
                       eng_time_mean=float(np.mean(x["t_eng"][x["engaged"]])) if x["engaged"].any() else float("nan"),
                       frac_active_mean=float(x["frac_active"].mean()), effort_mean=float(x["effort"].mean()),
                       goal_reached=float(x["reached"].mean()),
                       qp_infeasible_rate=float(x["qp_infeasible"].mean()))
    return rows


def latex_table(rows):
    f = [("Collision rate", "collision_rate", "{:.3f}"), ("Mean min. clearance [m]", "min_clear_mean", "{:.3f}"),
         ("Mean engagement dist. $\\|p\\|$ [m]", "eng_dist_true_mean", "{:.2f}"),
         ("Mean engagement time [s]", "eng_time_mean", "{:.2f}"),
         ("Active fraction", "frac_active_mean", "{:.3f}"), ("Intervention $\\int\\|u-u_{nom}\\|dt$", "effort_mean", "{:.2f}"),
         ("Goal reached", "goal_reached", "{:.3f}"), ("QP infeasible (rollouts)", "qp_infeasible_rate", "{:.3f}")]
    ms = list(rows)
    out = ["\\begin{tabular}{l" + "c" * len(ms) + "}", "\\toprule", "Metric & " + " & ".join(LAB[m] for m in ms) + " \\\\", "\\midrule"]
    for name, key, fmt in f:
        out.append(name + " & " + " & ".join(fmt.format(rows[m][key]) for m in ms) + " \\\\")
    return "\n".join(out + ["\\bottomrule", "\\end{tabular}"])


# ------------------------------------------------------------- BRT slice utils
def common_ranges(dp, de):
    rp, re = dp.state_range_.cpu().numpy(), de.state_range_.cpu().numpy()
    return np.stack([np.maximum(rp[:6, 0], re[:6, 0]), np.minimum(rp[:6, 1], re[:6, 1])], 1)


def brt_slice(orc, base, i, j, rng, n):
    xs, ys = np.linspace(*rng[i], n), np.linspace(*rng[j], n)
    X, Y = np.meshgrid(xs, ys)
    z = np.tile(base, (n * n, 1)).astype(np.float32)
    z[:, i], z[:, j] = X.ravel(), Y.ravel()
    return xs, ys, orc.value(z).reshape(n, n)


def base_states(Hp, He, T):
    """Shared state at the first engagement of either method (trajectories coincide up to there)."""
    kp = np.where(Hp["eng"].any(0), Hp["eng"].argmax(0), T)
    ke = np.where(He["eng"].any(0), He["eng"].argmax(0), T)
    kb = np.minimum(kp, ke)
    N = len(kb)
    idx = np.arange(N)
    ok = kb < T
    kb_c = np.minimum(kb, T)
    zs = {"s": He["s"][kb_c, idx], "xh": He["xh"][kb_c, idx], "P": He["P"][kb_c, idx]}
    return kb, ok, zs


# ------------------------------------------------------------------- plotting
def fig_single(i, Hs, Ms, a, ph, path):
    t = np.arange(a.steps + 1) * a.dt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].add_patch(plt.Circle((0, 0), ph["R"], color="0.8", label="obstacle"))
    for m, H in Hs.items():
        ax[0].plot(H["s"][:, i, 0], H["s"][:, i, 1], color=COL[m], lw=2, label=LAB[m], ls=LS[m])
        if m != "none":
            ax[0].plot(H["xh"][:, i, 0], H["xh"][:, i, 1], color=COL[m], lw=0.8, alpha=0.5)
            if Ms[m]["engaged"][i]:
                k = Ms[m]["k_eng"][i]
                ax[0].plot(*H["s"][k, i, 0:2], "*", ms=15, color=COL[m], mec="k")
        ax[1].plot(t, np.linalg.norm(H["s"][:, i, 0:2], axis=1) - ph["R"], color=COL[m], lw=2, ls=LS[m])
    ax[0].plot(*a.goal, "gx", ms=10, mew=3, label="goal")
    ax[0].plot(*Hs["none"]["s"][0, i, 0:2], "ko", label="start")
    ax[0].set(aspect="equal", xlabel="x [m]", ylabel="y [m]", title="True trajectories (thin: $\\hat p$; star: filter engages)")
    ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set(xlabel="t [s]", ylabel="true clearance $\\|p\\|-R_c$ [m]", title="True clearance")
    for m in ("plain", "est"):
        ax[2].plot(t[:-1], Hs[m]["V"][:, i], color=COL[m], lw=2, ls=LS[m], label=f"V ({m})")
    ax[2].axhline(a.delta, color="k", ls=":", lw=1, label="$\\delta$")
    ax[2].axhline(0, color="k", lw=1)
    ax[2].set(xlabel="t [s]", ylabel="V along own trajectory", title="Value / engagement")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_slices(i, kb, zs, orc_p, orc_e, rng, a, ph, path, pairs=((0, 1), (3, 4))):
    names = ["$\\hat x$", "$\\hat y$", "$\\theta$", "$v_x$", "$v_y$", "$\\omega$"]
    base_e = np.concatenate([zs["xh"][i], zs["s"][i, 2:6], zs["P"][i]])
    base_p = base_e[:6]
    fig, axes = plt.subplots(len(pairs), 3, figsize=(15, 4.6 * len(pairs)), squeeze=False)
    for r, (p_i, p_j) in enumerate(pairs):
        xs, ys, Vp = brt_slice(orc_p, base_p, p_i, p_j, rng, a.slice_n)
        _, _, Ve = brt_slice(orc_e, base_e, p_i, p_j, rng, a.slice_n)
        vmax = max(abs(Vp).max(), abs(Ve).max())
        for c, (V, ttl, m) in enumerate([(Vp, "plain BRT (6D)", "plain"), (Ve, "estimation-aware BRT (9D)", "est")]):
            ax = axes[r, c]
            pm = ax.pcolormesh(xs, ys, V, cmap=VCMAP, vmin=-vmax, vmax=vmax, shading="auto")
            ax.contour(xs, ys, V, [0.0], colors=COL[m], linestyles=LS[m], linewidths=2)
            ax.contour(xs, ys, V, [a.delta], colors=COL[m], linestyles=":", linewidths=1)
            ax.set_title(ttl)
            fig.colorbar(pm, ax=ax, fraction=0.046)
        ax = axes[r, 2]
        for V, m in ((Vp, "plain"), (Ve, "est")):
            ax.contour(xs, ys, V, [0.0], colors=COL[m], linestyles=LS[m], linewidths=2)
            ax.contour(xs, ys, V, [a.delta], colors=COL[m], linestyles=":", linewidths=1)
        ax.plot([], [], color=COL["plain"], ls=LS["plain"], lw=2, label="plain  $V=0$")
        ax.plot([], [], color=COL["est"], ls=LS["est"], lw=2, label="est.-aware  $V=0$")
        ax.plot([], [], color="k", ls=":", lw=1, label="$V=\\delta$")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title("zero / $\\delta$ level sets")
        for c in range(3):
            ax = axes[r, c]
            ax.plot(base_e[p_i], base_e[p_j], "k*", ms=14, mfc="yellow", label="engagement state")
            if (p_i, p_j) == (0, 1):
                ax.plot(*zs["s"][i, 0:2], "m^", ms=9)  # true position
                ax.add_patch(plt.Circle((0, 0), ph["R"], fill=False, ls=":", color="k"))
                ax.set_aspect("equal")
            ax.set_xlabel(names[p_i]); ax.set_ylabel(names[p_j])
    P = zs["P"][i]
    fig.suptitle(f"Slices at filter engagement (t={kb[i] * a.dt:.2f}s); other coords fixed at engagement state; "
                 f"$P_{{xx}}$={P[0]:.3f}, $P_{{yy}}$={P[1]:.3f}, $P_{{xy}}$={P[2]:.3f}.  "
                 "star: $\\hat p$, triangle: true $p$", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_slice_pair(i, kb, zs, orc_p, orc_e, rng, a, ph, path):
    """Paper figure: x-y slices of the plain (6D) and estimation-aware (9D) value functions at the
    engagement state of rollout i (star: x_hat at engagement, triangle: true position)."""
    base_e = np.concatenate([zs["xh"][i], zs["s"][i, 2:6], zs["P"][i]])
    xs, ys, Vp = brt_slice(orc_p, base_e[:6], 0, 1, rng, a.slice_n)
    _, _, Ve = brt_slice(orc_e, base_e, 0, 1, rng, a.slice_n)
    vmax = max(abs(Vp).max(), abs(Ve).max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, V, ttl, m in ((axes[0], Vp, "plain BRT (6D)", "plain"), (axes[1], Ve, "estimation-aware BRT (9D)", "est")):
        pm = ax.pcolormesh(xs, ys, V, cmap=VCMAP, vmin=-vmax, vmax=vmax, shading="auto")
        ax.contour(xs, ys, V, [0.0], colors=COL[m], linestyles=LS[m], linewidths=2)
        ax.contour(xs, ys, V, [a.delta], colors=COL[m], linestyles=":", linewidths=1)
        ax.add_patch(plt.Circle((0, 0), ph["R"], fill=False, ls=":", color="k"))
        ax.plot(*zs["s"][i, 0:2], "m^", ms=9)
        ax.plot(base_e[0], base_e[1], "k*", ms=14, mfc="yellow")
        ax.set(aspect="equal", xlabel="$\\hat x$", ylabel="$\\hat y$", title=ttl)
        fig.colorbar(pm, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_value_figs(i, zs, orc_p, orc_e, rng, a, out):
    """x-y slices of the 6D and 9D value functions at the engagement state of rollout i.
    Styling follows the Air3D figure: plain = black dashed, estimation-aware = coloured solid."""
    base_e = np.concatenate([zs["xh"][i], zs["s"][i, 2:6], zs["P"][i]])
    base_p = base_e[:6]
    xs, ys, Vp = brt_slice(orc_p, base_p, 0, 1, rng, a.slice_n)
    _, _, Ve = brt_slice(orc_e, base_e, 0, 1, rng, a.slice_n)

    # ---- 1. value function, 6D vs 9D, side by side ----
    vmax = max(abs(Vp).max(), abs(Ve).max())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for k, (V, ttl, m) in enumerate([(Vp, "Value function, plain (6D)", "plain"),
                                     (Ve, "Value function, estimation-aware (9D)", "est")]):
        pm = ax[k].pcolormesh(xs, ys, V, cmap=VCMAP, vmin=-vmax, vmax=vmax, shading="auto")
        ax[k].contour(xs, ys, V, levels=[0], colors=COL[m], linestyles=LS[m], linewidths=2)
        ax[k].plot([], [], color=COL[m], ls=LS[m], lw=2, label="BRT boundary $V=0$")
        ax[k].set_aspect("equal"); ax[k].grid(alpha=0.3); ax[k].legend(fontsize=8, loc="upper right")
        ax[k].set_xlabel("x_hat"); ax[k].set_title(ttl)
    ax[0].set_ylabel("y_hat")
    fig.colorbar(pm, ax=ax, fraction=0.025, pad=0.02, label="V")
    fig.savefig(f"{out}/value_function_6d_vs_9d.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- 2. filled BRTs on a plain background ----
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    ax.contourf(xs, ys, Vp, levels=[-1e9, 0.0], colors=["0.6"], alpha=0.35)
    ax.contourf(xs, ys, Ve, levels=[-1e9, 0.0], colors=[COL["est"]], alpha=0.35)
    ax.contour(xs, ys, Vp, [0.0], colors=COL["plain"], linestyles=LS["plain"], linewidths=2)
    ax.contour(xs, ys, Ve, [0.0], colors=COL["est"], linestyles=LS["est"], linewidths=2)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(f"{out}/brt_filled.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 3. boundaries only on a plain background ----
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    ax.contour(xs, ys, Vp, [0.0], colors=COL["plain"], linestyles=LS["plain"], linewidths=2)
    ax.contour(xs, ys, Ve, [0.0], colors=COL["est"], linestyles=LS["est"], linewidths=2)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(f"{out}/brt_boundary.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_summary(M, init, rows, a, ph, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    ms = list(M)
    # (a) collision rate
    for k, m in enumerate(ms):
        r = rows[m]
        ax[0, 0].bar(k, r["collision_rate"], color=COL[m], yerr=[[max(0., r["collision_rate"] - r["ci_lo"])], [max(0., r["ci_hi"] - r["collision_rate"])]], capsize=4)
    ax[0, 0].set_xticks(range(len(ms))); ax[0, 0].set_xticklabels([LAB[m] for m in ms], rotation=12, fontsize=8)
    ax[0, 0].set(ylabel="collision rate (95% Wilson CI)", title="(a) True-state collisions")
    # (b) paired engagement distance
    both = M["plain"]["engaged"] & M["est"]["engaged"]
    ax[0, 1].scatter(M["plain"]["dist_true_eng"][both], M["est"]["dist_true_eng"][both], s=14, alpha=0.6, color=EST_COL)
    lim = [ph["R"], np.nanmax([M["plain"]["dist_true_eng"], M["est"]["dist_true_eng"]]) + 0.1]
    ax[0, 1].plot(lim, lim, "k--", lw=1)
    ax[0, 1].set(xlabel="engagement $\\|p\\|$, plain [m]", ylabel="engagement $\\|p\\|$, est.-aware [m]",
                 title="(b) Engagement distance (above diagonal = est.-aware earlier)")
    # (c) ECDF of min clearance
    for m in ms:
        x = np.sort(M[m]["min_clear"])
        ax[1, 0].step(x, np.arange(1, len(x) + 1) / len(x), color=COL[m], ls=LS[m], label=LAB[m], lw=2)
    ax[1, 0].axvline(0, color="k", lw=1)
    ax[1, 0].set(xlabel="min true clearance $\\min_t\\|p\\|-R_c$ [m]", ylabel="ECDF", title="(c) Safety margin")
    ax[1, 0].legend(fontsize=8)
    # (d) initial states by outcome
    cp, ce = M["plain"]["collided"], M["est"]["collided"]
    cat = cp.astype(int) + 2 * ce.astype(int)
    names = ["both safe", "only plain collides", "only est.-aware collides", "both collide"]
    cols = ["tab:green", STD_COL, EST_COL, "tab:red"]
    p0 = init["s0"][:, 0:2]
    for c in range(4):
        if (cat == c).any():
            ax[1, 1].scatter(p0[cat == c, 0], p0[cat == c, 1], c=cols[c], s=18, label=f"{names[c]} ({(cat == c).sum()})")
    ax[1, 1].set(xlabel="$x_0$", ylabel="$y_0$", title="(d) Outcome by initial position"); ax[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_sweep(sweep, path):
    sc = sorted(sweep)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m in [m for m in METHODS if m in sweep[sc[0]]]:
        y = np.array([sweep[s][m]["collision_rate"] for s in sc])
        lo = np.array([sweep[s][m]["ci_lo"] for s in sc]); hi = np.array([sweep[s][m]["ci_hi"] for s in sc])
        ax[0].plot(sc, y, marker="o", color=COL[m], ls=LS[m], label=LAB[m]); ax[0].fill_between(sc, lo, hi, color=COL[m], alpha=0.15)
        if m != "none":
            ax[1].plot(sc, [sweep[s][m]["effort_mean"] for s in sc], marker="o", color=COL[m], ls=LS[m], label=LAB[m])
    ax[0].set(xlabel="true noise scale (x model $q,r$)", ylabel="collision rate", title="Robustness to noise mismatch")
    ax[1].set(xlabel="true noise scale", ylabel="$\\int\\|u-u_{nom}\\|dt$", title="Cost of conservatism")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ---------------------------------------------------------- belief CBF / checks
def make_bcbf(a, ph):
    from ea_reach.belief_cbf import QuadBelief, BeliefCBF, theorem3_margin
    sys_b = QuadBelief(q=ph["q"], r=ph["r"], gamma=ph["gamma"], a_max=a.bcbf_amax, d_max=ph["dmax"])
    probe = BeliefCBF(sys_b, R_c=ph["R"], kappa=ph["kappa"])
    margin = 0.0
    if a.bcbf_eps is not None:
        margin = theorem3_margin((1e-3, 0.05), ph["r"], a.dt, probe.delta, a.bcbf_eps)
    cbf = BeliefCBF(sys_b, R_c=ph["R"], kappa=ph["kappa"], pole=a.bcbf_pole, margin=margin,
                    robust=not a.bcbf_nonrobust)
    print(f"belief CBF: delta={cbf.delta:.4f} (kappa={ph['kappa']}), pole={a.bcbf_pole}, "
          f"a_max={a.bcbf_amax}, margin gamma={margin:.4f}, robust={cbf.robust}")
    return cbf


def training_values(ckpt, cls):
    """Constructor arguments of `cls` found in the last config_*.txt of the checkpoint's experiment dir,
    cast to the type of their default.  Every one of them can change V: kappa enters boundary_fn
    (added by io_to_value), and the ranges (xy_max, P_hi, ...) set the input normalisation."""
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))))
    cfgs = sorted(f for f in os.listdir(exp_dir) if f.startswith("config_") and f.endswith(".txt")) \
        if os.path.isdir(exp_dir) else []
    if not cfgs:
        print(f"WARNING: no training config next to {ckpt}; using {cls.__name__} defaults (check kappa!)")
        return {}
    train = {}
    for line in open(os.path.join(exp_dir, cfgs[-1])):
        if " = " in line:
            k, v = [t.strip() for t in line.split(" = ", 1)]
            train[k] = v
    out = {}
    for name, prm in inspect.signature(cls).parameters.items():
        if name in train and name != "set_mode" and prm.default is not inspect.Parameter.empty:
            try:
                out[name] = type(prm.default)(train[name])
            except (TypeError, ValueError):
                pass
    changed = {k: v for k, v in out.items() if v != inspect.signature(cls).parameters[k].default}
    print(f"{cls.__name__}: parameters from {cfgs[-1]}"
          + (" (differ from class defaults: " + ", ".join(f"{k}={v}" for k, v in changed.items()) + ")"
             if changed else " (all equal to class defaults)"))
    return out


def check_training_config(dyn, ckpt, keys=("kappa", "gamma", "q", "r", "collisionR", "dist_max", "thrust_max")):
    """Warn if the dynamics used here differ from the config the checkpoint was trained with.
    (io_to_value adds boundary_fn, which depends on kappa, so a mismatch changes V.)"""
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))))
    cfgs = sorted(f for f in os.listdir(exp_dir) if f.startswith("config_") and f.endswith(".txt")) \
        if os.path.isdir(exp_dir) else []
    if not cfgs:
        return
    train = {}
    for line in open(os.path.join(exp_dir, cfgs[-1])):
        if " = " in line:
            k, v = line.split(" = ", 1)
            train[k.strip()] = v.strip()
    attr = {"dist_max": "dist_max", "thrust_max": "thrust_max"}
    for k in keys:
        if k in train and hasattr(dyn, attr.get(k, k)):
            try:
                tv, uv = float(train[k]), float(getattr(dyn, attr.get(k, k)))
            except ValueError:
                continue
            if abs(tv - uv) > 1e-9:
                print(f"WARNING: {k}={uv} here but {k}={tv} in training config {cfgs[-1]} "
                      f"-> pass --est_kwargs '{{\"{k}\": {tv}}}'")


# ----------------------------------------------------------------------- main
def make_init(a, ph, rng):
    N = a.n_rollouts
    s0 = np.zeros((N, 6))
    s0[:, 0] = rng.uniform(*a.x0_range, N)
    s0[:, 1] = rng.uniform(*a.y0_range, N)
    s0[:, 2] = rng.uniform(-0.1, 0.1, N)
    s0[:, 3:5] = rng.uniform(-0.2, 0.2, (N, 2))
    P0 = np.tile([a.P0, a.P0, 0.0], (N, 1))
    xh0 = s0[:, 0:2] + np.sqrt(a.P0) * rng.standard_normal((N, 2))
    return dict(s0=s0, xh0=xh0, P0=P0)


def make_noise(a, N, scale, rng):
    return dict(w=rng.standard_normal((a.steps, N, 2)), n=rng.standard_normal((a.steps, N, 2)),
                wind_const=a.wind_scale * a.dmax * rng.choice([-1., 1.], (N, 2)), scale=scale)


def run_batch(a, ph, orcs, init, scale, seed):
    noise = make_noise(a, a.n_rollouts, scale, np.random.default_rng(seed + 1))
    H = {m: simulate(m, orcs.get(m), init, noise, a, ph, wind_orc=orcs["plain"])
         for m in METHODS if m == "none" or m in orcs}
    return H, {m: metrics(H[m], a, ph) for m in H}


def main():
    ap = argparse.ArgumentParser()
    # Checkpoint paths default to the constants at the top of the file (EDIT 0);
    # pass --plain_ckpt / --est_ckpt only if you want to override them from the terminal.
    ap.add_argument("--plain_ckpt", default=PLAIN_CKPT)
    ap.add_argument("--est_ckpt", default=EST_CKPT)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--t_max", type=float, default=1.0, help="DeepReach horizon used in training")
    ap.add_argument("--model", default="sine"); ap.add_argument("--model_mode", default="mlp")
    ap.add_argument("--num_nl", type=int, default=512); ap.add_argument("--num_hl", type=int, default=3)
    ap.add_argument("--plain_kwargs", default="{}"); ap.add_argument("--est_kwargs", default="{}")
    ap.add_argument("--n_rollouts", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.01); ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--delta", type=float, default=0.05, help="filter engages when V <= delta")
    ap.add_argument("--goal", type=float, nargs=2, default=[2.0, 0.0]); ap.add_argument("--goal_tol", type=float, default=0.3)
    ap.add_argument("--x0_range", type=float, nargs=2, default=[-2.2, -1.8])
    ap.add_argument("--y0_range", type=float, nargs=2, default=[-0.6, 0.6])
    ap.add_argument("--P0", type=float, default=0.0231)
    ap.add_argument("--wind", choices=["none", "corner", "toward", "hj"], default="hj",
                    help="hj: HJ-optimal wind of the plain value fn at the true state (training disturbance model)")
    ap.add_argument("--wind_scale", type=float, default=1.0, help="fraction of dist_max (1 = trained bound)")
    ap.add_argument("--noise_scales", type=float, nargs="+", default=[1.0], help="first = main; >1 value => sweep")
    ap.add_argument("--show_idx", type=int, default=-1, help="-1: auto-pick an informative rollout")
    ap.add_argument("--slice_n", type=int, default=201); ap.add_argument("--area_n", type=int, default=41, help="0 disables")
    # nominal controller gains
    ap.add_argument("--kp", type=float, default=1.0); ap.add_argument("--v_nom", type=float, default=1.0)
    ap.add_argument("--kv", type=float, default=3.0); ap.add_argument("--a_max", type=float, default=2.5)
    ap.add_argument("--kth", type=float, default=36.0); ap.add_argument("--kom", type=float, default=12.0)
    # belief CBF baseline (belief_cbf.py); delta is set from the 9D model's kappa (same sigma multiplier)
    ap.add_argument("--no_bcbf", action="store_true", help="skip the belief CBF baseline")
    ap.add_argument("--no_inflated", action="store_true", help="skip the inflated plain-BRT baselines")
    ap.add_argument("--bcbf_pole", type=float, default=2.0, help="ECBF gains zeta = (pole^2, 2*pole)")
    ap.add_argument("--bcbf_amax", type=float, default=4.0,
                    help="acceleration box of the CBF QP [m/s^2] (nominal uses --a_max)")
    ap.add_argument("--bcbf_eps", type=float, default=None, help="Thm. 3 eps -> adds margin gamma (default 0)")
    ap.add_argument("--bcbf_nonrobust", action="store_true", help="ignore the wind bound in the CBF")
    a = ap.parse_args()
    a.plain_ckpt, a.est_ckpt = resolve_path(a.plain_ckpt), resolve_path(a.est_ckpt)

    for name, path in (("plain", a.plain_ckpt), ("est", a.est_ckpt)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} checkpoint not found: '{path}'. "
                                    f"Set PLAIN_CKPT / EST_CKPT at the top of this file (EDIT 0).")

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    # dynamics parameters come from each checkpoint's training config (they change V: kappa enters
    # boundary_fn, the ranges set the input normalisation); --plain_kwargs / --est_kwargs override them
    plain_kw = {**training_values(a.plain_ckpt, D.PlanarQuadrotor), **json.loads(a.plain_kwargs)}
    est_kw = {**training_values(a.est_ckpt, D.EstimationAwarePlanarQuadrotor), **json.loads(a.est_kwargs)}
    dp = D.PlanarQuadrotor(set_mode="avoid", **plain_kw)
    de = D.EstimationAwarePlanarQuadrotor(set_mode="avoid", **est_kw)
    for d in (dp, de):
        if not hasattr(d, "tMax"):
            d.tMax = a.t_max
    orcs = {"plain": ValueOracle(dp, load_net(dp, a.plain_ckpt, a), dp.tMax),
            "est": ValueOracle(de, load_net(de, a.est_ckpt, a), de.tMax)}
    ph = dict(m=de.m, g=de.g, arm=de.arm, I=de.I, umax=de.thrust_max, dmax=de.dist_max, R=de.collisionR,
              q=de.q, r=de.r, gamma=de.gamma, kappa=de.kappa)
    a.dmax = ph["dmax"]
    check_training_config(de, a.est_ckpt)
    if not a.no_bcbf:
        orcs["bcbf"] = make_bcbf(a, ph)
    if not a.no_inflated:
        P_ss = math.sqrt(de.q / de.c_ric)                 # isotropic fixed point of P' = qI - c P^2
        P_max = de.P_hi + de.Pxy_max                      # largest lambda_max(P) in the training box
        for key, Pv in (("plain_ss", P_ss), ("plain_max", P_max)):
            orcs[key] = InflatedPlain(orcs["plain"], de.kappa * math.sqrt(Pv))
            print(f"{key}: plain target inflated by kappa*sqrt({Pv:.4f}) = {orcs[key].margin:.4f} "
                  f"(engages when V_plain <= {a.delta + orcs[key].margin:.4f})")
    rng_common = common_ranges(dp, de)

    init = make_init(a, ph, np.random.default_rng(a.seed))
    sweep = {}
    for si, sc in enumerate(a.noise_scales):
        H, M = run_batch(a, ph, orcs, init, sc, a.seed)
        sweep[sc] = summarize(M)
        print(f"\n=== noise scale {sc} ===")
        for m, r in sweep[sc].items():
            print(f"{m:>6}: " + ", ".join(f"{k}={v:.3f}" for k, v in r.items()))
        if si > 0:
            continue

        # sanity: trajectories must coincide before first engagement
        kb, ok, zs = base_states(H["plain"], H["est"], a.steps)
        chk = [np.allclose(H["plain"]["s"][kb[i], i], H["est"]["s"][kb[i], i], atol=1e-6) for i in np.where(ok)[0]]
        if not all(chk):
            print("WARNING: plain/est histories differ at the shared engagement step")

        # pick a rollout to show
        if a.show_idx >= 0:
            i = a.show_idx
        else:
            score = (M["plain"]["collided"] & ~M["est"]["collided"]).astype(float) * 1e3 \
                + (M["plain"]["k_eng"] - M["est"]["k_eng"]).astype(float)
            score[~ok] = -1e9
            i = int(np.argmax(score))
        print(f"showing rollout {i}: plain engages k={M['plain']['k_eng'][i]}, est engages k={M['est']['k_eng'][i]}")

        fig_single(i, H, M, a, ph, f"{a.out}/single_rollout.pdf")
        if ok[i]:
            fig_slices(i, kb, zs, orcs["plain"], orcs["est"], rng_common, a, ph, f"{a.out}/brt_slices_engagement.pdf")
            save_value_figs(i, zs, orcs["plain"], orcs["est"], rng_common, a, a.out)
            fig_slice_pair(i, kb, zs, orcs["plain"], orcs["est"], rng_common, a, ph, f"{a.out}/quad_brt.png")
        else:
            print("WARNING: chosen rollout never engages the filter; slice figures skipped (try --show_idx)")
        fig_summary(M, init, sweep[sc], a, ph, f"{a.out}/multi_rollout_summary.pdf")

        # BRT size at the engagement state over all rollouts (x-y slice, common range)
        if a.area_n > 0:
            ratio = []
            cell = (np.diff(rng_common[0])[0] / (a.area_n - 1)) * (np.diff(rng_common[1])[0] / (a.area_n - 1))
            for j in np.where(ok)[0]:
                be = np.concatenate([zs["xh"][j], zs["s"][j, 2:6], zs["P"][j]])
                _, _, Vp = brt_slice(orcs["plain"], be[:6], 0, 1, rng_common, a.area_n)
                _, _, Ve = brt_slice(orcs["est"], be, 0, 1, rng_common, a.area_n)
                ap_, ae_ = (Vp < 0).sum() * cell, (Ve < 0).sum() * cell
                if ap_ > 0:
                    ratio.append(ae_ / ap_)
            if ratio:
                print(f"x-y BRT area ratio (est/plain) at engagement: mean={np.mean(ratio):.2f}, "
                      f"median={np.median(ratio):.2f}, n={len(ratio)}")
                sweep[sc]["area_ratio"] = dict(mean=float(np.mean(ratio)), median=float(np.median(ratio)), n=len(ratio))
        np.savez(f"{a.out}/rollouts_main.npz", **{f"{m}_{k}": v for m, x in M.items() for k, v in x.items()}, **init)
        with open(f"{a.out}/table.tex", "w") as f:
            f.write(latex_table({m: sweep[sc][m] for m in METHODS if m in sweep[sc]}))

    if len(a.noise_scales) > 1:
        fig_sweep({s: {m: v for m, v in r.items() if m in METHODS} for s, r in sweep.items()},
                  f"{a.out}/noise_sweep.pdf")
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump({str(k): v for k, v in sweep.items()}, f, indent=2)


if __name__ == "__main__":
    main()