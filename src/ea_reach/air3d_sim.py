"""
Air3D + CT-EKF simulation utilities for the estimation-aware safety filter.

Library part (used by scripts/experiments/air3d_safety_filter.py):
  CFG             system / filter parameters (must match the checkpoint's training config)
  ValueModel      DeepReach value network wrapper: V and dV at (t, z)
  est_kwargs      EstimationAwareAir3D constructor arguments from CFG
  check_plant     checks the numpy simulator against EstimationAwareAir3D.dsdt
  est_dsdt, rk4, wrap

Demo part (``python -m ea_reach.air3d_sim``; not used for the paper tables): one animated
rollout per scenario, plus summary.json.
  A  evader heading north, initial covariance Sigma0 = 0.001 * I
  B  same geometry, initial covariance Sigma0 = 0.5 * I (outside the k=3 model's training range)
  C  standard Air3D filter vs estimation-aware filter under simulated noise (needs --std-ckpt,
     a DeepReach Air3D network)

Nominal evader policy: hold heading north (theta_E = pi/2).
Safety filter: when V(t_query, x_hat, Sigma) <= threshold, the evader switches to
the HJ-optimal turn rate; otherwise it applies the nominal command.

The simulation is stochastic: process noise Q on the pursuer, measurement noise R,
and an estimator whose covariance obeys exactly the Riccati equation
used in EstimationAwareAir3D.dsdt. The pursuer does pure pursuit on the TRUE
positions. Capture is judged on the TRUE distance, not the estimate.

Usage:
  python -m ea_reach.air3d_sim --scenario A
  python -m ea_reach.air3d_sim --scenario C --std-ckpt <air3d_model.pth>
"""
import os
import sys
import json
import inspect
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse, Circle, Polygon

from ea_reach.paths import add_deepreach_to_path, checkpoint, OUTPUTS

add_deepreach_to_path()
from utils import modules
from dynamics import dynamics as dyn_mod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = dict(
    vP=0.75, vE=0.75, uP_max=3.0, uE_max=4.0,
    kappa=3.0, R=0.05, Q=0.05, Rc=0.25,      # kappa, xy_max: must match the checkpoint's training config
    sigma_eps=1e-4, xy_max=3.0,
    sigma_lo=0.001, sigma_hi=0.05, sigma_xy_max=0.025,
    angle_alpha_factor=1.2,
    dt=0.01, T_sim=5.0, t_query=1.0,
    pursuer_gain=3.0, heading_gain=3.0,
    num_nl=512, num_hl=3, net="sine", net_mode="mlp",
)
CFG["sigma_ss"] = float(np.sqrt(CFG["Q"] * CFG["R"]))  # steady-state sigma sqrt(QR) = 0.05

SCENARIOS = {
    "A": dict(title="Scenario A: low initial uncertainty (Sigma0 = 0.001)",
              p0=[2.0, -0.5], psi0=np.pi/2, sigma0=0.001),
    "B": dict(title="Scenario B: high initial uncertainty (Sigma0 = 0.5)",
              p0=[2.0, -0.5], psi0=np.pi/2, sigma0=0.5),
    "C": dict(title="Scenario C: standard Air3D vs estimation-aware (steady-state noise)",
              p0=[2.0, -0.5], psi0=np.pi/2, sigma0=CFG["sigma_ss"]),
}


def est_kwargs(c=CFG):
    return dict(vP=c["vP"], vE=c["vE"], uP_max=c["uP_max"], uE_max=c["uE_max"],
                kappa=c["kappa"], Rx=c["R"], Ry=c["R"], Qx=c["Q"], Qy=c["Q"],
                Rc=c["Rc"], sigma_eps=c["sigma_eps"], xy_max=c["xy_max"],
                sigma_lo=c["sigma_lo"], sigma_hi=c["sigma_hi"],
                sigma_xy_max=c["sigma_xy_max"],
                angle_alpha_factor=c["angle_alpha_factor"])


def std_kwargs(cls, c=CFG):
    """Build Air3D kwargs from whatever the constructor of YOUR fork accepts.
    The standard model should be trained with the same speeds / turn rates as the
    estimation-aware one, otherwise the comparison is not apples to apples."""
    pool = dict(collisionR=c["Rc"], velocity=c["vE"],
                omega_max_control=c["uE_max"],   # evader turn rate (4)
                omega_max_dist=c["uP_max"],      # pursuer turn rate (3)
                omega_max=c["uE_max"],
                evader_speed=c["vE"], pursuer_speed=c["vP"],
                evader_max_turn_rate=c["uE_max"], pursuer_max_turn_rate=c["uP_max"],
                angle_alpha_factor=c["angle_alpha_factor"],
                set_mode="avoid")
    params = inspect.signature(cls.__init__).parameters
    missing = [k for k in params if k not in ("self", "args", "kwargs") and k not in pool]
    if missing:
        raise TypeError(f"Air3D needs arguments I can't fill: {missing}. Edit std_kwargs().")
    return {k: pool[k] for k in params if k in pool}



def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def to_t(x):
    return torch.tensor(np.asarray(x, dtype=np.float64))[None]


def est_dsdt(z, uE, uP, c=CFG):
    """Numpy copy of EstimationAwareAir3D.dsdt with e_x = e_y = 0 (the nominal
    estimator dynamics). Checked against the torch class in check_plant()."""
    x, y, psi, Sxx, Syy, Sxy = z
    R, Q = c["R"], c["Q"]
    return np.array([
        c["vP"] * np.cos(psi) - c["vE"] + uE * y,
        c["vP"] * np.sin(psi) - uE * x,
        uP - uE,
        2 * uE * Sxy + Q - (Sxx ** 2 / R + Sxy ** 2 / R),
        -2 * uE * Sxy + Q - (Sxy ** 2 / R + Syy ** 2 / R),
        uE * (Syy - Sxx) - (Sxx * Sxy / R + Sxy * Syy / R),
    ])


def rk4(f, z, dt):
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def check_plant(dyn, c=CFG):
    """Make sure the numpy estimator matches the class you trained."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        z = np.array([*rng.uniform(-2, 2, 2), rng.uniform(-3, 3),
                      *rng.uniform(0.05, 0.4, 2), rng.uniform(-0.05, 0.05)])
        uE, uP = rng.uniform(-4, 4), rng.uniform(-3, 3)
        ref = dyn.dsdt(to_t(z), to_t([uE]), to_t([uP, 0.0, 0.0]))[0].numpy()
        assert np.allclose(ref, est_dsdt(z, uE, uP, c), atol=1e-8), \
            "numpy dynamics != EstimationAwareAir3D.dsdt (check CFG Q/R/speeds)"


class ValueModel:
    def __init__(self, ckpt, dyn_name, dyn_kw, cfg=CFG):
        cls = getattr(dyn_mod, dyn_name)
        self.dyn = cls(**(dyn_kw(cls) if callable(dyn_kw) else dyn_kw))
        self.n = self.dyn.state_dim
        self.model = modules.SingleBVPNet(
            in_features=self.dyn.input_dim, out_features=1,
            type=cfg["net"], mode=cfg["net_mode"],
            hidden_features=cfg["num_nl"], num_hidden_layers=cfg["num_hl"],
            periodic_transform_fn=self.dyn.periodic_transform_fn,
        ).to(DEVICE)
        obj = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        if isinstance(obj, dict):
            obj = obj.get("model", obj.get("model_state_dict", obj))
        self.model.load_state_dict(obj)
        self.model.eval()
        self.rng = np.array(self.dyn.state_test_range(), dtype=float)

    def clamp(self, z):
        z = np.array(z[:self.n], dtype=float)
        z[2] = wrap(z[2])
        zc = np.clip(z, self.rng[:, 0], self.rng[:, 1])
        return zc, bool(np.any(np.abs(zc - z) > 1e-9))

    def query(self, t, z):
        zc, oob = self.clamp(z)
        coord = torch.tensor([[t, *zc]], dtype=torch.float32, device=DEVICE)
        inp = self.dyn.coord_to_input(coord)
        out = self.model({"coords": inp})
        mo = out["model_out"].squeeze(-1)
        V = self.dyn.io_to_value(out["model_in"], mo)
        dv = self.dyn.io_to_dv(out["model_in"], mo)
        return V.item(), dv[0, 1:].detach().cpu().numpy(), zc, oob

    def optimal_u(self, zc, dV):
        return float(self.dyn.optimal_control(to_t(zc), to_t(dV))[0, 0])

    def value_batch(self, t, Z):
        Zc = np.array([self.clamp(z)[0] for z in Z])
        coord = torch.tensor(np.c_[np.full(len(Zc), t), Zc], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            out = self.model({"coords": self.dyn.coord_to_input(coord)})
            V = self.dyn.io_to_value(out["model_in"], out["model_out"].squeeze(-1))
        return V.detach().cpu().numpy()


class SafetyFilter:
    """Least-restrictive filter: nominal command unless V <= threshold."""

    def __init__(self, model, uE_max, threshold=0.05, hysteresis=0.02,
                 t_query=1.0, enabled=True):
        self.m, self.uE_max = model, uE_max
        self.thr, self.hyst, self.t, self.enabled = threshold, hysteresis, t_query, enabled
        self.engaged = False

    def reset(self):
        self.engaged = False

    def __call__(self, z, u_nom):
        V, dV, zc, oob = self.m.query(self.t, z)
        if self.enabled:
            if V <= self.thr:
                self.engaged = True
            elif V > self.thr + self.hyst:
                self.engaged = False
        u = self.m.optimal_u(zc, dV) if self.engaged else float(np.clip(u_nom, -self.uE_max, self.uE_max))
        return u, dict(V=V, active=self.engaged, oob=oob)



def heading_hold_north(thE, c=CFG):
    return float(np.clip(c["heading_gain"] * wrap(np.pi / 2 - thE), -c["uE_max"], c["uE_max"]))


def pure_pursuit(P, E, thP, c=CFG):
    err = wrap(np.arctan2(*(E - P)[::-1]) - thP)
    return float(c["uP_max"] * np.clip(c["pursuer_gain"] * err, -1.0, 1.0))


def simulate(sc, filt, c=CFG, seed=0):
    """World frame: evader starts at the origin heading north. The pursuer starts
    at p0 (evader body frame, x forward / y left) with relative heading psi0."""
    rng = np.random.default_rng(seed)
    dt, n_steps = c["dt"], int(round(c["T_sim"] / c["dt"]))
    thE, E = np.pi / 2, np.zeros(2)
    p0 = np.asarray(sc["p0"], float)
    P = E + rot(thE) @ p0
    thP = thE + sc["psi0"]
    s0 = sc["sigma0"]
    z = np.array([*(p0 + np.sqrt(s0) * rng.standard_normal(2)),
                  wrap(sc["psi0"]), s0, s0, 0.0])
    filt.reset()

    log = {k: [] for k in ["E", "thE", "P", "thP", "p", "xhat", "S", "V", "active", "uE", "u_nom"]}
    captured = left = False
    for _ in range(n_steps):
        p = rot(-thE) @ (P - E)              # true relative position (evader body frame)
        z[2] = wrap(thP - thE)               # psi is known to the estimator
        u_nom = heading_hold_north(thE, c)
        uE, info = filt(z, u_nom)
        uP = pure_pursuit(P, E, thP, c)

        for k, v in dict(E=E.copy(), thE=thE, P=P.copy(), thP=thP, p=p.copy(),
                         xhat=z[:2].copy(), S=z[3:].copy(), V=info["V"],
                         active=info["active"], uE=uE, u_nom=u_nom).items():
            log[k].append(v)

        if np.hypot(*p) <= c["Rc"]:
            captured = True
            break
        if np.any(np.abs(z[:2]) > c["xy_max"]):
            left = True
            break

        zn = rk4(lambda s: est_dsdt(s, uE, uP, c), z, dt)
        meas = p + np.sqrt(c["R"] / dt) * rng.standard_normal(2)
        K = np.array([[z[3], z[5]], [z[5], z[4]]]) / c["R"]
        zn[:2] += dt * K @ (meas - z[:2])
        zn[3] = np.clip(zn[3], c["sigma_lo"], c["sigma_hi"])
        zn[4] = np.clip(zn[4], c["sigma_lo"], c["sigma_hi"])
        lim = np.sqrt(zn[3] * zn[4])
        zn[5] = np.clip(zn[5], -lim, lim)

        # true world motion (midpoint heading is exact for constant turn rate)
        hE, hP = thE + 0.5 * dt * uE, thP + 0.5 * dt * uP
        E = E + dt * c["vE"] * np.array([np.cos(hE), np.sin(hE)])
        P = P + dt * c["vP"] * np.array([np.cos(hP), np.sin(hP)]) \
            + np.sqrt(c["Q"] * dt) * rng.standard_normal(2)
        thE, thP = thE + dt * uE, thP + dt * uP
        z = zn

    out = {k: np.array(v) for k, v in log.items()}
    out["t"] = np.arange(len(out["V"])) * dt
    out["captured"], out["left_domain"] = captured, left
    dist = np.linalg.norm(out["p"], axis=1)
    out["min_dist"] = float(dist.min())
    out["min_V"] = float(out["V"].min())
    out["frac_active"] = float(out["active"].mean())
    out["t_first_active"] = float(out["t"][np.argmax(out["active"])]) if out["active"].any() else None
    return out


def monte_carlo(sc, columns, n, seed0, c=CFG):
    """Re-run every column over n noise seeds; report TRUE capture rates."""
    flags = {lab: [] for lab, _ in columns}
    mind = {lab: [] for lab, _ in columns}
    for s in range(seed0, seed0 + n):
        for lab, filt in columns:
            r = simulate(sc, filt, c, seed=s)
            flags[lab].append(bool(r["captured"]))
            mind[lab].append(r["min_dist"])
    summary = {lab: dict(capture_rate=float(np.mean(flags[lab])),
                         mean_min_dist=float(np.mean(mind[lab])), n=n) for lab in flags}
    return summary, flags

def tri(x, y, th, s=0.2):
    pts = np.array([[1.0, 0.0], [-0.7, 0.6], [-0.7, -0.6]]) * s
    return (rot(th) @ pts.T).T + np.array([x, y])


def world_extent(runs, pad=0.8):
    pts = np.vstack([np.vstack([r["E"], r["P"]]) for r in runs])
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    ctr, half = (lo + hi) / 2, max(hi - lo) / 2
    return ctr[0] - half, ctr[0] + half, ctr[1] - half, ctr[1] + half


def animate(runs, labels, title, path, c=CFG, threshold=0.05, fps=25, stride=4):
    ncol = len(runs)
    fig, axs = plt.subplots(2, ncol, figsize=(5.2 * ncol, 8.2),
                            gridspec_kw=dict(height_ratios=[3, 1.3]), squeeze=False)
    x0, x1, y0, y1 = world_extent(runs)
    vmax = min(2.0, max(r["V"].max() for r in runs))
    vmin = min(r["V"].min() for r in runs) - 0.2
    tmax = max(r["t"][-1] for r in runs) + 0.2

    art = []
    for j, (r, lab) in enumerate(zip(runs, labels)):
        aw, av = axs[0, j], axs[1, j]
        aw.set_xlim(x0, x1); aw.set_ylim(y0, y1); aw.set_aspect("equal"); aw.grid(alpha=0.25)
        aw.set_title(lab, fontsize=11)
        aw.set_xlabel("east"); aw.set_ylabel("north")
        lnE, = aw.plot([], [], color="tab:blue", lw=1.6)
        lnP, = aw.plot([], [], color="tab:red", lw=1.6)
        pE = Polygon(tri(0, 0, 0), closed=True, fc="tab:blue", ec="k", zorder=5)
        pP = Polygon(tri(0, 0, 0), closed=True, fc="tab:red", ec="k", zorder=5)
        aw.add_patch(pE); aw.add_patch(pP)
        cap = Circle((0, 0), c["Rc"], fill=False, ec="tab:red", ls="--", lw=1)
        aw.add_patch(cap)
        ell = Ellipse((0, 0), 1, 1, fill=False, ec="tab:orange", lw=1.3)
        aw.add_patch(ell)
        est, = aw.plot([], [], "x", color="tab:orange", ms=7)
        txt = aw.text(0.02, 0.98, "", transform=aw.transAxes, va="top",
                      family="monospace", fontsize=9,
                      bbox=dict(fc="white", ec="none", alpha=0.8))
        if j == 0:
            aw.plot([], [], color="tab:blue", label="evader (true)")
            aw.plot([], [], color="tab:red", label="pursuer (true)")
            aw.plot([], [], "x", color="tab:orange", label=f"estimate + {c['kappa']:g}-sigma ellipse")
            aw.legend(loc="lower left", fontsize=8)

        av.plot(r["t"], r["V"], color="k", lw=1.1)
        av.axhline(threshold, color="tab:orange", ls="--", lw=1, label=f"threshold = {threshold:g}")
        av.axhline(0, color="k", lw=0.8)
        act = r["active"].astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, act, 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            av.axvspan(r["t"][a], r["t"][min(b, len(r["t"])) - 1] + c["dt"], color="tab:orange", alpha=0.25, lw=0)
        av.set_xlim(0, tmax); av.set_ylim(vmin, vmax)
        av.set_xlabel("t [s]"); av.set_ylabel("V(x_hat, Sigma)"); av.grid(alpha=0.25)
        if j == 0:
            av.legend(loc="upper right", fontsize=8)
        cur = av.axvline(0, color="tab:blue", lw=1)
        dot, = av.plot([], [], "o", color="tab:blue")
        art.append(dict(r=r, lnE=lnE, lnP=lnP, pE=pE, pP=pP, cap=cap, ell=ell, est=est,
                        txt=txt, cur=cur, dot=dot))

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    # last frame lands exactly on the final step, then hold ~0.8 s so the outcome is visible
    n_frames = int(np.ceil((max(len(r["V"]) for r in runs) - 1) / stride)) + 1 + int(0.8 * fps)

    def update(f):
        for a in art:
            r = a["r"]
            i = min(f * stride, len(r["V"]) - 1)
            E, P, thE, thP = r["E"], r["P"], r["thE"][i], r["thP"][i]
            a["lnE"].set_data(E[:i + 1, 0], E[:i + 1, 1])
            a["lnP"].set_data(P[:i + 1, 0], P[:i + 1, 1])
            a["pE"].set_xy(tri(*E[i], thE))
            a["pE"].set_facecolor("tab:orange" if r["active"][i] else "tab:blue")
            a["pP"].set_xy(tri(*P[i], thP))
            a["cap"].center = tuple(P[i])
            ctr = E[i] + rot(thE) @ r["xhat"][i]
            Sxx, Syy, Sxy = r["S"][i]
            w, V = np.linalg.eigh(np.array([[Sxx, Sxy], [Sxy, Syy]]))
            a["ell"].set_center(tuple(ctr))
            a["ell"].set_width(2 * c["kappa"] * np.sqrt(max(w[1], 0)))
            a["ell"].set_height(2 * c["kappa"] * np.sqrt(max(w[0], 0)))
            a["ell"].set_angle(np.degrees(np.arctan2(V[1, 1], V[0, 1]) + thE))
            a["est"].set_data([ctr[0]], [ctr[1]])
            a["cur"].set_xdata([r["t"][i], r["t"][i]])
            a["dot"].set_data([r["t"][i]], [r["V"][i]])
            end = ""
            if i == len(r["V"]) - 1:
                end = "\nCAPTURED" if r["captured"] else ""
            a["txt"].set_text(f"t={r['t'][i]:.2f}s  V={r['V'][i]:+.2f}\n"
                              f"filter: {'ACTIVE' if r['active'][i] else 'off'}\n"
                              f"Sigma_xx={Sxx:.3f}{end}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    if path.endswith(".mp4") and animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps)
    else:
        if path.endswith(".mp4"):
            path = path[:-4] + ".gif"
        writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=90)
    plt.close(fig)
    return path


def plot_brt_comparison(est, std, path, c=CFG, psi=np.pi, lim=2.5, n=81):
    """The one static figure: zero level sets of V (the BRT boundary) for the standard
    and the estimation-aware model at a few covariance levels."""
    xs = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, xs)
    fig, ax = plt.subplots(figsize=(6, 6))
    Zs = np.c_[X.ravel(), Y.ravel(), np.full(X.size, psi)]
    ax.contour(xs, xs, std.value_batch(c["t_query"], Zs).reshape(X.shape), levels=[0],
               colors="k", linestyles="--", linewidths=2)
    ax.plot([], [], "k--", label="standard Air3D")
    for s, col in zip([c["sigma_lo"], c["sigma_ss"], c["sigma_hi"]], ["tab:green", "tab:orange", "tab:red"]):
        Z = np.c_[Zs, np.full(X.size, s), np.full(X.size, s), np.zeros(X.size)]
        ax.contour(xs, xs, est.value_batch(c["t_query"], Z).reshape(X.shape), levels=[0],
                   colors=col, linewidths=2)
        ax.plot([], [], color=col, label=f"estimation-aware, Sigma={s:.3g}")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("x_hat (evader forward)"); ax.set_ylabel("y_hat (evader left)")
    ax.set_title(f"BRT boundary V=0 at t={c['t_query']:g}, psi={psi:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_columns(key, est, std, a):
    mk = lambda m, en=True: SafetyFilter(m, CFG["uE_max"], a.threshold, a.hysteresis, CFG["t_query"], en)
    cols = [("Nominal only (north), no filter", mk(est, False))]
    if key in ("A", "B"):
        cols.append(("Estimation-aware safety filter", mk(est)))
    else:
        cols += [("Standard Air3D filter (ignores Sigma)", mk(std)),
                 ("Estimation-aware filter", mk(est))]
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=checkpoint("air3d_ea_6d_k3"))
    ap.add_argument("--std-ckpt", default=None, help="DeepReach Air3D network (scenario C only)")
    ap.add_argument("--scenario", default="A", choices=["A", "B", "C", "all"])
    ap.add_argument("--out", default=os.path.join(OUTPUTS, "air3d_animations"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--hysteresis", type=float, default=0.02)
    ap.add_argument("--mc", type=int, default=20, help="Monte Carlo runs per scenario (0 = skip)")
    ap.add_argument("--pick", action="store_true",
                    help="scenario C: animate the first seed where the standard filter fails "
                         "and the estimation-aware one survives (cherry-picked; MC rates are the evidence)")
    ap.add_argument("--fmt", default="gif", choices=["gif", "mp4"])
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    keys = ["A", "B", "C"] if a.scenario == "all" else [a.scenario]
    if "C" in keys and a.std_ckpt is None:
        ap.error("scenario C needs --std-ckpt")

    est = ValueModel(a.ckpt, "EstimationAwareAir3D", est_kwargs())
    check_plant(est.dyn)
    std = ValueModel(a.std_ckpt, "Air3D", std_kwargs) if "C" in keys else None
    summary = {}

    for key in keys:
        sc = SCENARIOS[key]
        for i, (lo, hi) in enumerate(est.rng):
            z0 = [*sc["p0"], sc["psi0"], sc["sigma0"], sc["sigma0"], 0.0][i]
            assert lo <= z0 <= hi, f"scenario {key}: state[{i}]={z0} outside [{lo}, {hi}]"
        cols = build_columns(key, est, std, a)
        seed = a.seed
        entry = {}
        if a.mc > 0:
            entry["monte_carlo"], flags = monte_carlo(sc, cols, a.mc, a.seed)
            print(f"\n{sc['title']}  ({a.mc} noise seeds, true capture):")
            for lab, v in entry["monte_carlo"].items():
                print(f"  {lab:40s} capture rate {v['capture_rate']:.2f}   mean min dist {v['mean_min_dist']:.2f}")
            if key == "C" and a.pick:
                labs = [l for l, _ in cols]
                for k in range(a.mc):
                    if flags[labs[1]][k] and not flags[labs[2]][k]:
                        seed = a.seed + k
                        break
                else:
                    print("  --pick: no seed with that pattern, using the default seed")
        runs = [simulate(sc, f, CFG, seed=seed) for _, f in cols]
        entry["animated_seed"] = seed
        entry["animated_run"] = {lab: dict(captured=bool(r["captured"]), min_dist=r["min_dist"],
                                           min_V=r["min_V"], frac_active=r["frac_active"],
                                           t_first_active=r["t_first_active"])
                                 for (lab, _), r in zip(cols, runs)}
        path = animate(runs, [l for l, _ in cols], f"{sc['title']}  (seed {seed})",
                       os.path.join(a.out, f"scenario_{key}.{a.fmt}"), CFG, a.threshold, a.fps)
        print("saved", path)
        summary[key] = entry
        if key == "C":
            plot_brt_comparison(est, std, os.path.join(a.out, "C_brt_comparison.png"))

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("saved", os.path.join(a.out, "summary.json"))


if __name__ == "__main__":
    main()