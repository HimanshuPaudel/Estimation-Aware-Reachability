"""Overlay the reduced 4D estimation-aware Air3D grid BRT on the vanilla
(perfect-state-knowledge) 3D Air3D grid BRT, and quantify how much the
estimation-aware BRT grows relative to vanilla.

Estimation-aware 4D (from scripts/grid/air3d_ea_4d_grid.py), PREFIX = data/air3d_ea4d
(R = Q = 0.05, s in [1e-3, 0.05]):
  <PREFIX>_values_all.npy   (nt, x, y, psi, s)
  <PREFIX>_grid_coords.npy  object array [x, y, psi, s]
  <PREFIX>_taus.npy         (optional)

Vanilla 3D Air3D (from scripts/grid/air3d_nominal_grid.py, PREFIX = data/air3d_nominal_xy1):
  <VAN_PREFIX>_values_all.npy  (nt, x, y, psi)   or (x, y, psi) if only the
                               final-time BRT was saved
  <VAN_PREFIX>_grid_coords.npy (optional) object array [x, y, psi]; if missing,
                               x, y in [-xy_max, xy_max] and psi in [-pi, pi)
                               (periodic, endpoint excluded) are assumed
  <VAN_PREFIX>_taus.npy        (optional) else linspace(0, T, nt)
The vanilla solve must use the same vP, vE, uP_max, uE_max and capture radius
Rc as the estimation-aware run (the EA target additionally inflates Rc by
kappa*sqrt(s + eps)).

Convention: {V <= 0} is the BRT for both. If your vanilla solution uses the
opposite sign, pass --vanilla-sign -1.

The EA 4D grid is cropped to |x|, |y| <= --xy-lim (default 2.5, the DeepReach
training domain; 0 disables) before plotting and metrics.

Rendering: EA grid set = smoothed contours (cubic upsampling, default; output
in <outdir>_smooth) or raw jagged cells with --raw (output in <outdir>);
vanilla = smooth contour (blue, dashed) on its own grid. Both use the nearest
periodic psi node. Metrics use native grid values, never the upsampled drawing.

Metrics (EA set E vs vanilla set V, both on the EA grid nodes; expectation:
the EA BRT contains the vanilla BRT, since estimation uncertainty only hurts):
  iou           |E & V| / |E | V|
  containment   |E & V| / |V|    (fraction of vanilla BRT inside EA; ~1 expected)
  expansion     |E| / |V|        (area growth of the BRT due to estimation)
  extra_area    area of E & ~V   (x/y units^2)
  hd_ea_to_van  max distance from an EA-BRT cell to the vanilla BRT (how far
                the EA BRT sticks out)
  hd_van_to_ea  max distance from a vanilla-BRT cell to the EA BRT (~0 expected)
  hausdorff     symmetric Hausdorff distance

Outputs (in <outdir> or <outdir>_smooth):
  fig_overlay_vanilla_s<sigma>.png          rows = tau, cols = psi
  paper_overlay_vanilla_tau_<tau>.{png,pdf} one row, one panel per
                                            (sigma, psi) pair, fixed tau
  metrics_vs_vanilla.csv                    per (sigma, tau, psi)

Usage:
  python scripts/plotting/air3d_grid_vs_nominal.py
  python scripts/plotting/air3d_grid_vs_nominal.py --raw
  python scripts/plotting/air3d_grid_vs_nominal.py --paper-sigmas 0.001 0.05 --paper-psis 3.14159 1.5708
  python scripts/plotting/air3d_grid_vs_nominal.py --vanilla-prefix data/air3d_nominal_xy3
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt, zoom

# ---------------- settings ----------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
PREFIX = os.path.join(REPO_ROOT, "data", "air3d_ea4d")             # scripts/grid/air3d_ea_4d_grid.py
VAN_PREFIX = os.path.join(REPO_ROOT, "data", "air3d_nominal_xy1")  # scripts/grid/air3d_nominal_grid.py --xy-max 1 ...
OUT_DIR = os.path.join(REPO_ROOT, "figures", "air3d_grid_vs_nominal")
T = 1.0
XY_MAX = 4.0                         # vanilla grid extent if coords file missing
XY_LIM = 2.5                         # crop EA grid to this (0 = no crop)
S_LIST = [0.001, 0.019, 0.033, 0.05]   # span the 4D grid's s range [1e-3, 0.05]
T_LIST = [0.0, 0.5, 1.0]
THETA_LIST = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
DPI = 100
CELL = 3.2
EA_COLOR = "tab:red"       # estimation-aware 4D grid
VAN_COLOR = "tab:blue"     # vanilla Air3D
# ------------------------------------------

METRIC_KEYS = ["iou", "containment", "expansion", "extra_area",
               "hd_ea_to_van", "hd_van_to_ea", "hausdorff"]


def nearest(vec, val):
    return int(np.argmin(np.abs(np.asarray(vec) - val)))


def nearest_periodic(grid, v):
    d = np.abs((np.asarray(grid) - v + np.pi) % (2 * np.pi) - np.pi)
    return int(np.argmin(d))


def draw_set_smooth(ax, x, y, V, color, alpha_fill, lw, ls="--"):
    """Fill {V <= 0} and draw the interpolated zero contour. V: (nx, ny)."""
    vmin, vmax = np.nanmin(V), np.nanmax(V)
    if not (vmin <= 0.0 <= vmax):
        if vmax <= 0.0:
            ax.axhspan(y[0], y[-1], color=color, alpha=alpha_fill, lw=0)
        return
    X, Y = np.meshgrid(x, y, indexing="ij")
    ax.contourf(X, Y, V, levels=[-np.inf, 0.0], colors=[color], alpha=alpha_fill)
    ax.contour(X, Y, V, levels=[0.0], colors=[color], linewidths=lw, linestyles=ls)


def draw_set_raw(ax, x, y, V, color, alpha_fill, lw, ls="-", upsample=8):
    """Raw {V <= 0}: filled grid cells plus stair-step outline. V: (nx, ny)."""
    mask = np.asarray(V) <= 0.0
    if not mask.any():
        return
    nx, ny = mask.shape
    dx, dy = x[1] - x[0], y[1] - y[0]
    xe = np.concatenate([x - dx / 2, [x[-1] + dx / 2]])
    ye = np.concatenate([y - dy / 2, [y[-1] + dy / 2]])
    Xe, Ye = np.meshgrid(xe, ye, indexing="ij")
    ax.pcolormesh(Xe, Ye, np.ma.masked_where(~mask, np.ones(mask.shape)),
                  cmap=ListedColormap([color]), alpha=alpha_fill,
                  shading="flat", rasterized=True)
    if mask.all():
        return
    up = np.repeat(np.repeat(mask, upsample, axis=0), upsample, axis=1)
    xs = xe[0] + (np.arange(nx * upsample) + 0.5) * dx / upsample
    ys = ye[0] + (np.arange(ny * upsample) + 0.5) * dy / upsample
    Xs, Ys = np.meshgrid(xs, ys, indexing="ij")
    ax.contour(Xs, Ys, up.astype(float), levels=[0.5], colors=[color],
               linewidths=lw, linestyles=ls)


def smooth_upsample(x, y, V, factor):
    """Cubic upsample of V (nx, ny) for drawing only; returns (xs, ys, Vs)."""
    if factor <= 1:
        return x, y, V
    Vs = zoom(np.asarray(V, dtype=float), factor, order=3)
    return (np.linspace(x[0], x[-1], Vs.shape[0]),
            np.linspace(y[0], y[-1], Vs.shape[1]), Vs)


class Data:
    def __init__(self, args):
        # ---- estimation-aware 4D ----
        V_ea = np.load(f"{args.ea_prefix}_values_all.npy", mmap_mode="r")
        cv = np.load(f"{args.ea_prefix}_grid_coords.npy", allow_pickle=True)
        x_e, y_e, self.psi_e, self.s_e = [np.asarray(c, float) for c in cv]
        if args.xy_lim > 0:               # crop to the DeepReach training domain
            ix = np.where(np.abs(x_e) <= args.xy_lim + 1e-9)[0]
            iy = np.where(np.abs(y_e) <= args.xy_lim + 1e-9)[0]
            V_ea = V_ea[:, ix[0]:ix[-1] + 1, iy[0]:iy[-1] + 1]
            x_e, y_e = x_e[ix], y_e[iy]
        self.V_ea, self.x_e, self.y_e = V_ea, x_e, y_e
        self.dx, self.dy = float(x_e[1] - x_e[0]), float(y_e[1] - y_e[0])
        p = f"{args.ea_prefix}_taus.npy"
        self.tau_e = (np.load(p) if os.path.exists(p)
                      else np.linspace(0.0, args.T, V_ea.shape[0]))
        assert V_ea.shape[0] == len(self.tau_e), "EA taus must match snapshots"

        # ---- vanilla 3D ----
        V = np.load(f"{args.vanilla_prefix}_values_all.npy", mmap_mode="r")
        if V.ndim == 3:                      # final-time only
            V = V[None]
            print("Vanilla: single snapshot; assumed to be tau = T and reused "
                  "for every tau (check this is what you want).")
            self.tau_v = np.array([args.T])
        else:
            p = f"{args.vanilla_prefix}_taus.npy"
            self.tau_v = (np.load(p) if os.path.exists(p)
                          else np.linspace(0.0, args.T, V.shape[0]))
        self.V_van = V
        assert V.shape[0] == len(self.tau_v), "vanilla taus must match snapshots"
        p = f"{args.vanilla_prefix}_grid_coords.npy"
        if os.path.exists(p):
            cv = np.load(p, allow_pickle=True)
            self.x_v, self.y_v, self.psi_v = [np.asarray(c, float) for c in cv[:3]]
        else:
            nx, ny, npsi = V.shape[1:]
            self.x_v = np.linspace(-args.xy_max, args.xy_max, nx)
            self.y_v = np.linspace(-args.xy_max, args.xy_max, ny)
            self.psi_v = np.linspace(-np.pi, np.pi, npsi, endpoint=False)
        self.sign_v = args.vanilla_sign

    def ea_slice(self, tau, psi, s):
        r = nearest(self.tau_e, tau)
        j = nearest(self.s_e, s)
        ip = nearest_periodic(self.psi_e, psi)
        return (np.asarray(self.V_ea[r][:, :, ip, j]), self.tau_e[r],
                self.psi_e[ip], self.s_e[j])

    def van_slice(self, tau, psi):
        r = nearest(self.tau_v, tau)
        ip = nearest_periodic(self.psi_v, psi)
        return self.sign_v * np.asarray(self.V_van[r][:, :, ip]), self.tau_v[r]


def draw_ea(args, ax, D, Ve, lw):
    if args.smooth_grid:
        xs, ys, Vs = smooth_upsample(D.x_e, D.y_e, Ve, args.smooth_factor)
        draw_set_smooth(ax, xs, ys, Vs, EA_COLOR, 0.35, lw, ls="-")
    else:
        draw_set_raw(ax, D.x_e, D.y_e, Ve, EA_COLOR, 0.35, lw)


def panel(args, ax, D, tau, psi, s, lw):
    Ve, tau_e, psi_e, s_e = D.ea_slice(tau, psi, s)
    Vv, tau_v = D.van_slice(tau, psi)
    draw_ea(args, ax, D, Ve, lw)
    draw_set_smooth(ax, D.x_v, D.y_v, Vv, VAN_COLOR, 0.25, lw, ls="--")
    ax.set_xlim(D.x_e[0], D.x_e[-1])
    ax.set_ylim(D.y_e[0], D.y_e[-1])
    ax.set_aspect("equal")
    return Ve, Vv, tau_e, psi_e, s_e


def legend_handles():
    return [
        Patch(facecolor=EA_COLOR, alpha=0.35, edgecolor=EA_COLOR,
              label="Estimation-aware (4D grid)"),
        Patch(facecolor=VAN_COLOR, alpha=0.25, edgecolor=VAN_COLOR,
              label="Vanilla Air3D"),
    ]


# -------------------------------- metrics ----------------------------------

def vanilla_mask_on_ea_grid(D, Vv):
    """Vanilla {V<=0} sampled at the EA grid nodes (bilinear in V, then sign)."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((D.x_v, D.y_v), Vv, bounds_error=False,
                                fill_value=None)
    Xe, Ye = np.meshgrid(D.x_e, D.y_e, indexing="ij")
    return f(np.column_stack([Xe.ravel(), Ye.ravel()])).reshape(Xe.shape) <= 0.0


def directed_hd(A, B, dx, dy):
    """max over cells of A of the distance to the nearest cell of B."""
    if not A.any():
        return 0.0
    if not B.any():
        return np.nan
    return float(distance_transform_edt(~B, sampling=(dx, dy))[A].max())


def vs_vanilla_metrics(E, V, dx, dy):
    """E: EA BRT mask, V: vanilla BRT mask, both on the EA grid nodes."""
    inter, union = int((E & V).sum()), int((E | V).sum())
    nE, nV = int(E.sum()), int(V.sum())
    h_ev, h_ve = directed_hd(E, V, dx, dy), directed_hd(V, E, dx, dy)
    return dict(
        iou=inter / union if union else 1.0,
        containment=inter / nV if nV else np.nan,
        expansion=nE / nV if nV else np.nan,
        extra_area=float((E & ~V).sum()) * dx * dy,
        hd_ea_to_van=h_ev,
        hd_van_to_ea=h_ve,
        hausdorff=float(np.nanmax([h_ev, h_ve])) if not (np.isnan(h_ev) and np.isnan(h_ve)) else np.nan,
    )


def save_csv(rows, path):
    fields = ["sigma", "tau", "psi"] + METRIC_KEYS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print("saved", path)


def summarize(rows):
    print("\nSummary vs vanilla (mean over psi)")
    print(f"  {'sigma':>7} {'tau':>5} {'IoU':>7} {'contain':>8} {'expans':>7} "
          f"{'extraA':>8} {'HD EA>V':>8} {'HD V>EA':>8}")
    keys = sorted({(r["sigma"], r["tau"]) for r in rows})
    for s, t in keys:
        sel = [r for r in rows if r["sigma"] == s and r["tau"] == t]
        m = {k: float(np.nanmean([r[k] for r in sel])) for k in METRIC_KEYS}
        print(f"  {s:7.4f} {t:5.2f} {m['iou']:7.3f} {m['containment']:8.3f} "
              f"{m['expansion']:7.2f} {m['extra_area']:8.3f} "
              f"{m['hd_ea_to_van']:8.3f} {m['hd_van_to_ea']:8.3f}")


# -------------------------------- figures ----------------------------------

def full_figure(args, D, s, outdir):
    nr, nc = len(args.taus), len(args.psis)
    fig, axes = plt.subplots(nr, nc, figsize=(CELL * nc, CELL * nr),
                             squeeze=False, sharex=True, sharey=True)
    print(f"sigma={s:.3f}")
    print(f"  {'tau':>5} {'IoU':>6} {'contain':>8} {'expans':>7} {'HD EA>V':>8}"
          f"   (mean over psi columns)")
    s_used, rows = None, []
    for r, tau in enumerate(args.taus):
        trows = []
        for c, psi in enumerate(args.psis):
            Ve, Vv, tau_e, psi_e, s_used = panel(args, axes[r, c], D, tau, psi, s, 1.8)
            m = vs_vanilla_metrics(Ve <= 0.0, vanilla_mask_on_ea_grid(D, Vv),
                                   D.dx, D.dy)
            m.update(sigma=float(s_used), tau=float(tau_e), psi=float(psi_e))
            rows.append(m)
            trows.append(m)
            axes[r, c].set_title(rf"$\tau$ = {tau_e:.2f}, $\psi$ = {psi_e:.2f}"
                                 f"\nIoU = {m['iou']:.2f}", fontsize=9)
            if c == 0:
                axes[r, c].set_ylabel("y")
            if r == nr - 1:
                axes[r, c].set_xlabel("x")
        avg = {k: float(np.nanmean([t[k] for t in trows])) for k in METRIC_KEYS}
        print(f"  {tau:5.2f} {avg['iou']:6.3f} {avg['containment']:8.3f} "
              f"{avg['expansion']:7.2f} {avg['hd_ea_to_van']:8.3f}")
    fig.legend(handles=legend_handles(), loc="upper center", ncol=2,
               frameon=False, fontsize=10)
    fig.suptitle(rf"Estimation-aware (4D) vs vanilla Air3D BRT, "
                 rf"$\sigma$ = {s_used:.3f}", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(outdir, f"fig_overlay_vanilla_s{s_used:.3f}.png")
    fig.savefig(out, dpi=args.dpi)
    print("saved", out)
    if args.show:
        plt.show()
    plt.close(fig)
    return rows


def paper_figure(args, D, outdir):
    pairs = [(s, p) for s in args.paper_sigmas for p in args.paper_psis]
    n = len(pairs)
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10}):
        fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2), squeeze=False,
                                 sharex=True, sharey=True)
        tau_used = None
        for c, (s, psi) in enumerate(pairs):
            ax = axes[0, c]
            _, _, tau_used, psi_e, s_e = panel(args, ax, D, args.paper_tau, psi, s, 1.5)
            ax.set_title(rf"$\sigma$ = {s_e:.3f}, $\psi$ = {psi_e:.2f}")
            ax.set_xlabel("$x$")
            if c == 0:
                ax.set_ylabel("$y$")
        fig.legend(handles=legend_handles(), loc="lower center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        base = os.path.join(outdir, f"paper_overlay_vanilla_tau_{tau_used:.2f}")
        fig.savefig(base + ".png", dpi=max(args.dpi, 300))
        fig.savefig(base + ".pdf")
        print("saved", base + ".png /.pdf")
        if args.show:
            plt.show()
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ea-prefix", default=PREFIX)
    ap.add_argument("--vanilla-prefix", default=VAN_PREFIX)
    ap.add_argument("--outdir", default=OUT_DIR,
                    help="base output dir; smoothed plots go to <outdir>_smooth")
    ap.add_argument("--T", type=float, default=T)
    ap.add_argument("--xy-max", type=float, default=XY_MAX,
                    help="used only if the vanilla coords file is missing")
    ap.add_argument("--xy-lim", type=float, default=XY_LIM,
                    help="crop the EA grid to |x|,|y| <= this (0 = no crop)")
    ap.add_argument("--sigmas", type=float, nargs="+", default=S_LIST)
    ap.add_argument("--taus", type=float, nargs="+", default=T_LIST)
    ap.add_argument("--psis", type=float, nargs="+", default=THETA_LIST)
    ap.add_argument("--paper-sigmas", type=float, nargs="+", default=[0.001, 0.05])
    ap.add_argument("--paper-psis", type=float, nargs="+",
                    default=[np.pi, np.pi / 2])
    ap.add_argument("--paper-tau", type=float, default=1.0)
    ap.add_argument("--no-paper", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="draw the raw jagged EA grid set (default: smoothed)")
    ap.add_argument("--smooth-factor", type=int, default=4,
                    help="cubic upsampling factor for the smoothed EA contours")
    ap.add_argument("--vanilla-sign", type=float, default=1.0,
                    help="-1 if the vanilla solution has the opposite sign "
                         "convention (inside BRT = V >= 0)")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    args.smooth_grid = not args.raw

    if not args.show:
        matplotlib.use("Agg")

    D = Data(args)
    outdir = args.outdir + ("_smooth" if args.smooth_grid else "")
    os.makedirs(outdir, exist_ok=True)
    print(f"EA grid: {D.V_ea.shape} | vanilla grid: {D.V_van.shape}")
    print("EA rendering:",
          f"smooth (cubic x{args.smooth_factor})" if args.smooth_grid else "raw jagged")
    print(f"Saving figures to {outdir}")

    if D.s_e.min() > min(args.sigmas) + 1e-9 or D.s_e.max() < max(args.sigmas) - 1e-9:
        print(f"WARNING: requested sigmas exceed the 4D grid s range "
              f"[{D.s_e.min():.4f}, {D.s_e.max():.4f}]; they snap to the edge nodes.")

    rows = []
    for s in args.sigmas:
        rows += full_figure(args, D, s, outdir)
    save_csv(rows, os.path.join(outdir, "metrics_vs_vanilla.csv"))
    summarize(rows)
    if not args.no_paper:
        paper_figure(args, D, outdir)


if __name__ == "__main__":
    main()