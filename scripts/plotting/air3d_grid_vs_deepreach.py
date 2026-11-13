"""Overlay the reduced 4D grid-based (HJ PDE) estimation-aware Air3D BRT on the
DeepReach estimation-aware Air3D BRT (air3d_ea_6d_k3), and quantify the
agreement with BRT-shape metrics.

Rows    : saved times tau = 0.00, 0.50, 1.00 (time-to-go)
Columns : psi = -pi, -pi/2, 0, pi/2, pi
Panel   : 4D grid HJ BRT (red fill, solid outline) overlaid with the
          DeepReach model BRT (blue fill, dashed outline); {V <= 0}.
Slice   : isotropic covariance Sxx = Syy = sigma (snapped to the nearest 4D
          grid node), Sxy = --sxy (0).

Metrics (4D grid HJ = reference set A, DeepReach = predicted set B, both
evaluated on the native 4D (x, y) grid at each (sigma, tau, psi)):
  IoU          |A & B| / |A | B|
  Dice         2|A & B| / (|A| + |B|)
  miss_rate    |A & ~B| / |A|     (HJ-unsafe cells DeepReach calls safe; the
                                   dangerous error)
  spurious     |~A & B| / |B|     (DeepReach-unsafe cells HJ calls safe)
  mismatch     fraction of all cells whose BRT membership differs
  area_ratio   |B| / |A|          (>1: DeepReach over-approximates)
  hausdorff    symmetric Hausdorff distance between the sets (x/y units)
  mean|dV|, max|dV|  value-function error on the native grid
The 4D grid is cropped to |x|, |y| <= DYN_KWARGS["xy_max"] (the DeepReach
training domain) before comparing.
DeepReach-paper metrics (volume_metrics.csv): value MSE and BRT volume error
(% of domain volume false-safe / false-unsafe) on the full (x, y, psi, s) grid.
Per-(sigma, tau, psi) rows go to a CSV; a (sigma, tau) summary is printed and
plotted (metrics_vs_sigma.png).

Paper figure (paper_overlay_4d_vs_deepreach_psi_pi_tau_<tau>.{png,pdf}):
  Single row, one panel per sigma in --sigmas, fixed psi / tau. --no-paper off.

Grid HJ rendering:
  default        SMOOTH contours (cubic upsampling of the grid values, linear
                 interpolation in psi); output goes to figures/<run>_smooth/.
  --raw          RAW jagged set (cells + stair-step outline, nearest periodic
                 psi node, no interpolation); output goes to figures/<run>/.
  Metrics are always computed on the native grid values, never on the
  upsampled drawing.

Inputs:
  data/air3d_ea4d_values_all.npy    shape (nt, nx, ny, npsi, ns)   (scripts/grid/air3d_ea_4d_grid.py)
  data/air3d_ea4d_grid_coords.npy   object array [x, y, psi, s]
  data/air3d_ea4d_taus.npy          (optional) else linspace(0, T, nt)
  checkpoints/air3d_ea_6d_k3/training/checkpoints/model_final.pth

Output (figures/<run>_smooth/, or figures/<run>/ with --raw):
  overlay_4d_vs_deepreach_sigma_<sigma>.png
  paper_overlay_4d_vs_deepreach_psi_pi_tau_<tau>.{png,pdf}
  metrics.csv, metrics_vs_sigma.png, volume_metrics.csv

Usage (DeepReach / torch environment):
  python scripts/plotting/air3d_grid_vs_deepreach.py
  python scripts/plotting/air3d_grid_vs_deepreach.py --sigmas 0.001 0.017 0.034 0.05 --show
  python scripts/plotting/air3d_grid_vs_deepreach.py --raw
  python scripts/plotting/air3d_grid_vs_deepreach.py --no-paper --no-overlay   # metrics only
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt, zoom

# deepreach repo (so `from dynamics import ...` and `from utils import ...` work)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "src"))
from ea_reach.paths import add_deepreach_to_path, checkpoint, repo_path, FIGURES   # noqa: E402

DEEPREACH_ROOT = add_deepreach_to_path()

import torch
from dynamics import dynamics as dyn_mod
from utils import modules

# ---------------- settings ----------------
PREFIX = repo_path("data", "air3d_ea4d")
CKPT = checkpoint("air3d_ea_6d_k3")     # kappa = 3, trained on |x|, |y| <= 3
T = 1.0            # horizon; saved grid values live at tau = 0, T/2, T
SIGMAS = [0.001, 0.019, 0.033, 0.05]   # span the 4D grid's s range [1e-3, 0.05]
SXY = 0.0
NN_RES = 101       # resolution of the (finer) mesh the DeepReach model is queried on
DPI = 100
CELL = 3.2         # panel size in inches
OUTDIR = os.path.join(FIGURES, "air3d_grid_vs_deepreach")
SHOW = False
# ------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_COLOR = "tab:red"      # 4D grid HJ solution
NN_COLOR = "tab:blue"       # DeepReach model

# dynamics / network parameters: MUST match the DeepReach training config
# (xy_max also sets the network's input/output normalization, so it must equal
# the training value, 3.0) and the 4D solver (R = Q = 0.05, s in [1e-3, 0.05]).
DYN_KWARGS = dict(
    vP=0.75, vE=0.75, uP_max=3.0, uE_max=4.0, kappa=3.0,
    Rx=0.05, Ry=0.05, Qx=0.05, Qy=0.05, Rc=0.25, sigma_eps=1e-4,
    xy_max=3.0, sigma_lo=1e-3, sigma_hi=0.05, sigma_xy_max=0.025,
    angle_alpha_factor=1.2,
)

METRIC_KEYS = ["iou", "dice", "miss_rate", "spurious", "mismatch",
               "area_ratio", "hausdorff", "mean_dV", "max_dV"]


def nearest(grid, v):
    return int(np.argmin(np.abs(np.asarray(grid) - v)))


def nearest_periodic(grid, v):
    """Index of the grid angle closest to v, with wrap-around at 2*pi."""
    d = np.abs((np.asarray(grid) - v + np.pi) % (2 * np.pi) - np.pi)
    return int(np.argmin(d))


def ckpt_run_dir(ckpt):
    """Run-folder name taken from the checkpoint path.

    checkpoints/<run>/training/checkpoints/x.pth                  -> <run>  (fallback)
    DeepReach_Training_Completed/<run>/training/checkpoints/x.pth -> <run>
    .../runs/<run>/training/checkpoints/x.pth                     -> <run>
    """
    parts = os.path.normpath(os.path.abspath(ckpt)).split(os.sep)
    for marker in ("DeepReach_Training_Completed", "runs"):
        if marker in parts:
            i = len(parts) - 1 - parts[::-1].index(marker)   # last occurrence
            if i + 1 < len(parts) - 1:
                return parts[i + 1]
    return parts[-4] if len(parts) >= 4 else parts[-2]


class DeepReachValueModel:
    """The estimation-aware Air3D value network from the deepreach repo."""

    def __init__(self, ckpt, dyn_kwargs=DYN_KWARGS,
                 net="sine", net_mode="mlp", num_nl=512, num_hl=3):
        self.dyn = dyn_mod.EstimationAwareAir3D(**dyn_kwargs)
        self.model = modules.SingleBVPNet(
            in_features=self.dyn.input_dim, out_features=1,
            type=net, mode=net_mode,
            hidden_features=num_nl, num_hidden_layers=num_hl,
            periodic_transform_fn=self.dyn.periodic_transform_fn,
        ).to(DEVICE)
        obj = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        if isinstance(obj, dict):
            obj = obj.get("model", obj.get("model_state_dict", obj))
        self.model.load_state_dict(obj)
        self.model.eval()
        self.rng = np.array(self.dyn.state_test_range(), dtype=float)

    def value_batch(self, t, Z, batch=200000):
        """V(t, z) for Z: (N, 6) = [x_hat, y_hat, psi, Sxx, Syy, Sxy]."""
        Z = np.array(Z, dtype=float)
        Z = np.clip(Z, self.rng[:, 0], self.rng[:, 1])
        outs = []
        with torch.no_grad():
            for s in range(0, len(Z), batch):
                Zc = torch.tensor(Z[s:s + batch], dtype=torch.float32,
                                  device=DEVICE)
                coord = torch.cat(
                    [torch.full((len(Zc), 1), t, dtype=torch.float32,
                                device=DEVICE), Zc], dim=1)
                out = self.model({"coords": self.dyn.coord_to_input(coord)})
                V = self.dyn.io_to_value(out["model_in"],
                                         out["model_out"].squeeze(-1))
                outs.append(V.detach().cpu().numpy())
        return np.concatenate(outs)


# ---------------------------- slicing / drawing ----------------------------

def psi_slice(V2, psi, p):
    """V2: (x, y, psi). Linear interpolation in periodic psi."""
    two_pi = 2 * np.pi
    dpsi = two_pi / len(psi)                    # periodic grid: endpoint excluded
    u = ((p - psi[0]) % two_pi) / dpsi
    i0 = int(np.floor(u)) % len(psi)
    i1 = (i0 + 1) % len(psi)
    w = u - np.floor(u)
    return (1 - w) * V2[:, :, i0] + w * V2[:, :, i1]


def psi_slice_raw(V2, psi, p):
    """V2: (x, y, psi). Nearest periodic psi node, no interpolation."""
    return V2[:, :, nearest_periodic(psi, p)]


def draw_set(ax, x, y, V, color, alpha_fill, lw, ls="-"):
    """Fill {V <= 0} and draw its boundary (smooth contours). V: (nx, ny)."""
    vmin, vmax = np.nanmin(V), np.nanmax(V)
    if not (vmin <= 0.0 <= vmax):
        if vmax <= 0.0:   # entire slice inside the BRT
            ax.axhspan(y[0], y[-1], color=color, alpha=alpha_fill, lw=0)
        return
    X, Y = np.meshgrid(x, y, indexing="ij")
    ax.contourf(X, Y, V, levels=[-np.inf, 0.0], colors=[color],
                alpha=alpha_fill)
    ax.contour(X, Y, V, levels=[0.0], colors=[color], linewidths=lw,
               linestyles=ls)


def draw_set_raw(ax, x, y, V, color, alpha_fill, lw, ls="-", upsample=8):
    """Raw {V <= 0}: filled grid cells plus a stair-step outline. V: (nx, ny)."""
    mask = np.asarray(V) <= 0.0
    if not mask.any():
        return
    nx, ny = mask.shape
    dx, dy = x[1] - x[0], y[1] - y[0]
    xe = np.concatenate([x - dx / 2, [x[-1] + dx / 2]])   # cell edges
    ye = np.concatenate([y - dy / 2, [y[-1] + dy / 2]])
    Xe, Ye = np.meshgrid(xe, ye, indexing="ij")

    ax.pcolormesh(Xe, Ye, np.ma.masked_where(~mask, np.ones(mask.shape)),
                  cmap=ListedColormap([color]), alpha=alpha_fill,
                  shading="flat", rasterized=True)

    if mask.all():        # no interior boundary to trace
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
    return np.linspace(x[0], x[-1], Vs.shape[0]), np.linspace(y[0], y[-1], Vs.shape[1]), Vs


def draw_grid_set(args, ax, x, y, Vg, lw):
    """Draw the 4D grid HJ set: raw jagged cells, or smoothed contours."""
    if args.smooth_grid:
        xs, ys, Vs = smooth_upsample(x, y, Vg, args.smooth_factor)
        draw_set(ax, xs, ys, Vs, GRID_COLOR, 0.35, lw)
    else:
        draw_set_raw(ax, x, y, Vg, GRID_COLOR, 0.35, lw)


def grid_slice(args, V_all, r, js, psi, p):
    """4D grid slice at time index r, sigma index js, heading p -> (x, y)."""
    V2 = np.asarray(V_all[r][:, :, :, js])                  # (x, y, psi)
    return psi_slice(V2, psi, p) if args.smooth_grid else psi_slice_raw(V2, psi, p)


def nn_slice(model, tau, Xm, Ym, p, s, sxy):
    """DeepReach V on the isotropic slice (Sxx = Syy = s, Sxy = sxy)."""
    Z = np.column_stack([Xm.ravel(), Ym.ravel(),
                         np.full(Xm.size, p),
                         np.full(Xm.size, s),
                         np.full(Xm.size, s),
                         np.full(Xm.size, sxy)])
    return model.value_batch(tau, Z).reshape(Xm.shape)


# -------------------------------- metrics ----------------------------------

def _ratio(num, den, empty=np.nan):
    return float(num) / float(den) if den > 0 else empty


def hausdorff_distance(A, B, dx, dy):
    """Symmetric Hausdorff distance between boolean sets A, B (x/y units).

    0 if both empty, NaN if exactly one is empty.
    """
    if not A.any() and not B.any():
        return 0.0
    if not A.any() or not B.any():
        return np.nan
    dist_to_A = distance_transform_edt(~A, sampling=(dx, dy))
    dist_to_B = distance_transform_edt(~B, sampling=(dx, dy))
    return float(max(dist_to_B[A].max(), dist_to_A[B].max()))


def brt_metrics(Vref, Vpred, dx, dy):
    """Shape/value metrics; Vref = 4D grid HJ (reference), Vpred = DeepReach."""
    A, B = Vref <= 0.0, Vpred <= 0.0
    inter = int((A & B).sum())
    union = int((A | B).sum())
    nA, nB = int(A.sum()), int(B.sum())
    dV = np.abs(Vref - Vpred)
    return dict(
        iou=_ratio(inter, union, empty=1.0),               # both empty -> perfect
        dice=_ratio(2 * inter, nA + nB, empty=1.0),
        miss_rate=_ratio((A & ~B).sum(), nA),
        spurious=_ratio((~A & B).sum(), nB),
        mismatch=float((A != B).mean()),
        area_ratio=_ratio(nB, nA),
        hausdorff=hausdorff_distance(A, B, dx, dy),
        mean_dV=float(dV.mean()),
        max_dV=float(dV.max()),
        area_hj=nA * dx * dy,
        area_nn=nB * dx * dy,
    )


def save_metrics_csv(rows, path):
    fields = ["sigma_req", "sigma_node", "tau", "psi"] + METRIC_KEYS + \
             ["area_hj", "area_nn"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print("saved", path)


def summarize(rows, sigma_nodes, taus):
    """Print mean-over-psi (and worst-case over psi) per (sigma, tau)."""
    print("\nSummary (mean over psi; min IoU / max Hausdorff over psi in brackets)")
    print(f"  {'sigma':>7} {'tau':>5} {'IoU':>7} {'Dice':>7} {'miss':>7} "
          f"{'spur':>7} {'mism':>7} {'areaR':>7} {'HD':>7} {'mean|dV|':>9}   worst")
    summ = {}
    for s in sigma_nodes:
        for tau in taus:
            sel = [r for r in rows
                   if abs(r["sigma_node"] - s) < 1e-12 and abs(r["tau"] - tau) < 1e-12]
            if not sel:
                continue
            m = {k: float(np.nanmean([r[k] for r in sel])) for k in METRIC_KEYS}
            m["iou_min"] = float(np.nanmin([r["iou"] for r in sel]))
            m["hd_max"] = float(np.nanmax([r["hausdorff"] for r in sel]))
            summ[(s, tau)] = m
            print(f"  {s:7.4f} {tau:5.2f} {m['iou']:7.3f} {m['dice']:7.3f} "
                  f"{m['miss_rate']:7.3f} {m['spurious']:7.3f} {m['mismatch']:7.3f} "
                  f"{m['area_ratio']:7.3f} {m['hausdorff']:7.3f} {m['mean_dV']:9.4f}"
                  f"   [{m['iou_min']:.3f} / {m['hd_max']:.3f}]")
    return summ


def metrics_figure(summ, sigma_nodes, taus, outdir, args):
    """IoU, miss rate, Hausdorff, mean|dV| vs sigma, one line per tau."""
    panels = [("iou", "IoU"), ("miss_rate", "Miss rate"),
              ("hausdorff", "Hausdorff dist."), ("mean_dV", r"mean $|\Delta V|$")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.2),
                             squeeze=False)
    for ax, (key, lab) in zip(axes[0], panels):
        for tau in taus:
            ys = [summ[(s, tau)][key] for s in sigma_nodes if (s, tau) in summ]
            xs = [s for s in sigma_nodes if (s, tau) in summ]
            ax.plot(xs, ys, "o-", label=rf"$\tau={tau:.2f}$")
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(outdir, "metrics_vs_sigma.png")
    fig.savefig(out, dpi=max(args.dpi, 200))
    print("saved", out)
    if args.show:
        plt.show()
    plt.close(fig)


def volume_metrics(args, V_all, x, y, psi, s_grid, taus, model, sigma_nodes,
                   outdir, band=0.25):
    """DeepReach-paper metrics on the full (x, y, psi, s) grid (NN queried at the
    grid nodes, Sxx = Syy = s, Sxy = args.sxy): value MSE and BRT volume error
    (% of domain volume marked safe-but-unsafe / unsafe-but-safe)."""
    X, Y, P, S = np.meshgrid(x, y, psi, s_grid, indexing="ij")
    Z = np.column_stack([X.ravel(), Y.ravel(), P.ravel(), S.ravel(), S.ravel(),
                         np.full(X.size, args.sxy)])
    sel = [("all", slice(None))] + [(f"{s:.4f}", nearest(s_grid, s)) for s in sigma_nodes]
    rows = []
    print("\nDeepReach-paper metrics (full x,y,psi grid; volume error in % of domain)")
    print(f"  {'sigma':>7} {'tau':>5} {'MSE':>10} {'MSE_band':>10} {'FalseSafe%':>10} "
          f"{'FalseUnsafe%':>12} {'Total%':>8} {'FS/BRT%':>8}")
    for r, tau in enumerate(taus):
        VH = np.asarray(V_all[r])
        VN = model.value_batch(float(tau), Z).reshape(VH.shape)
        for name, idx in sel:
            h, n = VH[..., idx], VN[..., idx]
            A, B = h <= 0.0, n <= 0.0
            fs, fu = (A & ~B).sum(), (~A & B).sum()
            near = np.abs(h) <= band
            m = dict(sigma=name, tau=float(tau),
                     mse=float(np.mean((h - n) ** 2)),
                     mse_band=float(np.mean((h - n)[near] ** 2)) if near.any() else np.nan,
                     false_safe_pct=100.0 * fs / A.size,
                     false_unsafe_pct=100.0 * fu / A.size,
                     total_pct=100.0 * (fs + fu) / A.size,
                     fs_of_brt_pct=100.0 * fs / A.sum() if A.any() else np.nan)
            rows.append(m)
            print(f"  {name:>7} {tau:5.2f} {m['mse']:10.2e} {m['mse_band']:10.2e} "
                  f"{m['false_safe_pct']:10.3f} {m['false_unsafe_pct']:12.3f} "
                  f"{m['total_pct']:8.3f} {m['fs_of_brt_pct']:8.2f}")
    path = os.path.join(outdir, "volume_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


# ------------------------------ paper figure --------------------------------

def paper_figure(args, V_all, x, y, psi, s_grid, taus, model, outdir,
                 psi_val=np.pi, tau_val=1.0):
    """Paper figure: fixed psi and fixed tau, one panel per sigma (single row)."""
    r = nearest(taus, tau_val)
    tau = taus[r]
    ncol = len(args.sigmas)
    xf = np.linspace(x[0], x[-1], args.nn_res)
    yf = np.linspace(y[0], y[-1], args.nn_res)
    Xf, Yf = np.meshgrid(xf, yf, indexing="ij")

    with plt.rc_context({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10}):
        fig, axes = plt.subplots(1, ncol, figsize=(2.6 * ncol, 3.2),
                                 squeeze=False, sharex=True, sharey=True)
        for c, sv in enumerate(args.sigmas):
            js = nearest(s_grid, sv)
            Vg = grid_slice(args, V_all, r, js, psi, psi_val)
            Vnf = nn_slice(model, tau, Xf, Yf, psi_val, s_grid[js], args.sxy)

            ax = axes[0, c]
            draw_grid_set(args, ax, x, y, Vg, 1.5)
            draw_set(ax, xf, yf, Vnf, NN_COLOR, 0.35, 1.5, ls="--")
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(y[0], y[-1])
            ax.set_aspect("equal")
            ax.set_title(rf"$\sigma = {s_grid[js]:.3f}$")
            ax.set_xlabel("$x$")
            if c == 0:
                ax.set_ylabel("$y$")

        handles = [
            Patch(facecolor=GRID_COLOR, alpha=0.35, edgecolor=GRID_COLOR,
                  label="Grid HJ (4D)"),
            Patch(facecolor=NN_COLOR, alpha=0.35, edgecolor=NN_COLOR,
                  label="DeepReach"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))

        base = os.path.join(outdir,
                            f"paper_overlay_4d_vs_deepreach_psi_pi_tau_{tau:.2f}")
        fig.savefig(base + ".png", dpi=max(args.dpi, 300))
        fig.savefig(base + ".pdf")        # vector, for the paper
        print("saved", base + ".png /.pdf")
        if args.show:
            plt.show()
        plt.close(fig)


# ---------------------------------- main ------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--T", type=float, default=T)
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--sxy", type=float, default=SXY)
    ap.add_argument("--nn-res", type=int, default=NN_RES)
    ap.add_argument("--outdir", default=OUTDIR,
                    help="base output dir; figures go to <outdir>/<run>/")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--raw", action="store_true",
                    help="draw the raw jagged grid set (default: smoothed, "
                         "written to figures/<run>_smooth/)")
    ap.add_argument("--smooth-factor", type=int, default=4,
                    help="cubic upsampling factor for the smoothed grid contours")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip the per-sigma overlay grids (metrics still run)")
    ap.add_argument("--no-paper", action="store_true",
                    help="skip the paper figure (psi, tau fixed; sigma panels)")
    ap.add_argument("--no-metrics-fig", action="store_true",
                    help="skip metrics_vs_sigma.png (CSV and printout remain)")
    ap.add_argument("--paper-psi", type=float, default=np.pi,
                    help="heading for the paper figure (default: pi)")
    ap.add_argument("--paper-tau", type=float, default=1.0,
                    help="time-to-go for the paper figure (default: 1.0)")
    ap.add_argument("--no-volume", action="store_true",
                    help="skip the full-grid MSE / BRT volume-error metrics")
    ap.add_argument("--band", type=float, default=0.25,
                    help="|V_HJ| band around the zero level set for MSE_band")
    ap.add_argument("--dyn-kwargs", default=None,
                    help='JSON overrides of DYN_KWARGS, e.g. \'{"kappa": 3.0, "xy_max": 3.0}\' (must match training)')
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    args.smooth_grid = not args.raw
    if args.dyn_kwargs:
        import json
        DYN_KWARGS.update(json.loads(args.dyn_kwargs))   # in place: also the DeepReachValueModel default
        print("DYN_KWARGS:", DYN_KWARGS)

    if not args.show:
        matplotlib.use("Agg")

    # ---- 4D grid HJ estimation-aware solution ----
    V_all = np.load(f"{args.prefix}_values_all.npy", mmap_mode="r")  # (nt, x, y, psi, s)
    coords = np.load(f"{args.prefix}_grid_coords.npy", allow_pickle=True)
    x, y, psi, s_grid = [np.asarray(a, dtype=float) for a in coords]
    # compare only inside the DeepReach training domain (4D grid may be larger)
    ix = np.where(np.abs(x) <= DYN_KWARGS["xy_max"] + 1e-9)[0]
    iy = np.where(np.abs(y) <= DYN_KWARGS["xy_max"] + 1e-9)[0]
    V_all = V_all[:, ix[0]:ix[-1] + 1, iy[0]:iy[-1] + 1]
    x, y = x[ix], y[iy]
    nt = V_all.shape[0]
    tau_path = f"{args.prefix}_taus.npy"
    taus = np.load(tau_path) if os.path.exists(tau_path) else np.linspace(0.0, args.T, nt)
    assert nt == len(taus), "taus length must match number of snapshots"
    psi_targets = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
    ncol = len(psi_targets)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    if s_grid.min() > min(args.sigmas) + 1e-9 or s_grid.max() < max(args.sigmas) - 1e-9:
        print(f"WARNING: requested sigmas {args.sigmas} exceed the 4D grid s range "
              f"[{s_grid.min():.4f}, {s_grid.max():.4f}]; they snap to the edge nodes.")

    # ---- DeepReach estimation-aware model ----
    model = DeepReachValueModel(args.ckpt)
    print(f"DeepReach model: {args.ckpt}  (device={DEVICE})")
    print("Grid HJ rendering:",
          f"smooth (cubic x{args.smooth_factor})" if args.smooth_grid else "raw jagged")

    # query meshes: fine mesh for smooth NN contours, native grid for metrics
    xf = np.linspace(x[0], x[-1], args.nn_res)
    yf = np.linspace(y[0], y[-1], args.nn_res)
    Xf, Yf = np.meshgrid(xf, yf, indexing="ij")
    Xg, Yg = np.meshgrid(x, y, indexing="ij")

    run = ckpt_run_dir(args.ckpt)
    outdir = os.path.join(args.outdir, run + ("_smooth" if args.smooth_grid else ""))
    os.makedirs(outdir, exist_ok=True)
    print(f"Saving figures to {outdir}")

    rows = []
    sigma_nodes = []
    for sv in args.sigmas:
        js = nearest(s_grid, sv)
        s_node = float(s_grid[js])
        if s_node not in sigma_nodes:
            sigma_nodes.append(s_node)
        print(f"sigma={sv:.3f} -> 4D grid node s={s_node:.4f}, Sxy={args.sxy:.4f}")

        fig = axes = None
        if not args.no_overlay:
            fig, axes = plt.subplots(nt, ncol, figsize=(CELL * ncol, CELL * nt),
                                     squeeze=False, sharex=True, sharey=True)

        print(f"  {'tau':>5} {'IoU':>6} {'Dice':>6} {'miss':>6} {'spur':>6} "
              f"{'areaR':>6} {'HD':>6} {'mean|dV|':>9} {'max|dV|':>8}")
        for r in range(nt):
            tau_rows = []
            for c, pt in enumerate(psi_targets):
                Vg = grid_slice(args, V_all, r, js, psi, pt)
                Vng = nn_slice(model, taus[r], Xg, Yg, pt, s_node, args.sxy)

                m = brt_metrics(Vg, Vng, dx, dy)
                A = Vg <= 0.0
                m["hj_touches_edge"] = bool(A[0, :].any() or A[-1, :].any()
                                            or A[:, 0].any() or A[:, -1].any())
                m.update(sigma_req=sv, sigma_node=s_node, tau=float(taus[r]),
                         psi=float(pt))
                rows.append(m)
                tau_rows.append(m)

                if axes is not None:
                    Vnf = nn_slice(model, taus[r], Xf, Yf, pt, s_node, args.sxy)
                    ax = axes[r, c]
                    draw_grid_set(args, ax, x, y, Vg, 1.8)
                    draw_set(ax, xf, yf, Vnf, NN_COLOR, 0.35, 1.8, ls="--")
                    ax.set_xlim(x[0], x[-1])
                    ax.set_ylim(y[0], y[-1])
                    ax.set_aspect("equal")
                    ax.set_title(f"tau = {taus[r]:.2f}, psi = {pt:.2f}\n"
                                 f"IoU = {m['iou']:.3f}", fontsize=9)
                    if c == 0:
                        ax.set_ylabel("y")
                    if r == nt - 1:
                        ax.set_xlabel("x")

            avg = {k: float(np.nanmean([t[k] for t in tau_rows])) for k in METRIC_KEYS}
            dV_max = max(t["max_dV"] for t in tau_rows)
            if any(t["hj_touches_edge"] for t in tau_rows):
                print(f"  WARNING tau={taus[r]:.2f}: HJ BRT touches the cropped "
                      f"domain edge; metrics there are truncated")
            print(f"  {taus[r]:5.2f} {avg['iou']:6.3f} {avg['dice']:6.3f} "
                  f"{avg['miss_rate']:6.3f} {avg['spurious']:6.3f} "
                  f"{avg['area_ratio']:6.3f} {avg['hausdorff']:6.3f} "
                  f"{avg['mean_dV']:9.4f} {dV_max:8.3f}")

        if fig is not None:
            grid_label = "Grid HJ 4D" if args.smooth_grid else "Grid HJ 4D (raw)"
            handles = [
                Patch(facecolor=GRID_COLOR, alpha=0.35, edgecolor=GRID_COLOR,
                      label=f"{grid_label} (Sxx=Syy={s_node:.3f}, Sxy={args.sxy:.3f})"),
                Patch(facecolor=NN_COLOR, alpha=0.35, edgecolor=NN_COLOR,
                      label="DeepReach"),
            ]
            fig.legend(handles=handles, loc="upper center", ncol=2,
                       frameon=False, fontsize=10)
            fig.suptitle("Estimation-aware Air3D BRT overlay: 4D grid HJ vs DeepReach",
                         y=0.995, fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))

            out = os.path.join(outdir,
                               f"overlay_4d_vs_deepreach_sigma_{sv:.3f}.png")
            fig.savefig(out, dpi=args.dpi)
            print("saved", out)
            if args.show:
                plt.show()
            plt.close(fig)

    # ---- metrics: CSV, summary table, trend figure ----
    save_metrics_csv(rows, os.path.join(outdir, "metrics.csv"))
    summ = summarize(rows, sigma_nodes, [float(t) for t in taus])
    if not args.no_metrics_fig:
        metrics_figure(summ, sigma_nodes, [float(t) for t in taus], outdir, args)

    # ---- DeepReach-paper metrics: full-grid value MSE and BRT volume error ----
    if not args.no_volume:
        volume_metrics(args, V_all, x, y, psi, s_grid, taus, model,
                       sigma_nodes, outdir, band=args.band)

    # ---- paper figure: fixed psi (default pi), columns = sigma values ----
    if not args.no_paper:
        paper_figure(args, V_all, x, y, psi, s_grid, taus, model,
                     outdir, psi_val=args.paper_psi, tau_val=args.paper_tau)


if __name__ == "__main__":
    main()