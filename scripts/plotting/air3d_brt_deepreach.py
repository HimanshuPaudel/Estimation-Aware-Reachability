"""Plot the DeepReach estimation-aware Air3D BRT in the binary red/blue style.

Rows    : t = 0.00, 0.50, 1.00
Columns : psi = -pi, -pi/2, 0, pi/2, pi
Panel   : indicator of {V <= 0} (1 = inside BRT = red, 0 = outside = blue),
          bwr colormap with a 0..1 colorbar, same layout as the reference figure.
Slice   : isotropic covariance Sxx = Syy = sigma, Sxy = --sxy.
One figure per sigma (default 0.001, 0.019, 0.033, 0.05).

Additionally, one paper-ready comparison figure shows the final-time BRT
(t = T) at psi = pi for every sigma: one filled panel per sigma plus an
overlay of all the V = 0 boundaries (no title/caption, saved as PNG + PDF).

Parameters (kappa = 3, R = Q = 0.05, sigma in [1e-3, 0.05], xy_max = 3) must match the
training config of the checkpoint: xy_max also sets the network's input/output
normalization, and the plotted extent must stay within the trained domain.

Usage (DeepReach / torch environment):
  python scripts/plotting/air3d_brt_deepreach.py
  python scripts/plotting/air3d_brt_deepreach.py --ckpt path/to/model.pth --dyn-kwargs '{"kappa": 2.0, "xy_max": 2.5}'
  python scripts/plotting/air3d_brt_deepreach.py --cmp-time 1.0 --cmp-psi 3.14159
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "src"))
from ea_reach.paths import add_deepreach_to_path, checkpoint, repo_path, FIGURES   # noqa: E402

DEEPREACH_ROOT = add_deepreach_to_path()

import torch
from dynamics import dynamics as dyn_mod
from utils import modules

# ---------------- settings ----------------
CKPT = checkpoint("air3d_ea_6d_k3")     # kappa = 3, trained on |x|, |y| <= 3
DYNAMICS = "EstimationAwareAir3D"      # class name in deepreach/dynamics/dynamics.py
T = 1.0
TIMES = [0.0, 0.5, 1.0]
PSIS = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
SIGMAS = [0.001, 0.019, 0.033, 0.05]
SXY = 0.0
RES = 301
XY_MAX = 2.5          # plot extent; must be <= the trained xy_max
OUTDIR = os.path.join(FIGURES, "air3d_brt_deepreach")
DPI = 100
CMP_TIME = T          # "final" BRT = full horizon
CMP_PSI = np.pi
CMP_DPI = 300         # resolution of the paper comparison figure
# ------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# must match the training config
DYN_KWARGS = dict(
    vP=0.75, vE=0.75, uP_max=3.0, uE_max=4.0, kappa=3.0,
    Rx=0.05, Ry=0.05, Qx=0.05, Qy=0.05, Rc=0.25, sigma_eps=1e-4,
    xy_max=3.0, sigma_lo=1e-3, sigma_hi=0.05, sigma_xy_max=0.025,
    angle_alpha_factor=1.2,
)

# paper style for the comparison figure
PAPER_RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,     # embed TrueType fonts (editable, journal-friendly)
    "ps.fonttype": 42,
}


class ValueModel:
    def __init__(self, ckpt, dynamics, dyn_kwargs,
                 net="sine", net_mode="mlp", num_nl=512, num_hl=3):
        self.dyn = getattr(dyn_mod, dynamics)(**dyn_kwargs)
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

    def value(self, t, Z, batch=200000):
        """Z: (N, 6) = [x, y, psi, Sxx, Syy, Sxy]."""
        Z = np.clip(np.asarray(Z, dtype=float), self.rng[:, 0], self.rng[:, 1])
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
                outs.append(V.cpu().numpy())
        return np.concatenate(outs)


def value_slice(model, X, Y, t, psi, sigma, sxy):
    """V on the (x, y) grid for a given t, psi and isotropic sigma slice."""
    Z = np.column_stack([X.ravel(), Y.ravel(),
                         np.full(X.size, psi),
                         np.full(X.size, sigma),
                         np.full(X.size, sigma),
                         np.full(X.size, sxy)])
    return model.value(t, Z).reshape(X.shape)


def fill_binary(ax, X, Y, V, cmap):
    # contourf interpolates the V = 0 crossing between nodes, so the
    # boundary is smooth (a binary imshow shows jagged pixel edges)
    lo, hi = min(V.min(), -1.0), max(V.max(), 1.0)
    ax.contourf(X, Y, V, levels=[lo, 0.0, hi],
                colors=[cmap(1.0), cmap(0.0)])   # red in, blue out


def plot_comparison(model, X, Y, args, cmap):
    """Final-time BRT at a fixed psi, compared across sigma values.

    Paper-ready: no suptitle/caption, tight bounding box so nothing
    (axis labels, ticks) is cropped, saved as PNG and vector PDF.
    """
    t, psi = args.cmp_time, args.cmp_psi
    sigmas = args.sigmas
    ns = len(sigmas)
    xy = args.xy_max
    ticks = np.linspace(-xy, xy, 5)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            1, ns + 1, figsize=(2.9 * (ns + 1), 3.2), squeeze=False,
            sharex=True, sharey=True, layout="constrained")
        axes = axes[0]
        colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, ns))
        handles = []

        for i, sv in enumerate(sigmas):
            V = value_slice(model, X, Y, t, psi, sv, args.sxy)

            # individual filled panel
            ax = axes[i]
            fill_binary(ax, X, Y, V, cmap)
            ax.set_title(rf"$\sigma = {sv:g}$")

            # overlay panel: V = 0 boundary only
            axes[-1].contour(X, Y, V, levels=[0.0], colors=[colors[i]],
                             linewidths=1.8)
            handles.append(matplotlib.lines.Line2D(
                [0], [0], color=colors[i], lw=1.8, label=rf"$\sigma = {sv:g}$"))

        axes[-1].set_title("Overlay of $V = 0$")
        axes[-1].grid(alpha=0.3, lw=0.5)
        axes[-1].legend(handles=handles, loc="upper right", frameon=True,
                        framealpha=0.9, borderpad=0.4, handlelength=1.6)

        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xlim(-xy, xy)
            ax.set_ylim(-xy, xy)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xlabel("$x$")
            ax.tick_params(direction="out", length=3)
        axes[0].set_ylabel("$y$")

        base = os.path.join(
            args.outdir, f"brt_deepreach_compare_t_{t:.2f}_psi_{psi:.2f}")
        for ext in ("png", "pdf"):
            out = f"{base}.{ext}"
            # bbox_inches="tight" keeps the full figure (labels, ticks) uncropped
            fig.savefig(out, dpi=args.cmp_dpi, bbox_inches="tight",
                        pad_inches=0.05, facecolor="white")
            print("saved", out)
        if args.show:
            plt.show()
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--dynamics", default=DYNAMICS)
    ap.add_argument("--dyn-kwargs", default=None,
                    help="JSON dict overriding/replacing the default dynamics kwargs")
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--times", type=float, nargs="+", default=TIMES)
    ap.add_argument("--sxy", type=float, default=SXY)
    ap.add_argument("--res", type=int, default=RES)
    ap.add_argument("--xy-max", type=float, default=XY_MAX,
                    help="plot extent; keep <= the trained xy_max")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--cmp-time", type=float, default=CMP_TIME,
                    help="time for the sigma-comparison figure (default: T)")
    ap.add_argument("--cmp-psi", type=float, default=CMP_PSI,
                    help="heading for the sigma-comparison figure (default: pi)")
    ap.add_argument("--cmp-dpi", type=int, default=CMP_DPI,
                    help="dpi of the comparison figure PNG (PDF is vector)")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the sigma-comparison figure")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    dyn_kwargs = dict(DYN_KWARGS)
    if args.dyn_kwargs:
        dyn_kwargs.update(json.loads(args.dyn_kwargs))
    if args.xy_max > dyn_kwargs["xy_max"] + 1e-9:
        print(f"WARNING: plot extent {args.xy_max} exceeds the trained "
              f"xy_max {dyn_kwargs['xy_max']}; inputs are clipped there.")
    model = ValueModel(args.ckpt, args.dynamics, dyn_kwargs)
    print(f"{args.dynamics}: {args.ckpt} (device={DEVICE})")

    xs = np.linspace(-args.xy_max, args.xy_max, args.res)
    ys = np.linspace(-args.xy_max, args.xy_max, args.res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    os.makedirs(args.outdir, exist_ok=True)

    cmap = plt.get_cmap("bwr")
    nr, nc = len(args.times), len(PSIS)
    for sv in args.sigmas:
        fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.4 * nr),
                                 squeeze=False, sharex=True, sharey=True)
        for r, t in enumerate(args.times):
            for c, psi in enumerate(PSIS):
                V = value_slice(model, X, Y, t, psi, sv, args.sxy)
                ax = axes[r, c]
                fill_binary(ax, X, Y, V, cmap)
                ax.set_title(rf"t = {t:.2f}, $\psi$ = {psi:.2f}", fontsize=10)
                ax.set_aspect("equal")
                if c == 0:
                    ax.set_ylabel("y")
                if r == nr - 1:
                    ax.set_xlabel("x")

        fig.suptitle(rf"BRT ({args.dynamics}), $\sigma$ = {sv:g} "
                     rf"(Sxx = Syy = {sv:g}, Sxy = {args.sxy:g})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 0.93, 0.96))
        cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(0, 1), cmap=cmap), cax=cax)

        out = os.path.join(args.outdir, f"brt_deepreach_sigma_{sv:.3f}.png")
        fig.savefig(out, dpi=args.dpi)
        print("saved", out)
        if args.show:
            plt.show()
        plt.close(fig)

    if not args.no_compare:
        plot_comparison(model, X, Y, args, cmap)


if __name__ == "__main__":
    main()