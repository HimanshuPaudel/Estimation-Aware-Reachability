"""Grid-format figures for the reduced 4D EstimationAwareAir3D solution (scripts/grid/air3d_ea_4d_grid.py).

Loads <prefix>_values_all.npy  (T+1, x, y, psi, s)
      <prefix>_grid_coords.npy (x, y, psi, s)
      <prefix>_taus.npy        (optional)

Produces one figure per sigma value s (s = Sigma_xx = Sigma_yy):
  fig_grid_s<val>.png   rows = time t, cols = theta (psi),
                        red = inside BRT (V <= 0), blue = outside, 0-1 colorbar per panel
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- settings (edit here) ----
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
PREFIX = os.path.join(REPO_ROOT, "data", "air3d_ea4d")     # scripts/grid/air3d_ea_4d_grid.py
OUT_DIR = os.path.join(REPO_ROOT, "figures", "air3d_grid_4d")
N_S = 4   # number of sigma slices, picked evenly from the grid's own s range
T_LIST = [0.0, 0.5, 1.0]
THETA_LIST = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
DPI = 100


def nearest(vec, val):
    return int(np.argmin(np.abs(np.asarray(vec) - val)))


def wrap_nearest_psi(psi_vec, psi_val):
    d = np.abs(((np.asarray(psi_vec) - psi_val) + np.pi) % (2 * np.pi) - np.pi)
    return int(np.argmin(d))


def fig_grid(values, cv, taus, t_list, theta_list, s_val, out, dpi):
    """One figure for one sigma (s) value. Rows = t, cols = psi.
    Red = inside BRT (V <= 0), blue = outside, 0-1 colorbar on every panel."""
    X, Y = np.meshgrid(cv[0], cv[1], indexing="ij")
    js = nearest(cv[3], s_val)
    nr, nc = len(t_list), len(theta_list)

    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 25.0 / 3.0 * nr), squeeze=False)
    for i, t in enumerate(t_list):
        it = nearest(taus, t)
        for j, th in enumerate(theta_list):
            ip = wrap_nearest_psi(cv[2], th)
            ax = axes[i, j]
            inside = (values[it][:, :, ip, js] <= 0).astype(float)
            im = ax.pcolormesh(X, Y, inside, cmap="bwr", vmin=0.0, vmax=1.0, shading="auto")
            ax.set_aspect("equal")
            ax.set_title(rf"t = {taus[it]:.2f}, $\psi$ = {th:.2f}")
            fig.colorbar(im, ax=ax)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main():
    values = np.load(f"{PREFIX}_values_all.npy")
    cv = [np.array(c, dtype=float) for c in np.load(f"{PREFIX}_grid_coords.npy",
                                                    allow_pickle=True)]
    tau_path = f"{PREFIX}_taus.npy"
    taus = np.load(tau_path) if os.path.exists(tau_path) else np.arange(values.shape[0], dtype=float)
    assert values.shape[0] == len(taus), "taus length must match number of snapshots"

    # quick check that the file being plotted is the new run
    print(f"s range in data: [{cv[3][0]:.4f}, {cv[3][-1]:.4f}], taus: {taus}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(OUT_DIR, "fig_grid_s*.png")):
        os.remove(f)  # remove stale figures from earlier runs

    s_list = np.linspace(cv[3][0], cv[3][-1], N_S)
    for sv in s_list:
        js = nearest(cv[3], sv)
        fig_grid(values, cv, taus, T_LIST, THETA_LIST, sv,
                 os.path.join(OUT_DIR, f"fig_grid_s{cv[3][js]:.3f}.png"), DPI)

    print(f"Saved {len(s_list)} figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()