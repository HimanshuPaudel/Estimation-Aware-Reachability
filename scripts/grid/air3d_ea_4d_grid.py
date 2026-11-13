"""Reduced 4D estimation-aware Air3D BRT with hj_reachability (grid validation, App. A).

On the isotropic covariance slice Sxx = Syy = s, Sxy = 0 the 6D estimation-aware Air3D
problem reduces exactly to 4D (x_hat, y_hat, psi, s).  Run in the hj_reachability env:

    python scripts/grid/air3d_ea_4d_grid.py
    -> data/air3d_ea4d_{values_all,grid_coords,taus}.npy   (121 x 121 x 48 x 15 on [-4, 4]^2)
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import time

import jax.numpy as jnp
import numpy as np
import hj_reachability as hj

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


class EstimationAwareAir3D4D(hj.dynamics.Dynamics):
    """Reduced 4D model: state = (x_hat, y_hat, psi, s)

    s = Sxx = Syy, Sxy = 0. Exact reduction of the 6D model when
    Qx = Qy = Q and Rx = Ry = R (the slice is invariant).

    control     = uE            (evader turn rate, maximizes)
    disturbance = (uP, ex, ey)  (pursuer turn rate, innovation; minimize)

    The innovation adversary is bounded by the innovation covariance
    S = Sigma + R (H = I):  |e_i| <= kappa * sqrt(s + R).
    """

    def __init__(self,
                 vP=0.75,
                 vE=0.75,
                 uP_max=3.0,
                 uE_max=4.0,
                 kappa=3.0,       # |e_i| <= kappa * sqrt(s + R)
                 R=0.05,          # measurement-noise intensity (Rx = Ry)
                 Q=0.05,          # process-noise intensity (Qx = Qy)
                 Rc=0.25,         # capture radius
                 sigma_eps=1e-4,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.vP, self.vE = vP, vE
        self.uP_max, self.uE_max = uP_max, uE_max
        self.kappa = kappa
        self.R, self.Q = R, Q
        self.Rc = Rc
        self.sigma_eps = sigma_eps

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-uE_max]), jnp.array([uE_max]))
        if disturbance_space is None:
            # placeholders; the true state-dependent bound kappa*sqrt(s + R)
            # is applied analytically in _e_max / optimal_control_and_disturbance
            # / partial_max_magnitudes.
            e_placeholder = 1e2
            disturbance_space = hj.sets.Box(
                jnp.array([-uP_max, -e_placeholder, -e_placeholder]),
                jnp.array([uP_max, e_placeholder, e_placeholder]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def _e_max(self, s):
        return self.kappa * jnp.sqrt(jnp.clip(s, 0.0, None) + self.R + self.sigma_eps)

    def __call__(self, state, control, disturbance, time):
        x, y, psi, s = state
        uE, = control
        uP, ex, ey = disturbance
        K = s / self.R  # K11 = K22 = s/R, K12 = K21 = 0 on the slice

        x_dot = self.vP * jnp.cos(psi) - self.vE + uE * y + K * ex
        y_dot = self.vP * jnp.sin(psi) - uE * x + K * ey
        psi_dot = uP - uE
        s_dot = self.Q - s**2 / self.R
        return jnp.array([x_dot, y_dot, psi_dot, s_dot])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        x, y, psi, s = state
        l_x, l_y, l_psi, l_s = grad_value
        K = s / self.R
        em = self._e_max(s)

        c_uE = l_x * y - l_y * x - l_psi
        c_uP = l_psi
        c_ex = l_x * K
        c_ey = l_y * K

        uE_star = jnp.sign(c_uE) * self.uE_max       # maximize
        uP_star = -jnp.sign(c_uP) * self.uP_max      # minimize
        ex_star = -jnp.sign(c_ex) * em               # minimize
        ey_star = -jnp.sign(c_ey) * em               # minimize

        # (control, disturbance) = ([uE], [uP, ex, ey]), matching __call__
        return jnp.array([uE_star]), jnp.array([uP_star, ex_star, ey_star])

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        del value, grad_value_box
        x, y, psi, s = state
        K = jnp.abs(s / self.R)
        em = self._e_max(s)
        return jnp.array([
            jnp.abs(self.vP * jnp.cos(psi) - self.vE) + self.uE_max * jnp.abs(y) + K * em,
            jnp.abs(self.vP * jnp.sin(psi)) + self.uE_max * jnp.abs(x) + K * em,
            self.uP_max + self.uE_max,
            jnp.abs(self.Q - s**2 / self.R),
        ])


def build_grid_4d(xy_max=3.0, s_lo=1e-3, s_hi=0.05, shape=(121, 121, 48, 15)):
    domain = hj.sets.Box(
        lo=jnp.array([-xy_max, -xy_max, -jnp.pi, s_lo]),
        hi=jnp.array([xy_max, xy_max, jnp.pi, s_hi]))
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain, shape, periodic_dims=(2,))


def target_4d(grid, kappa, Rc, sigma_eps=1e-4):
    x = grid.states[..., 0]
    y = grid.states[..., 1]
    s = grid.states[..., 3]
    # matches the 6D boundary_fn on Sxx = Syy = s, Sxy = 0: lam_max = s + eps
    # (target margin is governed by Sigma only; it covers the estimation gap)
    return jnp.sqrt(x**2 + y**2) - kappa * jnp.sqrt(s + sigma_eps) - Rc


def compare_with_6d(final4, cv4, path6_values, path6_coords):
    """Compare the 4D result against a 6D hj_reachability result on the
    Sxx = Syy (diagonal), Sxy = 0 nodes of the 6D grid."""
    from scipy.interpolate import RegularGridInterpolator

    v6 = np.load(path6_values)[-1]  # (x, y, psi, Sxx, Syy, Sxy) at final time
    cv6 = [np.array(c, dtype=float) for c in np.load(path6_coords, allow_pickle=True)]
    k0 = int(np.argmin(np.abs(cv6[5])))
    n_s = min(len(cv6[3]), len(cv6[4]))

    # pad psi periodically so interpolation works across the wrap
    psi = cv4[2]
    period = 2 * np.pi
    psi_pad = np.concatenate([[psi[-1] - period], psi, [psi[0] + period]])
    v4_pad = np.concatenate([final4[:, :, -1:, :], final4, final4[:, :, :1, :]], axis=2)
    interp = RegularGridInterpolator((cv4[0], cv4[1], psi_pad, cv4[3]), v4_pad,
                                     bounds_error=False, fill_value=None)

    X, Y, P = np.meshgrid(cv6[0], cv6[1], cv6[2], indexing="ij")
    print("\n4D vs 6D (diagonal Sxx=Syy, Sxy=0 nodes):")
    for i in range(n_s):
        if abs(cv6[3][i] - cv6[4][i]) > 1e-9:
            continue
        pts = np.stack([X.ravel(), Y.ravel(), P.ravel(),
                        np.full(X.size, cv6[3][i])], axis=-1)
        v_4 = interp(pts).reshape(X.shape)
        v_6 = v6[:, :, :, i, i, k0]
        err = np.abs(v_4 - v_6)
        sign_mismatch = float(((v_4 <= 0) != (v_6 <= 0)).mean())
        print(f"  s={cv6[3][i]:.3f}: max|dV|={err.max():.4f}, mean|dV|={err.mean():.4f}, "
              f"BRT membership mismatch={sign_mismatch:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs=4, default=[121, 121, 48, 15],
                    metavar=("NX", "NY", "NPSI", "NS"))
    ap.add_argument("--xy-max", type=float, default=4.0)
    ap.add_argument("--s-lo", type=float, default=1e-3)
    ap.add_argument("--s-hi", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=1.0, help="horizon (positive)")
    ap.add_argument("--n-snapshots", type=int, default=2,
                    help="number of time steps; snapshots saved at tau = 0, T/n, ..., T")
    ap.add_argument("--accuracy", default="very_high",
                    choices=["low", "medium", "high", "very_high"])
    ap.add_argument("--kappa", type=float, default=3.0)
    ap.add_argument("--Rc", type=float, default=0.25)
    ap.add_argument("--Q", type=float, default=0.05)
    ap.add_argument("--R", type=float, default=0.05)
    ap.add_argument("--out-prefix", default=os.path.join(REPO_ROOT, "data", "air3d_ea4d"))
    ap.add_argument("--compare-6d-values", default=None,
                    help="path to 6D *_values_all.npy for validation")
    ap.add_argument("--compare-6d-coords", default=None,
                    help="path to 6D *_grid_coords.npy for validation")
    args = ap.parse_args()

    start_time = time.perf_counter()

    grid = build_grid_4d(xy_max=args.xy_max, s_lo=args.s_lo, s_hi=args.s_hi,
                         shape=tuple(args.shape))
    dynamics = EstimationAwareAir3D4D(kappa=args.kappa, Rc=args.Rc, Q=args.Q, R=args.R)

    values0 = target_4d(grid, args.kappa, args.Rc, dynamics.sigma_eps)

    solver_settings = hj.SolverSettings.with_accuracy(
        args.accuracy,
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
    )

    # solver time runs 0 -> -T; tau = -t
    taus = np.linspace(0.0, args.T, args.n_snapshots + 1)
    times = [-float(t) for t in taus]
    print(f"Grid shape {grid.shape}, {int(np.prod(grid.shape))} points")
    print(f"s fixed point sqrt(QR) = {np.sqrt(args.Q * args.R):.4f}")
    em_hi = args.kappa * np.sqrt(args.s_hi + args.R + dynamics.sigma_eps)
    print(f"At s_hi: innovation bound = {em_hi:.3f}, "
          f"K*e_max = {(args.s_hi / args.R) * em_hi:.3f} (vs v = {dynamics.vP})")

    all_values = [np.array(values0)]
    v, t = values0, times[0]
    for t_next in times[1:]:
        v = hj.step(solver_settings, dynamics, grid, t, v, t_next, progress_bar=True)
        v.block_until_ready()
        all_values.append(np.array(v))
        t = t_next

    final = all_values[-1]
    if np.isnan(final).any():
        print("WARNING: NaNs in the final value function")
    print(f"Fraction of grid in BRT (unsafe) at tau={taus[-1]}: {float((final <= 0).mean()):.3f}")

    # sanity check at psi ~ 0: BRT radius should be at least Rc + kappa*sqrt(s)
    cv = [np.array(c) for c in grid.coordinate_vectors]
    ip = int(np.argmin(np.abs(cv[2])))
    X, Y = np.meshgrid(cv[0], cv[1], indexing="ij")
    R_xy = np.sqrt(X**2 + Y**2)
    for si in sorted({0, len(cv[3]) // 2, len(cv[3]) - 1}):
        s_val = cv[3][si]
        base = args.Rc + args.kappa * np.sqrt(s_val + dynamics.sigma_eps)
        print(f"  s={s_val:.3f} (initial radius {base:.2f}):")
        for r, tau in enumerate(taus):
            m = all_values[r][:, :, ip, si] <= 0
            print(f"    tau={tau:.2f}: max BRT radius = {R_xy[m].max() if m.any() else 0.0:.2f}")

    brt = final <= 0
    edge = np.zeros(final.shape[:2], dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    print("BRT touches x/y boundary:", bool((brt.any(axis=(2, 3)) & edge).any()))

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    np.save(f"{args.out_prefix}_values_all.npy", np.stack(all_values))  # (T+1, x, y, psi, s)
    np.save(f"{args.out_prefix}_grid_coords.npy", np.array(cv, dtype=object), allow_pickle=True)
    np.save(f"{args.out_prefix}_taus.npy", taus)

    if args.compare_6d_values and args.compare_6d_coords:
        compare_with_6d(final, cv, args.compare_6d_values, args.compare_6d_coords)

    print(f"Total execution time: {time.perf_counter() - start_time:.2f} seconds")
    return grid, all_values


if __name__ == "__main__":
    main()