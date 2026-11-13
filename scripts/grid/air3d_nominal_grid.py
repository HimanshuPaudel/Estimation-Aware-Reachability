"""Nominal (perfect-state-knowledge) Air3D avoid BRT with hj_reachability.

State (x, y, psi) relative to the evader; evader turn rate uE maximises V, pursuer turn
rate uP minimises it.  Target l = ||p|| - Rc.  Run in the hj_reachability environment.

Two grids are used in the paper:
  * safety-filter experiments (scripts/experiments/air3d_safety_filter.py), the default:
        python scripts/grid/air3d_nominal_grid.py
    -> data/air3d_nominal_xy3_{values_all,grid_coords}.npy   ([-3, 3]^2, 241 x 241 x 120)
  * nominal vs estimation-aware overlay (scripts/plotting/air3d_grid_vs_nominal.py):
        python scripts/grid/air3d_nominal_grid.py --xy-max 1 --shape 201 201 200 \\
            --out-prefix data/air3d_nominal_xy1
    -> data/air3d_nominal_xy1_{values_all,grid_coords}.npy   ([-1, 1]^2, 201 x 201 x 200)

Saved values have shape (3, nx, ny, npsi) at tau = 0, T/2, T.
"""
import argparse
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
import time
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


class Air3D(hj.dynamics.Dynamics):
    """Relative pursuit-evasion in the evader's frame.
    state = (x, y, psi); control = uE (evader turn rate, maximizes);
    disturbance = uP (pursuer turn rate, minimizes).
    """

    def __init__(self,
                 vP=0.75,
                 vE=0.75,
                 uP_max=3.0,
                 uE_max=4.0,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.vP, self.vE = vP, vE
        self.uP_max, self.uE_max = uP_max, uE_max

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-uE_max]), jnp.array([uE_max]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array([-uP_max]), jnp.array([uP_max]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def __call__(self, state, control, disturbance, time):
        x, y, psi = state
        uE, = control
        uP, = disturbance
        return jnp.array([
            self.vP * jnp.cos(psi) - self.vE + uE * y,
            self.vP * jnp.sin(psi) - uE * x,
            uP - uE,
        ])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        x, y, psi = state
        l_x, l_y, l_psi = grad_value

        c_uE = l_x * y - l_y * x - l_psi   # coefficient of uE in p . f
        c_uP = l_psi                       # coefficient of uP in p . f

        uE_star = jnp.sign(c_uE) * self.uE_max     # maximize
        uP_star = -jnp.sign(c_uP) * self.uP_max    # minimize
        return jnp.array([uE_star]), jnp.array([uP_star])

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        del value, grad_value_box
        x, y, psi = state
        return jnp.array([
            jnp.abs(self.vP * jnp.cos(psi) - self.vE) + self.uE_max * jnp.abs(y),
            jnp.abs(self.vP * jnp.sin(psi)) + self.uE_max * jnp.abs(x),
            self.uP_max + self.uE_max,
        ])


def build_grid(xy_max=1.0, shape=(201, 201, 200)):
    domain = hj.sets.Box(
        lo=jnp.array([-xy_max, -xy_max, -jnp.pi]),
        hi=jnp.array([ xy_max,  xy_max,  jnp.pi]))
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain, shape, periodic_dims=(2,))


def target_function(grid, Rc):
    x = grid.states[..., 0]
    y = grid.states[..., 1]
    return jnp.sqrt(x**2 + y**2) - Rc


def main():
    ap = argparse.ArgumentParser(description="Nominal Air3D BRT (hj_reachability)")
    ap.add_argument("--xy-max", type=float, default=3.0)
    ap.add_argument("--shape", type=int, nargs=3, default=[241, 241, 120])
    ap.add_argument("--T", type=float, default=1.0, help="horizon (positive)")
    ap.add_argument("--Rc", type=float, default=0.25)
    ap.add_argument("--out-prefix", default=os.path.join(REPO_ROOT, "data", "air3d_nominal_xy3"))
    args = ap.parse_args()
    start_time = time.perf_counter()

    Rc = args.Rc
    grid = build_grid(args.xy_max, tuple(args.shape))
    dynamics = Air3D()
    values0 = target_function(grid, Rc)

    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high",
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
    )

    # solver time runs 0 -> -T; tau = -t is 0, T/2, T
    times = [0.0, -0.5 * args.T, -args.T]
    print(f"Grid shape {grid.shape}, {np.prod(grid.shape)} points")

    all_values = [np.array(values0)]
    v, t = values0, times[0]
    for t_next in times[1:]:
        v = hj.step(solver_settings, dynamics, grid, t, v, t_next, progress_bar=True)
        v.block_until_ready()
        all_values.append(np.array(v))
        t = t_next

    final = all_values[-1]
    brt = final <= 0
    edge = np.zeros_like(brt)
    edge[0, :, :] = edge[-1, :, :] = True
    edge[:, 0, :] = edge[:, -1, :] = True
    print("BRT touches x/y boundary:", bool((brt & edge).any()))

    # max extent of the BRT in x and y
    xs, ys = grid.coordinate_vectors[0], grid.coordinate_vectors[1]
    print("max |x| in BRT:", float(np.abs(xs)[brt.any(axis=(1, 2))].max()))
    print("max |y| in BRT:", float(np.abs(ys)[brt.any(axis=(0, 2))].max()))
    print(f"Fraction of grid in BRT (unsafe) at t={times[-1]}: {float((final <= 0).mean()):.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    np.save(f"{args.out_prefix}_values_all.npy", np.stack(all_values))  # (3, x, y, psi)
    np.save(f"{args.out_prefix}_grid_coords.npy",
            np.array([np.array(c) for c in grid.coordinate_vectors], dtype=object),
            allow_pickle=True)

    print(f"Total execution time: {time.perf_counter() - start_time:.2f} seconds")
    return grid, all_values


if __name__ == "__main__":
    main()