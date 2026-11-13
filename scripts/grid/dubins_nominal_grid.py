"""
Nominal (perfect-state) Dubins3D avoid BRT with hj_reachability -- the "nominal" and "inflated"
baselines of scripts/experiments/dubins_safety_filter.py.  Run in the hj_reachability env:

    python scripts/grid/dubins_nominal_grid.py

State (x, y, theta), control omega (maximises V, |omega| <= omega_max), no disturbance:
    x' = v cos theta,  y' = v sin theta,  theta' = omega
Target l(x) = ||p|| - Rc (obstacle disc at the origin), BRT for horizon T.

Saves (same format as scripts/grid/air3d_nominal_grid.py):
    data/dubins_nominal_values_all.npy   (nt, nx, ny, ntheta), times linspace(0, T, nt)
    data/dubins_nominal_grid_coords.npy  [x, y, theta] coordinate vectors

Parameters must match EstimationAwareDubins3D (v, omega_max, Rc, domain).
"""
import argparse
import os
import time

import jax.numpy as jnp
import numpy as np
import hj_reachability as hj

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


class Dubins3D(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, v=1.0, omega_max=1.2, control_mode="max", disturbance_mode="min"):
        self.v = v
        super().__init__(control_mode, disturbance_mode,
                         hj.sets.Box(jnp.array([-omega_max]), jnp.array([omega_max])),
                         hj.sets.Box(jnp.zeros(1), jnp.zeros(1)))

    def open_loop_dynamics(self, state, time):
        _, _, th = state
        return jnp.array([self.v * jnp.cos(th), self.v * jnp.sin(th), 0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [1.0]])

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((3, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=1.0)
    ap.add_argument("--omega_max", type=float, default=1.2)
    ap.add_argument("--Rc", type=float, default=0.5)
    ap.add_argument("--xy_max", type=float, default=3.0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--nt", type=int, default=11, help="saved time frames (incl. t=0)")
    ap.add_argument("--shape", type=int, nargs=3, default=[161, 161, 120])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data"))
    a = ap.parse_args()

    t0 = time.perf_counter()
    domain = hj.sets.Box(lo=jnp.array([-a.xy_max, -a.xy_max, -jnp.pi]),
                         hi=jnp.array([a.xy_max, a.xy_max, jnp.pi]))
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(domain, tuple(a.shape), periodic_dims=2)
    dyn = Dubins3D(a.v, a.omega_max)
    l = jnp.linalg.norm(grid.states[..., :2], axis=-1) - a.Rc
    settings = hj.SolverSettings.with_accuracy("very_high",
                                               hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    times = -np.linspace(0.0, a.T, a.nt)          # solver time runs 0 -> -T
    vals = hj.solve(settings, dyn, grid, times, l, progress_bar=True)
    vals = np.asarray(vals)                        # (nt, nx, ny, ntheta), index k <-> tau = k T/(nt-1)

    brt = vals[-1] <= 0
    print(f"grid {grid.shape}; BRT fraction at T={a.T}: {brt.mean():.3f}; "
          f"min V {vals[-1].min():.3f}; time {time.perf_counter() - t0:.1f}s")
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "dubins_nominal_values_all.npy"), vals)
    np.save(os.path.join(a.out, "dubins_nominal_grid_coords.npy"),
            np.array([np.asarray(c) for c in grid.coordinate_vectors], dtype=object), allow_pickle=True)
    print("saved", os.path.join(a.out, "dubins_nominal_values_all.npy"))


if __name__ == "__main__":
    main()
