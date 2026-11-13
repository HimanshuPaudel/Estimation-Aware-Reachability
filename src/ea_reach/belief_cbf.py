#!/usr/bin/env python
"""
Belief Control Barrier Function (BCBF) safety filter -- baseline for the estimation-aware HJ paper.

Reference
---------
M. Vahs, C. Pek, J. Tumova, "Belief Control Barrier Functions for Risk-aware Control",
IEEE RA-L 8(12):8565-8572, 2023, doi:10.1109/LRA.2023.3330662; arXiv:2309.06499.
Equation numbers below refer to the arXiv version.

What is implemented (paper eq. -> function in this file)
---------------------------------------------------------
  (12),(28)  risk-aware barrier on the Gaussian belief b = [mu, vec(Sigma)]
                 h_b(b) = alpha^T (mu - c) - r - erfinv(1 - 2 delta) * sqrt(2 alpha^T Sigma alpha)
             with the linearised obstacle normal alpha = (mu - c) / ||mu - c||   -> BeliefCBF.barrier
             (obstacle centre c = 0 in both of our systems, r = R_c)
  (15)       control-affine belief dynamics  b' = f_b(b) + g_b(b) u            -> Air3DBelief / QuadBelief .F/.G
  (4)        second-order exponential CBF   h'' + zeta^T [h, h'] >= 0           -> BeliefCBF.constraint
  (25)       QP  min ||u - u_ref||^2  s.t. the CBF constraint, u in U          -> solve_qp_1d / solve_qp_2d
  Thm. 2     natural bound on Pr[h_b(b+) < 0] for one measurement update         -> theorem2_bound
  Thm. 3     margin gamma so that Pr[b+ in C_b] >= 1 - eps;  h~ = h_b - gamma   -> theorem3_margin

Adaptations to our setting (state these in the paper)
-----------------------------------------------------
  A1  Continuous-time estimators.  The paper uses a continuous-discrete EKF: between measurements
      Sigma' = A Sigma + Sigma A^T + Q (eq. 14).  Our estimators (CT-EKF, CT H-inf) are continuous-time,
      so f_b uses the estimator's own Riccati equation (it contains the -Sigma C^T R^-1 C Sigma term),
      and mu' uses the zero-innovation drift.  The innovation enters as zero-mean noise and is not seen
      by the CBF -- the same treatment the paper gives to the stochastic measurement jump.
  A2  Thm. 3 margin.  Our simulators inject a measurement every dt with discrete covariance R/dt, so
      theorem3_margin uses R_d = R/dt, H = I.  gamma is a constant: the max over the covariance range.
  A3  Bounded adversarial inputs (pursuer turn rate in Air3D, wind in the quadrotor) are not part of
      the paper.  With robust=True the CBF constraint holds for the worst case input, so the CBF gets
      the same information as the HJ value function.  robust=False ignores them (as in the paper).
  A4  Quadrotor: the CBF acts on the commanded acceleration of a double integrator, as in the paper's
      hardware experiment (eq. 29); the comparison script converts it to thrusts with the same inner
      loop as the nominal controller.
  A5  delta is chosen so that sqrt(2) erfinv(1 - 2 delta) = kappa, i.e. the CBF uses the SAME
      sigma-multiplier as the HJ target.  Note: the CBF uses the directional variance alpha^T Sigma alpha
      (paper's half-space VaR), the HJ target uses lambda_max(Sigma).
  A6  For the H-inf filter the Riccati variable P is used in place of Sigma (same as in the HJ model).
  A7  Dubins (DubinsBelief): the belief is the full 9D EKF state; the barrier uses the position block
      of Sigma.  There is no physical adversary, and (A1) the innovation is unseen by the CBF.  The
      measurement noise r(p) is evaluated at the estimate, as in the filter.

Derivatives
-----------
Lie derivatives are computed with torch autograd (float64, CPU), so the only hand-written pieces you
need to check are: barrier(), the F/G/D matrices of each system, and the two QP solvers.
`python -m ea_reach.belief_cbf` runs a self-test: autograd Lie derivatives vs finite differences, the
relative-degree-2 property, the paper's Thm. 2 numerical example (~0.09), the QP solvers vs brute
force, and a closed-loop invariance check.
"""
import math
from statistics import NormalDist

import numpy as np
import torch

DT = torch.float64
_SQRT2 = math.sqrt(2.0)


def erfinv(x):
    return float(torch.special.erfinv(torch.tensor(x, dtype=DT)))


def erf(x):
    return float(torch.special.erf(torch.tensor(x, dtype=DT)))


# =============================================================================================== #
# Belief dynamics  b' = F(b) + G(b) u + D(b) d     (paper eq. 15, plus adversary d for A3)
#   F: (N, n)   G: (N, n, m)   D: (N, n, k)
# Every system exposes: n, m, k, i_pos (2 indices of mu), i_sig (Sxx, Syy, Sxy indices),
#                       u_lo, u_hi (m,), d_bound (k,)
# =============================================================================================== #
class Air3DBelief:
    """Air3D with CT-EKF on the relative position (matches EstimationAwareAir3D / est_dsdt).

    b = [x, y, psi, Sxx, Syy, Sxy]  -- (x, y): estimated pursuer position in the evader frame.
    u = uE (evader turn rate, controlled);  d = uP (pursuer turn rate, adversary).
        x'   = vP cos psi - vE + uE y
        y'   = vP sin psi      - uE x
        psi' = uP - uE
        Sxx' =  2 uE Sxy + Q - (Sxx^2 + Sxy^2)/R
        Syy' = -2 uE Sxy + Q - (Sxy^2 + Syy^2)/R
        Sxy' =  uE (Syy - Sxx) - Sxy (Sxx + Syy)/R
    """
    n, m, k = 6, 1, 1
    i_pos, i_sig = (0, 1), (3, 4, 5)

    def __init__(self, vP, vE, uE_max, uP_max, Q, R):
        self.vP, self.vE, self.Q, self.R = vP, vE, Q, R
        self.u_lo, self.u_hi = np.array([-uE_max]), np.array([uE_max])
        self.d_bound = np.array([uP_max])

    def F(self, b):
        x, y, psi, Sxx, Syy, Sxy = b.unbind(-1)
        Q, R = self.Q, self.R
        return torch.stack([self.vP * torch.cos(psi) - self.vE,
                            self.vP * torch.sin(psi),
                            torch.zeros_like(x),
                            Q - (Sxx ** 2 + Sxy ** 2) / R,
                            Q - (Sxy ** 2 + Syy ** 2) / R,
                            -Sxy * (Sxx + Syy) / R], -1)

    def G(self, b):
        x, y, psi, Sxx, Syy, Sxy = b.unbind(-1)
        return torch.stack([y, -x, -torch.ones_like(x), 2 * Sxy, -2 * Sxy, Syy - Sxx], -1)[..., None]

    def D(self, b):
        g = torch.zeros(b.shape[:-1] + (6, 1), dtype=b.dtype)
        g[..., 2, 0] = 1.0
        return g


class QuadBelief:
    """Planar quadrotor, H-inf filter on position, CBF on commanded acceleration (adaptation A4).

    b = [x, y, vx, vy, Pxx, Pyy, Pxy];  u = (ax, ay) commanded acceleration;  d = (dx, dy) wind.
        p' = v,   v' = u + d,   P' = q I - (1/r - 1/gamma^2) P^2
    """
    n, m, k = 7, 2, 2
    i_pos, i_sig = (0, 1), (4, 5, 6)

    def __init__(self, q, r, gamma, a_max, d_max):
        self.q, self.c = q, 1.0 / r - 1.0 / gamma ** 2
        self.u_lo, self.u_hi = np.array([-a_max, -a_max]), np.array([a_max, a_max])
        self.d_bound = np.array([d_max, d_max])

    def F(self, b):
        x, y, vx, vy, Pxx, Pyy, Pxy = b.unbind(-1)
        z = torch.zeros_like(x)
        return torch.stack([vx, vy, z, z,
                            self.q - self.c * (Pxx ** 2 + Pxy ** 2),
                            self.q - self.c * (Pxy ** 2 + Pyy ** 2),
                            -self.c * Pxy * (Pxx + Pyy)], -1)

    def G(self, b):
        g = torch.zeros(b.shape[:-1] + (7, 2), dtype=b.dtype)
        g[..., 2, 0] = 1.0
        g[..., 3, 1] = 1.0
        return g

    D = G   # wind enters exactly like the commanded acceleration


class DubinsBelief:
    """Dubins car, CT-EKF over the full state (x, y, theta), beacon noise r(p) = r0 + r1 ||p - b||^2
    (matches EstimationAwareDubins3D with zero innovation, adaptation A1).

    b = [x, y, th, Sxx, Syy, Stt, Sxy, Sxt, Syt];  u = omega (turn rate);  no adversary (k = 0).
        x'   = v cos th,   y' = v sin th,   th' = omega
        Sigma' = A(th) Sigma + Sigma A(th)^T + Q - Sigma C^T C Sigma / r(p),   C = [I_2 0]
    The barrier depends on (x, y, Sxx, Syy, Sxy) only and omega enters only th', so the relative
    degree is 2: omega appears in h'' through d(h')/d th (heading changes p' and Sigma').
    """
    n, m, k = 9, 1, 0
    i_pos, i_sig = (0, 1), (3, 4, 6)

    def __init__(self, v, omega_max, qx, qy, qt, r0, r1, beacon):
        self.v, self.qx, self.qy, self.qt = v, qx, qy, qt
        self.r0, self.r1, self.bx, self.by = r0, r1, float(beacon[0]), float(beacon[1])
        self.u_lo, self.u_hi = np.array([-omega_max]), np.array([omega_max])
        self.d_bound = np.zeros(0)

    def F(self, b):
        x, y, th, Sxx, Syy, Stt, Sxy, Sxt, Syt = b.unbind(-1)
        s, c, v = torch.sin(th), torch.cos(th), self.v
        ir = 1.0 / (self.r0 + self.r1 * ((x - self.bx) ** 2 + (y - self.by) ** 2))
        return torch.stack([v * c, v * s, torch.zeros_like(x),
                            -2 * v * s * Sxt + self.qx - (Sxx ** 2 + Sxy ** 2) * ir,
                            2 * v * c * Syt + self.qy - (Sxy ** 2 + Syy ** 2) * ir,
                            self.qt - (Sxt ** 2 + Syt ** 2) * ir,
                            v * (c * Sxt - s * Syt) - Sxy * (Sxx + Syy) * ir,
                            -v * s * Stt - (Sxx * Sxt + Sxy * Syt) * ir,
                            v * c * Stt - (Sxy * Sxt + Syy * Syt) * ir], -1)

    def G(self, b):
        g = torch.zeros(b.shape[:-1] + (9, 1), dtype=b.dtype)
        g[..., 2, 0] = 1.0
        return g

    def D(self, b):
        return torch.zeros(b.shape[:-1] + (9, 0), dtype=b.dtype)


# =============================================================================================== #
# QP (paper eq. 25):  min ||u - u_ref||^2  s.t.  A u >= c,  u_lo <= u <= u_hi      (batched, exact)
# Returns u, feasible.  If the constraint cannot be met inside the box, u maximises A u over the box
# (best effort) and feasible = False.
# =============================================================================================== #
def solve_qp_1d(u_ref, A, c, lo, hi, tol=1e-12):
    u_ref, A, c = u_ref[:, 0], A[:, 0], c
    lo, hi = float(lo[0]), float(hi[0])
    u_box = np.clip(u_ref, lo, hi)
    # A u >= c  ->  u >= c/A (A > 0)  or  u <= c/A (A < 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        bnd = c / A
    l = np.where(A > tol, np.maximum(lo, bnd), lo)
    h = np.where(A < -tol, np.minimum(hi, bnd), hi)
    flat_ok = (np.abs(A) > tol) | (c <= 0)              # A ~ 0: constraint is just 0 >= c
    feas = (l <= h + 1e-12) & flat_ok
    u = np.where(feas, np.clip(u_box, l, np.maximum(l, h)), np.where(A >= 0, hi, lo))
    return u[:, None], feas


def solve_qp_2d(u_ref, A, c, lo, hi, tol=1e-12):
    N = len(c)
    u_box = np.clip(u_ref, lo, hi)
    ok_box = (A * u_box).sum(1) >= c                     # box optimum already satisfies the CBF
    nA2 = (A ** 2).sum(1)
    nA = np.sqrt(nA2)
    # otherwise the optimum lies on the line A u = c:  u = u0 + s t,  u0 = projection of u_ref,
    # t = unit direction of the line;  ||u - u_ref||^2 = ||u0 - u_ref||^2 + s^2  ->  s* = clip(0, s_lo, s_hi)
    safe = np.maximum(nA2, tol)
    u0 = u_ref + ((c - (A * u_ref).sum(1)) / safe)[:, None] * A
    t = np.stack([-A[:, 1], A[:, 0]], 1) / np.maximum(nA, tol)[:, None]
    s_lo, s_hi = np.full(N, -np.inf), np.full(N, np.inf)
    empty = np.zeros(N, bool)
    for i in range(2):
        ti = t[:, i]
        par = np.abs(ti) < tol
        empty |= par & ((u0[:, i] < lo[i] - 1e-12) | (u0[:, i] > hi[i] + 1e-12))
        with np.errstate(divide="ignore", invalid="ignore"):
            a1, a2 = (lo[i] - u0[:, i]) / ti, (hi[i] - u0[:, i]) / ti
        s_lo = np.where(par, s_lo, np.maximum(s_lo, np.minimum(a1, a2)))
        s_hi = np.where(par, s_hi, np.minimum(s_hi, np.maximum(a1, a2)))
    line_ok = (nA > tol) & ~empty & (s_lo <= s_hi + 1e-12)
    s = np.clip(0.0, s_lo, s_hi)
    u_line = u0 + np.where(line_ok, s, 0.0)[:, None] * t
    u_best = np.where(A >= 0, hi, lo)                     # maximiser of A u over the box
    u = np.where(ok_box[:, None], u_box, np.where(line_ok[:, None], u_line, u_best))
    feas = ok_box | line_ok
    return np.clip(u, lo, hi), feas


# =============================================================================================== #
# Thm. 2 / Thm. 3  (one discrete measurement update, position measured directly: H = I)
# =============================================================================================== #
def _update_terms(Sigma, alpha, R_d, delta):
    S = Sigma + R_d
    K = Sigma @ np.linalg.inv(S)                                       # eq. 16
    Lam = K @ S @ K.T                                                  # eq. 17
    xi = erfinv(1 - 2 * delta) * (math.sqrt(2 * alpha @ Sigma @ alpha)
                                   - math.sqrt(2 * alpha @ (np.eye(len(alpha)) - K) @ Sigma @ alpha))  # eq. 21-22
    return xi, math.sqrt(2 * alpha @ Lam @ alpha)


def theorem2_bound(Sigma, alpha, R_d, delta):
    """Pr[h_b(b+) < 0] for a belief on the boundary of C_b (eq. 20)."""
    xi, s = _update_terms(np.atleast_2d(Sigma), np.atleast_1d(alpha), np.atleast_2d(R_d), delta)
    return 0.5 * (1 - erf(xi / s))


def theorem3_margin(sigma_range, R_cont, dt, delta, eps, n_grid=50, n_dir=36):
    """Constant gamma >= sqrt(2 a^T Lam a) erfinv(1 - 2 eps) - xi  (Thm. 3), maximised over isotropic
    and anisotropic covariances in sigma_range and over obstacle directions alpha.  R_d = R/dt (A2)."""
    R_d = (R_cont / dt) * np.eye(2)
    lo, hi = sigma_range
    g = 0.0
    for sx in np.linspace(lo, hi, n_grid):
        for sy in (lo, sx, hi):
            Sig = np.diag([sx, sy])
            for th in np.linspace(0, np.pi, n_dir, endpoint=False):
                al = np.array([math.cos(th), math.sin(th)])
                xi, s = _update_terms(Sig, al, R_d, delta)
                g = max(g, s * erfinv(1 - 2 * eps) - xi)
    return g


# =============================================================================================== #
# Belief CBF
# =============================================================================================== #
class BeliefCBF:
    """Second-order (exponential) belief CBF safety filter around a circular obstacle at the origin.

    Parameters
    ----------
    sys    : Air3DBelief or QuadBelief
    R_c    : obstacle radius r in eq. (27)
    kappa  : sigma multiplier; sets delta so that sqrt(2) erfinv(1 - 2 delta) = kappa (A5)
    delta  : risk level of eq. (12) (overrides kappa if given)
    pole   : ECBF gains from pole placement with a double pole at -pole: zeta = (pole^2, 2 pole)
    margin : gamma of Thm. 3 (h~ = h_b - gamma); 0 = the paper's "eps = 0.5" variant
    robust : worst case over the bounded adversary d (A3)
    """

    def __init__(self, sys, R_c, kappa=None, delta=None, pole=2.0, margin=0.0, robust=True):
        if delta is None:
            assert kappa is not None, "give kappa or delta"
            delta = 1.0 - NormalDist().cdf(kappa)
        self.sys, self.R_c, self.delta = sys, R_c, delta
        self.c_var = erfinv(1 - 2 * delta)                 # erfinv(1 - 2 delta) of eq. (12)
        self.zeta1, self.zeta2 = pole ** 2, 2 * pole      # eq. (4): h'' + zeta2 h' + zeta1 h >= 0
        self.margin, self.robust = margin, robust

    # ---------------- eq. (12)/(28) -------------------------------------------------------------
    def barrier(self, b):
        ix, iy = self.sys.i_pos
        sxx, syy, sxy = self.sys.i_sig
        x, y = b[..., ix], b[..., iy]
        Sxx, Syy, Sxy = b[..., sxx], b[..., syy], b[..., sxy]
        rho = torch.sqrt(x ** 2 + y ** 2 + 1e-12)                       # alpha^T (mu - c),  c = 0
        var = (Sxx * x ** 2 + 2 * Sxy * x * y + Syy * y ** 2) / rho ** 2  # alpha^T Sigma alpha
        return rho - self.R_c - self.c_var * torch.sqrt(2 * torch.clamp(var, min=0.0) + 1e-12) - self.margin

    # ---------------- Lie derivatives with autograd ----------------------------------------------
    def lie(self, b_np):
        b = torch.as_tensor(np.asarray(b_np), dtype=DT).clone().requires_grad_(True)
        F, G, D = self.sys.F(b), self.sys.G(b), self.sys.D(b)
        h = self.barrier(b)
        dh = torch.autograd.grad(h.sum(), b, create_graph=True)[0]      # per-sample gradients
        Lf = (dh * F).sum(-1)
        Lg = torch.einsum("ni,nim->nm", dh, G)
        Ld = torch.einsum("ni,nik->nk", dh, D)
        dLf = torch.autograd.grad(Lf.sum(), b)[0]
        out = dict(h=h, Lfh=Lf, Lgh=Lg, Ldh=Ld,
                   LfLfh=(dLf * F).sum(-1),
                   LgLfh=torch.einsum("ni,nim->nm", dLf, G),
                   LdLfh=torch.einsum("ni,nik->nk", dLf, D))
        return {k: v.detach().numpy() for k, v in out.items()}

    # ---------------- eq. (4) as a linear constraint A u >= c ------------------------------------
    def constraint(self, b_np):
        """Relative degree 2 (Lgh = Ldh = 0, checked in the self-test), so
           h'' = LfLfh + LgLfh u + LdLfh d,  h' = Lfh  and eq. (4) reads
           LgLfh u >= -(LfLfh + zeta2 Lfh + zeta1 h) - LdLfh d   for every admissible d."""
        L = self.lie(b_np)
        worst_d = -np.abs(L["LdLfh"]) @ self.sys.d_bound if self.robust else 0.0
        A = L["LgLfh"]
        c = -(L["LfLfh"] + self.zeta2 * L["Lfh"] + self.zeta1 * L["h"]) - worst_d
        return A, c, L

    # ---------------- eq. (25) -------------------------------------------------------------------
    def filter(self, b_np, u_ref):
        """Returns (u, h, active, feasible); active = the filter changed the (box-clipped) reference."""
        u_ref = np.asarray(u_ref, float).reshape(len(b_np), self.sys.m)
        A, c, L = self.constraint(b_np)
        solve = solve_qp_1d if self.sys.m == 1 else solve_qp_2d
        u, feas = solve(u_ref, A, c, self.sys.u_lo, self.sys.u_hi)
        active = np.abs(u - np.clip(u_ref, self.sys.u_lo, self.sys.u_hi)).max(1) > 1e-9
        return u, L["h"], active, feas


# =============================================================================================== #
# Self-test
# =============================================================================================== #
def _rand_air3d(n, rng):
    r, a = rng.uniform(0.4, 2.0, n), rng.uniform(-np.pi, np.pi, n)
    sxx, syy = rng.uniform(0.001, 0.05, (2, n))
    sxy = rng.uniform(-1, 1, n) * 0.9 * np.sqrt(sxx * syy)
    return np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(-np.pi, np.pi, n), sxx, syy, sxy]


def _rand_quad(n, rng):
    r, a = rng.uniform(0.7, 2.5, n), rng.uniform(-np.pi, np.pi, n)
    pxx, pyy = rng.uniform(0.001, 0.05, (2, n))
    pxy = rng.uniform(-1, 1, n) * 0.9 * np.sqrt(pxx * pyy)
    return np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(-2, 2, (n, 2)), pxx, pyy, pxy]


def _rand_dubins(n, rng):
    r, a = rng.uniform(0.7, 2.8, n), rng.uniform(-np.pi, np.pi, n)
    sxx, syy, stt = rng.uniform(0.001, 0.18, n), rng.uniform(0.001, 0.18, n), rng.uniform(0.001, 0.05, n)
    sxy = rng.uniform(-1, 1, n) * 0.5 * np.sqrt(sxx * syy)
    sxt = rng.uniform(-1, 1, n) * 0.5 * np.sqrt(sxx * stt)
    syt = rng.uniform(-1, 1, n) * 0.5 * np.sqrt(syy * stt)
    return np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(-np.pi, np.pi, n), sxx, syy, stt, sxy, sxt, syt]


def _check_lie(cbf, b, rng, name, tau=1e-6):
    sys = cbf.sys
    n = len(b)
    u = rng.uniform(sys.u_lo, sys.u_hi, (n, sys.m))
    d = rng.uniform(-sys.d_bound, sys.d_bound, (n, sys.k))
    L = cbf.lie(b)

    def total_field(bb):
        t = torch.as_tensor(bb, dtype=DT)
        return (sys.F(t) + (sys.G(t) @ torch.as_tensor(u, dtype=DT)[..., None])[..., 0]
                + (sys.D(t) @ torch.as_tensor(d, dtype=DT)[..., None])[..., 0]).numpy()

    h = lambda bb: cbf.barrier(torch.as_tensor(bb, dtype=DT)).numpy()
    v = total_field(b)
    hdot_fd = (h(b + tau * v) - h(b - tau * v)) / (2 * tau)
    hdot_ad = L["Lfh"] + (L["Lgh"] * u).sum(1) + (L["Ldh"] * d).sum(1)
    Lf = lambda bb: cbf.lie(bb)["Lfh"]
    hdd_fd = (Lf(b + tau * v) - Lf(b - tau * v)) / (2 * tau)
    hdd_ad = L["LfLfh"] + (L["LgLfh"] * u).sum(1) + (L["LdLfh"] * d).sum(1)
    # relative errors (central differences: O(tau^2) truncation; |h''| reaches ~1e2 near the obstacle)
    e1 = (np.abs(hdot_fd - hdot_ad) / np.maximum(1.0, np.abs(hdot_ad))).max()
    e2 = (np.abs(hdd_fd - hdd_ad) / np.maximum(1.0, np.abs(hdd_ad))).max()
    rd = max(np.abs(L["Lgh"]).max(), np.abs(L["Ldh"]).max(initial=0.0))
    print(f"[{name}] rel. err h' AD vs FD={e1:.1e}  h''={e2:.1e}  max|Lgh|,|Ldh| (rel. degree 2 => 0)={rd:.1e}")
    assert e1 < 1e-6 and e2 < 1e-6 and rd < 1e-9


def _check_qp(rng, n=3000):
    # 1D vs brute force
    lo, hi = np.array([-4.0]), np.array([4.0])
    ur, A, c = rng.uniform(-6, 6, (n, 1)), rng.normal(0, 1, (n, 1)), rng.normal(0, 3, n)
    u, f = solve_qp_1d(ur, A, c, lo, hi)
    grid = np.linspace(-4, 4, 80001)
    worst = 0.0
    for i in range(0, n, 10):
        okg = A[i, 0] * grid >= c[i]
        if okg.any():
            assert f[i]
            worst = max(worst, abs(u[i, 0] - ur[i, 0]) - np.abs(grid[okg] - ur[i, 0]).min())
        else:
            assert not f[i]
    print(f"[QP 1D] feasibility agrees with brute force; objective gap {worst:.1e}")
    # 2D vs brute force on a grid
    lo2, hi2 = np.array([-4.0, -4.0]), np.array([4.0, 4.0])
    ur, A, c = rng.uniform(-6, 6, (n, 2)), rng.normal(0, 1, (n, 2)), rng.normal(0, 4, n)
    u, f = solve_qp_2d(ur, A, c, lo2, hi2)
    g = np.linspace(-4, 4, 401)
    GX, GY = np.meshgrid(g, g)
    P = np.c_[GX.ravel(), GY.ravel()]
    worst = 0.0
    for i in range(0, n, 30):
        okg = P @ A[i] >= c[i]
        if okg.sum() > 3:
            assert f[i] and A[i] @ u[i] >= c[i] - 1e-9
            worst = max(worst, np.linalg.norm(u[i] - ur[i]) - np.linalg.norm(P[okg] - ur[i], axis=1).min())
    print(f"[QP 2D] constraint satisfied; objective vs grid optimum {worst:+.1e} (<= 0 up to grid spacing 0.02)")
    assert worst < 0.03


def _closed_loop(rng):
    """Double integrator around the obstacle, constant P, worst-case-ish wind, no estimation noise:
    with robust=True, h must stay >= 0 (forward invariance)."""
    sys = QuadBelief(q=0.0, r=0.04, gamma=0.4, a_max=4.0, d_max=0.5)   # q = 0 and P on the fixed point 0
    cbf = BeliefCBF(sys, R_c=0.5, kappa=2.0, pole=2.0, robust=True)
    n, dt = 400, 1e-3
    b = np.zeros((n, 7))
    b[:, 0] = rng.uniform(-2.5, -1.5, n)
    b[:, 1] = rng.uniform(-0.3, 0.3, n)
    b[:, 2] = rng.uniform(0.5, 1.0, n)              # flying toward the obstacle
    b[:, 4:6] = 0.0                                  # P = 0 is an equilibrium when q = 0
    h0 = cbf.lie(b)["h"]
    keep = (h0 > 0) & (cbf.lie(b)["Lfh"] + 2.0 * h0 > 0)   # admissible initial conditions for the ECBF
    b = b[keep]
    hmin = np.inf
    for _ in range(4000):
        u, h, _, feas = cbf.filter(b, np.tile([2.0, 0.0], (len(b), 1)))
        hmin = min(hmin, h.min())
        A, c, L = cbf.constraint(b)
        d = -0.5 * np.sign(L["LdLfh"])                # worst-case wind for the current state
        b = b + dt * np.c_[b[:, 2:4], u + d, np.zeros((len(b), 3))]
    print(f"[closed loop] {len(b)} rollouts, worst-case wind: min h = {hmin:+.2e} (>= 0 up to Euler error)")
    assert hmin > -1e-3


def selftest():
    rng = np.random.default_rng(0)
    # A5: delta from kappa
    cbf = BeliefCBF(Air3DBelief(0.75, 0.75, 4.0, 3.0, 0.05, 0.05), R_c=0.25, kappa=2.0)
    print(f"[A5] kappa=2 -> delta={cbf.delta:.4f}, sqrt(2) erfinv(1-2 delta)={_SQRT2 * cbf.c_var:.6f}")
    assert abs(_SQRT2 * cbf.c_var - 2.0) < 1e-9
    # barrier closed form: isotropic Sigma = s I -> h = ||mu|| - R_c - kappa sqrt(s)
    b = np.array([[1.0, 0.5, 0.3, 0.02, 0.02, 0.0]])
    hb = cbf.lie(b)["h"][0]
    assert abs(hb - (math.hypot(1.0, 0.5) - 0.25 - 2.0 * math.sqrt(0.02))) < 1e-9
    print(f"[barrier] isotropic closed form matches: h={hb:.6f}")
    # paper's numerical example after Thm. 2: sigma- = 0.3, r = 0.1, delta = 0.01 -> ~0.09
    p = theorem2_bound(0.3, 1.0, 0.1, 0.01)
    print(f"[Thm 2] paper example -> {p:.4f} (paper: ~0.09)")
    assert abs(p - 0.09) < 0.005
    _check_lie(cbf, _rand_air3d(500, rng), rng, "Air3D")
    qcbf = BeliefCBF(QuadBelief(0.01, 0.04, 0.4, 4.0, 0.5), R_c=0.5, kappa=2.0)
    _check_lie(qcbf, _rand_quad(500, rng), rng, "Quad ")
    dcbf = BeliefCBF(DubinsBelief(1.0, 1.2, 0.005, 0.005, 0.01, 0.01, 0.01, (2.0, 0.0)), R_c=0.5, kappa=3.0)
    _check_lie(dcbf, _rand_dubins(500, rng), rng, "Dubins")
    Lg = dcbf.lie(_rand_dubins(500, rng))["LgLfh"]
    print(f"[Dubins] omega enters h'': median |LgLfh| = {np.median(np.abs(Lg)):.2e} (must be > 0)")
    assert np.median(np.abs(Lg)) > 1e-3
    _check_qp(rng)
    _closed_loop(rng)
    g = theorem3_margin((0.001, 0.05), 0.05, 0.01, cbf.delta, 0.01)
    print(f"[Thm 3] Air3D margin gamma (eps=0.01, R=0.05, dt=0.01) = {g:.4f}")
    print("all self-tests passed")


if __name__ == "__main__":
    selftest()
