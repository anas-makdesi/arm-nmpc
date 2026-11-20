"""Joint-torque NMPC for the Panda with acados.

State x = [q, dq] (14), input u = joint torques (7).
    dq' = ABA(q, dq, u - f(dq))                 f: optional friction model (parameters)
    minimise  sum_k w_p ||p_tcp(q_k) - p_ref(t_k)||^2 + w_q ||q_k - q_ready||^2
                    + w_dq ||dq_k||^2 + w_u ||u_k - g(q_k)||^2       (+ terminal terms)
    subject to |u| <= torque limits (hard), |dq| <= velocity limits and q in joint range
               (soft)
The posture term resolves the arm's redundancy; the torque term penalises effort beyond
gravity compensation. Horizon 0.35 s: 5 steps of the control period (10 ms at 100 Hz)
then 10 steps of 30 ms. One real-time iteration per step (Gauss-Newton, HPIPM), warm
started from the previous solution shifted forward in time.

Friction model parameters p = [fv (7), fc (7)] for f(dq) = fv dq + fc tanh(dq / eps) can
be set at run time (zeros = no friction model).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np

from . import dynamics as dy
from .task import Q_READY, Limits

BUILD_DIR = Path(os.environ.get("ANMPC_BUILD_DIR", Path.cwd() / "build"))
NJ, NX, NU, NP = 7, 14, 7, 14


@dataclass
class NMPCConfig:
    dt0: float = 0.01
    n_fine: int = 5
    dt_tail: float = 0.03
    n_tail: int = 10
    mode: str = "rti"
    max_iter: int = 30
    w_p: float = 1e6
    w_q: float = 1.0
    w_dq: float = 0.1
    w_u: float = 1e-4
    terminal_p: float = 1e6
    terminal_dq: float = 1.0
    slack: float = 1e5
    eps: float = 0.3  # friction-model smoothing (rad/s); smoother than the plant for RK4
    qp_cond_N: int = 5
    rk_steps: int = 1
    compile_flags: str = "-O2"
    name: str = ""

    @property
    def N(self) -> int:
        return self.n_fine + self.n_tail

    @property
    def time_steps(self) -> np.ndarray:
        return np.r_[np.full(self.n_fine, self.dt0), np.full(self.n_tail, self.dt_tail)]


def deploy_config(name: str = "") -> NMPCConfig:
    """Configuration of the ROS 2 node: a cheaper grid (N = 10, same 0.35 s horizon) for fewer
    solve-time overruns of the 10 ms budget, and stronger posture and joint-velocity
    weights that damp the motions the TCP cost does not see (the arm's null space, e.g.
    joint 7 turning about the tool axis) against the 1 kHz joint-impedance loop."""
    return NMPCConfig(n_fine=4, n_tail=6, dt_tail=0.31 / 6, w_q=5.0, w_dq=1.0, name=name)


def build_ocp(cfg: NMPCConfig, limits: Limits | None = None):
    from acados_template import AcadosModel, AcadosOcp

    lim = limits or Limits()
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    p = ca.SX.sym("p", NP)
    q, dq = x[:NJ], x[NJ:]
    fric = p[:NJ] * dq + p[NJ:] * ca.tanh(dq / cfg.eps)
    ddq = dy.aba_expr(q, dq, u - fric)
    ptcp, _ = dy.fk_expr(q)
    grav = dy.rnea_expr(q, ca.SX.zeros(NJ), ca.SX.zeros(NJ))

    model = AcadosModel()
    model.name = cfg.name or f"anmpc_{cfg.mode}_N{cfg.N}"
    model.x, model.u, model.p = x, u, p
    model.f_expl_expr = ca.vertcat(dq, ddq)
    model.cost_y_expr = ca.vertcat(ptcp, q, dq, u - grav)
    model.cost_y_expr_e = ca.vertcat(ptcp, dq)

    ocp = AcadosOcp()
    ocp.model = model
    so = ocp.solver_options
    so.N_horizon = cfg.N
    so.tf = float(cfg.time_steps.sum())
    so.time_steps = cfg.time_steps
    ocp.parameter_values = np.zeros(NP)

    c = ocp.cost
    c.cost_type = c.cost_type_e = "NONLINEAR_LS"
    c.W = np.diag(np.r_[np.full(3, cfg.w_p), np.full(NJ, cfg.w_q), np.full(NJ, cfg.w_dq),
                        np.full(NJ, cfg.w_u)])
    c.W_e = np.diag(np.r_[np.full(3, cfg.terminal_p), np.full(NJ, cfg.terminal_dq)])
    c.yref = np.zeros(3 + 3 * NJ)
    c.yref_e = np.zeros(3 + NJ)

    k = ocp.constraints
    k.x0 = np.r_[Q_READY, np.zeros(NJ)]
    k.idxbu = np.arange(NU)
    k.lbu, k.ubu = -lim.tau_max, lim.tau_max
    k.idxbx = np.arange(NX)
    k.lbx = np.r_[lim.q_min, -lim.dq_max]
    k.ubx = np.r_[lim.q_max, lim.dq_max]
    k.idxsbx = np.arange(NX)
    k.idxbx_e = np.arange(NX)
    k.lbx_e, k.ubx_e = k.lbx.copy(), k.ubx.copy()
    k.idxsbx_e = np.arange(NX)
    for a in ("zl", "zu", "Zl", "Zu"):
        setattr(c, a, cfg.slack * np.ones(NX))
        setattr(c, a + "_e", cfg.slack * np.ones(NX))

    so.integrator_type = "ERK"
    so.sim_method_num_stages = 4
    so.sim_method_num_steps = cfg.rk_steps  # RK4 substeps per interval (light wrist links)
    so.hessian_approx = "GAUSS_NEWTON"
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    so.qp_solver_cond_N = cfg.qp_cond_N
    so.qp_solver_warm_start = 1
    so.qp_solver_iter_max = 50
    if cfg.mode == "rti":
        so.nlp_solver_type = "SQP_RTI"
    else:
        so.nlp_solver_type = "SQP"
        so.nlp_solver_max_iter = cfg.max_iter
        so.tol = 1e-6
    so.print_level = 0
    so.ext_fun_compile_flags = cfg.compile_flags
    return ocp


class NMPC:
    """Cartesian tracking NMPC. ``step(t, q, dq)`` returns the joint torques."""

    def __init__(self, task, cfg: NMPCConfig | None = None, limits: Limits | None = None,
                 build: bool = True):
        from acados_template import AcadosOcpSolver

        self.task = task
        self.cfg = cfg = cfg or NMPCConfig()
        self.ocp = build_ocp(cfg, limits)
        out = BUILD_DIR / self.ocp.model.name
        out.mkdir(parents=True, exist_ok=True)
        self.ocp.code_export_directory = str(out / "c_generated_code")
        self.ocp.json_file = "acados_ocp.json"
        self.solver = AcadosOcpSolver(self.ocp, build=build, generate=build, verbose=False)
        self.offsets = np.r_[0.0, np.cumsum(cfg.time_steps)]
        self.initialized = False
        self._t_last = None
        self.solve_times, self.wall_times, self.status, self.iters = [], [], [], []
        self.set_friction_model(np.zeros(NJ), np.zeros(NJ))

    def _shift(self, dt):
        """Warm start: move the previous solution forward by dt in time (interpolated on
        the non-uniform grid), so the real-time iteration linearises at the right point."""
        if dt <= 0:
            return
        N = self.cfg.N
        X = np.array([self.solver.get(k, "x") for k in range(N + 1)])
        U = np.array([self.solver.get(k, "u") for k in range(N)])
        tk = self.offsets
        tn = np.minimum(tk + dt, tk[-1])
        Xn = np.array([np.interp(tn, tk, X[:, i]) for i in range(X.shape[1])]).T
        Un = np.array([np.interp(tn[:-1], tk[:-1], U[:, i]) for i in range(U.shape[1])]).T
        for k in range(N + 1):
            self.solver.set(k, "x", Xn[k])
            if k < N:
                self.solver.set(k, "u", Un[k])

    def set_friction_model(self, fv, fc):
        """Friction the model compensates (no rebuild needed)."""
        p = np.r_[fv, fc]
        for k in range(self.cfg.N + 1):
            self.solver.set(k, "p", p)

    def feedback_gain(self) -> np.ndarray:
        """Stage-0 feedback gain K (7 × 14) of the QP solved in the last step:
        u0(x0 + dx) = u0 + K dx, x = [q, dq].

        Backward Riccati recursion over the QP's stage matrices (the Gauss-Newton
        linearisation the real-time iteration just used). Exact while no inequality
        constraint is active; a saturated input really has zero sensitivity, and the robot
        saturates it anyway. Used by the 1 kHz loop between NMPC steps: it applies
        u0 + K (x - x_plan(t)), a stiffness that follows the NMPC's own cost, instead of a
        fixed joint-space PD that also pulls on the arm's null space."""
        g = self.solver.get_from_qp_in
        N = self.cfg.N
        P = np.asarray(g(N, "Q"))
        K = None
        for k in range(N - 1, -1, -1):
            A, B, Q, R, S = (np.asarray(g(k, f)) for f in "ABQRS")
            G = S + B.T @ P @ A
            K = -np.linalg.solve(R + B.T @ P @ B, G)
            P = Q + A.T @ P @ A + G.T @ K
            P = 0.5 * (P + P.T)
        return K

    def step(self, t, q, dq) -> np.ndarray:
        t0 = time.perf_counter()
        P, _, _ = self.task.reference(t + self.offsets)
        z7 = np.zeros(NJ)
        for k in range(self.cfg.N):
            self.solver.set(k, "yref", np.r_[P[k], Q_READY, z7, z7])
        self.solver.set(self.cfg.N, "yref", np.r_[P[-1], z7])
        x0 = np.r_[q, dq]
        if self.initialized and self._t_last is not None:
            self._shift(t - self._t_last)
        if not self.initialized:
            g = dy.rnea(q, np.zeros(NJ), np.zeros(NJ))
            for k in range(self.cfg.N + 1):
                self.solver.set(k, "x", x0)
                if k < self.cfg.N:
                    self.solver.set(k, "u", g)
            self._t_last = None
            if self.cfg.mode == "rti":
                for _ in range(10):
                    self.solver.solve_for_x0(x0, False, False)
            self.initialized = True
        u = self.solver.solve_for_x0(x0, fail_on_nonzero_status=False,
                                     print_stats_on_failure=False)
        self._t_last = t
        self.wall_times.append(time.perf_counter() - t0)
        self.solve_times.append(float(self.solver.get_stats("time_tot")))
        self.status.append(int(self.solver.status))
        self.iters.append(int(self.solver.get_stats("sqp_iter")) if self.cfg.mode == "sqp" else 1)
        return np.asarray(u, float)
