"""Baseline controllers for Cartesian tracking.

* :class:`OSC` - operational-space (task-space inverse dynamics) control, the textbook
  and industrial standard (Khatib 1987): desired TCP acceleration from feedforward plus PD,
  joint accelerations from the Jacobian pseudo-inverse, a null-space posture task,
  computed torque M ddq + h, then clipped to the torque limits. No knowledge of the
  velocity limits.
* :class:`TaskQP` - one-step task-space QP (as in whole-body control frameworks): the
  same desired TCP acceleration, but the joint accelerations and torques come from a QP
  that respects torque limits and keeps the next-step joint velocities and positions
  within limits. It sees one step ahead only.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from . import dynamics as dy
from .task import Q_READY, Limits

NJ = 7


class _TaskSpace:
    def __init__(self, task, kp=400.0, kd=40.0, k_null=20.0, d_null=8.0, limits=None):
        self.task = task
        self.kp, self.kd, self.k_null, self.d_null = kp, kd, k_null, d_null
        self.lim = limits or Limits()
        f = dy.functions()
        self._fk, self._J, self._Jd = f["fk"], f["jac"], f["jdot_dq"]
        self._M, self._rnea = f["mass"], f["rnea"]
        self.solve_times: list[float] = []

    def _terms(self, t, q, dq):
        p = np.array(self._fk(q)[0]).ravel()
        J = np.array(self._J(q))
        Jd_dq = np.array(self._Jd(q, dq)).ravel()
        pr, vr, ar = (a[0] for a in self.task.reference(t))
        a_des = ar + self.kp * (pr - p) + self.kd * (vr - J @ dq)
        M = np.array(self._M(q))
        h = np.array(self._rnea(q, dq, np.zeros(NJ))).ravel()  # C dq + g
        ddq_null = self.k_null * (Q_READY - q) - self.d_null * dq
        return J, Jd_dq, a_des, M, h, ddq_null


class OSC(_TaskSpace):
    def step(self, t, q, dq):
        t0 = time.perf_counter()
        J, Jd_dq, a_des, M, h, ddq_null = self._terms(t, q, dq)
        Jp = np.linalg.pinv(J)
        N = np.eye(NJ) - Jp @ J
        ddq = Jp @ (a_des - Jd_dq) + N @ ddq_null
        tau = np.clip(M @ ddq + h, -self.lim.tau_max, self.lim.tau_max)
        self.solve_times.append(time.perf_counter() - t0)
        return tau


class TaskQP(_TaskSpace):
    """min ||J ddq + Jd dq - a_des||^2 + w_n ||ddq - ddq_null||^2 + w_t ||tau - h||^2
    s.t.  M ddq - tau = -h,  |tau| <= tau_max,
          |dq + dt ddq| <= dq_max,  q_min <= q + dt dq + dt^2/2 ddq <= q_max"""

    def __init__(self, task, dt=0.01, w_null=1e-3, w_tau=1e-6, **kw):
        import osqp

        super().__init__(task, **kw)
        self.dt, self.w_null, self.w_tau = dt, w_null, w_tau
        nv = 2 * NJ  # [ddq, tau]
        P = np.ones((nv, nv))  # dense pattern; values updated each step
        self.P_pat = sp.triu(sp.csc_matrix(P)).tocsc()
        A = np.ones((3 * NJ, nv))
        A[NJ:, NJ:] = 0
        A[NJ : 2 * NJ, :NJ] = np.eye(NJ)
        A[2 * NJ :, :NJ] = np.eye(NJ)
        A = np.vstack([A, np.c_[np.zeros((NJ, NJ)), np.eye(NJ)]])
        self.A_pat = sp.csc_matrix(A)
        self.solver = osqp.OSQP()
        self.solver.setup(
            P=self.P_pat,
            q=np.zeros(nv),
            A=self.A_pat,
            l=-np.ones(4 * NJ),
            u=np.ones(4 * NJ),
            verbose=False,
            warm_starting=True,
            polishing=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=4000,
        )
        self.status: list[str] = []
        self._last = None

    def step(self, t, q, dq):
        t0 = time.perf_counter()
        J, Jd_dq, a_des, M, h, ddq_null = self._terms(t, q, dq)
        dt, lim = self.dt, self.lim
        H = np.zeros((2 * NJ, 2 * NJ))
        H[:NJ, :NJ] = J.T @ J + self.w_null * np.eye(NJ)
        H[NJ:, NJ:] = self.w_tau * np.eye(NJ)
        g = np.r_[J.T @ (Jd_dq - a_des) - self.w_null * ddq_null, -self.w_tau * h]
        A = np.zeros((4 * NJ, 2 * NJ))
        A[:NJ, :NJ], A[:NJ, NJ:] = M, -np.eye(NJ)
        A[NJ : 2 * NJ, :NJ] = np.eye(NJ)
        A[2 * NJ : 3 * NJ, :NJ] = np.eye(NJ)
        A[3 * NJ :, NJ:] = np.eye(NJ)
        lo = np.r_[
            -h, (-lim.dq_max - dq) / dt, (lim.q_min - q - dt * dq) / (0.5 * dt**2), -lim.tau_max
        ]
        up = np.r_[
            -h, (lim.dq_max - dq) / dt, (lim.q_max - q - dt * dq) / (0.5 * dt**2), lim.tau_max
        ]
        # keep the fixed sparsity patterns of the setup (tiny offsets on structural zeros)
        Px = sp.triu(sp.csc_matrix(H + 1e-15)).tocsc()
        Ax = sp.csc_matrix(A + 1e-15 * (self.A_pat.toarray() != 0))
        self.solver.update(Px=Px.data, Ax=Ax.data, q=g, l=lo, u=up)
        res = self.solver.solve()
        self.status.append(res.info.status)
        if res.x is not None and res.info.status_val in (1, 2):
            self._last = res.x[NJ:]
        tau = np.zeros(NJ) + (h if self._last is None else self._last)
        self.solve_times.append(time.perf_counter() - t0)
        return np.clip(tau, -lim.tau_max, lim.tau_max)
