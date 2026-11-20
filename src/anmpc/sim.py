"""Closed-loop runs of the arm and tracking metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dynamics as dy
from .plant import ArmPlant
from .task import Friction, Limits


@dataclass
class Run:
    name: str
    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray  # applied (after saturation)
    tau_cmd: np.ndarray
    p: np.ndarray  # TCP position
    p_ref: np.ndarray
    failed: bool
    solve_ms: np.ndarray
    t_settle: float  # error statistics are taken after this time (end of the ramp)

    @property
    def err(self):
        return np.linalg.norm(self.p - self.p_ref, axis=1)

    def summary(self, limits: Limits | None = None) -> dict:
        lim = limits or Limits()
        e = self.err
        m = self.t >= self.t_settle
        st = self.solve_ms
        over_v = np.mean(np.any(np.abs(self.dq) > lim.dq_max * 1.01, axis=1))  # > 1 % over
        sat = np.mean(np.any(np.abs(self.tau_cmd) >= lim.tau_max * 0.999, axis=1))
        return dict(
            name=self.name, failed=bool(self.failed), duration=round(float(self.t[-1]), 2),
            rms_error_mm=round(1e3 * float(np.sqrt(np.mean(e[m] ** 2))), 2) if m.any() else None,
            max_error_mm=round(1e3 * float(e[m].max()), 2) if m.any() else None,
            velocity_limit_violation_pct=round(100 * float(over_v), 2),
            torque_saturated_pct=round(100 * float(sat), 2),
            max_dq_ratio=round(float(np.max(np.abs(self.dq) / lim.dq_max)), 3),
            max_tau_ratio=round(float(np.max(np.abs(self.tau) / lim.tau_max)), 3),
            solve_ms_median=round(float(np.median(st)), 3) if st.size else None,
            solve_ms_p99=round(float(np.percentile(st, 99)), 3) if st.size else None,
            solve_ms_max=round(float(st.max()), 3) if st.size else None,
        )


def run(controller, task, duration: float, name: str = "", dt: float = 0.005,
        friction: Friction | None = None, q0=None, fail_error: float = 0.1,
        on_step=None) -> Run:
    q0 = task.q0 if q0 is None else q0
    plant = ArmPlant(q0, friction=friction)
    T, Q, DQ, TA, TC, P, PR = [], [], [], [], [], [], []
    failed = False
    for k in range(int(round(duration / dt))):
        t = k * dt
        q, dq = plant.q.copy(), plant.dq.copy()
        tau = controller.step(t, q, dq)
        p, _ = dy.fk(q)
        pr, _, _ = task.reference(t)
        applied = plant.step(tau, dt)
        if on_step is not None:
            on_step(k, t, q, dq, applied, plant)
        T.append(t); Q.append(q); DQ.append(dq); TA.append(applied); TC.append(tau)
        P.append(p); PR.append(pr[0])
        if np.linalg.norm(p - pr[0]) > fail_error or not np.all(np.isfinite(plant.q)):
            failed = True
            break
    st = np.asarray(getattr(controller, "solve_times", []), float) * 1e3
    return Run(name, np.array(T), np.array(Q), np.array(DQ), np.array(TA), np.array(TC),
               np.array(P), np.array(PR), failed, st, task.ramp)
