"""Simulated arm: rigid-body dynamics (ABA) plus joint friction, torque saturation at the
motor limits, integrated with RK4 at 1 kHz."""
from __future__ import annotations

import numpy as np

from . import dynamics as dy
from .task import Friction, Limits


class ArmPlant:
    def __init__(self, q0, dq0=None, friction: Friction | None = None,
                 limits: Limits | None = None, dt_int: float = 1e-3):
        self.q = np.asarray(q0, float).copy()
        self.dq = np.zeros(7) if dq0 is None else np.asarray(dq0, float).copy()
        self.friction = friction or Friction()
        self.limits = limits or Limits()
        self.dt_int = dt_int
        self._aba = dy.functions()["aba"]

    def _f(self, q, dq, tau):
        ddq = np.array(self._aba(q, dq, tau - self.friction.torque(dq))).ravel()
        return dq, ddq

    def step(self, tau, dt):
        tau = np.clip(np.asarray(tau, float), -self.limits.tau_max, self.limits.tau_max)
        n = max(1, int(round(dt / self.dt_int)))
        h = dt / n
        q, dq = self.q, self.dq
        for _ in range(n):
            k1q, k1v = self._f(q, dq, tau)
            k2q, k2v = self._f(q + h / 2 * k1q, dq + h / 2 * k1v, tau)
            k3q, k3v = self._f(q + h / 2 * k2q, dq + h / 2 * k2v, tau)
            k4q, k4v = self._f(q + h * k3q, dq + h * k3v, tau)
            q = q + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
            dq = dq + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        self.q, self.dq = q, dq
        return tau  # the torque actually applied
