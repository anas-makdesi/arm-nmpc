"""Robot limits, friction of the simulated arm, and the Cartesian reference.

The reference is a vertical circle for the tool centre point (TCP), in front of the
robot, started from rest with a smooth speed ramp. The circle's period sets how hard the
task is: at short periods the wrist joints reach their torque (12 N m) and the joints their
velocity limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dynamics as dy
from . import panda_data as D

Q_READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])


@dataclass(frozen=True)
class Limits:
    q_min: np.ndarray = field(default_factory=lambda: D.Q_MIN.copy())
    q_max: np.ndarray = field(default_factory=lambda: D.Q_MAX.copy())
    dq_max: np.ndarray = field(default_factory=lambda: D.DQ_MAX.copy())
    tau_max: np.ndarray = field(default_factory=lambda: D.TAU_MAX.copy())


@dataclass(frozen=True)
class Friction:
    """Joint friction of the simulated arm: tau_f = fv dq + fc tanh(dq / eps).
    The controllers' models have no friction unless they are given one."""

    fv: tuple = (0.6, 0.6, 0.5, 0.5, 0.25, 0.25, 0.2)  # viscous (N m s/rad)
    fc: tuple = (0.8, 0.8, 0.7, 0.7, 0.4, 0.4, 0.3)  # Coulomb (N m)
    eps: float = 0.05

    def torque(self, dq):
        return np.array(self.fv) * dq + np.array(self.fc) * np.tanh(np.asarray(dq) / self.eps)


NO_FRICTION = Friction(fv=(0.0,) * 7, fc=(0.0,) * 7)


@dataclass
class Circle:
    period: float = 1.5  # s per revolution once at speed
    radius: float = 0.15
    ramp: float = 1.0  # s to reach full speed from rest
    q0: np.ndarray = None

    def __post_init__(self):
        if self.q0 is None:
            self.q0 = Q_READY.copy()
        p0, _ = dy.fk(self.q0)
        self.center = p0 - np.array([0.0, 0.0, self.radius])  # start at the top of the circle

    @property
    def omega(self) -> float:
        return 2 * np.pi / self.period

    @property
    def peak_speed(self) -> float:
        return self.omega * self.radius

    def phase(self, t):
        """Angle along the circle and its first two derivatives (C1 ramp from rest)."""
        t = np.asarray(t, float)
        w, T = self.omega, self.ramp
        a = np.where(t < T, w * t ** 2 / (2 * T), w * (t - T / 2))
        da = np.where(t < T, w * t / T, w)
        dda = np.where(t < T, w / T, 0.0)
        return np.where(t < 0, 0.0, a), np.where(t < 0, 0.0, da), np.where(t < 0, 0.0, dda)

    def reference(self, t):
        """TCP position, velocity and acceleration at times t: arrays (K, 3)."""
        a, da, dda = (np.atleast_1d(v) for v in self.phase(t))
        r = self.radius
        # circle in the y-z plane, starting at the top, going towards +y
        p = self.center + r * np.c_[np.zeros_like(a), np.sin(a), np.cos(a)]
        v = r * np.c_[np.zeros_like(a), np.cos(a) * da, -np.sin(a) * da]
        acc = r * np.c_[np.zeros_like(a), -np.sin(a) * da ** 2 + np.cos(a) * dda,
                        -np.cos(a) * da ** 2 - np.sin(a) * dda]
        return p, v, acc
