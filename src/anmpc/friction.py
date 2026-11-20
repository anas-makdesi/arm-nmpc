"""Joint friction from data, with a certified error bound (ddma).

Unmodelled joint friction costs tracking accuracy. This module learns each joint's
friction curve tau_f(dq) from a short calibration motion and gives:

1. a band that contains the true curve: ddma's :class:`ddma.GrowthMap` on a fine
   partition of the velocity range, from
   * the measurements: friction torque = applied torque - rigid-body torque (from the
     model and the measured motion), within +-``noise``;
   * a physical prior: friction does not decrease with speed (g_lo = 0) and its slope is
     at most ``g_hi``.
   The band holds for every curve consistent with the data and the prior.
2. a smooth model inside the band for the NMPC, tau_f = fv dq + fc tanh(dq / eps), fitted
   to the band's midpoints (non-negative least squares);
3. a certified bound on the model error: the largest distance from the smooth model to
   the far edge of the band, over the velocities the data cover (outside that range the
   band is valid too, but wide).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dynamics as dy
from .plant import ArmPlant
from .task import Q_READY, Friction, Limits

NJ = 7


@dataclass
class FrictionCertificate:
    limits: Limits = field(default_factory=Limits)
    cells: int = 120
    g_hi: float = 25.0  # max slope of friction vs velocity (N m s/rad)
    noise: float = 0.15  # measurement error bound (N m)
    eps: float = 0.3  # smoothing of the fitted model (as in the NMPC)
    models: list = field(init=False, repr=False)

    def __post_init__(self):
        from ddma import GrowthMap, RectangularPartition

        self.models = []
        for i in range(NJ):
            v = self.limits.dq_max[i]
            part = RectangularPartition([np.linspace(-v, v, self.cells + 1)])
            self.models.append(GrowthMap(part, g_lo=[0.0], g_hi=[self.g_hi],
                                         d_lo=-self.noise, d_hi=self.noise))
        self.n = 0
        self.v_lo = np.full(NJ, np.inf)
        self.v_hi = np.full(NJ, -np.inf)

    def add(self, dq, tau_f):
        dq, tau_f = np.atleast_2d(dq), np.atleast_2d(tau_f)
        for i in range(NJ):
            self.models[i].update(dq[:, i:i + 1], tau_f[:, i])
        self.v_lo = np.minimum(self.v_lo, dq.min(axis=0))
        self.v_hi = np.maximum(self.v_hi, dq.max(axis=0))
        self.n += len(dq)

    def band(self, i):
        lo, hi = self.models[i].f_bounds()
        v = self.limits.dq_max[i]
        e = np.linspace(-v, v, self.cells + 1)
        return 0.5 * (e[1:] + e[:-1]), np.asarray(lo).ravel(), np.asarray(hi).ravel()

    def fit(self):
        """Smooth model (fv, fc) per joint and its certified error bound (N m)."""
        from scipy.optimize import nnls

        fv, fc, err = np.zeros(NJ), np.zeros(NJ), np.zeros(NJ)
        for i in range(NJ):
            c, lo, hi = self.band(i)
            ok = self._covered(i, c, lo, hi)
            mid = 0.5 * (lo[ok] + hi[ok])
            A = np.c_[c[ok], np.tanh(c[ok] / self.eps)]
            (fv[i], fc[i]), _ = nnls(A, mid)
            model = fv[i] * c[ok] + fc[i] * np.tanh(c[ok] / self.eps)
            err[i] = float(np.max(np.maximum(hi[ok] - model, model - lo[ok])))
        return fv, fc, err

    def _covered(self, i, c, lo, hi):
        return np.isfinite(lo) & np.isfinite(hi) & (c >= self.v_lo[i]) & (c <= self.v_hi[i])

    def coverage(self, i):
        """Velocity range (rad/s) the data cover (the fit and the error bound hold there)."""
        return float(self.v_lo[i]), float(self.v_hi[i])


def calibration_run(friction: Friction, duration: float = 6.0, dt: float = 0.01,
                    amp: float = 0.5, noise: float = 0.15, seed: int = 0):
    """Joint-space sine sweep around the ready pose (each joint at its own frequency),
    tracked with computed torque on the nominal model. Returns measured (dq, tau_f)."""
    rng = np.random.default_rng(seed)
    f = dy.functions()
    M, rnea, aba = f["mass"], f["rnea"], f["aba"]
    plant = ArmPlant(Q_READY, friction=friction)
    w = 2 * np.pi * np.array([0.35, 0.45, 0.4, 0.5, 0.7, 0.6, 0.8])
    A = amp * np.array([1.0, 0.5, 1.0, 0.5, 1.2, 1.0, 1.2])
    DQ, TF = [], []
    for k in range(int(duration / dt)):
        t = k * dt
        s = np.minimum(t / 1.0, 1.0)  # ramp in amplitude
        qd = Q_READY + s * A * np.sin(w * t)
        dqd = s * A * w * np.cos(w * t)
        ddqd = -s * A * w ** 2 * np.sin(w * t)
        q, dq = plant.q.copy(), plant.dq.copy()
        ddq_cmd = ddqd + 400 * (qd - q) + 40 * (dqd - dq)
        tau = np.array(M(q)) @ ddq_cmd + np.array(rnea(q, dq, np.zeros(NJ))).ravel()
        applied = plant.step(tau, dt)
        # measured friction at the start of the step: applied torque minus rigid-body torque
        ddq_true = np.array(aba(q, dq, applied - friction.torque(dq))).ravel()
        tau_rb = np.array(rnea(q, dq, ddq_true)).ravel()
        DQ.append(dq)
        TF.append(applied - tau_rb + rng.uniform(-noise, noise, NJ))
    return np.array(DQ), np.array(TF)
