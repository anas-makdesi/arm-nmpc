"""Tests for the arm NMPC package (Python side). Pinocchio is the reference for the
dynamics; acados and ddma tests are skipped when those are not installed."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from anmpc import dynamics as dy
from anmpc.baselines import OSC, TaskQP
from anmpc.sim import run
from anmpc.task import NO_FRICTION, Q_READY, Circle, Friction, Limits

warnings.filterwarnings("ignore")


def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


needs_acados = pytest.mark.skipif(not _has("acados_template"), reason="acados not installed")
needs_ddma = pytest.mark.skipif(not _has("ddma"), reason="ddma not installed")
needs_pin = pytest.mark.skipif(not _has("pinocchio"), reason="pinocchio not installed")


@needs_pin
def test_dynamics_match_pinocchio():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import pinocchio as pin
    from extract_panda import reduced_model

    try:
        m = reduced_model()
    except SystemExit:
        pytest.skip("panda.urdf not available (pip install example-robot-data)")
    d = m.createData()
    fid = m.getFrameId("panda_hand_tcp")
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.uniform(m.lowerPositionLimit, m.upperPositionLimit)
        dq, tau, ddq = rng.normal(size=7), rng.normal(size=7) * 10, rng.normal(size=7)
        pin.forwardKinematics(m, d, q)
        pin.updateFramePlacements(m, d)
        np.testing.assert_allclose(dy.fk(q)[0], d.oMf[fid].translation, atol=1e-12)
        np.testing.assert_allclose(dy.rnea(q, dq, ddq), pin.rnea(m, d, q, dq, ddq), atol=1e-9)
        np.testing.assert_allclose(dy.aba(q, dq, tau), pin.aba(m, d, q, dq, tau),
                                   rtol=1e-6, atol=1e-6)
        M = pin.crba(m, d, q)
        np.testing.assert_allclose(dy.mass_matrix(q), np.triu(M) + np.triu(M, 1).T, atol=1e-10)


def test_aba_inverts_rnea():
    rng = np.random.default_rng(1)
    lim = Limits()
    for _ in range(10):
        q = rng.uniform(lim.q_min, lim.q_max)
        dq, tau = rng.normal(size=7), rng.normal(size=7) * 5
        ddq = dy.aba(q, dq, tau)
        np.testing.assert_allclose(dy.rnea(q, dq, ddq), tau, atol=1e-8)


def test_circle_reference():
    c = Circle(1.0)
    p0, v0, _ = c.reference(0.0)
    np.testing.assert_allclose(p0[0], dy.fk(Q_READY)[0], atol=1e-12)
    np.testing.assert_allclose(v0[0], 0.0, atol=1e-12)  # starts at rest
    t = np.linspace(c.ramp, c.ramp + c.period, 200)
    p, v, _ = c.reference(t)
    np.testing.assert_allclose(np.linalg.norm(p - c.center, axis=1), c.radius, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), c.peak_speed, rtol=1e-9)


def test_osc_tracks_slow_circle():
    task = Circle(2.0)
    r = run(OSC(task), task, task.ramp + task.period, dt=0.01, friction=NO_FRICTION)
    assert not r.failed and r.summary()["rms_error_mm"] < 0.5


def test_task_qp_respects_velocity_limits():
    task = Circle(0.7)
    r = run(TaskQP(task, dt=0.01), task, task.ramp + task.period, dt=0.01, friction=NO_FRICTION)
    assert r.summary()["max_dq_ratio"] < 1.03


@needs_acados
def test_nmpc_tracks_fast_circle_within_limits():
    from anmpc.nmpc import NMPC, NMPCConfig

    task = Circle(0.7)
    c = NMPC(task, NMPCConfig(name="test_arm"))
    r = run(c, task, task.ramp + task.period, dt=0.01, friction=NO_FRICTION)
    s = r.summary()
    assert not r.failed and s["rms_error_mm"] < 0.2
    assert s["max_dq_ratio"] < 1.01
    assert s["max_tau_ratio"] <= 1.0
    assert all(st == 0 for st in c.status)


@needs_acados
def test_feedback_gain_matches_finite_differences():
    """K predicts how the real-time iteration's torque responds to a change of the state."""
    from anmpc.nmpc import NMPC, deploy_config

    c = NMPC(Circle(1.0), deploy_config(name="test_arm_deploy"))
    q, dq = Q_READY.copy(), np.zeros(7)
    for k in range(30):  # 0.3 s into the ramp: no constraint active
        c.step(0.01 * k, q, dq)
        x1 = c.solver.get(1, "x")
        q, dq = x1[:7], x1[7:]
    N, S = c.cfg.N, c.solver
    X = [S.get(k, "x") for k in range(N + 1)]
    U = [S.get(k, "u") for k in range(N)]

    def solve_at(x0):  # one real-time iteration from the same linearisation point
        for k in range(N + 1):
            S.set(k, "x", X[k])
            if k < N:
                S.set(k, "u", U[k])
        S.solve_for_x0(x0, False, False)
        return S.get(0, "u")

    x0 = np.r_[q, dq]
    u0 = solve_at(x0)
    K = c.feedback_gain()
    assert K.shape == (7, 14)
    h = 1e-5
    Kfd = np.column_stack([(solve_at(x0 + h * e) - solve_at(x0 - h * e)) / (2 * h)
                           for e in np.eye(14)])
    assert np.linalg.norm(K - Kfd) < 1e-3 * np.linalg.norm(Kfd)  # QP tolerance
    dx = 1e-4 * np.random.default_rng(0).standard_normal(14)
    np.testing.assert_allclose(solve_at(x0 + dx) - u0, K @ dx, rtol=1e-3, atol=1e-6)


@needs_ddma
def test_friction_band_is_sound():
    from anmpc.friction import FrictionCertificate, calibration_run

    F = Friction()
    DQ, TF = calibration_run(F, duration=4.0, amp=0.8)
    cert = FrictionCertificate()
    cert.add(DQ, TF)
    for i in range(7):
        c, lo, hi = cert.band(i)
        true = np.array(F.fv)[i] * c + np.array(F.fc)[i] * np.tanh(c / F.eps)
        ok = np.isfinite(lo) & np.isfinite(hi)
        assert np.all(lo[ok] <= true[ok] + 1e-9) and np.all(hi[ok] >= true[ok] - 1e-9)
    fv, fc, err = cert.fit()
    assert np.all(fv >= 0) and np.all(fc >= 0) and np.all(err < 1.0)


@needs_acados
@needs_ddma
def test_friction_model_improves_tracking():
    from anmpc.friction import FrictionCertificate, calibration_run
    from anmpc.nmpc import NMPC, NMPCConfig

    F = Friction()
    DQ, TF = calibration_run(F, amp=0.8)
    cert = FrictionCertificate()
    cert.add(DQ, TF)
    fv, fc, _ = cert.fit()
    task = Circle(1.0)
    c = NMPC(task, NMPCConfig(name="test_arm"))
    base = run(c, task, task.ramp + task.period, dt=0.01, friction=F).summary()["rms_error_mm"]
    c.initialized = False
    c.set_friction_model(fv, fc)
    with_model = run(c, task, task.ramp + task.period, dt=0.01,
                     friction=F).summary()["rms_error_mm"]
    assert with_model < 0.5 * base


def test_inner_loop_follows_the_plan():
    """The 1 kHz law (mirrored in ros2/arm_nmpc_ros/include/arm_nmpc_ros/inner_loop.hpp)."""
    from anmpc.realtime import KD, KP, Command, inner_torque

    rng = np.random.default_rng(1)
    t = 0.01 * np.arange(5)
    X, U, K = rng.normal(size=(5, 14)), rng.normal(size=(4, 7)), rng.normal(size=(7, 14))
    cmd = Command(t_plan=2.0, t=t, X=X, U=U, K=K)
    x = 0.4 * X[1] + 0.6 * X[2]  # on the plan, 60 % into the second interval
    np.testing.assert_allclose(inner_torque(cmd, 2.016, x), U[1], atol=1e-12)
    dx = 1e-3 * rng.normal(size=14)
    np.testing.assert_allclose(inner_torque(cmd, 2.016, x + dx), U[1] + K @ dx, atol=1e-12)
    pd = inner_torque(cmd, 2.016, x + dx, inner="pd")
    np.testing.assert_allclose(pd, U[1] - KP * dx[:7] - KD * dx[7:], atol=1e-12)
    np.testing.assert_allclose(inner_torque(cmd, 9.0, X[4]), U[3], atol=1e-12)  # late
