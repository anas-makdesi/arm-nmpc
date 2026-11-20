"""Model of the free-running ROS 2 loop (``mode:=realtime``), deterministic and offline.

The plant runs at 1 kHz with joint friction. Every 10 ms the NMPC node takes the latest
state, predicts it forward by its expected delay, solves one real-time iteration for the
predicted state and time, and sends a command; the command reaches the robot after the
node's true delay. Between commands the robot's 1 kHz loop applies ``inner_torque``.

Two command designs (``plan_nodes``):

* 1: the first interval only (feedforward torque, planned state at the next node); a late
  command leaves the robot extrapolating that interval. The node predicts with the
  feedforward torque. (arm-nmpc 0.1.0)
* 4: the first four intervals of the plan (states and torques) and the NMPC's feedback
  gain; a late command leaves the robot following the plan. The node predicts through the
  robot's own law with the last command it sent. (current ROS nodes)

Two inner loops (``inner``): ``"riccati"`` (the NMPC's stage-0 gain,
``NMPC.feedback_gain``) and ``"pd"`` (fixed joint impedance, Franka's example gains).

Two timing models (``timing``): ``"constant"`` (fixed delay) and ``"starved"``: per-step
delays like the ROS runs (median about 3 ms, occasional 8–17 ms spikes) plus 150 ms
episodes, about one per second, in which every step takes 11.5 ms (seen on a 2-core
virtual machine whose host takes the CPU away). A step that overruns the 10 ms period
makes the node skip state samples, and the node predicts with its running-average delay,
not the true one.

The C++ mirror of ``inner_torque`` is ``ros2/arm_nmpc_ros/include/arm_nmpc_ros/inner_loop.hpp``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dynamics as dy
from .nmpc import NMPC, deploy_config
from .plant import ArmPlant
from .task import Circle, Friction, Limits

NJ = 7
KP = np.array([600, 600, 600, 600, 250, 150, 50.0])  # N m / rad
KD = np.array([50, 50, 50, 20, 20, 20, 10.0])  # N m s / rad
DT_CTRL = 0.01


@dataclass
class Command:
    t_plan: float  # time of plan node 0 (the predicted state)
    t: np.ndarray  # node times after t_plan, (n + 1,)
    X: np.ndarray  # planned states [q, dq] at the nodes, (n + 1, 14)
    U: np.ndarray  # planned torques per interval, (n, 7)
    K: np.ndarray | None  # stage-0 feedback gain (7, 14)


def inner_torque(cmd: Command, t, x, inner="riccati", extrapolate=False):
    """The robot's 1 kHz law at time t (plan time) and state x.

    ``extrapolate``: one-interval command of arm-nmpc 0.1.0 (the state reference goes on
    along the first interval, up to three intervals, and the torque is held)."""
    s = t - cmd.t_plan
    n = len(cmd.U)
    if extrapolate:
        w = np.clip(s / (cmd.t[1] - cmd.t[0]), 0.0, 3.0)
        xr = (1 - w) * cmd.X[0] + w * cmd.X[1]
        tau = cmd.U[0].copy()
    else:
        s = np.clip(s, 0.0, cmd.t[n])
        j = min(int(np.searchsorted(cmd.t, s, side="right") - 1), n - 1)
        w = (s - cmd.t[j]) / (cmd.t[j + 1] - cmd.t[j])
        xr = (1 - w) * cmd.X[j] + w * cmd.X[j + 1]
        tau = cmd.U[j].copy()
    if inner == "riccati" and cmd.K is not None:
        return tau + cmd.K @ (x - xr)
    return tau + KP * (xr[:NJ] - x[:NJ]) + KD * (xr[NJ:] - x[NJ:])


def _predict(q, dq, t0, d, torque, fv, fc, eps=0.3):
    """Model integration (RK4, steps of at most 1 ms); torque(t, x) gives the input."""
    aba = dy.functions()["aba"]
    n = max(1, int(np.ceil(d / 1e-3)))
    h = d / n

    def f(t, q, v):
        u = torque(t, np.r_[q, v]) - (fv * v + fc * np.tanh(v / eps))
        return v, np.array(aba(q, v, u)).ravel()

    t = t0
    for _ in range(n):
        k1q, k1v = f(t, q, dq)
        k2q, k2v = f(t + h / 2, q + h / 2 * k1q, dq + h / 2 * k1v)
        k3q, k3v = f(t + h / 2, q + h / 2 * k2q, dq + h / 2 * k2v)
        k4q, k4v = f(t + h, q + h * k3q, dq + h * k3v)
        q = q + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        dq = dq + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += h
    return q, dq


def run(inner="riccati", plan_nodes=4, timing="constant", delay_ms=3.5, period=1.0,
        duration=5.0, fv=None, fc=None, seed=0, name="anmpc_realtime") -> dict:
    """Simulate the free-running loop; returns tracking and velocity-limit metrics
    (errors after the 1 s speed ramp)."""
    fv = np.zeros(NJ) if fv is None else np.asarray(fv, float)
    fc = np.zeros(NJ) if fc is None else np.asarray(fc, float)
    task, lim = Circle(period=period), Limits()
    c = NMPC(task, deploy_config(name=name))
    c.set_friction_model(fv, fc)
    plant = ArmPlant(task.q0, friction=Friction())
    rng = np.random.default_rng(seed)
    one = plan_nodes == 1
    d_avg, busy_until = 4e-3, -1.0
    starve_from = 1.2 + rng.uniform(0.0, 0.6)
    active = pending = last = None
    t_err, err, dqs, slow = [], [], [], 0
    n_ms = int(round(duration * 1000))
    for k in range(n_ms):
        t = k * 1e-3
        if k % 10 == 0 and t >= busy_until - 1e-9:
            if timing == "starved":
                d_true = 2.3e-3 + rng.exponential(0.7e-3)
                if rng.random() < 0.03:
                    d_true += rng.uniform(5e-3, 14e-3)
                if starve_from <= t < starve_from + 0.15:
                    d_true = 11.5e-3
                elif t >= starve_from + 0.15:
                    starve_from = t + rng.uniform(0.6, 1.2)
                d_pred = d_avg
            else:
                d_true = d_pred = 1e-3 * delay_ms
            slow += d_true > DT_CTRL
            busy_until = t + d_true
            q, dq = plant.q.copy(), plant.dq.copy()
            if last is not None:
                if one:
                    u_ff = last.U[0]
                    q, dq = _predict(q, dq, t, d_pred, lambda tt, x: u_ff, fv, fc)
                else:
                    q, dq = _predict(q, dq, t, d_pred,
                                     lambda tt, x: inner_torque(last, tt, x, inner), fv, fc)
            t_plan = t + (d_pred if last is not None else 0.0)
            c.step(t_plan, q, dq)
            n = max(plan_nodes, 1)
            X = np.array([c.solver.get(j, "x") for j in range(n + 1)])
            U = np.array([c.solver.get(j, "u") for j in range(n)])
            K = c.feedback_gain() if inner == "riccati" else None
            pending = (t + (d_true if last is not None else 0.0),
                       Command(t_plan, c.offsets[:n + 1].copy(), X, U, K))
            last = pending[1]
            if timing == "starved" and t > 0:
                d_avg = 0.9 * d_avg + 0.1 * d_true
        if pending is not None and t >= pending[0] - 1e-9:
            active, pending = pending[1], None
        if active is None:
            tau = dy.rnea(plant.q, np.zeros(NJ), np.zeros(NJ))  # hold until the first command
        else:
            tau = inner_torque(active, t, np.r_[plant.q, plant.dq], inner, extrapolate=one)
        plant.step(tau, 1e-3)
        p, _ = dy.fk(plant.q)
        pr, _, _ = task.reference(t + 1e-3)
        t_err.append(t + 1e-3)
        err.append(float(np.linalg.norm(p - pr[0])))
        dqs.append(np.abs(plant.dq) / lim.dq_max)
        if not np.all(np.isfinite(plant.q)) or err[-1] > 0.2:
            break
    t_err, err, R = np.array(t_err), np.array(err), np.array(dqs)
    m = t_err >= 1.0
    return dict(
        inner=inner, plan_nodes=plan_nodes, timing=timing, seed=seed,
        finished=bool(len(err) == n_ms),
        rms_error_mm=round(1e3 * float(np.sqrt(np.mean(err[m] ** 2))), 3) if m.any() else None,
        max_error_mm=round(1e3 * float(err[m].max()), 3) if m.any() else None,
        max_velocity_ratio=round(float(R.max()), 3),
        max_velocity_ratio_per_joint=[round(float(v), 2) for v in R.max(axis=0)],
        slow_steps=int(slow), failed_solves=int(np.sum(np.array(c.status) != 0)),
    )
