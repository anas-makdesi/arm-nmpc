"""Reference data for the C++ tests (ros2/arm_nmpc_ros/test/fixtures).

    python scripts/make_cpp_fixtures.py

dynamics.csv    random (q, dq, tau) and the Python ABA, TCP position and gravity torques
plant_traj.csv  plant trajectory (with joint friction) under a fixed torque sequence
solver_seq.csv  closed-loop times, states and the torques the Python NMPC (RTI) returned
solver_gain.csv the NMPC's stage-0 feedback gain (7 x 14, row-major) at the same steps
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from anmpc import dynamics as dy  # noqa: E402
from anmpc.nmpc import NMPC, deploy_config  # noqa: E402
from anmpc.plant import ArmPlant  # noqa: E402
from anmpc.task import Q_READY, Circle, Friction, Limits  # noqa: E402

FIX = ROOT / "ros2" / "arm_nmpc_ros" / "test" / "fixtures"


def main():
    FIX.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lim = Limits()
    rows = []
    for _ in range(20):
        q = rng.uniform(lim.q_min, lim.q_max)
        dq, tau = rng.normal(size=7), rng.normal(size=7) * 5
        g = dy.rnea(q, np.zeros(7), np.zeros(7))
        rows.append(np.r_[q, dq, tau, dy.aba(q, dq, tau), dy.fk(q)[0], g])
    np.savetxt(
        FIX / "dynamics.csv",
        np.array(rows),
        delimiter=",",
        fmt="%.17g",
        header="q(7),dq(7),tau(7),ddq(7),p_tcp(3),gravity(7)",
    )
    plant = ArmPlant(Q_READY, friction=Friction())
    rows = []
    for k in range(100):
        t = 0.01 * k
        tau = dy.rnea(plant.q, np.zeros(7), np.zeros(7)) + 2.0 * np.sin(3 * t + np.arange(7))
        q, dq = plant.q.copy(), plant.dq.copy()
        plant.step(tau, 0.01)
        rows.append(np.r_[q, dq, tau, plant.q, plant.dq])
    np.savetxt(
        FIX / "plant_traj.csv",
        np.array(rows),
        delimiter=",",
        fmt="%.17g",
        header="q(7),dq(7),tau(7),q_next(7),dq_next(7); dt=0.01, simulated friction",
    )
    task = Circle(1.0)
    c = NMPC(task, deploy_config(name="anmpc_deploy"))
    plant = ArmPlant(Q_READY, friction=Friction())
    rows, gains = [], []
    for k in range(150):
        t = 0.01 * k
        q, dq = plant.q.copy(), plant.dq.copy()
        tau = c.step(t, q, dq)
        rows.append(np.r_[t, q, dq, tau])
        gains.append(c.feedback_gain().ravel())
        plant.step(tau, 0.01)
    np.savetxt(
        FIX / "solver_seq.csv",
        np.array(rows),
        delimiter=",",
        fmt="%.17g",
        header="t,q(7),dq(7),tau(7); RTI (deploy config), circle period 1 s",
    )
    np.savetxt(
        FIX / "solver_gain.csv",
        np.array(gains),
        delimiter=",",
        fmt="%.17g",
        header="K (7 x 14, row-major) after each step of solver_seq.csv",
    )
    print("fixtures written to", FIX)


if __name__ == "__main__":
    main()
