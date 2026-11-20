"""Timing benchmark, in closed loop on two laps of the circle at 0.7 s per lap.

    python examples/benchmark.py            # writes results/timing.json

Per controller: time per control step (median, p99, worst), share of steps over the 10 ms
(100 Hz) budget, tracking error and joint-velocity use. Timings depend on the machine.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from anmpc.baselines import OSC, TaskQP  # noqa: E402
from anmpc.nmpc import NMPC, NMPCConfig, deploy_config  # noqa: E402
from anmpc.sim import run  # noqa: E402
from anmpc.task import NO_FRICTION, Circle  # noqa: E402

OUT = ROOT / "results"


def stats(r):
    ms = r.solve_ms
    s = r.summary()
    return dict(
        steps=int(ms.size),
        median_ms=round(float(np.median(ms)), 3),
        p99_ms=round(float(np.percentile(ms, 99)), 3),
        max_ms=round(float(ms.max()), 3),
        over_10ms_pct=round(100 * float(np.mean(ms > 10)), 2),
        rms_error_mm=s["rms_error_mm"],
        max_dq_ratio=s["max_dq_ratio"],
    )


def main():
    OUT.mkdir(exist_ok=True)
    task = Circle(0.7)
    dur = task.ramp + 2 * task.period
    T = {}
    for key, cfg in [
        ("nmpc_N15", NMPCConfig(name="anmpc_main")),
        ("nmpc_N10_deploy", deploy_config(name="anmpc_deploy")),
    ]:
        c = NMPC(task, cfg)
        T[key] = stats(run(c, task, dur, key, dt=0.01, friction=NO_FRICTION))
        T[key]["python_step_median_ms"] = round(1e3 * float(np.median(c.wall_times)), 3)
    T["osc"] = stats(run(OSC(task), task, dur, "osc", dt=0.01, friction=NO_FRICTION))
    T["task_qp"] = stats(run(TaskQP(task, dt=0.01), task, dur, "qp", dt=0.01, friction=NO_FRICTION))
    for k, v in T.items():
        print(k, v)
    (OUT / "timing.json").write_text(json.dumps(T, indent=2))
    return T


if __name__ == "__main__":
    main()
