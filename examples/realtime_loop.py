"""Free-running loop (model of ``mode:=realtime``): command designs and inner loops compared,
with a constant delay and with CPU starvation episodes (see anmpc.realtime).

    python examples/realtime_loop.py        # -> results/realtime_loop.json

Each run is cached in build/realtime_cache/, so an interrupted run resumes where it stopped.
The NMPC uses the friction model fitted inside the ddma band (results/circle.json).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from anmpc.realtime import run  # noqa: E402

DESIGNS = [  # (label, inner loop, plan nodes per command)
    ("one interval + fixed impedance (0.1.0)", "pd", 1),
    ("one interval + NMPC gain", "riccati", 1),
    ("plan + fixed impedance", "pd", 4),
    ("plan + NMPC gain (current)", "riccati", 4),
]
TIMINGS = [("constant", 0), ("starved", 0), ("starved", 1), ("starved", 2)]


def main():
    fm = json.loads((ROOT / "results" / "circle.json").read_text())["friction_model"]
    cache = ROOT / "build" / "realtime_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for timing, seed in TIMINGS:
        for label, inner, nodes in DESIGNS:
            f = cache / f"{timing}_s{seed}_{inner}_n{nodes}.json"
            if f.exists():
                r = json.loads(f.read_text())
            else:
                r = run(inner, nodes, timing, delay_ms=3.5, fv=fm["fv"], fc=fm["fc"], seed=seed)
                f.write_text(json.dumps(r))
            r["design"] = label
            rows.append(r)
            print(f"{timing:8s} seed {seed}  {label:40s} {r['rms_error_mm']:7.3f} mm  "
                  f"max {r['max_error_mm']:7.2f} mm  velocity {r['max_velocity_ratio']:.2f}x",
                  flush=True)
    out = dict(
        description="Model of the free-running ROS 2 loop (anmpc.realtime): circle period 1 s, "
                    "5 s, NMPC with the ddma friction model; errors after the 1 s ramp; "
                    "velocity as a fraction of the joint limit (worst joint, worst time).",
        runs=rows,
    )
    (ROOT / "results" / "realtime_loop.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
