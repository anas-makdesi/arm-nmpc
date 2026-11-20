#!/usr/bin/env python3
"""Record the running ROS 2 closed loop for a while and print a summary (JSON).

ros2 launch arm_nmpc_ros circle.launch.py &
python3 arm_probe.py --seconds 4 --skip 1.0 [--out run.json]
"""

import argparse
import json
import time

import numpy as np
import rclpy
from arm_nmpc_msgs.msg import ControllerStats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--skip", type=float, default=1.0, help="ramp time excluded from errors")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rclpy.init()
    node = rclpy.create_node("arm_probe")
    S = []
    node.create_subscription(
        ControllerStats,
        "/nmpc/stats",
        lambda m: S.append(
            (
                m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec,
                m.solve_time_ms,
                m.step_time_ms,
                m.latency_ms,
                m.status,
                m.tcp_error,
                m.max_velocity_ratio,
            )
        ),
        100,
    )
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.02)
    node.destroy_node()
    rclpy.shutdown()
    S = np.array(S[1:])
    t = S[:, 0] - S[0, 0]
    e = S[t >= a.skip, 5]

    def q(v, p):
        return round(float(np.percentile(v, p)), 3)

    out = dict(
        seconds=a.seconds,
        steps=int(len(S)),
        failed_solves=int((S[:, 4] != 0).sum()),
        rms_error_mm=round(1e3 * float(np.sqrt(np.mean(e**2))), 3),
        max_error_mm=round(1e3 * float(e.max()), 3),
        max_velocity_ratio=round(float(S[:, 6].max()), 4),
        solve_ms_median=q(S[:, 1], 50),
        solve_ms_p99=q(S[:, 1], 99),
        step_ms_median=q(S[:, 2], 50),
        latency_ms_median=q(S[:, 3], 50),
        latency_ms_p99=q(S[:, 3], 99),
        latency_ms_max=round(float(S[:, 3].max()), 3),
    )
    print(json.dumps(out, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
