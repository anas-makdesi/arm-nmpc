"""Launch test: simulator (with joint friction, lockstep) and NMPC nodes track the TCP
circle (period 1 s) for 10 s of wall time. Checks tracking error, joint-velocity limits,
solver status and the controller's step time against the 10 ms control period.
"""

import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from arm_nmpc_msgs.msg import ArmState, ControllerStats
from rclpy.qos import QoSProfile, ReliabilityPolicy

RUN_S = 10.0


@pytest.mark.launch_test
def generate_test_description():
    sim = launch_ros.actions.Node(
        package="arm_nmpc_ros", executable="sim_node", parameters=[{"friction": True}]
    )
    nmpc = launch_ros.actions.Node(
        package="arm_nmpc_ros", executable="nmpc_node", parameters=[{"period": 1.0}]
    )
    return launch.LaunchDescription([sim, nmpc, launch_testing.actions.ReadyToTest()])


class TestClosedLoop(unittest.TestCase):
    def test_drives_in_real_time(self):
        rclpy.init()
        node = rclpy.create_node("closed_loop_probe")
        states, stats = [], []
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(ArmState, "/arm/joint_state", states.append, qos)
        node.create_subscription(ControllerStats, "/nmpc/stats", stats.append, 10)
        t_end = time.time() + RUN_S
        while time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.02)
        node.destroy_node()
        rclpy.shutdown()

        self.assertGreater(len(stats), 300)  # lockstep: as fast as the controller runs
        self.assertTrue(all(m.status == 0 for m in stats[1:]))
        err = [m.tcp_error for m in stats[300:]]  # well after the 1 s speed ramp
        self.assertLess((sum(e * e for e in err) / len(err)) ** 0.5, 0.002)
        self.assertLess(max(m.max_velocity_ratio for m in stats), 1.03)
        self.assertLess(max(m.tcp_error for m in stats[300:]), 0.002)
        lat = sorted(m.latency_ms for m in stats[1:])
        self.assertLess(lat[len(lat) // 2], 10.0)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
