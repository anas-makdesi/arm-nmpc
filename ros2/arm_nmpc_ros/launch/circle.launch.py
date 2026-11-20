"""Panda simulator + NMPC tracking the TCP circle.

    ros2 launch arm_nmpc_ros circle.launch.py   # free-running, NMPC gain in the 1 kHz loop
    ros2 launch arm_nmpc_ros circle.launch.py inner_loop:=pd   # fixed joint impedance instead
    ros2 launch arm_nmpc_ros circle.launch.py mode:=lockstep   # deterministic (as the tests)
    ros2 launch arm_nmpc_ros circle.launch.py period:=0.7
    ros2 launch arm_nmpc_ros circle.launch.py fv:="[...]" fc:="[...]"   # friction model

``friction`` switches joint friction in the simulator; ``fv`` / ``fc`` are the friction
model the NMPC compensates (e.g. the one fitted inside the ddma band).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _vec(ctx, name):
    s = LaunchConfiguration(name).perform(ctx).strip("[]")
    return [float(v) for v in s.split(",")]


def _nodes(ctx):
    period = float(LaunchConfiguration("period").perform(ctx))
    friction = LaunchConfiguration("friction").perform(ctx).lower() in ("1", "true", "yes")
    mode = LaunchConfiguration("mode").perform(ctx)
    inner = LaunchConfiguration("inner_loop").perform(ctx)
    return [
        Node(
            package="arm_nmpc_ros",
            executable="sim_node",
            name="arm_sim",
            parameters=[
                {"state_rate_hz": 100.0, "friction": friction, "mode": mode, "inner_loop": inner}
            ],
            output="screen",
        ),
        Node(
            package="arm_nmpc_ros",
            executable="nmpc_node",
            name="nmpc",
            parameters=[
                {
                    "period": period,
                    "friction_fv": _vec(ctx, "fv"),
                    "friction_fc": _vec(ctx, "fc"),
                    "delay_ms": float(LaunchConfiguration("delay_ms").perform(ctx)),
                    "lockstep": mode == "lockstep",
                    "feedback_gain": inner == "riccati",
                }
            ],
            output="screen",
        ),
    ]


def generate_launch_description():
    zeros = "[0,0,0,0,0,0,0]"
    return LaunchDescription(
        [
            DeclareLaunchArgument("period", default_value="1.0"),
            DeclareLaunchArgument("friction", default_value="true"),
            DeclareLaunchArgument(
                "mode",
                default_value="realtime",
                description="realtime (free-running) or lockstep (deterministic)",
            ),
            DeclareLaunchArgument(
                "inner_loop",
                default_value="riccati",
                description="1 kHz loop in realtime mode: riccati (NMPC gain) or pd",
            ),
            DeclareLaunchArgument("fv", default_value=zeros),
            DeclareLaunchArgument("fc", default_value=zeros),
            DeclareLaunchArgument(
                "delay_ms",
                default_value="-1.0",
                description="delay compensation (ms); -1 = measured average",
            ),
            OpaqueFunction(function=_nodes),
        ]
    )
