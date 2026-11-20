# NMPC command for the robot's 1 kHz loop (as MPC is run on torque-controlled arms): the
# first intervals of the plan and the NMPC's own feedback gain. At time s after
# header.stamp the robot applies, every millisecond,
#   tau = tau_plan(s) + feedback_gain (x - x_plan(s)),      x = [q, dq]
# (or, if feedback_gain is all zero, a fixed joint impedance
#   tau = tau_plan(s) + Kp (q_plan(s) - q) + Kd (dq_plan(s) - dq)),
# with x_plan interpolated linearly between the plan nodes and tau_plan held over each
# interval. A late command leaves the robot following the previous plan (up to its last
# node). See arm_nmpc_ros/inner_loop.hpp.
std_msgs/Header header       # time of plan node 0
float64[7] torque            # N m, torque of the first interval (= plan_torque[0:7])
float64[5] plan_time         # s after header.stamp, nodes 0..4
float64[35] plan_position    # rad, nodes 0..4 (7 per node)
float64[35] plan_velocity    # rad/s, nodes 0..4
float64[28] plan_torque      # N m, intervals 0..3 (7 per interval)
float64[98] feedback_gain    # 7 x 14, row-major: d tau / d [q, dq] of the solved QP (stage 0)
