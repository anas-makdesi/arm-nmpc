# Changelog

## Unreleased
- **Free-running ROS 2 loop (`mode:=realtime`) no longer experimental.** Each command now
  carries the NMPC's stage-0 feedback gain and the first four intervals of its plan. The
  robot's 1 kHz loop (`arm_nmpc_ros/inner_loop.hpp`) applies the plan's torque plus
  K (x − x_plan), and keeps following the plan when a command is late. The NMPC node
  predicts its delay through that same law. In 12 of 13 ROS 2 runs the joints stay
  within their velocity limits (94–101 %; before: 190–212 %), at 0.16–0.50 mm RMS. The
  13th run, with heavy CPU starvation on the test machine, reached 207 %.
  `inner_loop:=pd` keeps the fixed joint impedance for comparison. **`circle.launch.py`
  now starts the free-running mode by default**; `mode:=lockstep` gives the deterministic
  loop (unchanged, 0.09 mm), which the node defaults and the launch test keep using.
- `NMPC.feedback_gain()` (Python) and `NmpcSolver::set_compute_gain` (C++): backward Riccati
  recursion over the stage matrices of the QP just solved. The deployed solver is unchanged.
- `JointTorques` message changed: `plan_time`, `plan_position`, `plan_velocity`,
  `plan_torque`, `feedback_gain` replace `position_ref0/1`, `velocity_ref0/1`, `interval`.
- `anmpc.realtime` and `examples/realtime_loop.py`: an offline, deterministic model of the
  free-running loop (delays, CPU starvation episodes) → `results/realtime_loop.json`. ROS 2
  runs: `results/ros2_realtime_runs.json` (all runs; it replaces the single-run
  `ros2_circle_realtime_ddma.json`).
- Tests: the gain against finite differences of the real-time iteration (Python), the C++
  gain against Python (`solver_gain.csv` fixture), the 1 kHz law (gtest).
- ddma pinned to `v0.12.0` (the `friction` extra). CI installs it, so the friction
  certificate tests now run on every push instead of being skipped.

## 0.1.0 (2026-11-16)
First release.
- Franka Emika Panda rigid-body dynamics in CasADi (forward kinematics, RNEA, ABA), tested
  against Pinocchio; parameters extracted from the URDF (`scripts/extract_panda.py`).
- acados joint-torque NMPC (Gauss-Newton RTI, HPIPM, 0.35 s non-uniform horizon) with
  time-shifted warm start; joint torque, velocity and position limits; friction model as
  run-time parameters.
- Baselines: operational-space control, one-step task-space QP (OSQP).
- Joint-friction band from a calibration motion with ddma's `GrowthMap`, smooth model fitted
  inside it with a certified error bound.
- Speed sweep, friction experiment and timing benchmark with figures and JSON results.
- ROS 2 Jazzy packages: messages, C++ NMPC node (generated solver and CasADi-generated
  dynamics), C++ simulator node with lockstep (default) and real-time modes, delay
  compensation and a 1 kHz joint-impedance loop for the real-time mode, parity gtests,
  launch test, probe script.
- CI (ruff, pytest, colcon build/test in a ROS container), Dockerfile.
