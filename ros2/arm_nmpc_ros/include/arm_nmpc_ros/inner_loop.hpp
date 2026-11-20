// The robot's 1 kHz loop between NMPC steps (as MPC is run on torque-controlled arms), shared
// by the simulator (which applies it) and the NMPC node (which predicts the arm forward
// through it to compensate its own delay). Mirrors anmpc.realtime.inner_torque.
//
// A command carries the first kPlanNodes intervals of the NMPC's plan (states at the nodes,
// torque per interval) and the stage-0 feedback gain K of the QP it solved. At time s after
// the command's stamp the robot applies
//   tau = tau_plan(s) + K (x - x_plan(s))                     (gain "riccati")
//   tau = tau_plan(s) + Kp (q_plan(s) - q) + Kd (dq_plan(s) - dq)   (fixed impedance "pd")
// with x_plan interpolated linearly between nodes and tau_plan held over each interval.
// If the next command is late, the robot keeps following the plan (up to its last node),
// rather than extrapolating from the first interval.
#pragma once

#include <algorithm>
#include <array>

#include "arm_nmpc_ros/arm_model.hpp"

namespace arm_nmpc
{

constexpr int kPlanNodes = 4;  // intervals carried by a command (4 x 10 ms on the deploy grid)
constexpr int kNx = 2 * NJ;
using Gain = std::array<double, NJ * kNx>;  // 7 x 14, row-major

// joint impedance of Franka's example controller (the fixed-gain alternative)
constexpr double kImpedanceK[NJ] = {600, 600, 600, 600, 250, 150, 50};  // N m / rad
constexpr double kImpedanceD[NJ] = {50, 50, 50, 20, 20, 20, 10};  // N m s / rad

struct Plan
{
  std::array<double, kPlanNodes + 1> t{};  // node times after the stamp (s)
  std::array<Vec7, kPlanNodes + 1> q{}, dq{};
  std::array<Vec7, kPlanNodes> tau{};
  Gain gain{};
  bool has_gain = false;
};

// torque the robot applies at time s after the plan's stamp, in state (q, dq)
inline Vec7 inner_torque(const Plan & p, double s, const Vec7 & q, const Vec7 & dq,
  bool use_gain)
{
  s = std::clamp(s, 0.0, p.t[kPlanNodes]);
  int j = 0;
  while (j < kPlanNodes - 1 && s >= p.t[j + 1]) {++j;}
  const double w = std::clamp((s - p.t[j]) / std::max(p.t[j + 1] - p.t[j], 1e-9), 0.0, 1.0);
  double e[kNx];  // x - x_plan(s)
  for (int i = 0; i < NJ; ++i) {
    e[i] = q[i] - ((1 - w) * p.q[j][i] + w * p.q[j + 1][i]);
    e[NJ + i] = dq[i] - ((1 - w) * p.dq[j][i] + w * p.dq[j + 1][i]);
  }
  Vec7 tau = p.tau[j];
  for (int i = 0; i < NJ; ++i) {
    if (use_gain && p.has_gain) {
      for (int k = 0; k < kNx; ++k) {tau[i] += p.gain[i * kNx + k] * e[k];}
    } else {
      tau[i] -= kImpedanceK[i] * e[i] + kImpedanceD[i] * e[NJ + i];
    }
  }
  return tau;
}

}  // namespace arm_nmpc
