// The robot-side law shared by the simulator and the NMPC node's delay prediction.
#include <gtest/gtest.h>

#include "arm_nmpc_ros/inner_loop.hpp"

using namespace arm_nmpc;

static Plan make_plan()
{
  Plan p;
  for (int k = 0; k <= kPlanNodes; ++k) {
    p.t[k] = 0.01 * k;
    for (int i = 0; i < NJ; ++i) {p.q[k][i] = 0.1 * k + i; p.dq[k][i] = -0.2 * k + 0.5 * i;}
    if (k < kPlanNodes) {for (int i = 0; i < NJ; ++i) {p.tau[k][i] = 10.0 * k + i;}}
  }
  for (int i = 0; i < NJ * kNx; ++i) {p.gain[i] = 0.01 * (i % 17) - 0.05;}
  p.has_gain = true;
  return p;
}

TEST(InnerLoop, OnThePlanAppliesThePlannedTorque)
{
  const Plan p = make_plan();
  for (double s : {0.0, 0.004, 0.015, 0.0299, 0.035}) {
    int j = std::min(static_cast<int>(s / 0.01), kPlanNodes - 1);
    const double w = (s - p.t[j]) / 0.01;
    Vec7 q, dq;
    for (int i = 0; i < NJ; ++i) {
      q[i] = (1 - w) * p.q[j][i] + w * p.q[j + 1][i];
      dq[i] = (1 - w) * p.dq[j][i] + w * p.dq[j + 1][i];
    }
    for (bool g : {true, false}) {
      const Vec7 tau = inner_torque(p, s, q, dq, g);
      for (int i = 0; i < NJ; ++i) {EXPECT_NEAR(tau[i], p.tau[j][i], 1e-9) << s;}
    }
  }
}

TEST(InnerLoop, GainActsOnTheDeviation)
{
  const Plan p = make_plan();
  for (int c = 0; c < kNx; ++c) {
    Vec7 q = p.q[1], dq = p.dq[1];
    const double e = 1e-3;
    (c < NJ ? q[c] : dq[c - NJ]) += e;
    const Vec7 tau = inner_torque(p, 0.01, q, dq, true);
    for (int i = 0; i < NJ; ++i) {EXPECT_NEAR(tau[i], p.tau[1][i] + p.gain[i * kNx + c] * e, 1e-12);}
  }
}

TEST(InnerLoop, FallsBackToTheFixedImpedance)
{
  Plan p = make_plan();
  p.has_gain = false;  // a command without a gain
  Vec7 q = p.q[0], dq = p.dq[0];
  q[2] += 0.01;
  dq[5] -= 0.1;
  const Vec7 tau = inner_torque(p, 0.0, q, dq, true);
  EXPECT_NEAR(tau[2], p.tau[0][2] - kImpedanceK[2] * 0.01, 1e-9);
  EXPECT_NEAR(tau[5], p.tau[0][5] + kImpedanceD[5] * 0.1, 1e-9);
}

TEST(InnerLoop, LateCommandKeepsToTheLastNode)
{
  const Plan p = make_plan();
  const Vec7 tau = inner_torque(p, 1.0, p.q[kPlanNodes], p.dq[kPlanNodes], true);
  for (int i = 0; i < NJ; ++i) {EXPECT_NEAR(tau[i], p.tau[kPlanNodes - 1][i], 1e-9);}
}
