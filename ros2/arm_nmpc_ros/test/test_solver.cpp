// The C++ wrapper must return the torques the Python NMPC returned for the same states.
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "arm_nmpc_ros/nmpc_solver.hpp"

using namespace arm_nmpc;

static std::vector<std::vector<double>> read_csv(const std::string & path)
{
  std::ifstream f(path);
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') {continue;}
    std::vector<double> r;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {r.push_back(std::stod(tok));}
    rows.push_back(r);
  }
  return rows;
}

static Circle circle1()
{
  Circle c;
  c.period = 1.0;
  return c;
}

TEST(Solver, ReproducesPythonClosedLoop)
{
  NmpcSolver s(circle1());
  auto rows = read_csv(std::string(FIXTURES) + "/solver_seq.csv");
  ASSERT_EQ(rows.size(), 150u);
  double worst = 0;
  for (const auto & r : rows) {
    Vec7 q, dq;
    for (int i = 0; i < 7; ++i) {q[i] = r[1 + i]; dq[i] = r[8 + i];}
    auto res = s.step(r[0], q, dq);
    EXPECT_EQ(res.status, 0);
    for (int i = 0; i < 7; ++i) {
      worst = std::max(worst, std::abs(res.tau[i] - r[15 + i]) / params::tau_max[i]);
    }
  }
  // same code and data; differences come from floating-point details (compiler flags)
  // carried through the warm start, and stay within the QP solver's tolerance
  EXPECT_LT(worst, 2e-4);
}

TEST(Solver, FitsTheRealTimeBudget)
{
  NmpcSolver s(circle1());
  auto rows = read_csv(std::string(FIXTURES) + "/solver_seq.csv");
  std::vector<double> ms;
  for (const auto & r : rows) {
    Vec7 q, dq;
    for (int i = 0; i < 7; ++i) {q[i] = r[1 + i]; dq[i] = r[8 + i];}
    auto t0 = std::chrono::steady_clock::now();
    s.step(r[0], q, dq);
    ms.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(ms.begin() + 1, ms.end());
  EXPECT_LT(ms[1 + (ms.size() - 1) / 2], 10.0);  // 100 Hz budget
}

TEST(Solver, TorquesWithinLimits)
{
  NmpcSolver s(circle1());
  auto rows = read_csv(std::string(FIXTURES) + "/solver_seq.csv");
  for (const auto & r : rows) {
    Vec7 q, dq;
    for (int i = 0; i < 7; ++i) {q[i] = r[1 + i]; dq[i] = r[8 + i];}
    auto res = s.step(r[0], q, dq);
    for (int i = 0; i < 7; ++i) {EXPECT_LE(std::abs(res.tau[i]), params::tau_max[i] + 1e-6);}
  }
}

TEST(Solver, FeedbackGainMatchesPython)
{
  NmpcSolver s(circle1());
  s.set_compute_gain(true);
  auto rows = read_csv(std::string(FIXTURES) + "/solver_seq.csv");
  auto gains = read_csv(std::string(FIXTURES) + "/solver_gain.csv");
  ASSERT_EQ(gains.size(), rows.size());
  double worst = 0, gain_ms = 0;
  for (std::size_t k = 0; k < rows.size(); ++k) {
    const auto & r = rows[k];
    Vec7 q, dq;
    for (int i = 0; i < 7; ++i) {q[i] = r[1 + i]; dq[i] = r[8 + i];}
    auto res = s.step(r[0], q, dq);
    ASSERT_EQ(gains[k].size(), res.plan.gain.size());
    double scale = 0, diff = 0;
    for (std::size_t i = 0; i < res.plan.gain.size(); ++i) {
      scale = std::max(scale, std::abs(gains[k][i]));
      diff = std::max(diff, std::abs(res.plan.gain[i] - gains[k][i]));
    }
    worst = std::max(worst, diff / scale);
    gain_ms = std::max(gain_ms, res.gain_time_ms);
  }
  // same recursion on the same QP; the QP itself differs within the solver's tolerance
  EXPECT_LT(worst, 1e-3);
  EXPECT_LT(gain_ms, 1.0);  // small next to the solve
}
