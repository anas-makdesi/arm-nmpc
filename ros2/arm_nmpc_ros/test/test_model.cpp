// The generated dynamics, the C++ plant and the circle must reproduce the Python ones.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "arm_nmpc_ros/arm_model.hpp"

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

static Vec7 v7(const std::vector<double> & r, int o)
{
  Vec7 v;
  for (int i = 0; i < 7; ++i) {v[i] = r[o + i];}
  return v;
}

TEST(Dynamics, MatchesPython)
{
  Dynamics d;
  auto rows = read_csv(std::string(FIXTURES) + "/dynamics.csv");
  ASSERT_EQ(rows.size(), 20u);
  double e_ddq = 0, e_p = 0, e_g = 0;
  for (const auto & r : rows) {
    const Vec7 q = v7(r, 0), dq = v7(r, 7), tau = v7(r, 14);
    const Vec7 ddq = d.forward(q, dq, tau), g = d.g(q);
    const Vec3 p = d.tcp(q);
    for (int i = 0; i < 7; ++i) {
      e_ddq = std::max(e_ddq, std::abs(ddq[i] - r[21 + i]) / (1 + std::abs(r[21 + i])));
      e_g = std::max(e_g, std::abs(g[i] - r[31 + i]));
    }
    for (int i = 0; i < 3; ++i) {e_p = std::max(e_p, std::abs(p[i] - r[28 + i]));}
  }
  EXPECT_LT(e_ddq, 1e-10);
  EXPECT_LT(e_p, 1e-12);
  EXPECT_LT(e_g, 1e-10);
}

TEST(Plant, MatchesPythonWithFriction)
{
  auto rows = read_csv(std::string(FIXTURES) + "/plant_traj.csv");
  ASSERT_EQ(rows.size(), 100u);
  Plant p(v7(rows[0], 0), Friction::simulated());
  double worst = 0;
  for (const auto & r : rows) {
    p.step(v7(r, 14), 0.01);
    for (int i = 0; i < 7; ++i) {
      worst = std::max({worst, std::abs(p.q()[i] - r[21 + i]),
          std::abs(p.dq()[i] - r[28 + i]) / (1 + std::abs(r[28 + i]))});
    }
  }
  EXPECT_LT(worst, 1e-9);
}

TEST(Circle, StartsAtRestOnTheReadyPose)
{
  Dynamics d;
  Vec7 q;
  for (int i = 0; i < 7; ++i) {q[i] = params::q_ready[i];}
  const Vec3 p = d.tcp(q);
  Circle c;
  const Vec3 r0 = c.position(0.0), r1 = c.position(1e-3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(r0[i], p[i], 1e-12);
    EXPECT_NEAR(r1[i], p[i], 1e-6);  // speed ramps up from zero
  }
}
