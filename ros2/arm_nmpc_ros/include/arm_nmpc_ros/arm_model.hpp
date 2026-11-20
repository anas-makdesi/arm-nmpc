// Panda plant, friction and circle reference: C++ mirrors of anmpc.plant, anmpc.task
// (checked against Python in test/test_model.cpp). The rigid-body dynamics are the
// CasADi-generated functions of the Python package (generated/panda_dynamics.c).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "arm_params.hpp"

extern "C" {
#include "panda_dynamics.h"
}

namespace arm_nmpc
{

constexpr int NJ = 7;
using Vec7 = std::array<double, NJ>;
using Vec3 = std::array<double, 3>;

// Calls a CasADi-generated function with its own work memory.
class CasadiFn
{
public:
  using Eval = int (*)(const casadi_real **, casadi_real **, casadi_int *, casadi_real *, int);
  using Work = int (*)(casadi_int *, casadi_int *, casadi_int *, casadi_int *);
  CasadiFn(Eval f, Work w)
  : f_(f)
  {
    casadi_int a, r, i, ww;
    w(&a, &r, &i, &ww);
    arg_.resize(a);
    res_.resize(r);
    iw_.resize(i);
    w_.resize(ww);
  }
  void operator()(std::initializer_list<const double *> in, std::initializer_list<double *> out)
  {
    std::size_t k = 0;
    for (auto p : in) {arg_[k++] = p;}
    k = 0;
    for (auto p : out) {res_[k++] = p;}
    f_(arg_.data(), res_.data(), iw_.data(), w_.data(), 0);
  }

private:
  Eval f_;
  std::vector<const casadi_real *> arg_;
  std::vector<casadi_real *> res_;
  std::vector<casadi_int> iw_;
  std::vector<casadi_real> w_;
};

struct Dynamics
{
  CasadiFn aba{panda_aba, panda_aba_work};
  CasadiFn fk{panda_fk, panda_fk_work};
  CasadiFn gravity{panda_gravity, panda_gravity_work};

  Vec7 forward(const Vec7 & q, const Vec7 & dq, const Vec7 & tau)
  {
    Vec7 ddq;
    aba({q.data(), dq.data(), tau.data()}, {ddq.data()});
    return ddq;
  }
  Vec3 tcp(const Vec7 & q)
  {
    Vec3 p;
    fk({q.data()}, {p.data()});
    return p;
  }
  Vec7 g(const Vec7 & q)
  {
    Vec7 t;
    gravity({q.data()}, {t.data()});
    return t;
  }
};

struct Friction
{
  Vec7 fv{}, fc{};
  double eps = 0.05;
  Vec7 torque(const Vec7 & dq) const
  {
    Vec7 t;
    for (int i = 0; i < NJ; ++i) {t[i] = fv[i] * dq[i] + fc[i] * std::tanh(dq[i] / eps);}
    return t;
  }
  static Friction simulated()
  {
    Friction f;
    for (int i = 0; i < NJ; ++i) {
      f.fv[i] = params::sim_friction_fv[i];
      f.fc[i] = params::sim_friction_fc[i];
    }
    f.eps = params::sim_friction_eps;
    return f;
  }
};

// Simulated arm: ABA + joint friction + torque saturation, RK4 at dt_int.
class Plant
{
public:
  Plant(const Vec7 & q0, Friction fr, double dt_int = 1e-3)
  : q_(q0), fr_(fr), dt_int_(dt_int) {dq_.fill(0.0);}

  const Vec7 & q() const {return q_;}
  const Vec7 & dq() const {return dq_;}

  Vec7 step(const Vec7 & tau_cmd, double dt)
  {
    Vec7 tau;
    for (int i = 0; i < NJ; ++i) {
      tau[i] = std::clamp(tau_cmd[i], -params::tau_max[i], params::tau_max[i]);
    }
    const int n = std::max(1, static_cast<int>(std::lround(dt / dt_int_)));
    const double h = dt / n;
    auto f = [&](const Vec7 & q, const Vec7 & dq) {
        const Vec7 tf = fr_.torque(dq);
        Vec7 t;
        for (int i = 0; i < NJ; ++i) {t[i] = tau[i] - tf[i];}
        return dyn_.forward(q, dq, t);
      };
    auto add = [](const Vec7 & a, double c, const Vec7 & b) {
        Vec7 o;
        for (int i = 0; i < NJ; ++i) {o[i] = a[i] + c * b[i];}
        return o;
      };
    for (int it = 0; it < n; ++it) {
      const Vec7 k1q = dq_, k1v = f(q_, dq_);
      const Vec7 q2 = add(q_, h / 2, k1q), v2 = add(dq_, h / 2, k1v);
      const Vec7 k2q = v2, k2v = f(q2, v2);
      const Vec7 q3 = add(q_, h / 2, k2q), v3 = add(dq_, h / 2, k2v);
      const Vec7 k3q = v3, k3v = f(q3, v3);
      const Vec7 q4 = add(q_, h, k3q), v4 = add(dq_, h, k3v);
      const Vec7 k4q = v4, k4v = f(q4, v4);
      for (int i = 0; i < NJ; ++i) {
        q_[i] += h / 6 * (k1q[i] + 2 * k2q[i] + 2 * k3q[i] + k4q[i]);
        dq_[i] += h / 6 * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]);
      }
    }
    return tau;
  }

private:
  Vec7 q_, dq_;
  Friction fr_;
  double dt_int_;
  Dynamics dyn_;
};

// Vertical circle for the TCP with a smooth speed ramp from rest (anmpc.task.Circle).
struct Circle
{
  double period = 1.0;

  Vec3 position(double t) const
  {
    const double w = 2 * M_PI / period, T = params::circle_ramp;
    double a = 0.0;
    if (t >= 0) {a = t < T ? w * t * t / (2 * T) : w * (t - T / 2);}
    const double r = params::circle_radius;
    return {params::circle_center[0], params::circle_center[1] + r * std::sin(a),
      params::circle_center[2] + r * std::cos(a)};
  }
};

}  // namespace arm_nmpc
