#include "arm_nmpc_ros/nmpc_solver.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_arm_nmpc.h"
}

namespace arm_nmpc
{

namespace
{
inline arm_nmpc_solver_capsule * cap(void * p) {return static_cast<arm_nmpc_solver_capsule *>(p);}
constexpr int N = ARM_NMPC_N;
constexpr int NX = 2 * NJ;
constexpr int NY = 3 + 3 * NJ;

// linear interpolation of a sampled signal (non-uniform grid), held at the ends
double interp(double x, const double * xs, const std::vector<double> & ys, int n, int stride,
  int idx)
{
  if (x <= xs[0]) {return ys[idx];}
  if (x >= xs[n - 1]) {return ys[(n - 1) * stride + idx];}
  int k = 0;
  while (xs[k + 1] < x) {++k;}
  const double w = (x - xs[k]) / (xs[k + 1] - xs[k]);
  return (1 - w) * ys[k * stride + idx] + w * ys[(k + 1) * stride + idx];
}
// small dense matrices, row-major
template<int R, int C>
using Mat = std::array<double, R * C>;

// C = op(A) * B with A (m x k) or A^T (k x m stored), B (k x n)
template<int M, int K, int N, bool TA = false>
void matmul(const double * A, const double * B, double * C)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double s = 0;
      for (int l = 0; l < K; ++l) {s += (TA ? A[l * M + i] : A[i * K + l]) * B[l * N + j];}
      C[i * N + j] = s;
    }
  }
}

// read a QP stage matrix (acados stores it column-major) into a row-major array;
// Q and R hold only their lower triangle and are mirrored
template<int R, int C>
void qp_matrix(void * solver, int stage, const char * field, Mat<R, C> & M, bool sym = false)
{
  double cm[R * C];
  ocp_nlp_get_at_stage(static_cast<ocp_nlp_solver *>(solver), stage, field, cm);
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      M[i * C + j] = (sym && j > i) ? cm[j + i * R] : cm[i + j * R];
    }
  }
}
}  // namespace

NmpcSolver::NmpcSolver(Circle circle)
: circle_(circle)
{
  static_assert(N == params::N, "solver and parameter header disagree");
  auto * c = arm_nmpc_acados_create_capsule();
  const int st = arm_nmpc_acados_create(c);
  if (st != 0) {
    arm_nmpc_acados_free_capsule(c);
    throw std::runtime_error("acados create failed with status " + std::to_string(st));
  }
  capsule_ = c;
  offsets_[0] = 0.0;
  for (int k = 0; k < N; ++k) {offsets_[k + 1] = offsets_[k] + params::time_steps[k];}
  Vec7 z{};
  set_friction_model(z, z);
}

NmpcSolver::~NmpcSolver()
{
  if (capsule_) {
    arm_nmpc_acados_free(cap(capsule_));
    arm_nmpc_acados_free_capsule(cap(capsule_));
  }
}

void NmpcSolver::set_friction_model(const Vec7 & fv, const Vec7 & fc)
{
  double p[2 * NJ];
  for (int i = 0; i < NJ; ++i) {p[i] = fv[i]; p[NJ + i] = fc[i];}
  for (int k = 0; k <= N; ++k) {arm_nmpc_acados_update_params(cap(capsule_), k, p, 2 * NJ);}
}

void NmpcSolver::set_x0(const double * x)
{
  auto * c = cap(capsule_);
  ocp_nlp_constraints_model_set(c->nlp_config, c->nlp_dims, c->nlp_in, c->nlp_out, 0, "lbx",
    const_cast<double *>(x));
  ocp_nlp_constraints_model_set(c->nlp_config, c->nlp_dims, c->nlp_in, c->nlp_out, 0, "ubx",
    const_cast<double *>(x));
}

void NmpcSolver::shift(double dt)
{
  if (dt <= 0) {return;}
  auto * c = cap(capsule_);
  std::vector<double> X((N + 1) * NX), U(N * NJ);
  for (int k = 0; k <= N; ++k) {
    ocp_nlp_out_get(c->nlp_config, c->nlp_dims, c->nlp_out, k, "x", &X[k * NX]);
    if (k < N) {ocp_nlp_out_get(c->nlp_config, c->nlp_dims, c->nlp_out, k, "u", &U[k * NJ]);}
  }
  const double t_end = offsets_[N];
  for (int k = 0; k <= N; ++k) {
    const double tn = std::min(offsets_[k] + dt, t_end);
    double xk[NX];
    for (int i = 0; i < NX; ++i) {xk[i] = interp(tn, offsets_, X, N + 1, NX, i);}
    ocp_nlp_out_set(c->nlp_config, c->nlp_dims, c->nlp_out, c->nlp_in, k, "x", xk);
    if (k < N) {
      const double tu = std::min(tn, offsets_[N - 1]);
      double uk[NJ];
      for (int i = 0; i < NJ; ++i) {uk[i] = interp(tu, offsets_, U, N, NJ, i);}
      ocp_nlp_out_set(c->nlp_config, c->nlp_dims, c->nlp_out, c->nlp_in, k, "u", uk);
    }
  }
}

// Backward Riccati recursion over the stage matrices of the QP just solved (the Gauss-Newton
// linearisation of this real-time iteration): K = -(R + B'PB)^-1 (S + B'PA) at stage 0.
// Exact while no inequality constraint is active. Same as anmpc.nmpc.NMPC.feedback_gain.
void NmpcSolver::feedback_gain(Gain & Kout)
{
  auto * solver = cap(capsule_)->nlp_solver;
  Mat<NX, NX> P, A, Q, T1, T2;
  Mat<NX, NJ> B, PB;
  Mat<NJ, NJ> Rm, H;
  Mat<NJ, NX> S, G, K;
  qp_matrix<NX, NX>(solver, N, "Q", P, true);
  for (int k = N - 1; k >= 0; --k) {
    qp_matrix<NX, NX>(solver, k, "A", A);
    qp_matrix<NX, NJ>(solver, k, "B", B);
    qp_matrix<NX, NX>(solver, k, "Q", Q, true);
    qp_matrix<NJ, NJ>(solver, k, "R", Rm, true);
    qp_matrix<NJ, NX>(solver, k, "S", S);
    matmul<NX, NX, NJ>(P.data(), B.data(), PB.data());                 // P B
    matmul<NJ, NX, NJ, true>(B.data(), PB.data(), H.data());           // B' P B
    matmul<NJ, NX, NX, true>(PB.data(), A.data(), G.data());           // B' P A
    for (int i = 0; i < NJ * NJ; ++i) {H[i] += Rm[i];}
    for (int i = 0; i < NJ * NX; ++i) {G[i] += S[i];}
    // Cholesky H = L L' (H is positive definite: R > 0 from the torque cost)
    Mat<NJ, NJ> L{};
    for (int j = 0; j < NJ; ++j) {
      double d = H[j * NJ + j];
      for (int l = 0; l < j; ++l) {d -= L[j * NJ + l] * L[j * NJ + l];}
      L[j * NJ + j] = std::sqrt(d);
      for (int i = j + 1; i < NJ; ++i) {
        double v = H[i * NJ + j];
        for (int l = 0; l < j; ++l) {v -= L[i * NJ + l] * L[j * NJ + l];}
        L[i * NJ + j] = v / L[j * NJ + j];
      }
    }
    // K = -H^-1 G, column by column
    for (int c = 0; c < NX; ++c) {
      double y[NJ];
      for (int i = 0; i < NJ; ++i) {
        double v = G[i * NX + c];
        for (int l = 0; l < i; ++l) {v -= L[i * NJ + l] * y[l];}
        y[i] = v / L[i * NJ + i];
      }
      for (int i = NJ - 1; i >= 0; --i) {
        double v = y[i];
        for (int l = i + 1; l < NJ; ++l) {v -= L[l * NJ + i] * K[l * NX + c];}
        K[i * NX + c] = v / L[i * NJ + i];
      }
      for (int i = 0; i < NJ; ++i) {K[i * NX + c] = -K[i * NX + c];}
    }
    // P = Q + A' P A + G' K
    matmul<NX, NX, NX>(P.data(), A.data(), T1.data());                 // P A
    matmul<NX, NX, NX, true>(A.data(), T1.data(), T2.data());          // A' P A
    matmul<NX, NJ, NX, true>(G.data(), K.data(), T1.data());           // G' K
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NX; ++j) {
        P[i * NX + j] = Q[i * NX + j] + T2[i * NX + j] + T1[i * NX + j];
      }
    }
    for (int i = 0; i < NX; ++i) {
      for (int j = i + 1; j < NX; ++j) {
        const double m = 0.5 * (P[i * NX + j] + P[j * NX + i]);
        P[i * NX + j] = P[j * NX + i] = m;
      }
    }
  }
  for (int i = 0; i < NJ * NX; ++i) {Kout[i] = K[i];}
}

SolveResult NmpcSolver::step(double t, const Vec7 & q, const Vec7 & dq)
{
  auto * c = cap(capsule_);
  // references: [p_ref, q_ready, 0, 0] per stage, [p_ref, 0] at the end
  double y[NY] = {0};
  for (int k = 0; k <= N; ++k) {
    const Vec3 p = circle_.position(t + offsets_[k]);
    for (int i = 0; i < 3; ++i) {y[i] = p[i];}
    if (k < N) {
      for (int i = 0; i < NJ; ++i) {y[3 + i] = params::q_ready[i];}
      for (int i = 3 + NJ; i < NY; ++i) {y[i] = 0.0;}
    } else {
      for (int i = 3; i < NY; ++i) {y[i] = 0.0;}
    }
    ocp_nlp_cost_model_set(c->nlp_config, c->nlp_dims, c->nlp_in, k, "yref", y);
  }
  double x0[NX];
  for (int i = 0; i < NJ; ++i) {x0[i] = q[i]; x0[NJ + i] = dq[i];}
  if (initialized_ && t_last_) {shift(t - *t_last_);}
  if (!initialized_) {
    const Vec7 g = dyn_.g(q);
    double gu[NJ];
    for (int i = 0; i < NJ; ++i) {gu[i] = g[i];}
    for (int k = 0; k <= N; ++k) {
      ocp_nlp_out_set(c->nlp_config, c->nlp_dims, c->nlp_out, c->nlp_in, k, "x", x0);
      if (k < N) {ocp_nlp_out_set(c->nlp_config, c->nlp_dims, c->nlp_out, c->nlp_in, k, "u", gu);}
    }
    t_last_.reset();
    for (int i = 0; i < 10; ++i) {
      set_x0(x0);
      arm_nmpc_acados_solve(c);
    }
    initialized_ = true;
  }
  set_x0(x0);
  SolveResult r;
  r.status = arm_nmpc_acados_solve(c);
  t_last_ = t;
  double tt = 0;
  ocp_nlp_get(c->nlp_solver, "time_tot", &tt);
  r.solve_time_ms = 1e3 * tt;
  double u[NJ];
  ocp_nlp_out_get(c->nlp_config, c->nlp_dims, c->nlp_out, 0, "u", u);
  for (int i = 0; i < NJ; ++i) {r.tau[i] = u[i];}
  static_assert(kPlanNodes <= N, "plan longer than the horizon");
  for (int k = 0; k <= kPlanNodes; ++k) {
    double xk[NX];
    ocp_nlp_out_get(c->nlp_config, c->nlp_dims, c->nlp_out, k, "x", xk);
    r.plan.t[k] = offsets_[k];
    for (int i = 0; i < NJ; ++i) {r.plan.q[k][i] = xk[i]; r.plan.dq[k][i] = xk[NJ + i];}
    if (k < kPlanNodes) {
      double uk[NJ];
      ocp_nlp_out_get(c->nlp_config, c->nlp_dims, c->nlp_out, k, "u", uk);
      for (int i = 0; i < NJ; ++i) {r.plan.tau[k][i] = uk[i];}
    }
  }
  if (compute_gain_) {
    const auto g0 = std::chrono::steady_clock::now();
    feedback_gain(r.plan.gain);
    r.plan.has_gain = true;
    r.gain_time_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count();
  }
  return r;
}

}  // namespace arm_nmpc
