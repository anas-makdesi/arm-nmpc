// NMPC controller node for the Panda: tracks the TCP circle with one real-time iteration
// of the generated acados solver per joint-state message. No Python in the loop.
//
// Subscribes  /arm/joint_state  arm_nmpc_msgs/ArmState
// Publishes   /arm/torque_cmd   arm_nmpc_msgs/JointTorques (feedforward torque + plan)
//             /nmpc/stats       arm_nmpc_msgs/ControllerStats
// Parameters  period (circle period, s)
//             friction_fv, friction_fc (7 values each; e.g. the ddma-fitted model; run time)
//             lockstep (true: the simulator waits for each command; no delay compensation)
//             delay_ms (prediction horizon of the delay compensation; default: the
//             running average of this node's own step time)
//             feedback_gain (true: send the stage-0 Riccati gain of each solved QP, for the
//             robot's 1 kHz loop; default true)
// Time on the trajectory is counted from the first state message received.
//
// Delay compensation: the torque computed for a state reaches the motors a few
// milliseconds later. The node therefore predicts the state forward by that delay and
// solves the NMPC for the predicted state and time. The prediction integrates the model
// through the robot's own 1 kHz law (inner_loop.hpp) with the last command sent, i.e. with
// the torques the robot really applies over that time. Without delay compensation, this
// stiff tracking controller becomes unstable in the real-time loop.

#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#include "arm_nmpc_msgs/msg/arm_state.hpp"
#include "arm_nmpc_msgs/msg/controller_stats.hpp"
#include "arm_nmpc_msgs/msg/joint_torques.hpp"
#include "arm_nmpc_ros/inner_loop.hpp"
#include "arm_nmpc_ros/nmpc_solver.hpp"
#include "rclcpp/rclcpp.hpp"

using arm_nmpc_msgs::msg::ArmState;
using arm_nmpc_msgs::msg::ControllerStats;
using arm_nmpc_msgs::msg::JointTorques;
namespace an = arm_nmpc;

class NmpcNode : public rclcpp::Node
{
public:
  NmpcNode()
  : Node("nmpc"), solver_(make_circle(declare_parameter("period", 1.0)))
  {
    circle_ = make_circle(get_parameter("period").as_double());
    auto fv = declare_parameter("friction_fv", std::vector<double>(7, 0.0));
    auto fc = declare_parameter("friction_fc", std::vector<double>(7, 0.0));
    apply_friction(fv, fc);
    delay_param_ = declare_parameter("delay_ms", -1.0);
    lockstep_ = declare_parameter("lockstep", true);
    use_gain_ = declare_parameter("feedback_gain", true);
    solver_.set_compute_gain(use_gain_);
    if (lockstep_) {delay_param_ = 0.0;}  // the simulator waits: no delay to compensate
    param_cb_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & ps) {
        auto fv = get_parameter("friction_fv").as_double_array();
        auto fc = get_parameter("friction_fc").as_double_array();
        for (const auto & p : ps) {
          if (p.get_name() == "friction_fv") {fv = p.as_double_array();}
          if (p.get_name() == "friction_fc") {fc = p.as_double_array();}
        }
        rcl_interfaces::msg::SetParametersResult res;
        res.successful = fv.size() == 7 && fc.size() == 7;
        if (res.successful) {apply_friction(fv, fc);}
        return res;
      });
    auto qos = rclcpp::QoS(1).best_effort();
    cmd_pub_ = create_publisher<JointTorques>("/arm/torque_cmd", qos);
    stats_pub_ = create_publisher<ControllerStats>("/nmpc/stats", 10);
    sub_ = create_subscription<ArmState>(
      "/arm/joint_state", qos, [this](ArmState::ConstSharedPtr m) {on_state(*m);});
    RCLCPP_INFO(get_logger(), "NMPC ready: circle period %.2f s", circle_.period);
  }

private:
  static an::Circle make_circle(double T)
  {
    an::Circle c;
    c.period = T;
    return c;
  }

  void apply_friction(const std::vector<double> & fv, const std::vector<double> & fc)
  {
    an::Vec7 a, b;
    for (int i = 0; i < 7; ++i) {a[i] = fv[i]; b[i] = fc[i];}
    solver_.set_friction_model(a, b);
    fric_.fv = a;
    fric_.fc = b;
    fric_.eps = 0.3;  // as the NMPC model (anmpc.nmpc.NMPCConfig.eps)
  }

  // predict (q, dq) from time t (s after t0) forward by d seconds through the robot's 1 kHz
  // law with the last command sent (RK4, steps of at most 1 ms)
  void predict(an::Vec7 & q, an::Vec7 & dq, double t, double d)
  {
    const int n = std::max(1, static_cast<int>(std::ceil(d / 1e-3)));
    const double h = d / n;
    auto f = [&](double tt, const an::Vec7 & qq, const an::Vec7 & vv) {
        const an::Vec7 tau = an::inner_torque(last_plan_, tt - last_plan_t_, qq, vv, use_gain_);
        const an::Vec7 tf = fric_.torque(vv);
        an::Vec7 u;
        for (int i = 0; i < 7; ++i) {u[i] = tau[i] - tf[i];}
        return dyn_.forward(qq, vv, u);
      };
    for (int k = 0; k < n; ++k) {
      const double tk = t + k * h;
      const an::Vec7 a1 = f(tk, q, dq);
      an::Vec7 q2, v2, q3, v3, q4, v4;
      for (int i = 0; i < 7; ++i) {q2[i] = q[i] + h / 2 * dq[i]; v2[i] = dq[i] + h / 2 * a1[i];}
      const an::Vec7 a2 = f(tk + h / 2, q2, v2);
      for (int i = 0; i < 7; ++i) {q3[i] = q[i] + h / 2 * v2[i]; v3[i] = dq[i] + h / 2 * a2[i];}
      const an::Vec7 a3 = f(tk + h / 2, q3, v3);
      for (int i = 0; i < 7; ++i) {q4[i] = q[i] + h * v3[i]; v4[i] = dq[i] + h * a3[i];}
      const an::Vec7 a4 = f(tk + h, q4, v4);
      for (int i = 0; i < 7; ++i) {
        q[i] += h / 6 * (dq[i] + 2 * v2[i] + 2 * v3[i] + v4[i]);
        dq[i] += h / 6 * (a1[i] + 2 * a2[i] + 2 * a3[i] + a4[i]);
      }
    }
  }

  void on_state(const ArmState & m)
  {
    const auto c0 = std::chrono::steady_clock::now();
    const rclcpp::Time stamp(m.header.stamp);
    if (!t0_) {t0_ = stamp;}
    const double t = (stamp - *t0_).seconds();
    an::Vec7 q, dq;
    for (int i = 0; i < 7; ++i) {q[i] = m.position[i]; dq[i] = m.velocity[i];}
    const an::Vec7 q_meas = q, dq_meas = dq;
    const double d = 1e-3 * (delay_param_ >= 0 ? delay_param_ : delay_avg_ms_);
    if (started_ && d > 0) {predict(q, dq, t, d);}
    const double t_plan = t + (started_ ? d : 0.0);
    const auto res = solver_.step(t_plan, q, dq);

    JointTorques cmd;
    const rclcpp::Time t_pub = now();
    // the plan starts at the predicted state (state time + delay)
    cmd.header.stamp = stamp + rclcpp::Duration::from_seconds(started_ ? d : 0.0);
    const an::Plan & plan = res.plan;
    for (int i = 0; i < 7; ++i) {cmd.torque[i] = res.tau[i];}
    for (int k = 0; k <= an::kPlanNodes; ++k) {
      cmd.plan_time[k] = plan.t[k];
      for (int i = 0; i < 7; ++i) {
        cmd.plan_position[7 * k + i] = plan.q[k][i];
        cmd.plan_velocity[7 * k + i] = plan.dq[k][i];
        if (k < an::kPlanNodes) {cmd.plan_torque[7 * k + i] = plan.tau[k][i];}
      }
    }
    for (int i = 0; i < 7 * 14; ++i) {cmd.feedback_gain[i] = plan.gain[i];}
    cmd_pub_->publish(cmd);
    last_plan_ = res.plan;
    last_plan_t_ = t_plan;
    q = q_meas;
    dq = dq_meas;

    const an::Vec3 p = dyn_.tcp(q), pr = circle_.position(t);
    ControllerStats st;
    st.header = m.header;
    st.solve_time_ms = res.solve_time_ms;
    st.step_time_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
    // lockstep simulation stamps states with simulated time: the latency is then the step
    st.latency_ms = lockstep_ ? st.step_time_ms : (t_pub - stamp).seconds() * 1e3;
    // running average of the delay (only steady-state steps after the first one)
    if (started_) {delay_avg_ms_ = 0.9 * delay_avg_ms_ + 0.1 * st.latency_ms;}
    started_ = true;
    st.status = res.status;
    st.tcp_error = std::sqrt(std::pow(p[0] - pr[0], 2) + std::pow(p[1] - pr[1], 2) +
      std::pow(p[2] - pr[2], 2));
    double vr = 0;
    for (int i = 0; i < 7; ++i) {vr = std::max(vr, std::abs(dq[i]) / an::params::dq_max[i]);}
    st.max_velocity_ratio = vr;
    stats_pub_->publish(st);
  }

  an::NmpcSolver solver_;
  an::Circle circle_;
  an::Dynamics dyn_;
  an::Friction fric_;
  an::Plan last_plan_;  // last command sent, and its time (s after t0)
  double last_plan_t_ = 0.0;
  bool use_gain_ = true;
  double delay_param_ = -1.0, delay_avg_ms_ = 4.0;
  bool started_ = false, lockstep_ = true;
  std::optional<rclcpp::Time> t0_;
  rclcpp::Publisher<JointTorques>::SharedPtr cmd_pub_;
  rclcpp::Publisher<ControllerStats>::SharedPtr stats_pub_;
  rclcpp::Subscription<ArmState>::SharedPtr sub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NmpcNode>());
  rclcpp::shutdown();
  return 0;
}
