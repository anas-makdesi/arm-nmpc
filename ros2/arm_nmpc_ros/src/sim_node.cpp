// Panda simulator node: rigid-body dynamics (generated ABA), joint friction and torque
// saturation, integrated in real time at 1 kHz; joint state published at the control rate.
// It also plays the robot's 1 kHz loop (inner_loop.hpp): each millisecond it applies the
// plan's torque plus feedback around the planned trajectory, either the NMPC's own gain
// (inner_loop "riccati", default: K sent with every command) or a fixed joint impedance
// (inner_loop "pd", gains from Franka's joint-impedance example). Starts at the ready pose
// and holds it until the first command arrives.
//
// mode "lockstep" (default): the simulator waits for the command answering each state,
// applies its feedforward torque for one control interval (as the Python simulation does)
// and publishes the next state, stamped with simulated time. Deterministic; the
// controller's timing is still measured by the NMPC node.
// mode "realtime": free-running 1 kHz loop with the inner loop above. With the fixed
// joint impedance, joints 1 and 3 swing in the arm's null space (up to 2x their velocity
// limits); with the NMPC's gain they stay within limits (see README).
//
// Publishes   /arm/joint_state  arm_nmpc_msgs/ArmState  (state_rate_hz)
// Subscribes  /arm/torque_cmd   arm_nmpc_msgs/JointTorques
// Parameters  state_rate_hz, friction (bool: simulate joint friction), mode,
//             inner_loop ("riccati" or "pd"; "riccati" falls back to "pd" if a command
//             carries no gain)

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include "arm_nmpc_msgs/msg/arm_state.hpp"
#include "arm_nmpc_msgs/msg/joint_torques.hpp"
#include "arm_nmpc_ros/arm_model.hpp"
#include "arm_nmpc_ros/inner_loop.hpp"
#include "rclcpp/rclcpp.hpp"

using arm_nmpc_msgs::msg::ArmState;
using arm_nmpc_msgs::msg::JointTorques;
using namespace std::chrono_literals;
namespace an = arm_nmpc;

class SimNode : public rclcpp::Node
{
public:
  SimNode()
  : Node("arm_sim")
  {
    const double rate = declare_parameter("state_rate_hz", 100.0);
    const bool fr = declare_parameter("friction", true);
    lockstep_ = declare_parameter("mode", std::string("lockstep")) == "lockstep";
    const auto inner = declare_parameter("inner_loop", std::string("riccati"));
    if (inner != "riccati" && inner != "pd") {
      throw std::invalid_argument("inner_loop must be \"riccati\" or \"pd\"");
    }
    riccati_ = inner == "riccati";
    dt_ctrl_ = 1.0 / rate;
    an::Vec7 q0;
    for (int i = 0; i < 7; ++i) {q0[i] = an::params::q_ready[i];}
    plant_ = std::make_unique<an::Plant>(q0, fr ? an::Friction::simulated() : an::Friction{});
    every_ = std::max(1, static_cast<int>(std::lround(1000.0 / rate)));
    auto qos = rclcpp::QoS(1).best_effort();
    pub_ = create_publisher<ArmState>("/arm/joint_state", qos);
    sub_ = create_subscription<JointTorques>(
      "/arm/torque_cmd", qos, [this](JointTorques::ConstSharedPtr m) {
        // lockstep: only the command that answers the latest state advances the simulation
        if (lockstep_ && rclcpp::Time(m->header.stamp) != last_stamp_) {return;}
        plan_stamp_ = m->header.stamp;
        for (int k = 0; k <= an::kPlanNodes; ++k) {
          plan_.t[k] = m->plan_time[k];
          for (int i = 0; i < 7; ++i) {
            plan_.q[k][i] = m->plan_position[7 * k + i];
            plan_.dq[k][i] = m->plan_velocity[7 * k + i];
            if (k < an::kPlanNodes) {plan_.tau[k][i] = m->plan_torque[7 * k + i];}
          }
        }
        std::copy(m->feedback_gain.begin(), m->feedback_gain.end(), plan_.gain.begin());
        plan_.has_gain = std::any_of(m->feedback_gain.begin(), m->feedback_gain.end(),
            [](double v) {return v != 0.0;});
        started_ = true;
        if (lockstep_) {
          an::Vec7 tau;
          for (int i = 0; i < 7; ++i) {tau[i] = m->torque[i];}
          plant_->step(tau, dt_ctrl_);
          ++k_ctrl_;
          publish_state();
        }
      });
    t_start_ = now();
    timer_ = create_wall_timer(1ms, [this]() {tick();});
    RCLCPP_INFO(get_logger(), "simulating the Panda, joint friction %s", fr ? "on" : "off");
  }

private:
  void publish_state()
  {
    ArmState m;
    // lockstep: simulated time (offset from the node's start); realtime: wall clock
    m.header.stamp = lockstep_ ? t_start_ + rclcpp::Duration::from_seconds(k_ctrl_ * dt_ctrl_)
      : now();
    for (int i = 0; i < 7; ++i) {m.position[i] = plant_->q()[i]; m.velocity[i] = plant_->dq()[i];}
    last_stamp_ = m.header.stamp;
    pub_->publish(m);
  }

  void tick()
  {
    if (lockstep_) {  // only (re)publish the start state until the controller answers
      if (!started_ && k_ % every_ == 0) {publish_state();}
      ++k_;
      return;
    }
    if (k_ % every_ == 0) {publish_state();}
    if (started_) {
      const double s = (now() - plan_stamp_).seconds();
      plant_->step(an::inner_torque(plan_, s, plant_->q(), plant_->dq(), riccati_), 1e-3);
    }
    ++k_;
  }

  an::Dynamics dyn_;
  std::unique_ptr<an::Plant> plant_;
  an::Plan plan_;
  rclcpp::Time plan_stamp_;
  long k_ = 0;
  int every_ = 10;
  bool started_ = false;
  bool lockstep_ = true;
  bool riccati_ = true;
  double dt_ctrl_ = 0.01;
  long k_ctrl_ = 0;
  rclcpp::Time t_start_, last_stamp_;
  rclcpp::Publisher<ArmState>::SharedPtr pub_;
  rclcpp::Subscription<JointTorques>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimNode>());
  rclcpp::shutdown();
  return 0;
}
