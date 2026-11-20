cmake_minimum_required(VERSION 3.16)
project(arm_nmpc_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmState.msg"
  "msg/JointTorques.msg"
  "msg/ControllerStats.msg"
  DEPENDENCIES std_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
