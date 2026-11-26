# RGB-D Grasp System

ROS 2 Humble 기반 TurtleBot3 + OpenMANIPULATOR-X, RealSense D435i,
YOLO segmentation 및 PCA grasp-pose 추정을 결합한 파지 시스템입니다.

## Hardware interfaces

이 프로젝트는 `/home/user/eated_ws`에서 검증한 하드웨어 구성을 사용합니다.

- Planning frame: `base_link`
- End effector: `end_effector_link`
- Arm group: `arm`
- RGB: `/realsense/d435i/color/image_raw`
- Aligned depth: `/realsense/d435i/aligned_depth_to_color/image_raw`
- CameraInfo: `/realsense/d435i/color/camera_info`
- Camera optical frame: `d435i_color_optical_frame`
- Arm action: `/arm_controller/follow_joint_trajectory`
- Gripper action: `/gripper_controller/gripper_cmd`
- MoveIt action: `/move_action`

The eye-in-hand mount TF is published as:

```text
end_effector_link -> d435i_link
xyz = [-0.065, 0.0195, 0.052]
rpy = [0.0, 0.0, 0.0]
```

## Build

```bash
cd /home/user/grasp_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select grasp_control grasp_vision
source install/setup.bash
```

## Run

Run each command in a separate terminal and wait for the previous component to
be ready.

```bash
# 1. TurtleBot3 + OpenMANIPULATOR-X hardware
ros2 launch turtlebot3_manipulation_bringup hardware.launch.py \
  use_fake_hardware:=false start_rviz:=false

# 2. MoveIt
ros2 launch turtlebot3_manipulation_moveit_config move_group.launch.py \
  use_sim:=false

# 3. RealSense + eye-in-hand TF + grasp vision
ros2 launch grasp_vision grasp_vision.launch.py

# 4. Grasp controller
ros2 launch grasp_control grasp_control.launch.py
```

> **Safety:** `grasp_control.yaml` has `home_on_startup: true`. Starting the
> control node immediately commands the arm to the EATED HOME pose. Clear the
> robot workspace and keep the emergency stop available before starting it.

When the control node is idle, start one grasp attempt with:

```bash
ros2 topic pub --once /grasp/execute_trigger \
  std_msgs/msg/Bool "{data: true}"
```

The sequence is:

```text
HOME -> RECOGNITION -> fresh stable desired 6D grasp pose
     -> common router -> OpenMANIPULATOR-X feasibility check

YES -> PRE-GRASP -> PICK -> CLOSE -> vertical RETREAT -> CARGO
NO  -> HOME -> RB3-730 dummy adapter (logs the original 6D pose only)
```

Monitor status with:

```bash
ros2 topic echo /grasp/control_status
ros2 topic echo /grasp/router_status
```

The OpenMANIPULATOR-X executor asks MoveIt to plan PRE-GRASP and PICK with
`plan_only=true`, computes forward kinematics for the planned final joints, and
compares that reachable 4DOF pose with the desired 6D quaternion. The default
orientation residual limit is 20 degrees. A failed residual or plan is reported
as `feasible=false`; no pick motion is executed and the router calls the RB3
dummy adapter.

The dummy adapter is intentionally not a success simulation. It prints the
received frame, XYZ, and quaternion and reports `executed=false`, making it
safe to replace later with an RB3-730 MoveIt simulation adapter.

## Parameters requiring real-hardware verification

Edit the parameter files before the first pick test:

- `src/grasp_control/config/grasp_control.yaml`
- `src/grasp_vision/config/grasp_vision.yaml`

The initial workspace limit is:

```yaml
workspace_min_xyz_m: [0.05, -0.30, 0.01]
workspace_max_xyz_m: [0.45, 0.30, 0.45]
```

`pick_calibration_xyz_m` starts at `[0.0, 0.0, 0.0]`. The old EATED `+0.02 m`
X correction is intentionally not reused because the new segmentation/PCA
grasp point is different from the previous bounding-box-center point. Measure
the new error in RViz and at PRE-GRASP before changing this value.
