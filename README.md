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

The controller launch terminal also provides manual safety commands:

```text
1 = start the complete grasp sequence
o = open the gripper only
0 = cancel the active action and move HOME
p = cancel the active action and pause at the current position
s = show router state
h = show the menu
```

The same commands are available as one-shot topics:

```bash
ros2 topic pub --once /grasp/open_gripper_trigger std_msgs/msg/Bool "{data: true}"
ros2 topic pub --once /grasp/home_trigger std_msgs/msg/Bool "{data: true}"
ros2 topic pub --once /grasp/pause_trigger std_msgs/msg/Bool "{data: true}"
```

Pause cancels the active ROS action instead of preserving a trajectory index.
The next sequence and immediate-HOME command therefore start from current joint
feedback rather than accumulating an offset from the interrupted trajectory.

The sequence is:

```text
HOME -> RECOGNITION -> fresh stable desired 6D grasp pose
     -> common router -> OpenMANIPULATOR-X feasibility check

YES -> PRE-GRASP -> PICK -> CLOSE -> vertical RETREAT -> CARGO
NO  -> HOME -> RB3-730 dummy adapter (receives object + grasp 6D poses)
```

Monitor status with:

```bash
ros2 topic echo /grasp/control_status
ros2 topic echo /grasp/router_status
```

The OpenMANIPULATOR-X executor asks MoveIt to plan both equivalent parallel-jaw
PRE-GRASP orientations with configured orientation constraints. It computes
forward kinematics for each planned endpoint, selects the smallest valid
orientation residual, and plans a complete Cartesian line from that PRE-GRASP
endpoint to PICK. The verified trajectories are executed without replanning.
A failed constraint, residual, or Cartesian plan is reported as
`feasible=false`; no pick motion is executed and the router sends both the
original object pose and grasp pose to the RB3 dummy adapter on the new
`/grasp/rb3_dummy/candidate` topic. The existing
`/grasp/rb3_dummy/execute` topic remains available.

PRE-GRASP feasibility now searches from the table-normal vertical grasp toward
an oblique grasp in configured 15-degree steps. It stops at the first tilt
level with a valid plan, so the selected pose is as vertical as the 4-DOF arm
can reach. Tilting is about the gripper closing axis, preserving finger
alignment with the Duplo width. Planned FK is rejected if the approach does
not point downward, the closing axis drifts, or the EE PICK point lacks the
configured table clearance.

Immediately before evaluating the selected PICK request, the controller prints
the corresponding vision object pose once as base-frame XYZ and roll/pitch/yaw.
It does not continuously print the vision pose stream.

## Table-plane and camera-TF verification

The vision launch also publishes a depth ROI with the segmented object and its
boundary margin removed. A separate node fits this cloud in `base_link`, waits
for a stable multi-frame result, and then publishes:

```text
/grasp/table_cloud          sensor_msgs/msg/PointCloud2
/grasp/table_normal         geometry_msgs/msg/Vector3Stamped
/grasp/table_plane          grasp_interfaces/msg/TablePlane
/grasp/table_plane_marker   visualization_msgs/msg/MarkerArray
base_link -> table_frame    dynamic TF
```

The YOLO bounding-box camera view and `/grasp/debug_image` include the latest
object XYZ and roll/pitch/yaw transformed to `base_link`. The overlay is
cleared when segmentation disappears.

The table estimator does not print a periodic terminal stream. Its stable
height and inlier ratio are appended once to the existing `[PICK OBJECT 6D
POSE]` output immediately before PICK evaluation. Before changing object-pose
math, verify at several arm recognition poses that:

- table roll and pitch stay close to zero;
- table height stays close to the physical `base_link` table height (normally
  `z=0` for the current setup);
- the RViz table ROI, translucent plane, normal arrow, and `table_frame` remain
  coincident with the physical support surface.

If the base-frame plane changes with arm pose, calibrate the camera mount TF
first. Do not compensate for that error in the object pose. The image ROI,
depth range, object exclusion margin, RANSAC tolerance, and stability limits
are configurable in `src/grasp_vision/config/grasp_vision.yaml`.
Stable planes tilted more than five degrees from `base_link +Z` are not used
for object or grasp poses; this prevents a consistently wrong plane from
tilting an otherwise flat Duplo pose.

For Duplo detections, the filtered object cloud and table plane are matched by
timestamp. Visible height is compared with the fixed 63.1 x 31.0 x 23.1 mm
model to classify flat, side, or end placement. Yaw comes only from the cloud
projected into table XY; object roll/pitch come from the table normal rather
than a 3D PCA over mixed top and side surfaces. The box center is reconstructed
from the table contact center plus half the selected vertical dimension.
Ambiguous heights do not publish a grasp pose.

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
