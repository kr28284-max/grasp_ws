"""Control the TurtleBot3 OpenMANIPULATOR-X grasp sequence."""

import copy
import math
import threading
import time
from collections import deque
from typing import Optional, Sequence

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from grasp_interfaces.msg import GraspExecutionResult, TablePlane
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, String
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint


class MotionInterrupted(RuntimeError):
    """Raised when a pause or immediate-HOME command cancels an action."""

    def __init__(self, mode: str) -> None:
        super().__init__(f'motion interrupted by {mode} request')
        self.mode = mode


class GraspControlNode(Node):
    """Run a guarded vision-to-pick sequence for OpenMANIPULATOR-X."""

    JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']

    def __init__(self) -> None:
        super().__init__('grasp_control_node')
        self.callback_group = ReentrantCallbackGroup()
        self.state_lock = threading.RLock()
        self.pose_condition = threading.Condition(self.state_lock)

        self._declare_parameters()
        self._read_parameters()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self, spin_thread=False
        )

        self.current_class = ''
        self.latest_camera_pose: Optional[PoseStamped] = None
        self.latest_base_pose: Optional[PoseStamped] = None
        self.latest_object_base_pose: Optional[PoseStamped] = None
        self.latest_table_plane: Optional[TablePlane] = None
        self.stable_base_pose: Optional[PoseStamped] = None
        self.stable_pose_time = 0.0
        self.pose_samples = deque(maxlen=self.stable_sample_count)
        self.current_joint_positions: Optional[list[float]] = None
        self.is_busy = False
        self.home_pending = False
        self.interrupt_mode: Optional[str] = None
        self.active_goal_handles = []

        self.create_subscription(
            String, '/grasp/target_class', self.class_cb, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            PoseStamped, '/grasp/target_grasp_pose', self.grasp_pose_cb, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            PoseStamped, '/grasp/object_pose', self.object_pose_cb, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            TablePlane, '/grasp/table_plane', self.table_plane_cb, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            Bool,
            '/grasp/openmanipulator/prepare',
            self.prepare_cb,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            PoseStamped,
            '/grasp/openmanipulator/execute',
            self.execute_cb,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Bool,
            '/grasp/openmanipulator/home',
            self.home_cb,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Bool,
            '/grasp/openmanipulator/open_gripper',
            self.open_gripper_cb,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Bool,
            '/grasp/openmanipulator/pause',
            self.pause_cb,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10,
            callback_group=self.callback_group
        )

        self.base_pose_pub = self.create_publisher(
            PoseStamped, '/grasp/base/target_grasp_pose', 10
        )
        self.base_object_pose_pub = self.create_publisher(
            PoseStamped, '/grasp/base/object_pose', 10
        )
        self.pre_grasp_pub = self.create_publisher(
            PoseStamped, '/grasp/base/pre_grasp_pose', 10
        )
        self.status_pub = self.create_publisher(
            String, '/grasp/control_status', 10
        )
        self.desired_pose_pub = self.create_publisher(
            PoseStamped, '/grasp/base/desired_grasp_pose', 10
        )
        self.ready_pub = self.create_publisher(
            Bool, '/grasp/openmanipulator/ready', 10
        )
        self.result_pub = self.create_publisher(
            GraspExecutionResult, '/grasp/openmanipulator/result', 10
        )

        self.arm_joint_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            callback_group=self.callback_group
        )
        self.gripper_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd',
            callback_group=self.callback_group
        )
        self.move_group_client = ActionClient(
            self, MoveGroup, '/move_action',
            callback_group=self.callback_group
        )
        self.cartesian_path_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path',
            callback_group=self.callback_group
        )
        self.compute_fk_client = self.create_client(
            GetPositionFK, '/compute_fk',
            callback_group=self.callback_group,
        )
        self.execute_trajectory_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory',
            callback_group=self.callback_group
        )

        self.get_logger().info(
            'TurtleBot3 OpenMANIPULATOR-X grasp controller initialized.'
        )
        if self.home_on_startup:
            with self.state_lock:
                self.is_busy = True
            threading.Thread(
                target=self._startup_home,
                name='startup-home',
                daemon=True,
            ).start()
        else:
            self.publish_status('IDLE: Waiting for execute trigger')

    def _declare_parameters(self) -> None:
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('move_group_name', 'arm')
        self.declare_parameter('end_effector_link', 'end_effector_link')
        self.declare_parameter('home_on_startup', True)
        self.declare_parameter('home_pose_deg', [0.0, -101.0, 53.0, 70.0])
        self.declare_parameter(
            'recognition_pose_deg', [0.0, -19.0, 0.0, 75.0]
        )
        self.declare_parameter(
            'cargo_pose_deg', [-73.0, -30.0, 38.0, -10.0]
        )
        self.declare_parameter('joint_motion_duration_sec', 4.0)
        self.declare_parameter('recognition_settle_sec', 0.5)
        self.declare_parameter('gripper_open_position', 0.019)
        self.declare_parameter('gripper_closed_position', -0.010)
        self.declare_parameter('gripper_max_effort', 0.2)
        self.declare_parameter('grasp_position_threshold', -0.008)
        self.declare_parameter('pick_calibration_xyz_m', [0.0, 0.0, 0.0])
        self.declare_parameter('pregrasp_offset_xyz_m', [0.0, 0.0, 0.05])
        self.declare_parameter('retreat_offset_z_m', 0.15)
        self.declare_parameter(
            'pick_orientation_xyzw',
            [0.0, 0.7071068, 0.0, 0.7071068],
        )
        self.declare_parameter('position_tolerance_m', 0.01)
        self.declare_parameter('moveit_planning_time_sec', 3.0)
        self.declare_parameter('moveit_velocity_scale', 0.10)
        self.declare_parameter('moveit_acceleration_scale', 0.10)
        self.declare_parameter('approach_cartesian_step_m', 0.005)
        self.declare_parameter('approach_joint_jump_rad', 0.35)
        self.declare_parameter('retreat_cartesian_step_m', 0.005)
        self.declare_parameter('retreat_joint_jump_rad', 0.35)
        self.declare_parameter('action_server_timeout_sec', 10.0)
        self.declare_parameter('action_result_timeout_sec', 20.0)
        self.declare_parameter('pose_wait_timeout_sec', 10.0)
        self.declare_parameter('pose_max_age_sec', 0.5)
        self.declare_parameter('tf_timeout_sec', 0.2)
        self.declare_parameter('stable_sample_count', 5)
        self.declare_parameter('max_position_stddev_m', 0.005)
        self.declare_parameter('workspace_min_xyz_m', [0.05, -0.30, 0.01])
        self.declare_parameter('workspace_max_xyz_m', [0.45, 0.30, 0.45])
        self.declare_parameter('return_home_on_error', True)
        self.declare_parameter('max_orientation_residual_deg', 20.0)

    def _read_parameters(self) -> None:
        def value(name: str):
            return self.get_parameter(name).value

        self.base_frame = str(value('base_frame'))
        self.move_group_name = str(value('move_group_name'))
        self.end_effector_link = str(value('end_effector_link'))
        self.home_on_startup = bool(value('home_on_startup'))
        self.home_pose = self._degrees_pose(value('home_pose_deg'))
        self.recognition_pose = self._degrees_pose(value('recognition_pose_deg'))
        self.cargo_pose = self._degrees_pose(value('cargo_pose_deg'))
        self.joint_motion_duration = float(value('joint_motion_duration_sec'))
        self.recognition_settle = float(value('recognition_settle_sec'))
        self.gripper_open = float(value('gripper_open_position'))
        self.gripper_closed = float(value('gripper_closed_position'))
        self.gripper_effort = float(value('gripper_max_effort'))
        self.grasp_position_threshold = float(
            value('grasp_position_threshold')
        )
        self.pick_calibration = self._xyz(value('pick_calibration_xyz_m'))
        self.pregrasp_offset = self._xyz(value('pregrasp_offset_xyz_m'))
        self.retreat_offset_z = float(value('retreat_offset_z_m'))
        self.pick_orientation = self._quaternion(
            value('pick_orientation_xyzw')
        )
        self.position_tolerance = float(value('position_tolerance_m'))
        self.moveit_planning_time = float(value('moveit_planning_time_sec'))
        self.velocity_scale = float(value('moveit_velocity_scale'))
        self.acceleration_scale = float(value('moveit_acceleration_scale'))
        self.approach_step = float(value('approach_cartesian_step_m'))
        self.approach_joint_jump = float(value('approach_joint_jump_rad'))
        self.retreat_step = float(value('retreat_cartesian_step_m'))
        self.retreat_joint_jump = float(value('retreat_joint_jump_rad'))
        self.server_timeout = float(value('action_server_timeout_sec'))
        self.result_timeout = float(value('action_result_timeout_sec'))
        self.pose_wait_timeout = float(value('pose_wait_timeout_sec'))
        self.pose_max_age = float(value('pose_max_age_sec'))
        self.tf_timeout = float(value('tf_timeout_sec'))
        self.stable_sample_count = int(value('stable_sample_count'))
        self.max_position_stddev = float(value('max_position_stddev_m'))
        self.workspace_min = self._xyz(value('workspace_min_xyz_m'))
        self.workspace_max = self._xyz(value('workspace_max_xyz_m'))
        self.return_home_on_error = bool(value('return_home_on_error'))
        self.max_orientation_residual = math.radians(
            float(value('max_orientation_residual_deg'))
        )

        if self.stable_sample_count < 1:
            raise ValueError('stable_sample_count must be positive')
        if not (
            self.gripper_closed
            < self.grasp_position_threshold
            < self.gripper_open
        ):
            raise ValueError(
                'grasp_position_threshold must be between closed and open'
            )
        if any(
            low >= high
            for low, high in zip(self.workspace_min, self.workspace_max)
        ):
            raise ValueError('workspace minimums must be lower than maximums')
        if not 0.0 < self.velocity_scale <= 1.0:
            raise ValueError('moveit_velocity_scale must be in (0, 1]')
        if not 0.0 < self.acceleration_scale <= 1.0:
            raise ValueError('moveit_acceleration_scale must be in (0, 1]')

    @staticmethod
    def _degrees_pose(values: Sequence[float]) -> list[float]:
        if len(values) != 4:
            raise ValueError('joint poses must contain four values')
        return [math.radians(float(item)) for item in values]

    @staticmethod
    def _xyz(values: Sequence[float]) -> tuple[float, float, float]:
        if len(values) != 3:
            raise ValueError('XYZ parameters must contain three values')
        return tuple(float(item) for item in values)

    @staticmethod
    def _quaternion(
        values: Sequence[float],
    ) -> tuple[float, float, float, float]:
        if len(values) != 4:
            raise ValueError('quaternion parameters must contain four values')
        quaternion = np.asarray(values, dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-9:
            raise ValueError('pick quaternion must not be zero')
        return tuple(float(item) for item in quaternion / norm)

    def publish_status(self, text: str) -> None:
        self.status_pub.publish(String(data=text))
        self.get_logger().info(f'[STATUS] {text}')

    def class_cb(self, msg: String) -> None:
        with self.pose_condition:
            self.current_class = msg.data.strip()
            if not self.current_class:
                self._clear_pose_locked()

    def joint_state_cb(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position))
        if all(name in positions for name in self.JOINT_NAMES):
            with self.state_lock:
                self.current_joint_positions = [
                    float(positions[name]) for name in self.JOINT_NAMES
                ]

    def grasp_pose_cb(self, msg: PoseStamped) -> None:
        if not msg.header.frame_id:
            self.get_logger().warn('Rejected grasp pose with an empty frame_id')
            return
        self.latest_camera_pose = msg
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=self.tf_timeout),
            )
            base_pose = do_transform_pose_stamped(msg, transform)
            base_pose.header.frame_id = self.base_frame
            self.base_pose_pub.publish(base_pose)
            self._record_pose_sample(base_pose)
        except TransformException as exc:
            self.get_logger().warn(
                f'TF unavailable for {msg.header.frame_id} -> '
                f'{self.base_frame}: {exc}',
                throttle_duration_sec=2.0,
            )

    def object_pose_cb(self, msg: PoseStamped) -> None:
        """Cache the vision object pose without producing a stream of logs."""
        if not msg.header.frame_id:
            self.get_logger().warn('Rejected object pose with an empty frame_id')
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=self.tf_timeout),
            )
            base_pose = do_transform_pose_stamped(msg, transform)
            base_pose.header.frame_id = self.base_frame
            with self.state_lock:
                self.latest_object_base_pose = base_pose
            self.base_object_pose_pub.publish(base_pose)
        except TransformException as exc:
            self.get_logger().warn(
                f'Object-pose TF unavailable for {msg.header.frame_id} -> '
                f'{self.base_frame}: {exc}',
                throttle_duration_sec=2.0,
            )

    def table_plane_cb(self, msg: TablePlane) -> None:
        """Cache stable table metrics for the one-shot PICK pose log."""
        with self.state_lock:
            self.latest_table_plane = msg

    def _record_pose_sample(self, pose: PoseStamped) -> None:
        point = np.array([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ])
        if not np.isfinite(point).all():
            return

        with self.pose_condition:
            self.latest_base_pose = pose
            self.pose_samples.append(point)
            if len(self.pose_samples) < self.stable_sample_count:
                return
            samples = np.asarray(self.pose_samples)
            if float(np.max(np.std(samples, axis=0))) > self.max_position_stddev:
                return

            median = np.median(samples, axis=0)
            stable = PoseStamped()
            stable.header = pose.header
            stable.pose = copy.deepcopy(pose.pose)
            stable.pose.position.x = float(median[0])
            stable.pose.position.y = float(median[1])
            stable.pose.position.z = float(median[2])
            self.stable_base_pose = stable
            self.stable_pose_time = time.monotonic()
            self.pose_condition.notify_all()
            self.desired_pose_pub.publish(stable)
            self.pre_grasp_pub.publish(
                self._offset_pose(stable, self.pregrasp_offset)
            )

    def _clear_pose_locked(self) -> None:
        self.latest_camera_pose = None
        self.latest_base_pose = None
        self.stable_base_pose = None
        self.stable_pose_time = 0.0
        self.latest_object_base_pose = None
        self.pose_samples.clear()

    def prepare_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        with self.state_lock:
            if self.is_busy or self.home_pending:
                self.get_logger().warn('Controller is busy; prepare ignored.')
                self.ready_pub.publish(Bool(data=False))
                return
            self.interrupt_mode = None
            self.is_busy = True
        threading.Thread(
            target=self._prepare_for_candidate,
            name='prepare-recognition',
            daemon=True,
        ).start()

    def execute_cb(self, desired_pose: PoseStamped) -> None:
        with self.state_lock:
            if self.is_busy or self.home_pending:
                self.get_logger().warn('Controller is busy; execute ignored.')
                self._publish_result(
                    desired_pose,
                    desired_pose,
                    feasible=False,
                    executed=False,
                    success=False,
                    orientation_residual=math.inf,
                    message='OpenMANIPULATOR-X executor is busy.',
                )
                return
            self.interrupt_mode = None
            self.is_busy = True
            object_pose = copy.deepcopy(self.latest_object_base_pose)
            table_plane = copy.deepcopy(self.latest_table_plane)
        threading.Thread(
            target=self._execute_candidate,
            args=(desired_pose, object_pose, table_plane),
            name='openmanipulator-execution',
            daemon=True,
        ).start()

    def home_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        with self.state_lock:
            if self.home_pending:
                self.get_logger().warn('HOME is already pending; request ignored.')
                return
            self.home_pending = True
        threading.Thread(
            target=self._interrupt_and_home,
            name='interrupt-and-home',
            daemon=True,
        ).start()

    def open_gripper_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        with self.state_lock:
            if self.is_busy or self.home_pending:
                self.get_logger().warn(
                    'Controller is busy; gripper OPEN ignored.'
                )
                return
            self.interrupt_mode = None
            self.is_busy = True
        threading.Thread(
            target=self._command_open_gripper,
            name='command-open-gripper',
            daemon=True,
        ).start()

    def pause_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        self.publish_status('PAUSE: Cancelling active robot action')
        self._request_interrupt('pause')

    def _startup_home(self) -> None:
        try:
            self.publish_status('STARTUP: Moving to HOME')
            self._send_joint_pose(self.home_pose, 'HOME')
            self.publish_status('IDLE: HOME reached; waiting for execute trigger')
        except MotionInterrupted as exc:
            self._publish_interrupted_status(exc.mode)
        except Exception as exc:
            self.publish_status(f'ERROR: Startup HOME failed: {exc}')
        finally:
            self._release_busy()

    def _interrupt_and_home(self) -> None:
        self.publish_status('HOME: Cancelling active robot action')
        self._request_interrupt('home')
        with self.pose_condition:
            while self.is_busy:
                self.pose_condition.wait(timeout=0.1)
            self.interrupt_mode = None
            self.is_busy = True
        try:
            self.publish_status('HOME: Returning to HOME pose')
            self._send_joint_pose(self.home_pose, 'HOME')
            self.publish_status('IDLE: HOME reached')
        except MotionInterrupted as exc:
            self._publish_interrupted_status(exc.mode)
        except Exception as exc:
            self.publish_status(f'ERROR: HOME failed: {exc}')
        finally:
            with self.pose_condition:
                self.is_busy = False
                self.home_pending = False
                self.pose_condition.notify_all()

    def _command_open_gripper(self) -> None:
        try:
            self.publish_status('GRIPPER: Opening gripper')
            self._send_gripper(self.gripper_open, 'OPEN')
            self.publish_status('IDLE: Gripper is open')
        except MotionInterrupted as exc:
            self._publish_interrupted_status(exc.mode)
        except Exception as exc:
            self.publish_status(f'ERROR: Gripper OPEN failed: {exc}')
        finally:
            self._release_busy()

    def _prepare_for_candidate(self) -> None:
        ready = False
        try:
            self.publish_status('PREPARE: Moving to RECOGNITION pose')
            self._send_joint_pose(self.recognition_pose, 'RECOGNITION')
            time.sleep(self.recognition_settle)
            self._check_interrupted()
            with self.pose_condition:
                self._clear_pose_locked()
            ready = True
            self.publish_status(
                'READY: RECOGNITION reached; collecting fresh pose samples'
            )
        except MotionInterrupted as exc:
            self._publish_interrupted_status(exc.mode)
        except Exception as exc:
            self.publish_status(f'ERROR: RECOGNITION failed: {exc}')
        finally:
            self._release_busy()
            self.ready_pub.publish(Bool(data=ready))

    def _execute_candidate(
        self,
        desired_pose: PoseStamped,
        object_pose: Optional[PoseStamped] = None,
        table_plane: Optional[TablePlane] = None,
    ) -> None:
        result_published = False
        command_pose = copy.deepcopy(desired_pose)
        orientation_residual = math.inf
        try:
            self._log_object_pose_once(object_pose, table_plane)
            candidates = self._vision_grasp_to_ee_candidates(
                desired_pose.pose.orientation
            )
            valid_plans = []
            for index, orientation in enumerate(candidates, start=1):
                candidate_pick = self._pose_with_orientation(
                    desired_pose, orientation
                )
                candidate_pick = self._offset_pose(
                    candidate_pick, self.pick_calibration
                )
                self._validate_workspace(candidate_pick)
                candidate_pregrasp = self._offset_pose(
                    candidate_pick, self.pregrasp_offset
                )
                self._validate_workspace(candidate_pregrasp)
                self.publish_status(
                    f'CHECK: Planning PRE-GRASP candidate {index}/2 '
                    'without execution'
                )
                plan = self._move_group_pose(
                    candidate_pregrasp,
                    f'PRE-GRASP CANDIDATE {index}',
                    plan_only=True,
                )
                if plan is None:
                    continue
                planned_pose = self._planned_command_pose(plan)
                residual = self._orientation_error(
                    orientation, planned_pose.pose.orientation
                )
                self.get_logger().info(
                    f'PRE-GRASP candidate {index} FK orientation residual: '
                    f'{math.degrees(residual):.1f} deg'
                )
                if residual <= self.max_orientation_residual:
                    valid_plans.append((
                        residual,
                        index,
                        plan,
                        planned_pose,
                        candidate_pick,
                        orientation,
                    ))
                else:
                    self.get_logger().warn(
                        f'PRE-GRASP candidate {index} rejected: FK residual '
                        f'{math.degrees(residual):.1f} deg exceeds '
                        f'{math.degrees(self.max_orientation_residual):.1f} deg'
                    )

            if not valid_plans:
                raise ValueError(
                    'no PRE-GRASP orientation candidate passed planning and '
                    f'the {math.degrees(self.max_orientation_residual):.1f} deg '
                    'FK residual limit'
                )

            (
                orientation_residual,
                selected_index,
                pregrasp_plan,
                pregrasp_command_pose,
                pick,
                selected_orientation,
            ) = min(valid_plans, key=lambda candidate: candidate[0])
            self.publish_status(
                f'CHECK: Selected PRE-GRASP candidate {selected_index} '
                f'(FK residual={math.degrees(orientation_residual):.1f} deg)'
            )
            self.publish_status(
                'CHECK: Planning Cartesian PICK approach without execution'
            )
            pick_trajectory = self._plan_cartesian_approach(
                pregrasp_plan,
                pregrasp_command_pose,
                pick,
            )

            command_pose = self._planned_trajectory_pose(pick_trajectory)
            orientation_residual = self._orientation_error(
                selected_orientation, command_pose.pose.orientation
            )
            if orientation_residual > self.max_orientation_residual:
                reason = (
                    'planned 4DOF orientation exceeds the residual limit: '
                    f'{math.degrees(orientation_residual):.1f} deg > '
                    f'{math.degrees(self.max_orientation_residual):.1f} deg'
                )
                self.publish_status(f'INFEASIBLE: {reason}')
                self._safe_return_home('INFEASIBLE')
                self._publish_result(
                    desired_pose,
                    command_pose,
                    feasible=False,
                    executed=False,
                    success=False,
                    orientation_residual=orientation_residual,
                    message=reason,
                )
                result_published = True
                return

            self.publish_status('EXECUTE: Opening gripper')
            self._send_gripper(self.gripper_open, 'OPEN')
            self.publish_status('EXECUTE: Running verified PRE-GRASP trajectory')
            self._execute_trajectory(
                pregrasp_plan.result.planned_trajectory,
                'PRE-GRASP',
            )
            self.publish_status('EXECUTE: Running verified PICK trajectory')
            self._execute_trajectory(pick_trajectory, 'PICK')
            self.publish_status('EXECUTE: Closing gripper')
            self._send_gripper(self.gripper_closed, 'CLOSE', accept_stall=True)
            self.publish_status('EXECUTE: Vertical retreat')
            self._send_vertical_retreat()
            self.publish_status('EXECUTE: Moving to CARGO pose')
            self._send_joint_pose(self.cargo_pose, 'CARGO')
            self.publish_status('SUCCESS: OpenMANIPULATOR-X pick completed')
            self._publish_result(
                desired_pose,
                command_pose,
                feasible=True,
                executed=True,
                success=True,
                orientation_residual=orientation_residual,
                message='OpenMANIPULATOR-X pick completed.',
            )
            result_published = True
        except MotionInterrupted as exc:
            self._publish_interrupted_status(exc.mode)
            self._publish_result(
                desired_pose,
                command_pose,
                feasible=False,
                executed=True,
                success=False,
                orientation_residual=orientation_residual,
                message=str(exc),
            )
            result_published = True
        except ValueError as exc:
            self.publish_status(f'INFEASIBLE: {exc}')
            self._safe_return_home('INFEASIBLE')
            self._publish_result(
                desired_pose,
                command_pose,
                feasible=False,
                executed=False,
                success=False,
                orientation_residual=orientation_residual,
                message=str(exc),
            )
            result_published = True
        except Exception as exc:
            self.publish_status(f'ERROR: OpenMANIPULATOR-X execution failed: {exc}')
            if self.return_home_on_error:
                try:
                    self.publish_status('RECOVERY: Returning HOME')
                    self._send_joint_pose(self.home_pose, 'HOME')
                except Exception as home_exc:
                    self.publish_status(f'ERROR: HOME recovery failed: {home_exc}')
            if not result_published:
                self._publish_result(
                    desired_pose,
                    command_pose,
                    feasible=True,
                    executed=True,
                    success=False,
                    orientation_residual=orientation_residual,
                    message=str(exc),
                )
                result_published = True
        finally:
            self._release_busy()

    def _release_busy(self) -> None:
        with self.pose_condition:
            self.is_busy = False
            self.pose_condition.notify_all()

    def _request_interrupt(self, mode: str) -> None:
        with self.state_lock:
            self.interrupt_mode = mode
            goal_handles = list(self.active_goal_handles)
        for goal_handle in goal_handles:
            goal_handle.cancel_goal_async()
        if not goal_handles and mode == 'pause':
            self.publish_status(
                'PAUSED: No active action; waiting for the next trigger'
            )

    def _check_interrupted(self) -> None:
        with self.state_lock:
            mode = self.interrupt_mode
        if mode is not None:
            raise MotionInterrupted(mode)

    def _publish_interrupted_status(self, mode: str) -> None:
        if mode == 'pause':
            self.publish_status(
                'PAUSED: Robot action cancelled; holding current position'
            )
        else:
            self.publish_status(
                'HOME: Previous action cancelled; HOME command pending'
            )

    def _safe_return_home(self, reason: str) -> None:
        try:
            self.publish_status(f'{reason}: Returning HOME')
            self._send_joint_pose(self.home_pose, 'HOME')
        except Exception as exc:
            self.publish_status(f'ERROR: HOME after {reason} failed: {exc}')

    @staticmethod
    def _pose_with_orientation(source: PoseStamped, orientation) -> PoseStamped:
        result = copy.deepcopy(source)
        result.pose.orientation = copy.deepcopy(orientation)
        return result

    @staticmethod
    def _quaternion_to_rpy(quaternion) -> tuple[float, float, float]:
        values = np.asarray(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(values))
        if norm < 1e-9:
            raise ValueError('object-pose quaternion must not be zero')
        x, y, z, w = values / norm
        roll = math.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(pitch_term)
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        return roll, pitch, yaw

    def _log_object_pose_once(
        self,
        object_pose: Optional[PoseStamped],
        table_plane: Optional[TablePlane] = None,
    ) -> None:
        if object_pose is None:
            self.get_logger().warn(
                'PICK object 6D pose unavailable; no object-pose line printed.'
            )
            return
        roll, pitch, yaw = self._quaternion_to_rpy(
            object_pose.pose.orientation
        )
        position = object_pose.pose.position
        table_text = 'table_height=unavailable, inlier_ratio=unavailable'
        if table_plane is not None:
            table_text = (
                f'table_height={table_plane.height:+.4f} m, '
                f'inlier_ratio={table_plane.inlier_ratio:.1%}'
            )
        self.get_logger().info(
            '[PICK OBJECT 6D POSE] '
            f'frame={object_pose.header.frame_id}, '
            f'x={position.x:.4f} m, y={position.y:.4f} m, '
            f'z={position.z:.4f} m, '
            f'roll={math.degrees(roll):.2f} deg, '
            f'pitch={math.degrees(pitch):.2f} deg, '
            f'yaw={math.degrees(yaw):.2f} deg, '
            f'{table_text}'
        )

    @staticmethod
    def _orientation_error(first, second) -> float:
        first_array = np.array([first.x, first.y, first.z, first.w])
        second_array = np.array([second.x, second.y, second.z, second.w])
        first_norm = float(np.linalg.norm(first_array))
        second_norm = float(np.linalg.norm(second_array))
        if first_norm < 1e-9 or second_norm < 1e-9:
            return math.inf
        dot = abs(float(np.dot(
            first_array / first_norm,
            second_array / second_norm,
        )))
        return 2.0 * math.acos(min(1.0, max(-1.0, dot)))

    @staticmethod
    def _quaternion_product(first, second) -> Quaternion:
        """Return the Hamilton product of two normalized XYZW quaternions."""
        first_array = np.asarray(
            [first.x, first.y, first.z, first.w], dtype=np.float64
        )
        second_array = np.asarray(
            [second.x, second.y, second.z, second.w], dtype=np.float64
        )
        first_norm = float(np.linalg.norm(first_array))
        second_norm = float(np.linalg.norm(second_array))
        if first_norm < 1e-9 or second_norm < 1e-9:
            raise ValueError('grasp quaternion must not be zero')
        x1, y1, z1, w1 = first_array / first_norm
        x2, y2, z2, w2 = second_array / second_norm
        product = np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ])
        product /= np.linalg.norm(product)
        return Quaternion(
            x=float(product[0]),
            y=float(product[1]),
            z=float(product[2]),
            w=float(product[3]),
        )

    @classmethod
    def _vision_grasp_to_ee_candidates(cls, grasp) -> tuple[Quaternion, Quaternion]:
        """
        Map vision axes to EE axes and include parallel-jaw symmetry.

        Vision uses +Z for approach and +X for finger closing.  The
        OpenMANIPULATOR-X end effector uses +X and +Y respectively, so its
        rotation columns are [G_z, G_x, G_y].  A 180-degree local +X turn is
        an equivalent grasp for a parallel-jaw gripper.
        """
        axis_mapping = Quaternion(x=-0.5, y=-0.5, z=-0.5, w=0.5)
        converted = cls._quaternion_product(grasp, axis_mapping)
        flipped = cls._quaternion_product(
            converted, Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
        )
        return converted, flipped

    def _publish_result(
        self,
        desired_pose: PoseStamped,
        command_pose: PoseStamped,
        *,
        feasible: bool,
        executed: bool,
        success: bool,
        orientation_residual: float,
        message: str,
    ) -> None:
        result = GraspExecutionResult()
        result.header = desired_pose.header
        result.robot_name = 'openmanipulator_x'
        result.feasible = feasible
        result.executed = executed
        result.success = success
        result.desired_pose = desired_pose
        result.command_pose = command_pose
        delta = np.array([
            desired_pose.pose.position.x - command_pose.pose.position.x,
            desired_pose.pose.position.y - command_pose.pose.position.y,
            desired_pose.pose.position.z - command_pose.pose.position.z,
        ])
        result.position_residual_m = float(np.linalg.norm(delta))
        result.orientation_residual_rad = orientation_residual
        result.message = message
        self.result_pub.publish(result)

    @staticmethod
    def _offset_pose(
        source: PoseStamped,
        offset: Sequence[float],
    ) -> PoseStamped:
        result = PoseStamped()
        result.header = source.header
        result.pose = copy.deepcopy(source.pose)
        result.pose.position.x += float(offset[0])
        result.pose.position.y += float(offset[1])
        result.pose.position.z += float(offset[2])
        return result

    def _validate_workspace(self, pose: PoseStamped) -> None:
        point = (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )
        if not all(math.isfinite(value) for value in point):
            raise ValueError('target contains a non-finite coordinate')
        if any(
            value < low or value > high
            for value, low, high in zip(
                point, self.workspace_min, self.workspace_max
            )
        ):
            raise ValueError(
                f'target {point} is outside workspace '
                f'{self.workspace_min} .. {self.workspace_max}'
            )

    def _send_joint_pose(self, positions: Sequence[float], label: str) -> None:
        self._wait_action_server(self.arm_joint_client, 'arm trajectory')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = self._duration_message(
            self.joint_motion_duration
        )
        goal.trajectory.points.append(point)
        response = self._run_action(
            self.arm_joint_client,
            goal,
            f'arm {label}',
            self.joint_motion_duration + 8.0,
        )
        if (
            response.status != GoalStatus.STATUS_SUCCEEDED
            or response.result.error_code
            != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            raise RuntimeError(
                f'{label} trajectory failed: '
                f'{response.result.error_string or response.result.error_code}'
            )

    def _send_gripper(
        self,
        position: float,
        label: str,
        accept_stall: bool = False,
    ) -> None:
        self._wait_action_server(self.gripper_client, 'gripper')
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = self.gripper_effort
        response = self._run_action(
            self.gripper_client, goal, f'gripper {label}'
        )
        result = response.result
        if accept_stall:
            if (
                result.stalled
                and result.position >= self.grasp_position_threshold
            ):
                self.get_logger().info(
                    f'Grasp confirmed at gripper position {result.position:.4f}'
                )
                return
            if result.reached_goal:
                raise RuntimeError('gripper fully closed; no object was detected')
            raise RuntimeError(
                f'grasp not confirmed: position={result.position:.4f}, '
                f'stalled={result.stalled}'
            )
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(
                f'gripper {label} failed with status {response.status}'
            )

    def _move_group_pose(
        self,
        target: PoseStamped,
        label: str,
        *,
        plan_only: bool,
    ):
        self._wait_action_server(self.move_group_client, 'MoveIt')
        goal = MoveGroup.Goal()
        goal.request.group_name = self.move_group_name
        goal.request.allowed_planning_time = self.moveit_planning_time
        goal.request.num_planning_attempts = 3
        goal.request.max_velocity_scaling_factor = self.velocity_scale
        goal.request.max_acceleration_scaling_factor = self.acceleration_scale
        goal.planning_options.plan_only = plan_only
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2

        constraints = self._goal_constraints(
            target,
            self.base_frame,
            self.end_effector_link,
            self.position_tolerance,
            self.max_orientation_residual,
        )
        goal.request.goal_constraints.append(constraints)

        response = self._run_action(
            self.move_group_client, goal, f'MoveIt {label}'
        )
        failed = (
            response.status != GoalStatus.STATUS_SUCCEEDED
            or response.result.error_code.val != MoveItErrorCodes.SUCCESS
        )
        if failed:
            self.get_logger().warn(
                f'MoveIt {label} failed: error={response.result.error_code.val}'
            )
            return None
        return response

    @staticmethod
    def _goal_constraints(
        target: PoseStamped,
        base_frame: str,
        end_effector_link: str,
        position_tolerance: float,
        orientation_tolerance: float,
    ) -> Constraints:
        """Build coupled position and orientation goal constraints."""
        pose = copy.deepcopy(target.pose)
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = base_frame
        position_constraint.link_name = end_effector_link
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [position_tolerance] * 3
        position_constraint.constraint_region.primitives.append(box)
        position_constraint.constraint_region.primitive_poses.append(pose)
        position_constraint.weight = 1.0
        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = base_frame
        orientation_constraint.link_name = end_effector_link
        orientation_constraint.orientation = copy.deepcopy(
            target.pose.orientation
        )
        orientation_constraint.absolute_x_axis_tolerance = orientation_tolerance
        orientation_constraint.absolute_y_axis_tolerance = orientation_tolerance
        orientation_constraint.absolute_z_axis_tolerance = orientation_tolerance
        orientation_constraint.weight = 1.0
        constraints.orientation_constraints.append(orientation_constraint)
        return constraints

    @staticmethod
    def _planned_end_state(move_group_response):
        trajectory = move_group_response.result.planned_trajectory.joint_trajectory
        if not trajectory.joint_names or not trajectory.points:
            raise ValueError('MoveIt plan has no final joint state')
        final_positions = trajectory.points[-1].positions
        if len(final_positions) != len(trajectory.joint_names):
            raise ValueError('MoveIt final joint state is incomplete')
        start_state = copy.deepcopy(move_group_response.result.trajectory_start)
        start_state.is_diff = True
        positions = dict(zip(
            start_state.joint_state.name,
            start_state.joint_state.position,
        ))
        positions.update(zip(trajectory.joint_names, final_positions))
        start_state.joint_state.name = list(positions)
        start_state.joint_state.position = list(positions.values())
        start_state.joint_state.velocity = []
        start_state.joint_state.effort = []
        return start_state

    def _execute_trajectory(self, trajectory, label: str) -> None:
        """Execute the exact trajectory that passed feasibility checks."""
        self._wait_action_server(
            self.execute_trajectory_client, 'trajectory execution'
        )
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        execution = self._run_action(
            self.execute_trajectory_client, goal, f'{label} execution'
        )
        if (
            execution.status != GoalStatus.STATUS_SUCCEEDED
            or execution.result.error_code.val != MoveItErrorCodes.SUCCESS
        ):
            raise RuntimeError(
                f'{label} execution failed: '
                f'error={execution.result.error_code.val}'
            )

    def _planned_trajectory_pose(self, robot_trajectory) -> PoseStamped:
        trajectory = robot_trajectory.joint_trajectory
        if not trajectory.joint_names or not trajectory.points:
            raise ValueError('MoveIt plan has no final joint state for FK')
        final_positions = trajectory.points[-1].positions
        if len(final_positions) != len(trajectory.joint_names):
            raise ValueError('MoveIt final joint state is incomplete')

        deadline = time.monotonic() + self.server_timeout
        while not self.compute_fk_client.wait_for_service(timeout_sec=0.1):
            if time.monotonic() >= deadline:
                raise RuntimeError('MoveIt FK service is unavailable')

        request = GetPositionFK.Request()
        request.header.frame_id = self.base_frame
        request.fk_link_names = [self.end_effector_link]
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(trajectory.joint_names)
        request.robot_state.joint_state.position = list(final_positions)
        response = self._wait_future(
            self.compute_fk_client.call_async(request),
            'planned-pose FK',
            self.result_timeout,
        )
        if (
            response.error_code.val != MoveItErrorCodes.SUCCESS
            or not response.pose_stamped
        ):
            raise ValueError(
                f'MoveIt FK failed: error={response.error_code.val}'
            )
        return response.pose_stamped[0]

    def _planned_command_pose(self, move_group_response) -> PoseStamped:
        return self._planned_trajectory_pose(
            move_group_response.result.planned_trajectory
        )

    def _plan_cartesian_approach(
        self,
        pregrasp_plan,
        pregrasp_pose: PoseStamped,
        pick: PoseStamped,
    ):
        """Plan a straight approach while preserving PRE-GRASP orientation."""
        deadline = time.monotonic() + self.server_timeout
        while not self.cartesian_path_client.wait_for_service(timeout_sec=0.1):
            if time.monotonic() >= deadline:
                raise RuntimeError('Cartesian path service is unavailable')

        waypoint = Pose()
        waypoint.position = copy.deepcopy(pick.pose.position)
        waypoint.orientation = copy.deepcopy(pregrasp_pose.pose.orientation)

        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.start_state = self._planned_end_state(pregrasp_plan)
        request.group_name = self.move_group_name
        request.link_name = self.end_effector_link
        request.waypoints = [waypoint]
        request.max_step = self.approach_step
        request.revolute_jump_threshold = self.approach_joint_jump
        request.avoid_collisions = True
        response = self._wait_future(
            self.cartesian_path_client.call_async(request),
            'Cartesian PICK approach plan',
            self.result_timeout,
        )
        fraction = float(response.fraction)
        if (
            response.error_code.val != MoveItErrorCodes.SUCCESS
            or not math.isfinite(fraction)
            or fraction < 0.999
            or len(response.solution.joint_trajectory.points) < 2
        ):
            raise ValueError(
                'MoveIt could not plan the complete Cartesian PICK approach: '
                f'fraction={fraction:.3f}, error={response.error_code.val}'
            )
        self.get_logger().info(
            f'Cartesian PICK approach planned: {fraction * 100.0:.1f}%'
        )
        return response.solution

    def _send_vertical_retreat(self) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_link,
                Time(),
                timeout=Duration(seconds=1.0),
            )
        except TransformException as exc:
            raise RuntimeError(
                f'end-effector pose unavailable: {exc}'
            ) from exc

        with self.state_lock:
            joints = (
                list(self.current_joint_positions)
                if self.current_joint_positions is not None
                else None
            )
        if joints is None:
            raise RuntimeError('joint states unavailable for vertical retreat')

        waypoint = Pose()
        waypoint.position.x = transform.transform.translation.x
        waypoint.position.y = transform.transform.translation.y
        waypoint.position.z = (
            transform.transform.translation.z + self.retreat_offset_z
        )
        waypoint.orientation = transform.transform.rotation

        deadline = time.monotonic() + self.server_timeout
        while not self.cartesian_path_client.wait_for_service(timeout_sec=0.1):
            if time.monotonic() >= deadline:
                raise RuntimeError('Cartesian path service is unavailable')

        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.start_state.is_diff = True
        request.start_state.joint_state.name = list(self.JOINT_NAMES)
        request.start_state.joint_state.position = joints
        request.group_name = self.move_group_name
        request.link_name = self.end_effector_link
        request.waypoints = [waypoint]
        request.max_step = self.retreat_step
        request.revolute_jump_threshold = self.retreat_joint_jump
        request.avoid_collisions = True
        response = self._wait_future(
            self.cartesian_path_client.call_async(request),
            'Cartesian retreat plan',
            self.result_timeout,
        )
        if (
            response.error_code.val != MoveItErrorCodes.SUCCESS
            or not math.isfinite(response.fraction)
            or response.fraction <= 0.0
            or len(response.solution.joint_trajectory.points) < 2
        ):
            raise RuntimeError(
                f'no retreat path: fraction={response.fraction:.3f}, '
                f'error={response.error_code.val}'
            )
        if response.fraction < 0.999:
            self.get_logger().warn(
                f'Only {response.fraction * 100.0:.1f}% of the retreat '
                'was reachable; executing that collision-checked segment.'
            )

        self._wait_action_server(
            self.execute_trajectory_client, 'trajectory execution'
        )
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        execution = self._run_action(
            self.execute_trajectory_client, goal, 'retreat execution'
        )
        if (
            execution.status != GoalStatus.STATUS_SUCCEEDED
            or execution.result.error_code.val != MoveItErrorCodes.SUCCESS
        ):
            raise RuntimeError(
                'retreat execution failed: '
                f'error={execution.result.error_code.val}'
            )

    def _wait_action_server(self, client: ActionClient, label: str) -> None:
        if not client.wait_for_server(timeout_sec=self.server_timeout):
            raise RuntimeError(f'{label} action server is unavailable')

    def _run_action(
        self,
        client: ActionClient,
        goal,
        label: str,
        timeout: Optional[float] = None,
    ):
        self._check_interrupted()
        goal_handle = self._wait_future(
            client.send_goal_async(goal),
            f'{label} goal response',
            self.server_timeout,
        )
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f'{label} goal was rejected')
        with self.state_lock:
            self.active_goal_handles.append(goal_handle)
        try:
            self._check_interrupted()
            response = self._wait_future(
                goal_handle.get_result_async(),
                f'{label} result',
                self.result_timeout if timeout is None else timeout,
            )
            self._check_interrupted()
            return response
        except Exception:
            goal_handle.cancel_goal_async()
            raise
        finally:
            with self.state_lock:
                if goal_handle in self.active_goal_handles:
                    self.active_goal_handles.remove(goal_handle)

    @staticmethod
    def _duration_message(seconds: float) -> DurationMsg:
        whole = int(seconds)
        return DurationMsg(
            sec=whole,
            nanosec=int((seconds - whole) * 1_000_000_000),
        )

    @staticmethod
    def _wait_future(future, label: str, timeout: float):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                raise RuntimeError(f'{label} timed out')
            time.sleep(0.02)
        if not future.done():
            raise RuntimeError(f'{label} interrupted by shutdown')
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f'{label} failed: {exception}') from exception
        return future.result()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
