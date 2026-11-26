"""Integration test for the router's OpenMANIPULATOR NO branch."""

import os
from io import StringIO
import threading
import time

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from grasp_interfaces.msg import GraspCandidate, GraspExecutionResult
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool

from grasp_control.grasp_router_node import GraspRouterNode
from grasp_control.rb3_dummy_executor_node import Rb3DummyExecutorNode


def wait_for(predicate, timeout=3.0):
    """Wait until a ROS callback updates the expected test state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def ros_graph(tmp_path):
    os.environ['ROS_LOG_DIR'] = str(tmp_path)
    rclpy.init()
    router = GraspRouterNode()
    dummy = Rb3DummyExecutorNode()
    probe = Node('grasp_router_test_probe')
    executor = MultiThreadedExecutor(num_threads=3)
    for node in (router, dummy, probe):
        executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    yield router, dummy, probe
    executor.shutdown()
    spin_thread.join(timeout=1.0)
    for node in (probe, dummy, router):
        node.destroy_node()
    rclpy.shutdown()


def test_infeasible_openmanipulator_result_routes_to_dummy(ros_graph):
    router, _dummy, probe = ros_graph
    prepare_messages = []
    execute_messages = []
    dummy_results = []
    dummy_candidates = []
    probe.create_subscription(
        Bool,
        '/grasp/openmanipulator/prepare',
        prepare_messages.append,
        10,
    )
    probe.create_subscription(
        PoseStamped,
        '/grasp/openmanipulator/execute',
        execute_messages.append,
        10,
    )
    probe.create_subscription(
        GraspExecutionResult,
        '/grasp/rb3_dummy/result',
        dummy_results.append,
        10,
    )
    probe.create_subscription(
        GraspCandidate,
        '/grasp/rb3_dummy/candidate',
        dummy_candidates.append,
        10,
    )
    trigger_pub = probe.create_publisher(
        Bool, '/grasp/execute_trigger', 10
    )
    ready_pub = probe.create_publisher(
        Bool, '/grasp/openmanipulator/ready', 10
    )
    candidate_pub = probe.create_publisher(
        PoseStamped, '/grasp/base/desired_grasp_pose', 10
    )
    object_pub = probe.create_publisher(
        PoseStamped, '/grasp/base/object_pose', 10
    )
    result_pub = probe.create_publisher(
        GraspExecutionResult, '/grasp/openmanipulator/result', 10
    )

    time.sleep(0.2)
    trigger_pub.publish(Bool(data=True))
    assert wait_for(lambda: len(prepare_messages) == 1)

    ready_pub.publish(Bool(data=True))
    assert wait_for(lambda: router.state == router.WAITING_POSE)

    desired_pose = PoseStamped()
    desired_pose.header.frame_id = 'base_link'
    desired_pose.pose.orientation.w = 1.0
    object_pose = PoseStamped()
    object_pose.header.frame_id = 'base_link'
    object_pose.pose.position.x = 0.25
    object_pose.pose.orientation.w = 1.0
    object_pub.publish(object_pose)
    assert wait_for(lambda: router.latest_object_pose is not None)
    candidate_pub.publish(desired_pose)
    assert wait_for(lambda: len(execute_messages) == 1)

    result = GraspExecutionResult()
    result.robot_name = 'openmanipulator_x'
    result.feasible = False
    result.executed = False
    result.success = False
    result.desired_pose = desired_pose
    result.orientation_residual_rad = 0.5
    result.message = 'test infeasible pose'
    result_pub.publish(result)

    assert wait_for(lambda: len(dummy_results) == 1)
    assert wait_for(lambda: len(dummy_candidates) == 1)
    assert dummy_candidates[0].object_pose.pose.position.x == pytest.approx(0.25)
    assert dummy_candidates[0].grasp_pose.header.frame_id == 'base_link'
    assert dummy_results[0].robot_name == 'rb3_730_dummy'
    assert not dummy_results[0].executed
    assert wait_for(lambda: router.state == router.IDLE)


def test_manual_control_triggers_are_forwarded(ros_graph):
    router, _dummy, probe = ros_graph
    opened = []
    homed = []
    paused = []
    probe.create_subscription(
        Bool, '/grasp/openmanipulator/open_gripper', opened.append, 10
    )
    probe.create_subscription(
        Bool, '/grasp/openmanipulator/home', homed.append, 10
    )
    probe.create_subscription(
        Bool, '/grasp/openmanipulator/pause', paused.append, 10
    )

    time.sleep(0.2)
    router.open_gripper_cb(Bool(data=True))
    router.home_trigger_cb(Bool(data=True))
    router.pause_trigger_cb(Bool(data=True))

    assert wait_for(lambda: len(opened) == 1)
    assert wait_for(lambda: len(homed) == 1)
    assert wait_for(lambda: len(paused) == 1)
    assert opened[0].data and homed[0].data and paused[0].data
    assert router.state == router.IDLE


def test_keyboard_menu_lists_manual_controls():
    terminal = StringIO()

    GraspRouterNode.write_keyboard_menu(terminal)

    menu = terminal.getvalue()
    assert 'o : Open the gripper only' in menu
    assert '0 : Cancel the current action and move HOME' in menu
    assert 'p : Pause at the current position' in menu
