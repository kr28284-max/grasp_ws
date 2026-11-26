"""Integration test for the router's OpenMANIPULATOR NO branch."""

import os
import threading
import time

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from grasp_interfaces.msg import GraspExecutionResult
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
    trigger_pub = probe.create_publisher(
        Bool, '/grasp/execute_trigger', 10
    )
    ready_pub = probe.create_publisher(
        Bool, '/grasp/openmanipulator/ready', 10
    )
    candidate_pub = probe.create_publisher(
        PoseStamped, '/grasp/base/desired_grasp_pose', 10
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
    assert dummy_results[0].robot_name == 'rb3_730_dummy'
    assert not dummy_results[0].executed
    assert wait_for(lambda: router.state == router.IDLE)
