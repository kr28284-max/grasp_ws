"""Unit tests for hardware-independent grasp-control helpers."""

import math

import pytest
from geometry_msgs.msg import PoseStamped, Quaternion

from grasp_control.control_node import GraspControlNode


def test_degrees_pose_converts_four_joint_values():
    pose = GraspControlNode._degrees_pose([0.0, -90.0, 45.0, 180.0])
    assert pose == pytest.approx([0.0, -math.pi / 2, math.pi / 4, math.pi])


def test_offset_pose_does_not_mutate_source():
    source = PoseStamped()
    source.pose.position.x = 0.2
    source.pose.position.y = -0.1
    source.pose.position.z = 0.05

    result = GraspControlNode._offset_pose(source, (0.01, 0.02, 0.03))

    assert (
        source.pose.position.x,
        source.pose.position.y,
        source.pose.position.z,
    ) == pytest.approx((0.2, -0.1, 0.05))
    assert (
        result.pose.position.x,
        result.pose.position.y,
        result.pose.position.z,
    ) == pytest.approx((0.21, -0.08, 0.08))


def test_quaternion_is_normalized():
    quaternion = GraspControlNode._quaternion([0.0, 2.0, 0.0, 2.0])
    assert quaternion == pytest.approx(
        [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
    )


def test_orientation_error_handles_quaternion_sign_equivalence():
    first = Quaternion(x=0.0, y=0.7071068, z=0.0, w=0.7071068)
    second = Quaternion(x=0.0, y=-0.7071068, z=0.0, w=-0.7071068)
    assert GraspControlNode._orientation_error(first, second) == pytest.approx(0.0)


def test_orientation_error_reports_half_turn():
    identity = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    half_turn = Quaternion(x=0.0, y=0.0, z=1.0, w=0.0)
    assert GraspControlNode._orientation_error(
        identity, half_turn
    ) == pytest.approx(math.pi)


@pytest.mark.parametrize('values', [[], [1.0], [1.0, 2.0]])
def test_xyz_rejects_wrong_length(values):
    with pytest.raises(ValueError):
        GraspControlNode._xyz(values)
