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


def test_vision_grasp_axes_are_mapped_to_end_effector_axes():
    vision = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    converted, _ = GraspControlNode._vision_grasp_to_ee_candidates(vision)

    # The cyclic axis mapping [G_z, G_x, G_y] is a -120-degree turn about
    # (1, 1, 1), represented by this XYZW quaternion.
    assert [converted.x, converted.y, converted.z, converted.w] == pytest.approx(
        [-0.5, -0.5, -0.5, 0.5]
    )


def test_example_grasp_matches_pick_after_axis_mapping_and_symmetry():
    vision = Quaternion(x=-0.7351, y=0.6753, z=-0.0464, w=0.0367)
    fixed_pick = Quaternion(x=0.0, y=0.7071068, z=0.0, w=0.7071068)

    candidates = GraspControlNode._vision_grasp_to_ee_candidates(vision)
    raw_residual = GraspControlNode._orientation_error(vision, fixed_pick)
    residuals = [
        GraspControlNode._orientation_error(candidate, fixed_pick)
        for candidate in candidates
    ]

    assert math.degrees(raw_residual) == pytest.approx(119.5, abs=0.1)
    assert math.degrees(min(residuals)) == pytest.approx(8.3, abs=0.1)
    assert math.degrees(min(residuals)) < 20.0


def test_quaternion_to_rpy_reports_expected_angles():
    half_yaw = math.radians(45.0)
    quaternion = Quaternion(
        z=math.sin(half_yaw),
        w=math.cos(half_yaw),
    )

    roll, pitch, yaw = GraspControlNode._quaternion_to_rpy(quaternion)

    assert math.degrees(roll) == pytest.approx(0.0)
    assert math.degrees(pitch) == pytest.approx(0.0)
    assert math.degrees(yaw) == pytest.approx(90.0)


def test_goal_constraints_include_target_orientation_and_20_degree_limit():
    target = PoseStamped()
    target.pose.position.x = 0.3
    target.pose.orientation.y = math.sqrt(0.5)
    target.pose.orientation.w = math.sqrt(0.5)
    tolerance = math.radians(20.0)

    constraints = GraspControlNode._goal_constraints(
        target,
        'base_link',
        'end_effector_link',
        0.01,
        tolerance,
    )

    assert len(constraints.position_constraints) == 1
    assert len(constraints.orientation_constraints) == 1
    orientation = constraints.orientation_constraints[0]
    assert orientation.header.frame_id == 'base_link'
    assert orientation.link_name == 'end_effector_link'
    assert orientation.orientation.y == pytest.approx(math.sqrt(0.5))
    assert orientation.orientation.w == pytest.approx(math.sqrt(0.5))
    assert orientation.absolute_x_axis_tolerance == pytest.approx(tolerance)
    assert orientation.absolute_y_axis_tolerance == pytest.approx(tolerance)
    assert orientation.absolute_z_axis_tolerance == pytest.approx(tolerance)


@pytest.mark.parametrize('values', [[], [1.0], [1.0, 2.0]])
def test_xyz_rejects_wrong_length(values):
    with pytest.raises(ValueError):
        GraspControlNode._xyz(values)
