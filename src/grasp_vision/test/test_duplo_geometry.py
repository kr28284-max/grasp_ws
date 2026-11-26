"""Tests for table-referenced Duplo pose estimation."""

import math

import numpy as np
import pytest

from grasp_vision.duplo_geometry import classify_duplo_placement
from grasp_vision.duplo_geometry import estimate_duplo_table_pose


DIMENSIONS = np.array([0.0631, 0.0310, 0.0231])


def _flat_duplo_points(yaw_degrees: float, table_height: float):
    rng = np.random.default_rng(7)
    yaw = math.radians(yaw_degrees)
    long_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    short_axis = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    center = np.array([0.20, 0.04, table_height])

    long_samples = rng.uniform(-DIMENSIONS[0] / 2, DIMENSIONS[0] / 2, 800)
    short_samples = rng.uniform(-DIMENSIONS[1] / 2, DIMENSIONS[1] / 2, 800)
    top = (
        center
        + long_samples[:, None] * long_axis
        + short_samples[:, None] * short_axis
        + DIMENSIONS[2] * normal
    )

    # Add a large visible front side. A 3D PCA normal tilts toward this face,
    # while the table-referenced estimator must remain vertical.
    long_side = rng.uniform(-DIMENSIONS[0] / 2, DIMENSIONS[0] / 2, 500)
    height_side = rng.uniform(0.0, DIMENSIONS[2], 500)
    side = (
        center
        + long_side[:, None] * long_axis
        - (DIMENSIONS[1] / 2) * short_axis
        + height_side[:, None] * normal
    )
    return np.vstack((top, side))


def test_flat_pose_ignores_oblique_visible_side_for_roll_pitch():
    table_height = -0.0101
    points = _flat_duplo_points(91.0, table_height)

    result = estimate_duplo_table_pose(
        points,
        table_normal=np.array([0.0, 0.0, 1.0]),
        table_height=table_height,
        dimensions=DIMENSIONS,
        height_tolerance=0.010,
    )

    assert result is not None
    assert result.placement == 'flat'
    np.testing.assert_allclose(result.rotation[:, 2], [0.0, 0.0, 1.0])
    assert result.center[2] == pytest.approx(
        table_height + DIMENSIONS[2] / 2, abs=1e-6
    )
    estimated_yaw = math.degrees(math.atan2(
        result.rotation[1, 0], result.rotation[0, 0]
    ))
    yaw_error = abs((estimated_yaw - 91.0 + 90.0) % 180.0 - 90.0)
    assert yaw_error < 1.0


@pytest.mark.parametrize(
    ('height', 'expected'),
    [
        (0.023, (2, 'flat')),
        (0.031, (1, 'side')),
        (0.063, (0, 'end')),
    ],
)
def test_placement_uses_known_model_dimensions(height, expected):
    assert classify_duplo_placement(
        height, DIMENSIONS, tolerance=0.005
    ) == expected


def test_placement_rejects_ambiguous_height():
    assert classify_duplo_placement(
        0.045, DIMENSIONS, tolerance=0.005
    ) is None
    assert classify_duplo_placement(
        0.027, DIMENSIONS, tolerance=0.012, ambiguity_margin=0.002
    ) is None
