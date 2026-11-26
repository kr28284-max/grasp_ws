"""Tests for table-plane geometry helpers."""

import math

import numpy as np

from grasp_vision.table_geometry import matrix_to_quaternion
from grasp_vision.table_geometry import plane_height
from grasp_vision.table_geometry import quaternion_to_matrix
from grasp_vision.table_geometry import ransac_plane
from grasp_vision.table_geometry import table_basis
from grasp_vision.table_geometry import transform_points


def test_ransac_plane_rejects_outliers_and_recovers_height():
    rng = np.random.default_rng(12)
    xy = rng.uniform(-0.25, 0.25, size=(500, 2))
    z = 0.032 + 0.01 * xy[:, 0] - 0.02 * xy[:, 1]
    z += rng.normal(0.0, 0.0005, size=len(z))
    table = np.column_stack((xy, z))
    outliers = rng.uniform(-0.25, 0.25, size=(80, 3))
    points = np.vstack((table, outliers))

    result = ransac_plane(
        points,
        distance_threshold=0.002,
        iterations=200,
        min_inlier_ratio=0.70,
        rng=np.random.default_rng(4)
    )

    assert result is not None
    normal, offset, inliers, _ = result
    if normal[2] < 0.0:
        normal, offset = -normal, -offset
    assert np.count_nonzero(inliers) > 480
    assert math.isclose(plane_height(normal, offset), 0.032, abs_tol=0.001)
    expected = np.array([-0.01, 0.02, 1.0])
    expected /= np.linalg.norm(expected)
    assert np.dot(normal, expected) > 0.999


def test_table_basis_is_right_handed_and_projects_base_x():
    normal = np.array([0.1, -0.2, 1.0])
    basis = table_basis(normal)

    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    assert np.linalg.det(basis) > 0.999999
    assert basis[0, 0] > 0.0
    np.testing.assert_allclose(
        basis[:, 2], normal / np.linalg.norm(normal), atol=1e-12
    )


def test_rotation_conversions_and_point_transform_round_trip():
    basis = table_basis(np.array([0.15, 0.05, 1.0]))
    quaternion = matrix_to_quaternion(basis)

    np.testing.assert_allclose(
        quaternion_to_matrix(quaternion), basis, atol=1e-12
    )
    points = np.array([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.4]])
    translation = np.array([0.4, -0.1, 0.2])
    transformed = transform_points(points, translation, quaternion)
    expected = points @ basis.T + translation
    np.testing.assert_allclose(transformed, expected, atol=1e-12)
