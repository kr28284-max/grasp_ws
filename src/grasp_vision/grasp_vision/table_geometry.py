"""Geometry helpers for robust table-plane estimation."""

import math

import numpy as np


def fit_plane_svd(points: np.ndarray):
    """Fit a normalized plane ``normal dot point + offset = 0``."""
    if len(points) < 3:
        raise ValueError('at least three points are required')
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    return normal, -float(np.dot(normal, centroid))


def ransac_plane(
    points: np.ndarray,
    distance_threshold: float,
    iterations: int,
    min_inlier_ratio: float,
    rng=None
):
    """Return a refined plane and inlier mask, or ``None`` when invalid."""
    points = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 3:
        return None

    generator = rng if rng is not None else np.random.default_rng()
    best_mask = None
    best_count = 0
    for _ in range(max(1, iterations)):
        sample = points[generator.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(np.dot(normal, sample[0]))
        mask = np.abs(points @ normal + offset) <= distance_threshold
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_mask = mask

    required = max(3, int(math.ceil(min_inlier_ratio * len(points))))
    if best_mask is None or best_count < required:
        return None

    normal, offset = fit_plane_svd(points[best_mask])
    refined_mask = np.abs(points @ normal + offset) <= distance_threshold
    if int(np.count_nonzero(refined_mask)) < required:
        return None
    normal, offset = fit_plane_svd(points[refined_mask])
    return normal, offset, refined_mask, points


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert an ``[x, y, z, w]`` quaternion to a rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError('zero-length quaternion')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_points(
    points: np.ndarray,
    translation: np.ndarray,
    quaternion: np.ndarray
) -> np.ndarray:
    """Apply a ROS transform to an array of points."""
    rotation = quaternion_to_matrix(quaternion)
    return np.asarray(points) @ rotation.T + np.asarray(translation)


def table_basis(normal: np.ndarray) -> np.ndarray:
    """Build table axes with base X projected onto the table plane."""
    z_axis = np.asarray(normal, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)
    if z_axis[2] < 0.0:
        z_axis = -z_axis
    base_x = np.array([1.0, 0.0, 0.0])
    x_axis = base_x - np.dot(base_x, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        base_y = np.array([0.0, 1.0, 0.0])
        x_axis = base_y - np.dot(base_y, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to ``[x, y, z, w]``."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1]
                              - matrix[2, 2]) * 2.0
            quat = np.array([
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            ])
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0]
                              - matrix[2, 2]) * 2.0
            quat = np.array([
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            ])
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0]
                              - matrix[1, 1]) * 2.0
            quat = np.array([
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ])
    return quat / np.linalg.norm(quat)


def plane_height(normal: np.ndarray, offset: float) -> float:
    """Return plane Z where the base-frame Z axis intersects the plane."""
    if abs(normal[2]) < 1e-6:
        raise ValueError('plane is parallel to the base Z axis')
    return -float(offset) / float(normal[2])


def normal_roll_pitch(normal: np.ndarray):
    """Return diagnostic roll and pitch of a table normal in radians."""
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    roll = math.atan2(-normal[1], normal[2])
    pitch = math.atan2(normal[0], math.hypot(normal[1], normal[2]))
    return roll, pitch
