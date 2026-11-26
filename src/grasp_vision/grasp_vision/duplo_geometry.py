"""Table-referenced geometry for a fixed-dimension Duplo model."""

from dataclasses import dataclass

import numpy as np

from grasp_vision.table_geometry import table_basis


@dataclass(frozen=True)
class DuploTablePose:
    """Resolved Duplo model and top-down grasp geometry in the base frame."""

    center: np.ndarray
    rotation: np.ndarray
    grasp_center: np.ndarray
    grasp_rotation: np.ndarray
    placement: str
    observed_height: float
    vertical_dimension: float


def _planar_rectangle(points_xy: np.ndarray):
    """Estimate robust center and principal rectangle axes in table XY."""
    # A vertical side produces many 3D samples at almost identical XY. Collapse
    # those samples into 1.5 mm cells so it cannot bias the planar yaw merely
    # because it contains more depth pixels than the top face.
    grid = np.round(points_xy / 0.0015).astype(np.int64)
    _, unique_indices = np.unique(grid, axis=0, return_index=True)
    planar_samples = points_xy[unique_indices]
    center = np.mean(planar_samples, axis=0)
    covariance = np.cov(planar_samples - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    dominant_component = int(np.argmax(np.abs(major)))
    if major[dominant_component] < 0.0:
        major = -major
    minor = np.array([-major[1], major[0]])

    centered = points_xy - center
    major_projection = centered @ major
    minor_projection = centered @ minor
    major_bounds = np.percentile(major_projection, [2.0, 98.0])
    minor_bounds = np.percentile(minor_projection, [2.0, 98.0])
    rectangle_center = (
        center
        + major * float(np.mean(major_bounds))
        + minor * float(np.mean(minor_bounds))
    )
    return rectangle_center, major, minor


def classify_duplo_placement(
    observed_height: float,
    dimensions: np.ndarray,
    tolerance: float,
    ambiguity_margin: float = 0.002
):
    """Return the model-axis index vertical to the table and its label."""
    differences = np.abs(dimensions - observed_height)
    vertical_index = int(np.argmin(differences))
    if float(differences[vertical_index]) > tolerance:
        return None
    ordered = np.sort(differences)
    if len(ordered) > 1 and ordered[1] - ordered[0] < ambiguity_margin:
        return None
    labels = {0: 'end', 1: 'side', 2: 'flat'}
    return vertical_index, labels[vertical_index]


def estimate_duplo_table_pose(
    points_base: np.ndarray,
    table_normal: np.ndarray,
    table_height: float,
    dimensions,
    height_tolerance: float,
    height_ambiguity_margin: float = 0.002
):
    """Estimate fixed model axes without fitting roll/pitch to visible sides."""
    points = np.asarray(points_base, dtype=np.float64)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    if len(points) < 3 or dimensions.shape != (3,):
        return None

    normal = np.asarray(table_normal, dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        return None
    normal /= normal_norm
    if normal[2] < 0.0:
        normal = -normal

    table_rotation = table_basis(normal)
    table_origin = np.array([0.0, 0.0, float(table_height)])
    points_table = (points - table_origin) @ table_rotation
    finite = np.isfinite(points_table).all(axis=1)
    points_table = points_table[finite]
    if len(points_table) < 3:
        return None

    # The upper percentile represents the visible top surface even when the
    # oblique camera also sees a vertical side extending down to the table.
    observed_height = float(np.percentile(points_table[:, 2], 90.0))
    placement = classify_duplo_placement(
        observed_height,
        dimensions,
        height_tolerance,
        height_ambiguity_margin
    )
    if placement is None:
        return None
    vertical_index, placement_name = placement

    rectangle_center, major_2d, _ = _planar_rectangle(points_table[:, :2])
    major_base = table_rotation[:, :2] @ major_2d
    major_base /= np.linalg.norm(major_base)

    horizontal_indices = [
        index for index in range(3) if index != vertical_index
    ]
    major_model_index = max(
        horizontal_indices, key=lambda index: dimensions[index]
    )
    minor_model_index = min(
        horizontal_indices, key=lambda index: dimensions[index]
    )

    model_rotation = np.zeros((3, 3), dtype=np.float64)
    model_rotation[:, vertical_index] = normal
    model_rotation[:, major_model_index] = major_base
    minor_base = np.cross(normal, major_base)
    minor_base /= np.linalg.norm(minor_base)
    model_rotation[:, minor_model_index] = minor_base
    if np.linalg.det(model_rotation) < 0.0:
        model_rotation[:, minor_model_index] *= -1.0

    contact_center = (
        table_origin
        + table_rotation[:, 0] * rectangle_center[0]
        + table_rotation[:, 1] * rectangle_center[1]
    )
    vertical_dimension = float(dimensions[vertical_index])
    object_center = contact_center + normal * (vertical_dimension * 0.5)

    # Preserve the old flat-placement grasp depth: 3/4 of object height above
    # the contact plane.  This keeps the user's measured Z calibration valid.
    grasp_center = contact_center + normal * (vertical_dimension * 0.75)
    grasp_x = model_rotation[:, minor_model_index]
    grasp_z = -normal
    grasp_y = np.cross(grasp_z, grasp_x)
    grasp_y /= np.linalg.norm(grasp_y)
    grasp_x = np.cross(grasp_y, grasp_z)
    grasp_rotation = np.column_stack((grasp_x, grasp_y, grasp_z))

    return DuploTablePose(
        center=object_center,
        rotation=model_rotation,
        grasp_center=grasp_center,
        grasp_rotation=grasp_rotation,
        placement=placement_name,
        observed_height=observed_height,
        vertical_dimension=vertical_dimension,
    )
