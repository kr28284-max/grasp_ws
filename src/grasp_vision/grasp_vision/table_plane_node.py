"""Estimate and publish a stable table frame from the RGB-D table cloud."""

from collections import deque
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped, Vector3Stamped
from grasp_interfaces.msg import TablePlane
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

from grasp_vision.table_geometry import matrix_to_quaternion
from grasp_vision.table_geometry import plane_height
from grasp_vision.table_geometry import ransac_plane
from grasp_vision.table_geometry import table_basis
from grasp_vision.table_geometry import transform_points


class TablePlaneNode(Node):
    """Fit table planes in base_link and reject temporally unstable fits."""

    def __init__(self):
        super().__init__('table_plane_node')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('table_frame', 'table_frame')
        self.declare_parameter('min_points', 100)
        self.declare_parameter('ransac_iterations', 150)
        self.declare_parameter('distance_threshold', 0.004)
        self.declare_parameter('min_inlier_ratio', 0.45)
        self.declare_parameter('stability_window', 8)
        self.declare_parameter('min_stable_samples', 5)
        self.declare_parameter('max_normal_deviation_deg', 3.0)
        self.declare_parameter('max_height_std', 0.008)
        self.declare_parameter('marker_size', 0.40)

        self.base_frame = str(self.get_parameter('base_frame').value)
        self.table_frame = str(self.get_parameter('table_frame').value)
        self.min_points = int(self.get_parameter('min_points').value)
        self.ransac_iterations = int(
            self.get_parameter('ransac_iterations').value
        )
        self.distance_threshold = float(
            self.get_parameter('distance_threshold').value
        )
        self.min_inlier_ratio = float(
            self.get_parameter('min_inlier_ratio').value
        )
        window = max(1, int(self.get_parameter('stability_window').value))
        self.min_stable_samples = min(
            window, int(self.get_parameter('min_stable_samples').value)
        )
        self.max_normal_deviation = math.radians(float(
            self.get_parameter('max_normal_deviation_deg').value
        ))
        self.max_height_std = float(
            self.get_parameter('max_height_std').value
        )
        self.marker_size = float(self.get_parameter('marker_size').value)
        self.samples = deque(maxlen=window)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.normal_pub = self.create_publisher(
            Vector3Stamped, '/grasp/table_normal', 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, '/grasp/table_plane_marker', 10
        )
        self.plane_pub = self.create_publisher(
            TablePlane, '/grasp/table_plane', 10
        )
        self.create_subscription(
            PointCloud2, '/grasp/table_cloud', self.cloud_cb, 10
        )
        self.get_logger().info(
            'Table plane estimator started; output frame is '
            f'{self.base_frame}'
        )

    @staticmethod
    def unpack_xyz(msg: PointCloud2) -> np.ndarray:
        """Read XYZ float fields from the organized or flat cloud."""
        if msg.width * msg.height == 0 or msg.point_step < 12:
            return np.empty((0, 3), dtype=np.float32)
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        rows = raw.reshape(-1, msg.point_step)
        xyz = np.frombuffer(rows[:, :12].tobytes(), dtype=np.float32)
        xyz = xyz.reshape(-1, 3)
        return xyz[np.isfinite(xyz).all(axis=1)]

    def cloud_cb(self, msg: PointCloud2):
        points = self.unpack_xyz(msg)
        if len(points) < self.min_points:
            self.get_logger().debug(
                f'Table cloud has only {len(points)} valid points'
            )
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.15)
            )
        except tf2_ros.TransformException as exc:
            self.get_logger().debug(f'Table TF unavailable: {exc}')
            return

        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ])
        quaternion = np.array([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ])
        base_points = transform_points(points, translation, quaternion)
        fit = ransac_plane(
            base_points,
            self.distance_threshold,
            self.ransac_iterations,
            self.min_inlier_ratio
        )
        if fit is None:
            self.get_logger().debug(
                'No table plane passed the RANSAC checks'
            )
            return

        normal, offset, inlier_mask, fitted_points = fit
        if normal[2] < 0.0:
            normal = -normal
            offset = -offset
        try:
            height = plane_height(normal, offset)
        except ValueError as exc:
            self.get_logger().debug(str(exc))
            return
        self.samples.append((normal, offset, height))
        stable = self.stable_average()
        if stable is None:
            self.get_logger().debug('Collecting stable table-plane samples')
            return

        avg_normal, avg_offset, avg_height, _, _ = stable
        inlier_ratio = (
            float(np.count_nonzero(inlier_mask)) / len(fitted_points)
        )
        self.publish_outputs(
            msg, avg_normal, avg_offset, avg_height, inlier_ratio
        )

    def stable_average(self):
        if len(self.samples) < self.min_stable_samples:
            return None
        normals = np.array([sample[0] for sample in self.samples])
        avg_normal = np.mean(normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal)
        angles = np.arccos(np.clip(normals @ avg_normal, -1.0, 1.0))
        heights = np.array([sample[2] for sample in self.samples])
        angle_max = float(np.max(angles))
        height_std = float(np.std(heights))
        if (
            angle_max > self.max_normal_deviation
            or height_std > self.max_height_std
        ):
            return None
        avg_offset = float(np.mean([sample[1] for sample in self.samples]))
        avg_height = plane_height(avg_normal, avg_offset)
        return avg_normal, avg_offset, avg_height, angle_max, height_std

    def publish_outputs(
        self,
        source: PointCloud2,
        normal: np.ndarray,
        offset: float,
        height: float,
        inlier_ratio: float
    ):
        normal_msg = Vector3Stamped()
        normal_msg.header.stamp = source.header.stamp
        normal_msg.header.frame_id = self.base_frame
        normal_msg.vector.x = float(normal[0])
        normal_msg.vector.y = float(normal[1])
        normal_msg.vector.z = float(normal[2])
        self.normal_pub.publish(normal_msg)

        plane_msg = TablePlane()
        plane_msg.header = normal_msg.header
        plane_msg.normal = normal_msg.vector
        plane_msg.height = float(height)
        plane_msg.inlier_ratio = float(inlier_ratio)
        self.plane_pub.publish(plane_msg)

        basis = table_basis(normal)
        quaternion = matrix_to_quaternion(basis)
        origin = np.array([0.0, 0.0, height])

        frame = TransformStamped()
        frame.header = normal_msg.header
        frame.child_frame_id = self.table_frame
        frame.transform.translation.x = float(origin[0])
        frame.transform.translation.y = float(origin[1])
        frame.transform.translation.z = float(origin[2])
        frame.transform.rotation.x = float(quaternion[0])
        frame.transform.rotation.y = float(quaternion[1])
        frame.transform.rotation.z = float(quaternion[2])
        frame.transform.rotation.w = float(quaternion[3])
        self.tf_broadcaster.sendTransform(frame)

        markers = MarkerArray()
        plane = Marker()
        plane.header = normal_msg.header
        plane.ns = 'table_plane'
        plane.id = 0
        plane.type = Marker.CUBE
        plane.action = Marker.ADD
        plane.pose.position.z = float(height)
        plane.pose.orientation.x = float(quaternion[0])
        plane.pose.orientation.y = float(quaternion[1])
        plane.pose.orientation.z = float(quaternion[2])
        plane.pose.orientation.w = float(quaternion[3])
        plane.scale.x = self.marker_size
        plane.scale.y = self.marker_size
        plane.scale.z = 0.002
        plane.color.r = 0.1
        plane.color.g = 0.45
        plane.color.b = 1.0
        plane.color.a = 0.28
        markers.markers.append(plane)

        arrow = Marker()
        arrow.header = normal_msg.header
        arrow.ns = 'table_normal'
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.points = [
            Point(x=0.0, y=0.0, z=float(height)),
            Point(
                x=float(normal[0] * 0.10),
                y=float(normal[1] * 0.10),
                z=float(height + normal[2] * 0.10)
            ),
        ]
        arrow.scale.x = 0.008
        arrow.scale.y = 0.016
        arrow.scale.z = 0.025
        arrow.color.r = 0.0
        arrow.color.g = 0.9
        arrow.color.b = 1.0
        arrow.color.a = 1.0
        markers.markers.append(arrow)

        label = Marker()
        label.header = normal_msg.header
        label.ns = 'table_label'
        label.id = 2
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.z = float(height + 0.12)
        label.scale.z = 0.022
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0
        label.text = (
            f'table z={height:+.3f} m, inliers={inlier_ratio:.0%}, '
            f'd={offset:+.3f}'
        )
        markers.markers.append(label)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = TablePlaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
