import math
import numpy as np

import message_filters
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped, TransformStamped
from grasp_interfaces.msg import TablePlane
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

from grasp_vision.duplo_geometry import estimate_duplo_table_pose
from grasp_vision.table_geometry import transform_points


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3 회전 행렬을 [x, y, z, w] 쿼터니언으로 변환."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(q)
    return q / norm if norm > 0.0 else np.array([0.0, 0.0, 0.0, 1.0])


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')

        # 1. 파라미터 선언
        self.declare_parameter('min_points', 30)
        self.min_points = self.get_parameter('min_points').value
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('table_sync_slop_sec', 0.05)
        self.declare_parameter('tf_timeout_sec', 0.15)
        self.declare_parameter('duplo_height_tolerance', 0.012)
        self.declare_parameter('duplo_height_ambiguity_margin', 0.002)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.table_sync_slop = float(
            self.get_parameter('table_sync_slop_sec').value
        )
        self.tf_timeout = float(self.get_parameter('tf_timeout_sec').value)
        self.duplo_height_tolerance = float(
            self.get_parameter('duplo_height_tolerance').value
        )
        self.duplo_height_ambiguity_margin = float(
            self.get_parameter('duplo_height_ambiguity_margin').value
        )

        # 실측 치수 (미터 단위)
        # 4x2 듀플로 블록: 긴변 6.31cm, 짧은변 3.1cm, 높이 2.31cm(스터드포함) / 1.9cm, 중심높이 1.15cm
        self.declare_parameter('duplo_length', 0.0631)
        self.declare_parameter('duplo_width', 0.0310)
        self.declare_parameter('duplo_height', 0.0231)
        self.declare_parameter('duplo_center_offset', 0.0115)

        # 망치: 전체 길이 26.2cm, 헤드 너비 8.1cm
        self.declare_parameter('hammer_length', 0.262)
        self.declare_parameter('hammer_head_width', 0.081)

        self.duplo_dims = (
            self.get_parameter('duplo_length').value,
            self.get_parameter('duplo_width').value,
            self.get_parameter('duplo_height').value,
            self.get_parameter('duplo_center_offset').value
        )
        self.hammer_dims = (
            self.get_parameter('hammer_length').value,
            self.get_parameter('hammer_head_width').value
        )

        # 2. 토픽 구독
        self.current_class = ''
        self.create_subscription(
            String,
            '/grasp/target_class',
            self.class_cb,
            10
        )
        cloud_sub = message_filters.Subscriber(
            self,
            PointCloud2,
            '/grasp/target_cloud_filtered'
        )
        table_sub = message_filters.Subscriber(
            self,
            TablePlane,
            '/grasp/table_plane'
        )
        self.table_sync = message_filters.ApproximateTimeSynchronizer(
            [cloud_sub, table_sub],
            queue_size=10,
            slop=self.table_sync_slop
        )
        self.table_sync.registerCallback(self.cloud_cb)

        # 3. 퍼블리셔
        self.object_pose_pub = self.create_publisher(
            PoseStamped,
            '/grasp/object_pose',
            10
        )
        self.grasp_pose_pub = self.create_publisher(
            PoseStamped,
            '/grasp/target_grasp_pose',
            10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/grasp/pose_markers',
            10
        )

        # 4. TF 브로드캐스터
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info('Pose Estimation & Grasp Node Started.')

    def class_cb(self, msg: String):
        self.current_class = msg.data.strip().lower()

    def cloud_cb(self, msg: PointCloud2, table_plane: TablePlane):
        points = self.unpack_pointcloud2(msg)

        if len(points) < self.min_points or not self.current_class:
            self.clear_markers(msg.header)
            return

        # 1) 표면 이상치 제거 (Outlier rejection)
        centroid_raw = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid_raw, axis=1)
        valid_idx = dists < np.percentile(dists, 95)
        filtered_points = points[valid_idx]

        if len(filtered_points) < self.min_points:
            self.clear_markers(msg.header)
            return

        output_header = msg.header
        if 'duplo' in self.current_class:
            base_points = self.points_to_base(filtered_points, msg)
            if base_points is None:
                self.clear_markers(msg.header)
                return
            result = self.process_duplo_with_table(base_points, table_plane)
            if result is None:
                self.clear_markers(msg.header)
                return
            obj_pose, grasp_pose, box_dims = result
            output_header = Header()
            output_header.stamp = msg.header.stamp
            output_header.frame_id = self.base_frame
        else:
            # Preserve the existing PCA path for non-Duplo classes.
            obj_pose, grasp_pose, box_dims = self.estimate_pose(
                filtered_points,
                self.current_class
            )

        if obj_pose is None:
            self.clear_markers(msg.header)
            return

        # 3) ROS PoseStamped 생성 및 퍼블리시
        obj_msg = self.create_pose_stamped(
            output_header, obj_pose[0], obj_pose[1]
        )
        grasp_msg = self.create_pose_stamped(
            output_header, grasp_pose[0], grasp_pose[1]
        )

        self.object_pose_pub.publish(obj_msg)
        self.grasp_pose_pub.publish(grasp_msg)

        # 4) TF 브로드캐스트 (object_frame & grasp_frame)
        self.broadcast_tf(
            output_header, 'object_frame', obj_pose[0], obj_pose[1]
        )
        self.broadcast_tf(
            output_header, 'grasp_frame', grasp_pose[0], grasp_pose[1]
        )

        # 5) RViz 시각화 마커 발행
        self.publish_markers(
            output_header, obj_pose, grasp_pose, box_dims
        )

    def points_to_base(self, points: np.ndarray, msg: PointCloud2):
        """Transform a timestamped camera cloud to the planning frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=self.tf_timeout)
            )
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(
                f'Duplo cloud TF unavailable: {exc}',
                throttle_duration_sec=2.0
            )
            return None
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
        return transform_points(points, translation, quaternion)

    def process_duplo_with_table(
        self,
        points_base: np.ndarray,
        table_plane: TablePlane
    ):
        """Create a fixed-model Duplo pose referenced to the table plane."""
        normal = np.array([
            table_plane.normal.x,
            table_plane.normal.y,
            table_plane.normal.z,
        ])
        model_dims = self.duplo_dims[:3]
        result = estimate_duplo_table_pose(
            points_base,
            normal,
            table_plane.height,
            model_dims,
            self.duplo_height_tolerance,
            self.duplo_height_ambiguity_margin
        )
        if result is None:
            self.get_logger().warn(
                'Duplo placement is ambiguous: visible height does not match '
                '23.1, 31.0, or 63.1 mm.',
                throttle_duration_sec=2.0
            )
            return None
        obj_pose = (
            result.center,
            rotation_matrix_to_quaternion(result.rotation)
        )
        grasp_pose = (
            result.grasp_center,
            rotation_matrix_to_quaternion(result.grasp_rotation)
        )
        return obj_pose, grasp_pose, tuple(model_dims)

    def unpack_pointcloud2(self, cloud_msg: PointCloud2) -> np.ndarray:
        if cloud_msg.width * cloud_msg.height == 0:
            return np.empty((0, 3), dtype=np.float32)

        raw_bytes = np.frombuffer(cloud_msg.data, dtype=np.uint8)
        reshaped = raw_bytes.reshape(-1, cloud_msg.point_step)
        xyz = np.frombuffer(reshaped[:, :12].tobytes(), dtype=np.float32).reshape(-1, 3)

        valid = np.isfinite(xyz).all(axis=1)
        return xyz[valid]

    def estimate_pose(self, points: np.ndarray, class_name: str):
        """PCA를 이용해 표면 점군으로부터 물체 6D Pose 및 Grasp Pose 계산."""
        surface_centroid = np.mean(points, axis=0)
        centered = points - surface_centroid

        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 고유값이 큰 순서대로 정렬 (v0: 긴축, v1: 짧은축, v2: 법선축)
        sort_idx = np.argsort(eigenvalues)[::-1]
        v0 = eigenvectors[:, sort_idx[0]]  # Long axis
        v1 = eigenvectors[:, sort_idx[1]]  # Short axis
        v2 = eigenvectors[:, sort_idx[2]]  # Surface normal

        # 법선 벡터(v2)가 항상 카메라 시선(-Z) 쪽을 향하도록 부호 정렬
        # 카메라 광학 좌표계에서 카메라는 [0, 0, 0]에 위치함
        to_cam = -surface_centroid
        if np.dot(v2, to_cam) < 0.0:
            v2 = -v2

        # 오른손 법칙 직교좌표계 보장: v1 = v2 x v0
        v0 = v0 / np.linalg.norm(v0)
        v1 = np.cross(v2, v0)
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(v0, v1)
        v2 = v2 / np.linalg.norm(v2)

        if 'duplo' in class_name:
            return self.process_duplo(points, surface_centroid, v0, v1, v2)
        elif 'hammer' in class_name:
            return self.process_hammer(points, surface_centroid, v0, v1, v2)
        else:
            # 기본 직육면체 처리
            R = np.column_stack((v0, v1, v2))
            q = rotation_matrix_to_quaternion(R)
            box_dims = (0.05, 0.05, 0.05)
            obj_pose = (surface_centroid, q)
            grasp_pose = (surface_centroid, q)
            return obj_pose, grasp_pose, box_dims

    def process_duplo(
        self,
        points: np.ndarray,
        centroid: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray
    ):
        length, width, height, center_offset = self.duplo_dims

        # 대칭성 해결 규칙:
        # 카메라 좌표계 기준(상단 마운트)에서 로봇 베이스 방향(일반적으로 -Y 또는 -Z 방향)과
        # 일관성을 갖도록 긴 변(v0)과 짧은 변(v1)의 방향을 고정합니다.
        # 카메라 optical frame 기준 Y축(아래쪽)과의 내적이 양수가 되도록 정렬하여 180도 뒤집힘 방지
        if v0[1] < 0.0:
            v0 = -v0
            v1 = -v1

        # 물체 3D 중심점: 표면 중심에서 물체 내부(-v2 방향)로 center_offset(1.15cm) 이동
        obj_center = centroid - v2 * center_offset

        # 물체 좌표계 R_obj = [v0, v1, v2] (X: 긴변, Y: 짧은변, Z: 윗면 법선)
        R_obj = np.column_stack((v0, v1, v2))
        q_obj = rotation_matrix_to_quaternion(R_obj)

        # RH-P12-RN 그리퍼 파지 자세:
        # 상단 수직 접근(Top-down approach: 그리퍼 접근축 Z_g = -v2)
        # 핑거 닫힘 축 X_g = v1 (짧은 변 3.1cm을 양쪽에서 물도록 정렬)
        # Y_g = Z_g x X_g = -v2 x v1
        z_grasp = -v2
        x_grasp = v1
        y_grasp = np.cross(z_grasp, x_grasp)
        y_grasp = y_grasp / np.linalg.norm(y_grasp)

        R_grasp = np.column_stack((x_grasp, y_grasp, z_grasp))
        q_grasp = rotation_matrix_to_quaternion(R_grasp)

        # 파지 위치: 표면에서 살짝 내려간 위치 (스터드 포함 안전 파지점)
        grasp_center = centroid - v2 * (center_offset * 0.5)

        box_dims = (length, width, height)
        return (obj_center, q_obj), (grasp_center, q_grasp), box_dims

    def process_hammer(
        self,
        points: np.ndarray,
        centroid: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray
    ):
        length, head_width = self.hammer_dims

        # 망치의 긴 축(v0) 상에서 헤드(Head)와 손잡이(Handle) 구분
        # 헤드 부위는 v1 방향(너비) 점군 분산이 손잡이보다 훨씬 큼
        projections_v0 = np.dot(points - centroid, v0)
        side_pos = points[projections_v0 > 0.0] - centroid
        side_neg = points[projections_v0 <= 0.0] - centroid

        var_pos = np.var(np.dot(side_pos, v1)) if len(side_pos) > 5 else 0.0
        var_neg = np.var(np.dot(side_neg, v1)) if len(side_neg) > 5 else 0.0

        # var가 큰 쪽이 헤드, 작은 쪽이 손잡이.
        # 항상 v0가 손잡이(Handle) -> 헤드(Head) 방향을 향하도록 정렬
        if var_neg > var_pos:
            v0 = -v0
            v1 = -v1

        obj_center = centroid
        R_obj = np.column_stack((v0, v1, v2))
        q_obj = rotation_matrix_to_quaternion(R_obj)

        # 손잡이 파지점: 헤드 반대편(손잡이 영역)으로 중심 이동
        grasp_center = centroid - v0 * (length * 0.25)

        # RH-P12-RN 그리퍼가 손잡이를 감싸쥘 수 있는 자세:
        # Z_g: 상단 수직 접근 (-v2)
        # X_g: 손잡이 축에 수직 방향 (v1)
        z_grasp = -v2
        x_grasp = v1
        y_grasp = np.cross(z_grasp, x_grasp)
        y_grasp = y_grasp / np.linalg.norm(y_grasp)

        R_grasp = np.column_stack((x_grasp, y_grasp, z_grasp))
        q_grasp = rotation_matrix_to_quaternion(R_grasp)

        box_dims = (length, head_width, 0.04)
        return (obj_center, q_obj), (grasp_center, q_grasp), box_dims

    def create_pose_stamped(
        self,
        header: Header,
        position: np.ndarray,
        quaternion: np.ndarray
    ) -> PoseStamped:
        msg = PoseStamped()
        msg.header = header
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(quaternion[0])
        msg.pose.orientation.y = float(quaternion[1])
        msg.pose.orientation.z = float(quaternion[2])
        msg.pose.orientation.w = float(quaternion[3])
        return msg

    def broadcast_tf(
        self,
        header: Header,
        child_frame: str,
        position: np.ndarray,
        quaternion: np.ndarray
    ):
        t = TransformStamped()
        t.header = header
        t.child_frame_id = child_frame
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        t.transform.rotation.x = float(quaternion[0])
        t.transform.rotation.y = float(quaternion[1])
        t.transform.rotation.z = float(quaternion[2])
        t.transform.rotation.w = float(quaternion[3])
        self.tf_broadcaster.sendTransform(t)

    def publish_markers(
        self,
        header: Header,
        obj_pose: tuple,
        grasp_pose: tuple,
        box_dims: tuple
    ):
        marker_array = MarkerArray()
        obj_pos, obj_quat = obj_pose
        grasp_pos, grasp_quat = grasp_pose

        # 1. Bounding Box Marker (물체 실측 크기 박스)
        box_marker = Marker()
        box_marker.header = header
        box_marker.ns = 'bounding_box'
        box_marker.id = 0
        box_marker.type = Marker.CUBE
        box_marker.action = Marker.ADD
        box_marker.pose.position.x = float(obj_pos[0])
        box_marker.pose.position.y = float(obj_pos[1])
        box_marker.pose.position.z = float(obj_pos[2])
        box_marker.pose.orientation.x = float(obj_quat[0])
        box_marker.pose.orientation.y = float(obj_quat[1])
        box_marker.pose.orientation.z = float(obj_quat[2])
        box_marker.pose.orientation.w = float(obj_quat[3])
        box_marker.scale.x = float(box_dims[0])
        box_marker.scale.y = float(box_dims[1])
        box_marker.scale.z = float(box_dims[2])
        box_marker.color.r = 0.0
        box_marker.color.g = 1.0
        box_marker.color.b = 0.5
        box_marker.color.a = 0.45
        marker_array.markers.append(box_marker)

        # 2. Grasp Target Arrow Marker (그리퍼 접근 방향)
        grasp_marker = Marker()
        grasp_marker.header = header
        grasp_marker.ns = 'grasp_pose'
        grasp_marker.id = 1
        grasp_marker.type = Marker.ARROW
        grasp_marker.action = Marker.ADD
        grasp_marker.pose.position.x = float(grasp_pos[0])
        grasp_marker.pose.position.y = float(grasp_pos[1])
        grasp_marker.pose.position.z = float(grasp_pos[2])
        grasp_marker.pose.orientation.x = float(grasp_quat[0])
        grasp_marker.pose.orientation.y = float(grasp_quat[1])
        grasp_marker.pose.orientation.z = float(grasp_quat[2])
        grasp_marker.pose.orientation.w = float(grasp_quat[3])
        grasp_marker.scale.x = 0.06
        grasp_marker.scale.y = 0.015
        grasp_marker.scale.z = 0.015
        grasp_marker.color.r = 1.0
        grasp_marker.color.g = 0.8
        grasp_marker.color.b = 0.0
        grasp_marker.color.a = 0.9
        marker_array.markers.append(grasp_marker)

        # 3. Text Label Marker
        text_marker = Marker()
        text_marker.header = header
        text_marker.ns = 'label'
        text_marker.id = 2
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = float(obj_pos[0])
        text_marker.pose.position.y = float(obj_pos[1])
        text_marker.pose.position.z = float(obj_pos[2]) - 0.05
        text_marker.scale.z = 0.025
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = f'{self.current_class.upper()} 6D Pose'
        marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)

    def clear_markers(self, header: Header):
        marker_array = MarkerArray()
        del_marker = Marker()
        del_marker.header = header
        del_marker.action = Marker.DELETEALL
        marker_array.markers.append(del_marker)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
