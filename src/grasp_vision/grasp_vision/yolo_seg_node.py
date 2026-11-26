import os
import threading
import time
import traceback

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO


class YoloSegPointCloudNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_node')

        self.window_name = 'RealSense - YOLO Segmentation'
        self.display_frame = None
        self.display_lock = threading.Lock()
        self.display_stop = threading.Event()
        self.display_thread = threading.Thread(
            target=self.display_loop,
            name='realsense_display',
            daemon=True
        )
        self.display_thread.start()

        # ROS 수신 콜백과 무거운 YOLO 추론을 분리하기 위한 최신 프레임 버퍼
        self.pending_frame = None
        self.inference_lock = threading.Lock()
        self.inference_event = threading.Event()
        self.inference_stop = threading.Event()
        self.has_annotated_frame = False

        # EATED hardware-compatible camera interfaces and model location.
        self.declare_parameter(
            'model_path',
            '~/grasp_ws/weights/best_openvino_model'
        )
        self.declare_parameter(
            'rgb_topic',
            '/realsense/d435i/color/image_raw'
        )
        self.declare_parameter(
            'depth_topic',
            '/realsense/d435i/aligned_depth_to_color/image_raw'
        )
        self.declare_parameter(
            'camera_info_topic',
            '/realsense/d435i/color/camera_info'
        )
        self.declare_parameter(
            'camera_optical_frame',
            'd435i_color_optical_frame'
        )
        model_path = os.path.expanduser(
            str(self.get_parameter('model_path').value)
        )
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(
            self.get_parameter('camera_info_topic').value
        )
        self.camera_frame_id = str(
            self.get_parameter('camera_optical_frame').value
        )

        # 1. OpenVINO FP16 모델 로드
        self.model = YOLO(model_path, task='segment')
        self.bridge = CvBridge()

        # 카메라 내부 파라미터 (fx, fy, cx, cy)
        self.intrinsics = None
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_cb,
            qos_profile_sensor_data
        )

        # 2. RGB + Aligned Depth 시간 동기화 구독
        rgb_sub = message_filters.Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=qos_profile_sensor_data
        )
        depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.05
        )
        self.ts.registerCallback(self.sync_cb)

        # 3. 퍼블리셔 (중복 제거: filtered_cloud 및 raw_cloud로 단일화)
        self.raw_cloud_pub = self.create_publisher(
            PointCloud2,
            '/grasp/target_cloud_raw',
            10
        )
        self.filtered_cloud_pub = self.create_publisher(
            PointCloud2,
            '/grasp/target_cloud_filtered',
            10
        )
        self.cls_pub = self.create_publisher(String, '/grasp/target_class', 10)
        self.vis_pub = self.create_publisher(Image, '/grasp/debug_image', 10)
        self.reprojection_pub = self.create_publisher(
            Image,
            '/grasp/reprojection_image',
            10
        )

        self.inference_thread = threading.Thread(
            target=self.inference_loop,
            name='yolo_inference',
            daemon=True
        )
        self.inference_thread.start()

        self.get_logger().info('OpenVINO YOLOv8-seg 3D PointCloud Node Started.')

    def camera_info_cb(self, msg: CameraInfo):
        if self.intrinsics is None:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.get_logger().info(f'Camera Intrinsics received: {self.intrinsics}')

    def sync_cb(self, rgb_msg: Image, depth_msg: Image):
        # 추론이 밀려도 오래된 프레임을 쌓지 않고 가장 최신 프레임만 유지
        with self.inference_lock:
            self.pending_frame = (rgb_msg, depth_msg)
            self.inference_event.set()

    def inference_loop(self):
        while not self.inference_stop.is_set():
            self.inference_event.wait(timeout=0.1)
            if self.inference_stop.is_set():
                break

            with self.inference_lock:
                frame_pair = self.pending_frame
                self.pending_frame = None
                self.inference_event.clear()

            if frame_pair is None:
                continue

            try:
                self.process_frame(*frame_pair)
            except Exception as exc:
                self.get_logger().error(
                    'Frame processing failed: '
                    f'{type(exc).__name__}: {exc!r}\n'
                    f'{traceback.format_exc()}'
                )

    def process_frame(self, rgb_msg: Image, depth_msg: Image):
        color_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        h, w = color_img.shape[:2]

        # 최초 추론 결과가 나오기 전까지만 카메라 원본 화면을 표시
        if not self.has_annotated_frame:
            self.set_display_frame(color_img)

        # OpenVINO 추론
        start_time = time.perf_counter()
        results = self.model.predict(color_img, device='cpu', imgsz=640, conf=0.5, verbose=False)
        inference_time = time.perf_counter() - start_time
        inference_fps = 1.0 / inference_time if inference_time > 0.0 else 0.0
        res = results[0]

        # 디버그 이미지는 감지 여부와 관계없이 항상 퍼블리시
        vis_img = res.plot()
        cv2.putText(
            vis_img,
            f'YOLO FPS: {inference_fps:.1f}',
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        vis_msg = self.create_bgr8_image(rgb_msg.header, vis_img)
        self.vis_pub.publish(vis_msg)

        # 다음 추론이 끝날 때까지 현재 세그멘테이션 결과를 계속 유지
        self.set_display_frame(vis_img)
        self.has_annotated_frame = True

        # 검출이 사라지면 빈 메시지로 이전 RViz 점군을 즉시 제거한다.
        if self.intrinsics is None or res.masks is None or len(res.boxes) == 0:
            self.publish_empty_detection(rgb_msg.header, color_img)
            return

        best_idx = int(np.argmax(res.boxes.conf.cpu().numpy()))
        cls_id = int(res.boxes.cls[best_idx].item())
        cls_name = res.names[cls_id]

        # masks.data는 640x640 letterbox 좌표이므로 직접 resize하면
        # 640x480 영상에서 마스크가 세로로 밀린다. 원본 영상 좌표로
        # 복원된 polygon을 사용해 정확한 크기의 마스크를 만든다.
        polygon = np.rint(res.masks.xy[best_idx]).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(polygon) >= 3:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
            cv2.fillPoly(mask, [polygon], 1)
        mask = mask > 0

        # optical_frame 명시적 보장
        cloud_header = Header()
        cloud_header.stamp = rgb_msg.header.stamp
        cloud_header.frame_id = (
            rgb_msg.header.frame_id
            if rgb_msg.header.frame_id
            else self.camera_frame_id
        )

        # Raw cloud: 세그멘테이션 마스크와 기본 Depth 범위만 적용
        raw_points, raw_colors, _, _, _ = self.mask_to_points(
            mask,
            depth_img,
            color_img
        )
        self.raw_cloud_pub.publish(
            self.create_colored_pointcloud2(
                cloud_header,
                raw_points,
                raw_colors
            )
        )

        # Filtered cloud 1단계: 경계의 배경 Depth를 줄이기 위한 마스크 침식
        kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
        points, point_colors, u_valid, v_valid, z = self.mask_to_points(
            mask_eroded,
            depth_img,
            color_img
        )

        filtered_msg = self.create_colored_pointcloud2(
            cloud_header,
            points,
            point_colors
        )
        self.filtered_cloud_pub.publish(filtered_msg)

        reprojection_img = self.create_reprojection_image(
            color_img,
            mask_eroded,
            u_valid,
            v_valid,
            z,
            int(np.count_nonzero(mask_eroded))
        )
        reprojection_msg = self.create_bgr8_image(
            rgb_msg.header,
            reprojection_img
        )
        self.reprojection_pub.publish(reprojection_msg)

        cls_msg = String()
        cls_msg.data = cls_name
        self.cls_pub.publish(cls_msg)

    def mask_to_points(
        self,
        mask: np.ndarray,
        depth_img: np.ndarray,
        color_img: np.ndarray
    ):
        v_idx, u_idx = np.where(mask)
        z_mm = depth_img[v_idx, u_idx].astype(np.float32)
        valid = (z_mm > 100.0) & (z_mm < 1500.0)

        u_valid = u_idx[valid]
        v_valid = v_idx[valid]
        z = z_mm[valid] / 1000.0

        if len(z) == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                u_valid,
                v_valid,
                z
            )

        fx, fy, cx, cy = self.intrinsics
        x = (u_valid - cx) * z / fx
        y = (v_valid - cy) * z / fy
        points = np.stack((x, y, z), axis=-1).astype(np.float32)
        colors = color_img[v_valid, u_valid]
        return points, colors, u_valid, v_valid, z

    def publish_empty_detection(self, header: Header, color_img: np.ndarray):
        cloud_header = Header()
        cloud_header.stamp = header.stamp
        cloud_header.frame_id = (
            header.frame_id if header.frame_id else self.camera_frame_id
        )

        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        empty_colored = self.create_colored_pointcloud2(
            cloud_header,
            empty_points,
            empty_colors
        )

        self.raw_cloud_pub.publish(empty_colored)
        self.filtered_cloud_pub.publish(empty_colored)

        cls_msg = String()
        cls_msg.data = ''
        self.cls_pub.publish(cls_msg)

        image = color_img.copy()
        cv2.putText(
            image,
            'No segmentation detection - clouds cleared',
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        image_msg = self.create_bgr8_image(header, image)
        self.reprojection_pub.publish(image_msg)

    @staticmethod
    def create_bgr8_image(header: Header, image: np.ndarray) -> Image:
        """Create a bgr8 Image without the OpenCV-5 cv_bridge type check."""
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError(
                'bgr8 image must have uint8 HxWx3 shape; '
                f'got dtype={image.dtype}, shape={image.shape}'
            )
        contiguous = np.ascontiguousarray(image)
        return Image(
            header=header,
            height=int(contiguous.shape[0]),
            width=int(contiguous.shape[1]),
            encoding='bgr8',
            is_bigendian=False,
            step=int(contiguous.strides[0]),
            data=contiguous.tobytes(),
        )

    def create_colored_pointcloud2(
        self,
        header: Header,
        points: np.ndarray,
        bgr_colors: np.ndarray
    ) -> PointCloud2:
        blue = bgr_colors[:, 0].astype(np.uint32)
        green = bgr_colors[:, 1].astype(np.uint32)
        red = bgr_colors[:, 2].astype(np.uint32)
        packed_rgb = ((red << 16) | (green << 8) | blue).astype(np.uint32)

        # PointCloud2/PCL은 RGB 비트를 float32 필드에 담는다.
        rgb_as_float = packed_rgb.view(np.float32)
        colored_points = np.column_stack(
            (points, rgb_as_float)
        ).astype(np.float32)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        return PointCloud2(
            header=header,
            height=1,
            width=colored_points.shape[0],
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=16,
            row_step=16 * colored_points.shape[0],
            data=colored_points.tobytes()
        )

    def create_reprojection_image(
        self,
        color_img: np.ndarray,
        mask: np.ndarray,
        source_u: np.ndarray,
        source_v: np.ndarray,
        z: np.ndarray,
        mask_point_count: int
    ) -> np.ndarray:
        image = color_img.copy()
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        if len(z) == 0:
            cv2.putText(
                image,
                'No valid depth points',
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return image

        fx, fy, cx, cy = self.intrinsics
        x = (source_u - cx) * z / fx
        y = (source_v - cy) * z / fy
        projected_u = fx * x / z + cx
        projected_v = fy * y / z + cy

        reprojection_error = np.sqrt(
            (projected_u - source_u) ** 2
            + (projected_v - source_v) ** 2
        )
        rms_error = float(np.sqrt(np.mean(reprojection_error ** 2)))
        valid_ratio = 100.0 * len(z) / max(mask_point_count, 1)

        projected_u = np.rint(projected_u).astype(np.int32)
        projected_v = np.rint(projected_v).astype(np.int32)
        height, width = image.shape[:2]
        inside = (
            (projected_u >= 0)
            & (projected_u < width)
            & (projected_v >= 0)
            & (projected_v < height)
        )

        projected_u = projected_u[inside]
        projected_v = projected_v[inside]
        sample_step = max(1, len(projected_u) // 4000)
        image[
            projected_v[::sample_step],
            projected_u[::sample_step]
        ] = (255, 0, 255)

        cv2.putText(
            image,
            f'Reprojection RMS: {rms_error:.3f} px',
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        cv2.putText(
            image,
            f'Valid depth: {valid_ratio:.1f}%',
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        return image

    def set_display_frame(self, frame):
        with self.display_lock:
            self.display_frame = frame

    def display_loop(self):
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            while not self.display_stop.is_set():
                with self.display_lock:
                    frame = self.display_frame

                if frame is not None:
                    cv2.imshow(self.window_name, frame)

                # 추론과 무관하게 GUI 이벤트를 지속적으로 처리
                cv2.waitKey(10)
        except cv2.error as exc:
            self.get_logger().error(f'OpenCV display error: {exc}')
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    def destroy_node(self):
        self.inference_stop.set()
        self.inference_event.set()
        self.inference_thread.join()
        self.display_stop.set()
        self.display_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
