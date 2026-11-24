
import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# 절대 경로로 OpenVINO 모델 로드
model_path = os.path.expanduser('~/grasp_ws/weights/best_openvino_model')
model = YOLO(model_path, task="segment")

# RealSense D435i 파이프라인 시작
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("RealSense started. Press 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        t0 = time.perf_counter()
        results = model.predict(frame, device="cpu", imgsz=640, conf=0.5, verbose=False)
        fps = 1.0 / (time.perf_counter() - t0)

        annotated = results[0].plot()
        cv2.putText(annotated, f"OpenVINO FP16 FPS: {fps:.1f}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("YOLOv8s-seg OpenVINO Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

