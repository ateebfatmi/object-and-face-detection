from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO


MODEL_PATH = "yolov8m.pt"
CONFIDENCE = 0.30
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 960
CAMERA_INDEX = 0
FACE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "blaze_face_short_range.tflite"


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    label_y = max(y, height + baseline + 8)
    cv2.rectangle(frame, (x, label_y - height - baseline - 8),
                  (x + width + 10, label_y), color, -1)
    cv2.putText(frame, text, (x + 5, label_y - baseline - 3), font, scale,
                (10, 10, 10), thickness, cv2.LINE_AA)


def open_camera(preferred_index: int) -> tuple[cv2.VideoCapture, object]:
    """Open and warm up a webcam, falling back to other common indexes."""
    indexes = [preferred_index, *[index for index in range(3) if index != preferred_index]]
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY

    for index in indexes:
        camera = cv2.VideoCapture(index, backend)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not camera.isOpened():
            camera.release()
            continue

        # Some macOS cameras need a short warm-up before returning the first frame.
        for _ in range(30):
            ok, frame = camera.read()
            if ok and frame is not None and frame.size:
                print(f"Using camera index {index}")
                return camera, frame
            time.sleep(0.05)
        camera.release()

    raise RuntimeError(
        "Could not read a webcam frame. Close other camera apps and check "
        "System Settings > Privacy & Security > Camera."
    )


def main() -> None:
    model = YOLO(MODEL_PATH)
    camera, pending_frame = open_camera(CAMERA_INDEX)

    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    if not FACE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing MediaPipe face model: {FACE_MODEL_PATH}")

    face_options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.60,
        min_suppression_threshold=0.30,
    )

    try:
        with mp.tasks.vision.FaceDetector.create_from_options(face_options) as face_detector:
            frame_number = 0
            while True:
                if pending_frame is not None:
                    frame, pending_frame = pending_frame, None
                    ok = True
                else:
                    ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("The webcam stopped returning frames.")
                frame_height, frame_width = frame.shape[:2]

                result = model.predict(frame, conf=CONFIDENCE, iou=IOU_THRESHOLD,
                                       imgsz=IMAGE_SIZE, max_det=100, verbose=False)[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].round().int().cpu().tolist()
                        x1, x2 = clamp(x1, 0, frame_width - 1), clamp(x2, 0, frame_width - 1)
                        y1, y2 = clamp(y1, 0, frame_height - 1), clamp(y2, 0, frame_height - 1)
                        confidence = float(box.conf[0].item())
                        name = str(model.names[int(box.cls[0].item())]).title()
                        color = (0, 174, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        draw_label(frame, f"{name} {confidence:.0%}", x1, y1, color)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_pipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                frame_number += 1
                timestamp_ms = int(frame_number * 1000 / 30)
                faces = face_detector.detect_for_video(media_pipe_image, timestamp_ms)
                if faces.detections:
                    for face in faces.detections:
                        box = face.bounding_box
                        x1 = clamp(box.origin_x, 0, frame_width - 1)
                        y1 = clamp(box.origin_y, 0, frame_height - 1)
                        x2 = clamp(x1 + box.width, 0, frame_width - 1)
                        y2 = clamp(y1 + box.height, 0, frame_height - 1)
                        score = float(face.categories[0].score) if face.categories else 0.0
                        color = (50, 220, 80)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        draw_label(frame, f"Face {score:.0%}", x1, y1, color)

                now = time.perf_counter()
                fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                smoothed_fps = fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * fps
                cv2.putText(frame, f"FPS: {smoothed_fps:.1f}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("YOLOv8m + MediaPipe Faces", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
