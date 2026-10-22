from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
UNKNOWN_FACES_DIR = DATA_DIR / "unknown_faces"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
DATABASE_DIR = ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "visionguard.db"

YOLO11N_MODEL = MODELS_DIR / "yolo11n.pt"
YOLO11S_MODEL = MODELS_DIR / "yolo11s.pt"
YOLO_WORLD_MODEL = MODELS_DIR / "yolov8s-worldv2.pt"
OBJECTS365_MODEL = MODELS_DIR / "yolo26s-objv1-150.pt"
DAILY_OBJECTS_MODEL = MODELS_DIR / "daily_objects.pt"
KEY_OBJECT_MODEL = MODELS_DIR / "key_objects.pt"
BYTETRACK_CONFIG = MODELS_DIR / "bytetrack.yaml"
BYTETRACK_EXPERIMENTAL_CONFIG = MODELS_DIR / "bytetrack_extended.yaml"

YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def ensure_directories() -> None:
    for path in (
        KNOWN_FACES_DIR, UNKNOWN_FACES_DIR, SCREENSHOTS_DIR, UPLOADS_DIR,
        PROCESSED_DIR, MODELS_DIR, DATABASE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class DetectionSettings:
    object_detection: bool = True
    face_detection: bool = True
    face_recognition: bool = True
    tracking: bool = True
    save_unknown_faces: bool = False
    capture_screenshots: bool = False
    confidence: float = 0.35
    iou_threshold: float = 0.5
    custom_confidence: float = 0.02
    max_detections: int = 300
    face_threshold: float = 0.45
    # Preserve detail for small objects before YOLO performs its own letterbox
    # resize. Individual real-time views can still select a smaller width.
    max_width: int = 1440
    frame_stride: int = 1
    # Live demo safety: non-person objects must agree across this many
    # consecutive frames before they are drawn or persisted.
    temporal_confirmation_frames: int = 1
