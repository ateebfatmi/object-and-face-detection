from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import MODELS_DIR, ensure_directories


@dataclass(slots=True)
class ObjectDetection:
    box: tuple[int, int, int, int]
    name: str
    confidence: float
    tracking_id: int | None = None


class ObjectDetector:
    def __init__(self, mode: str = "accurate") -> None:
        from ultralytics import YOLO, YOLOWorld
        ensure_directories()
        if mode == "extended":
            self.model = YOLOWorld(str(MODELS_DIR / "yolov8s-worldv2.pt"))
            self.model.set_classes([
                "person", "headphones", "earbuds", "cell phone", "bottle", "laptop",
                "backpack", "chair", "blue ballpoint pen",
            ])
        elif mode == "fast":
            self.model = YOLO(str(MODELS_DIR / "yolo11n.pt"))
        else:
            self.model = YOLO(str(MODELS_DIR / "yolo11s.pt"))
        self.mode = mode
        self.lock = threading.Lock()

    def detect(self, frame: np.ndarray, confidence: float = 0.5, tracking: bool = False) -> list[ObjectDetection]:
        # Open-vocabulary confidence is not calibrated like COCO confidence; this
        # focused model scores clear headphones around 0.05-0.10.
        effective_confidence = min(confidence, 0.02) if self.mode == "extended" else confidence
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": effective_confidence,
            "imgsz": 1280 if self.mode == "extended" else 768,
            "iou": 0.55,
            "verbose": False,
        }
        with self.lock:
            tracker_config = str(MODELS_DIR / "bytetrack_extended.yaml") if self.mode == "extended" else "bytetrack.yaml"
            results = self.model.track(**kwargs, persist=True, tracker=tracker_config) if tracking else self.model.predict(**kwargs)
        detections: list[ObjectDetection] = []
        if not results or results[0].boxes is None:
            return detections
        boxes = results[0].boxes
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        for xyxy, cls_id, conf, track_id in zip(boxes.xyxy.int().cpu().tolist(), boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist(), ids):
            raw_name = str(self.model.names[cls_id])
            display_name = "pen" if raw_name == "blue ballpoint pen" else raw_name
            detections.append(ObjectDetection(tuple(xyxy), display_name, float(conf), track_id))
        return detections

    @staticmethod
    def draw(frame: np.ndarray, detections: list[ObjectDetection]) -> np.ndarray:
        for item in detections:
            x1, y1, x2, y2 = item.box
            color = (46, 204, 113) if item.name == "person" else (0, 174, 255)
            suffix = f" #{item.tracking_id}" if item.tracking_id is not None else ""
            label = f"{item.name.title()}{suffix} - {item.confidence:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 26)), (x1 + max(120, len(label) * 8), y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, .52, (10, 15, 25), 1, cv2.LINE_AA)
        return frame
