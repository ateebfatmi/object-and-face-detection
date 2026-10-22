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
        self.custom_model = None
        if mode in {"hybrid", "extended"}:
            self.model = YOLO(str(MODELS_DIR / "yolo11s.pt"))
            self.custom_model = YOLOWorld(str(MODELS_DIR / "yolov8s-worldv2.pt"))
            self.custom_model.set_classes([
                "headphones", "earbuds", "blue ballpoint pen",
                "blue scissors", "pair of scissors", "scissors held in hand",
            ])
        elif mode == "fast":
            self.model = YOLO(str(MODELS_DIR / "yolo11n.pt"))
        else:
            self.model = YOLO(str(MODELS_DIR / "yolo11s.pt"))
        self.mode = mode
        self.lock = threading.Lock()

    def detect(self, frame: np.ndarray, confidence: float = 0.5, tracking: bool = False) -> list[ObjectDetection]:
        standard_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": confidence,
            "imgsz": 768,
            "iou": 0.55,
            "verbose": False,
        }
        with self.lock:
            results = self.model.track(**standard_kwargs, persist=True, tracker="bytetrack.yaml") if tracking else self.model.predict(**standard_kwargs)
            detections = self._parse(results, self.model)
            if self.custom_model is not None:
                custom_kwargs = {**standard_kwargs, "conf": 0.02, "imgsz": 1280, "iou": 0.45}
                custom_tracker = str(MODELS_DIR / "bytetrack_extended.yaml")
                custom_results = (self.custom_model.track(**custom_kwargs, persist=True, tracker=custom_tracker)
                                  if tracking else self.custom_model.predict(**custom_kwargs))
                custom = self._parse(custom_results, self.custom_model, track_offset=1_000_000)
                detections.extend(self._remove_overlaps(custom, detections))
        return detections

    @staticmethod
    def _parse(results: Any, model: Any, track_offset: int = 0) -> list[ObjectDetection]:
        detections: list[ObjectDetection] = []
        if not results or results[0].boxes is None:
            return detections
        boxes = results[0].boxes
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        for xyxy, cls_id, conf, track_id in zip(boxes.xyxy.int().cpu().tolist(), boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist(), ids):
            raw_name = str(model.names[cls_id])
            if raw_name == "blue ballpoint pen":
                display_name = "pen"
            elif "scissors" in raw_name:
                display_name = "scissors"
            else:
                display_name = raw_name
            if track_id is not None:
                track_id += track_offset
            detections.append(ObjectDetection(tuple(xyxy), display_name, float(conf), track_id))
        return detections

    @staticmethod
    def _remove_overlaps(candidates: list[ObjectDetection], standard: list[ObjectDetection]) -> list[ObjectDetection]:
        """Suppress duplicate custom boxes while preserving genuinely new classes."""
        kept: list[ObjectDetection] = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            # Different text prompts can describe the same custom object.
            if any(ObjectDetector._iou(candidate.box, other.box) > 0.50 for other in kept):
                continue
            if any(ObjectDetector._iou(candidate.box, other.box) > 0.65 and candidate.name == other.name
                   for other in standard):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        return intersection / max(area_a + area_b - intersection, 1)

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
