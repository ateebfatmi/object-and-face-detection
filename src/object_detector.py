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
                "headphones", "earbuds", "ballpoint pen", "pen held in hand",
            ])
        elif mode == "fast":
            self.model = YOLO(str(MODELS_DIR / "yolo11n.pt"))
        else:
            self.model = YOLO(str(MODELS_DIR / "yolo11s.pt"))
        self.mode = mode
        # The old fixed 768px inference size discarded detail even when the
        # user selected the larger model. Keep the fast profile inexpensive,
        # while giving the accurate profiles enough pixels for small objects.
        self.inference_size = 640 if mode == "fast" else 960
        self.lock = threading.Lock()

    def detect(self, frame: np.ndarray, confidence: float = 0.5, tracking: bool = False) -> list[ObjectDetection]:
        standard_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": confidence,
            "imgsz": self.inference_size,
            "iou": 0.50,
            "max_det": 300,
            "verbose": False,
        }
        # Test-time augmentation improves recall for a single uploaded image.
        # It is deliberately disabled for tracking, where stable boxes and
        # frame rate matter more than the small recall gain.
        if self.mode == "accurate" and not tracking:
            standard_kwargs["augment"] = True
        with self.lock:
            results = self.model.track(**standard_kwargs, persist=True, tracker="bytetrack.yaml") if tracking else self.model.predict(**standard_kwargs)
            detections = self._filter_standard(self._parse(results, self.model), frame.shape[:2])
            detections = self._resolve_watch_phone_conflicts(detections)
            if self.custom_model is not None:
                # YOLO-World scores are lower than closed-set COCO scores, but
                # accepting 0.02 produced obvious false positives and stacks of
                # competing prompt boxes. Keep a conservative retrieval floor,
                # then apply class/shape validation below.
                custom_kwargs = {**standard_kwargs, "conf": 0.04, "imgsz": 1280, "iou": 0.40, "max_det": 40}
                custom_tracker = str(MODELS_DIR / "bytetrack_extended.yaml")
                custom_results = (self.custom_model.track(**custom_kwargs, persist=True, tracker=custom_tracker)
                                  if tracking else self.custom_model.predict(**custom_kwargs))
                custom = self._parse(custom_results, self.custom_model, track_offset=1_000_000)
                custom = self._filter_custom(custom, frame.shape[:2])
                custom = self._remove_overlaps(custom, detections)
                detections = self._resolve_pen_scissors_conflicts(detections, custom)
                detections.extend(custom)
        return detections

    @staticmethod
    def _filter_standard(detections: list[ObjectDetection], frame_shape: tuple[int, int]) -> list[ObjectDetection]:
        """Remove high-frequency COCO confusions with implausible geometry.

        COCO has no pen class. In desk/webcam scenes, a hand holding a pen and
        the edge of a chair are commonly returned as a portrait-shaped laptop
        or a very large pair of scissors. These conservative rules only remove
        those characteristic weak/oversized boxes.
        """
        frame_height, frame_width = frame_shape
        frame_area = max(frame_width * frame_height, 1)
        output: list[ObjectDetection] = []
        for item in detections:
            x1, y1, x2, y2 = item.box
            width, height = max(x2 - x1, 1), max(y2 - y1, 1)
            area_ratio = (width * height) / frame_area
            if item.name == "laptop" and height > width * 1.25 and item.confidence < 0.60:
                continue
            if item.name == "clock":
                elongation = max(width, height) / min(width, height)
                # COCO has a clock class but no wristwatch class. A close-up
                # wristwatch normally produces a small portrait-oriented box,
                # unlike the near-square box of a wall clock.
                if height > width and elongation >= 1.25 and area_ratio < 0.08:
                    output.append(ObjectDetection(item.box, "watch", item.confidence, item.tracking_id))
                    continue
            if item.name == "cell phone":
                elongation = max(width, height) / min(width, height)
                # Weak, almost-square phone boxes are usually hands, sleeves,
                # furniture edges, or the wrist around a detected watch.
                if item.confidence < 0.70 and elongation < 1.45:
                    continue
            if item.name == "scissors":
                elongation = max(width, height) / min(width, height)
                # COCO does not contain a pen class, so YOLO commonly maps a
                # tightly boxed pen to its nearest thin handheld class. Real
                # scissors normally have visible handles and a wider box.
                if elongation >= 3.0 and area_ratio < 0.06:
                    output.append(ObjectDetection(item.box, "pen", item.confidence, item.tracking_id))
                    continue
                if height > width * 1.60 and area_ratio > 0.08 and item.confidence < 0.85:
                    continue
            output.append(item)
        return output

    @staticmethod
    def _resolve_watch_phone_conflicts(detections: list[ObjectDetection]) -> list[ObjectDetection]:
        """Remove a phone hallucination covering the same wristwatch region."""
        watches = [item for item in detections if item.name == "watch"]
        if not watches:
            return detections
        return [item for item in detections if not (
            item.name == "cell phone"
            and any(
                ObjectDetector._overlap(item.box, watch.box) > 0.35
                and item.confidence <= watch.confidence + 0.10
                for watch in watches
            )
        )]

    @staticmethod
    def _parse(results: Any, model: Any, track_offset: int = 0) -> list[ObjectDetection]:
        detections: list[ObjectDetection] = []
        if not results or results[0].boxes is None:
            return detections
        boxes = results[0].boxes
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        for xyxy, cls_id, conf, track_id in zip(boxes.xyxy.int().cpu().tolist(), boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist(), ids):
            raw_name = str(model.names[cls_id])
            if "pen" in raw_name:
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
            if any(ObjectDetector._overlap(candidate.box, other.box) > 0.45 for other in kept):
                continue
            if any(ObjectDetector._overlap(candidate.box, other.box) > 0.60 and candidate.name == other.name
                   for other in standard):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _filter_custom(detections: list[ObjectDetection], frame_shape: tuple[int, int] | None = None) -> list[ObjectDetection]:
        """Reject weak or geometrically implausible open-vocabulary results."""
        minimum_confidence = {"pen": 0.055, "headphones": 0.08, "earbuds": 0.06}
        output: list[ObjectDetection] = []
        for item in detections:
            if item.confidence < minimum_confidence.get(item.name, 0.08):
                continue
            x1, y1, x2, y2 = item.box
            width, height = max(x2 - x1, 1), max(y2 - y1, 1)
            elongation = max(width, height) / min(width, height)
            if frame_shape:
                frame_height, frame_width = frame_shape
                area_ratio = (width * height) / max(frame_width * frame_height, 1)
                # Accessory prompts occasionally latch onto nearly the entire
                # image. Such a box cannot be a pen, earbud, or headphones.
                if area_ratio > 0.45:
                    continue
            # A pen should be visibly elongated. This removes the large hand,
            # face, and background regions that YOLO-World sometimes calls pens.
            if item.name == "pen" and (elongation < 2.4 or max(width, height) < 12):
                continue
            output.append(item)
        return output

    @staticmethod
    def _resolve_pen_scissors_conflicts(standard: list[ObjectDetection], custom: list[ObjectDetection]) -> list[ObjectDetection]:
        """Prefer a validated pen over an overlapping COCO scissors false positive."""
        pens = [item for item in custom if item.name == "pen"]
        if not pens:
            return standard
        return [item for item in standard if not (
            item.name == "scissors" and item.confidence < 0.50
            and any(ObjectDetector._overlap(item.box, pen.box) > 0.42 for pen in pens)
        )]

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        return intersection / max(area_a + area_b - intersection, 1)

    @staticmethod
    def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        """Intersection over the smaller box; better than IoU for nested thin objects."""
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        return intersection / max(min(area_a, area_b), 1)

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
