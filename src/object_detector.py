from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    BYTETRACK_CONFIG,
    BYTETRACK_EXPERIMENTAL_CONFIG,
    YOLO11N_MODEL,
    YOLO11S_MODEL,
    YOLO_WORLD_MODEL,
    ensure_directories,
)


CUSTOM_PROMPTS = ("ballpoint pen", "earbuds", "headphones")
CUSTOM_LABELS = {"ballpoint pen": "pen", "earbuds": "earbuds", "headphones": "headphones"}
MODELS = {
    "fast": (YOLO11N_MODEL, 640),
    "accurate": (YOLO11S_MODEL, 960),
    "hybrid": (YOLO11S_MODEL, 960),
}


@dataclass(slots=True)
class ObjectDetection:
    box: tuple[int, int, int, int]
    name: str
    confidence: float
    tracking_id: int | None = None
    detector_source: str = "yolo11"


class ObjectDetector:
    """Validated YOLO11 detector with an optional, clearly separated YOLO-World pass."""

    def __init__(self, mode: str = "accurate") -> None:
        from ultralytics import YOLO, YOLOWorld

        ensure_directories()
        if mode not in MODELS:
            choices = ", ".join(sorted(MODELS))
            raise ValueError(f"Unknown object detector mode {mode!r}. Expected one of: {choices}.")

        model_path, self.inference_size = MODELS[mode]
        self._require_file(model_path, f"{mode} YOLO11 weights")
        self._require_file(BYTETRACK_CONFIG, "ByteTrack configuration")
        self.model = YOLO(str(model_path))
        self._validate_standard_model()

        self.custom_model = None
        if mode == "hybrid":
            self._require_file(YOLO_WORLD_MODEL, "YOLO-World weights")
            self._require_file(BYTETRACK_EXPERIMENTAL_CONFIG, "experimental ByteTrack configuration")
            self.custom_model = YOLOWorld(str(YOLO_WORLD_MODEL))
            self.custom_model.set_classes(list(CUSTOM_PROMPTS))

        self.mode = mode
        self.model_path = model_path
        self.lock = threading.RLock()

    @staticmethod
    def _require_file(path: Path, role: str) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing {role}: {path}. The application will not silently switch models.")

    def _validate_standard_model(self) -> None:
        names = self.model.names
        if self.model.task != "detect" or len(names) != 80 or str(names.get(0)) != "person":
            raise RuntimeError("The selected YOLO11 checkpoint is not the expected 80-class COCO detection model.")

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Object detection expects a three-channel OpenCV BGR image.")
        if frame.size == 0 or frame.shape[0] < 2 or frame.shape[1] < 2:
            raise ValueError("Object detection received an empty image.")
        if frame.dtype != np.uint8:
            raise ValueError("Object detection expects uint8 pixels in the 0-255 range.")

    def detect(
        self,
        frame: np.ndarray,
        confidence: float = 0.35,
        tracking: bool = False,
        iou_threshold: float = 0.50,
        custom_confidence: float = 0.05,
        max_detections: int = 300,
    ) -> list[ObjectDetection]:
        self._validate_frame(frame)
        if not 0 < confidence <= 1 or not 0 < iou_threshold <= 1:
            raise ValueError("Confidence and IoU thresholds must be in the interval (0, 1].")
        if not 0 < custom_confidence <= 1 or max_detections < 1:
            raise ValueError("Experimental confidence must be in (0, 1] and max detections must be positive.")

        standard_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": confidence,
            "imgsz": self.inference_size,
            "iou": iou_threshold,
            "max_det": max_detections,
            "verbose": False,
        }
        # TTA is useful for single images, but unstable boxes and extra latency
        # make it inappropriate for ByteTrack video streams.
        if self.mode == "accurate" and not tracking:
            standard_kwargs["augment"] = True

        with self.lock:
            if tracking:
                results = self.model.track(
                    **standard_kwargs, persist=True, tracker=str(BYTETRACK_CONFIG)
                )
            else:
                results = self.model.predict(**standard_kwargs)
            standard = self._parse(results, self.model, frame.shape[:2], detector_source="yolo11")
            standard = self._postprocess_standard(standard, frame.shape[:2])

            if self.custom_model is None:
                return standard

            custom_kwargs = {
                **standard_kwargs,
                "conf": custom_confidence,
                "imgsz": 1280,
                "max_det": min(max_detections, 100),
            }
            custom_kwargs.pop("augment", None)
            if tracking:
                custom_results = self.custom_model.track(
                    **custom_kwargs,
                    persist=True,
                    tracker=str(BYTETRACK_EXPERIMENTAL_CONFIG),
                )
            else:
                custom_results = self.custom_model.predict(**custom_kwargs)
            custom = self._parse(
                custom_results,
                self.custom_model,
                frame.shape[:2],
                track_offset=1_000_000,
                detector_source="yolo_world",
            )
            return self._fuse(standard, custom)

    def reset_tracking(self) -> None:
        """Reset persistent ByteTrack state before a new camera/video source."""
        with self.lock:
            for model in (self.model, self.custom_model):
                predictor = getattr(model, "predictor", None) if model is not None else None
                trackers = getattr(predictor, "trackers", ()) or ()
                for tracker in trackers:
                    reset = getattr(tracker, "reset", None)
                    if callable(reset):
                        reset()
                if predictor is not None and hasattr(predictor, "vid_path"):
                    predictor.vid_path = [None] * max(len(trackers), 1)

    @staticmethod
    def _parse(
        results: Any,
        model: Any,
        frame_shape: tuple[int, int],
        track_offset: int = 0,
        detector_source: str = "yolo11",
    ) -> list[ObjectDetection]:
        detections: list[ObjectDetection] = []
        if not results or results[0].boxes is None:
            return detections

        height, width = frame_shape
        boxes = results[0].boxes
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        rows = zip(boxes.xyxy.cpu().tolist(), boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist(), ids)
        for xyxy, cls_id, confidence, track_id in rows:
            if not math.isfinite(float(confidence)) or cls_id not in model.names:
                continue
            x1 = min(max(int(round(xyxy[0])), 0), width)
            y1 = min(max(int(round(xyxy[1])), 0), height)
            x2 = min(max(int(round(xyxy[2])), 0), width)
            y2 = min(max(int(round(xyxy[3])), 0), height)
            if x2 <= x1 or y2 <= y1:
                continue
            raw_name = str(model.names[cls_id]).strip().lower()
            display_name = CUSTOM_LABELS.get(raw_name, raw_name)
            if track_id is not None:
                track_id = int(track_id) + track_offset
            detections.append(ObjectDetection(
                (x1, y1, x2, y2), display_name, float(confidence), track_id, detector_source
            ))
        return detections

    @staticmethod
    def _postprocess_standard(
        detections: list[ObjectDetection], frame_shape: tuple[int, int]
    ) -> list[ObjectDetection]:
        """Correct conservative, repeatable COCO confusions in webcam scenes."""
        frame_height, frame_width = frame_shape
        frame_area = max(frame_width * frame_height, 1)
        corrected: list[ObjectDetection] = []
        for item in detections:
            x1, y1, x2, y2 = item.box
            width, height = max(x2 - x1, 1), max(y2 - y1, 1)
            elongation = max(width, height) / min(width, height)
            area_ratio = (width * height) / frame_area

            # COCO has a clock class but no wristwatch class. A close-up watch
            # is generally a small portrait box, while a wall clock is close
            # to square. Preserve the confidence, track, and detector source.
            if item.name == "clock" and height > width and elongation >= 1.25 and area_ratio < 0.08:
                corrected.append(ObjectDetection(
                    item.box, "watch", item.confidence, item.tracking_id, item.detector_source
                ))
                continue

            # Weak, nearly square phone boxes are commonly produced by hands,
            # sleeves, furniture edges, and the wrist around a detected watch.
            if item.name == "cell phone" and item.confidence < 0.70 and elongation < 1.45:
                continue
            corrected.append(item)

        watches = [item for item in corrected if item.name == "watch"]
        if not watches:
            return corrected
        return [item for item in corrected if not (
            item.name == "cell phone"
            and any(
                ObjectDetector._overlap(item.box, watch.box) > 0.35
                and item.confidence <= watch.confidence + 0.10
                for watch in watches
            )
        )]

    @staticmethod
    def _fuse(standard: list[ObjectDetection], custom: list[ObjectDetection]) -> list[ObjectDetection]:
        """Class-aware fusion for the optional open-vocabulary pass."""
        custom_kept: list[ObjectDetection] = []
        for candidate in sorted(custom, key=lambda item: item.confidence, reverse=True):
            if any(
                candidate.name == other.name and ObjectDetector._iou(candidate.box, other.box) >= 0.50
                for other in custom_kept
            ):
                continue
            custom_kept.append(candidate)

        standard_kept = list(standard)
        for candidate in custom_kept:
            if candidate.name == "pen":
                standard_kept = [item for item in standard_kept if not (
                    item.name in {"cell phone", "scissors"}
                    and ObjectDetector._overlap(item.box, candidate.box) >= 0.65
                )]
        return standard_kept + custom_kept

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        intersection = ObjectDetector._intersection(a, b)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        return intersection / max(area_a + area_b - intersection, 1)

    @staticmethod
    def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        """Intersection over the smaller box, useful for nested detections."""
        intersection = ObjectDetector._intersection(a, b)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        return intersection / max(min(area_a, area_b), 1)

    @staticmethod
    def _intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def draw(frame: np.ndarray, detections: list[ObjectDetection]) -> np.ndarray:
        for item in detections:
            x1, y1, x2, y2 = item.box
            if item.detector_source == "yolo_world":
                color = (255, 190, 70)
            else:
                color = (46, 204, 113) if item.name == "person" else (0, 174, 255)
            suffix = f" #{item.tracking_id}" if item.tracking_id is not None else ""
            source = " · WORLD" if item.detector_source == "yolo_world" else ""
            label = f"{item.name.title()}{suffix} - {item.confidence:.0%}{source}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_right = min(frame.shape[1], x1 + max(120, len(label) * 8))
            cv2.rectangle(frame, (x1, max(0, y1 - 26)), (label_right, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, max(14, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .52, (10, 15, 25), 1, cv2.LINE_AA)
        return frame
