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
    DAILY_OBJECTS_MODEL,
    KEY_OBJECT_MODEL,
    OBJECTS365_MODEL,
    YOLO11N_MODEL,
    YOLO11S_MODEL,
    YOLO_WORLD_MODEL,
    ensure_directories,
)


CUSTOM_PROMPTS = (
    "bottle",
    "water bottle",
    "metal water bottle",
    "stainless steel bottle",
    "thermos bottle",
    "flask",
    "electric ceiling fan",
    "car key with black keychain",
    # A visual description works materially better than the bare COCO-missing
    # noun for thin pens held in front of a webcam.
    "long thin pen",
    "earbuds",
    "headphones",
)
ACCESSORY_PROMPTS = ("long thin pen", "earbuds", "headphones")
BACKPACK_VERIFY_PROMPTS = ("backpack", "hanging clothes")
WATCH_VERIFY_PROMPTS = (
    "bottle",
    "water bottle",
    "analog wristwatch",
    "wrist watch",
    "handheld wristwatch",
)
CUSTOM_LABELS = {
    "bottle": "bottle",
    "water bottle": "bottle",
    "metal water bottle": "bottle",
    "stainless steel bottle": "bottle",
    "thermos bottle": "bottle",
    "flask": "bottle",
    "electric ceiling fan": "fan",
    "long thin pen": "pen",
    "earbuds": "earbuds",
    "headphones": "headphones",
    "car key with black keychain": "key",
    "ceiling fan": "fan",
    "mobile phone": "cell phone",
    "coffee cup": "cup",
    "pen/pencil": "pen",
    "analog wristwatch": "watch",
    "wrist watch": "watch",
    "handheld wristwatch": "watch",
}

DAILY_MODEL_CLASSES = (
    "backpack", "baseball bat", "bottle", "ceiling fan", "coffee cup",
    "headphones", "laptop", "mobile phone", "pen", "watch",
)
# Conservative per-class gates measured against the held-out Open Images test
# split. Demo Safe Mode adds temporal confirmation on top of these thresholds.
DAILY_THRESHOLDS = {
    "backpack": 0.45,
    "baseball bat": 0.30,
    "bottle": 0.30,
    "fan": 0.25,
    "cup": 0.30,
    "headphones": 0.25,
    "laptop": 0.35,
    "cell phone": 0.40,
    "pen": 0.12,
    "watch": 0.25,
}
KEY_MODEL_THRESHOLD = 0.20

# These are common COCO-only guesses for objects that the targeted World pass
# can describe more precisely. Suppression only happens when the boxes overlap
# strongly, so unrelated objects remain visible.
CUSTOM_OVERRIDES = {
    "bottle": {"remote", "cell phone", "cup", "vase"},
    "fan": {"airplane", "umbrella", "clock", "cell phone"},
    "pen": {"baseball bat", "cell phone", "scissors"},
    "key": {"bottle", "cell phone", "remote", "scissors"},
    "watch": {"bottle", "cell phone", "clock"},
}
MODELS = {
    "fast": (YOLO11N_MODEL, 640),
    "accurate": (YOLO11S_MODEL, 960),
    "hybrid": (YOLO11S_MODEL, 960),
    "diverse": (OBJECTS365_MODEL, 768),
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
        self._validate_primary_model(mode)

        self.custom_model = None
        self.daily_model = None
        self.key_model = None
        if mode in {"hybrid", "diverse"}:
            self._require_file(YOLO_WORLD_MODEL, "YOLO-World weights")
            self._require_file(BYTETRACK_EXPERIMENTAL_CONFIG, "experimental ByteTrack configuration")
            self.custom_model = YOLOWorld(str(YOLO_WORLD_MODEL))
            self.custom_model.set_classes(list(CUSTOM_PROMPTS))
            if DAILY_OBJECTS_MODEL.is_file() and DAILY_OBJECTS_MODEL.stat().st_size > 0:
                self.daily_model = YOLO(str(DAILY_OBJECTS_MODEL))
                self._validate_daily_model()
            if KEY_OBJECT_MODEL.is_file() and KEY_OBJECT_MODEL.stat().st_size > 0:
                self.key_model = YOLO(str(KEY_OBJECT_MODEL))
                self._validate_key_model()

        self.mode = mode
        self.model_path = model_path
        self.lock = threading.RLock()

    @staticmethod
    def _require_file(path: Path, role: str) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing {role}: {path}. The application will not silently switch models.")

    def _validate_primary_model(self, mode: str) -> None:
        names = self.model.names
        expected_classes = 365 if mode == "diverse" else 80
        expected_role = "Objects365" if mode == "diverse" else "COCO"
        if (
            self.model.task != "detect"
            or len(names) != expected_classes
            or str(names.get(0)).strip().lower() != "person"
        ):
            raise RuntimeError(
                f"The selected primary checkpoint is not the expected {expected_classes}-class {expected_role} model."
            )

    def _validate_daily_model(self) -> None:
        names = tuple(str(self.daily_model.names[index]).strip().lower() for index in range(10))
        if self.daily_model.task != "detect" or names != DAILY_MODEL_CLASSES:
            raise RuntimeError(
                "daily_objects.pt does not contain the expected daily-use classes."
            )

    def _validate_key_model(self) -> None:
        names = tuple(str(value).strip().lower() for value in self.key_model.names.values())
        if self.key_model.task != "detect" or names != ("key",):
            raise RuntimeError("key_objects.pt is not the expected one-class key detector.")

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
        custom_confidence: float = 0.02,
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
            primary_source = "objects365" if self.mode == "diverse" else "yolo11"
            standard = self._parse(results, self.model, frame.shape[:2], detector_source=primary_source)
            standard = self._postprocess_standard(standard, frame.shape[:2])

            if self.custom_model is None:
                return standard

            standard = self._remove_verified_clothing_backpacks(frame, standard)

            daily: list[ObjectDetection] = []
            if self.daily_model is not None:
                daily_result = self.daily_model.predict(
                    source=frame,
                    conf=min(DAILY_THRESHOLDS.values()),
                    imgsz=768,
                    iou=iou_threshold,
                    max_det=min(max_detections, 100),
                    verbose=False,
                )
                daily = self._parse(
                    daily_result,
                    self.daily_model,
                    frame.shape[:2],
                    detector_source="daily_yolo",
                )
                daily = self._postprocess_daily(daily, frame.shape[:2])

            supervised_keys: list[ObjectDetection] = []
            if self.key_model is not None:
                key_result = self.key_model.predict(
                    source=frame,
                    conf=0.05,
                    imgsz=768,
                    iou=iou_threshold,
                    max_det=min(max_detections, 30),
                    verbose=False,
                )
                supervised_keys = self._parse(
                    key_result,
                    self.key_model,
                    frame.shape[:2],
                    detector_source="key_yolo",
                )
                supervised_keys = self._postprocess_keys(supervised_keys)

            custom_kwargs = {
                "conf": custom_confidence,
                "imgsz": 960,
                "iou": iou_threshold,
                "max_det": min(max_detections, 100),
                "verbose": False,
            }
            # Run the full frame independently for large scene objects. Mixing
            # its aspect ratio with portrait tiles in one batch adds padding
            # and can erase the weak bottle/fan signals this pass is for.
            full_result = self.custom_model.predict(source=frame, **custom_kwargs)
            custom = self._parse(
                full_result,
                self.custom_model,
                frame.shape[:2],
                detector_source="yolo_world",
            )

            # Two overlapping vertical tiles preserve small-accessory detail.
            # Large-object detections from tiles are ignored because tile edges
            # commonly produce partial bottle/fan false positives.
            tiles = self._custom_tiles(frame)
            self.custom_model.set_classes(list(ACCESSORY_PROMPTS))
            try:
                custom_results = self.custom_model.predict(
                    source=[tile for tile, _, _ in tiles], **custom_kwargs
                )
                for result, (tile, offset_x, offset_y) in zip(custom_results, tiles):
                    parsed = self._parse(
                        [result],
                        self.custom_model,
                        tile.shape[:2],
                        detector_source="yolo_world",
                    )
                    custom.extend(self._offset_detections(parsed, offset_x, offset_y))
            finally:
                # Focused bottle/fan verification below needs the complete
                # prompt set, and the cached model persists between frames.
                self.custom_model.set_classes(list(CUSTOM_PROMPTS))

            # Reflective bottles are often localized very well by COCO as a
            # phone/remote but classified incorrectly in the full scene. Ask
            # YOLO-World to verify that already-localized region at a lower
            # threshold; relabel only when the second model agrees.
            focus_kwargs = {**custom_kwargs, "conf": min(custom_confidence, 0.01)}
            for standard_item, target_name in self._focus_candidates(standard):
                x1, y1, x2, y2 = standard_item.box
                crop = frame[y1:y2, x1:x2]
                focused_result = self.custom_model.predict(source=crop, **focus_kwargs)
                focused = self._parse(
                    focused_result,
                    self.custom_model,
                    crop.shape[:2],
                    detector_source="yolo_world",
                )
                confirmations = [item for item in focused if item.name == target_name]
                if confirmations:
                    best = max(confirmations, key=lambda item: item.confidence)
                    custom.append(ObjectDetection(
                        standard_item.box,
                        target_name,
                        best.confidence,
                        None,
                        "yolo_world",
                    ))
            custom = self._replace_bottles_verified_as_watches(frame, custom)
            custom = self._postprocess_custom(custom, frame.shape[:2])
            # Once the supervised specialist is available, do not surface the
            # much noisier open-vocabulary key guesses. The specialist was
            # explicitly trained against pen and bottle hard negatives.
            if self.key_model is not None:
                custom = [item for item in custom if item.name != "key"]
            return self._fuse(standard, daily + supervised_keys + custom)

    @staticmethod
    def _postprocess_daily(
        detections: list[ObjectDetection], frame_shape: tuple[int, int]
    ) -> list[ObjectDetection]:
        """Apply held-out confidence gates and simple physical-shape checks."""
        frame_height, frame_width = frame_shape
        frame_area = max(frame_width * frame_height, 1)
        kept: list[ObjectDetection] = []
        for item in detections:
            if item.confidence < DAILY_THRESHOLDS.get(item.name, 1.0):
                continue
            x1, y1, x2, y2 = item.box
            width, height = max(x2 - x1, 1), max(y2 - y1, 1)
            elongation = max(width, height) / min(width, height)
            area_ratio = width * height / frame_area
            if item.name in {"pen", "baseball bat"} and elongation < 2.0:
                continue
            if item.name == "bottle" and (height <= width or elongation < 1.25):
                continue
            if item.name == "fan" and (elongation > 2.5 or area_ratio < 0.005):
                continue
            if item.name == "watch" and area_ratio > 0.12:
                continue
            kept.append(item)
        return kept

    @staticmethod
    def _postprocess_keys(detections: list[ObjectDetection]) -> list[ObjectDetection]:
        """Use the held-out hard-negative operating point for key boxes."""
        kept: list[ObjectDetection] = []
        for item in detections:
            x1, y1, x2, y2 = item.box
            width, height = max(x2 - x1, 1), max(y2 - y1, 1)
            elongation = max(width, height) / min(width, height)
            if item.confidence >= KEY_MODEL_THRESHOLD and elongation >= 1.35:
                kept.append(item)
        return kept

    def _replace_bottles_verified_as_watches(
        self, frame: np.ndarray, detections: list[ObjectDetection]
    ) -> list[ObjectDetection]:
        """Disambiguate bottle/key candidates using a close-up prompt ensemble.

        Watches are small in a webcam frame, so verifying against the complete
        scene repeats the same scale problem that caused the original bottle
        guess.  Each candidate is cropped with context, enlarged by inference,
        and compared against several equivalent watch descriptions.
        """
        candidates = [item for item in detections if item.name in {"bottle", "key"}]
        if not candidates or self.custom_model is None:
            return detections

        replacements: dict[int, ObjectDetection] = {}
        frame_height, frame_width = frame.shape[:2]
        self.custom_model.set_classes(list(WATCH_VERIFY_PROMPTS))
        try:
            for candidate in candidates:
                x1, y1, x2, y2 = candidate.box
                pad_x = max(int((x2 - x1) * 0.35), 8)
                pad_y = max(int((y2 - y1) * 0.25), 8)
                crop_x1, crop_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                crop_x2, crop_y2 = min(frame_width, x2 + pad_x), min(frame_height, y2 + pad_y)
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size == 0:
                    continue
                result = self.custom_model.predict(
                    source=crop,
                    conf=0.005,
                    imgsz=640,
                    iou=0.50,
                    max_det=20,
                    verbose=False,
                )
                verified = self._parse(
                    result,
                    self.custom_model,
                    crop.shape[:2],
                    detector_source="yolo_world",
                )
                verified = self._offset_detections(verified, crop_x1, crop_y1)
                nearby = [
                    item for item in verified
                    if ObjectDetector._overlap(item.box, candidate.box) >= 0.35
                ]
                bottle_score = max(
                    (item.confidence for item in nearby if item.name == "bottle"),
                    default=0.0,
                )
                watches = [item for item in nearby if item.name == "watch"]
                if not watches:
                    continue
                watch = max(watches, key=lambda item: item.confidence)
                if self._watch_evidence_wins(bottle_score, watch.confidence):
                    replacements[id(candidate)] = ObjectDetection(
                        candidate.box,
                        "watch",
                        watch.confidence,
                        candidate.tracking_id,
                        "yolo_world",
                    )
        finally:
            self.custom_model.set_classes(list(CUSTOM_PROMPTS))
        return [replacements.get(id(item), item) for item in detections]

    @staticmethod
    def _watch_evidence_wins(bottle_score: float, watch_score: float) -> bool:
        """Require a useful margin before changing a model-backed bottle label."""
        return watch_score >= 0.025 and watch_score >= max(
            bottle_score * 1.10,
            bottle_score + 0.005,
        )

    def _remove_verified_clothing_backpacks(
        self, frame: np.ndarray, detections: list[ObjectDetection]
    ) -> list[ObjectDetection]:
        """Use YOLO-World to reject standard backpack boxes that are hanging clothes."""
        backpacks = [item for item in detections if item.name == "backpack"]
        if not backpacks or self.custom_model is None:
            return detections

        rejected_ids: set[int] = set()
        frame_height, frame_width = frame.shape[:2]
        self.custom_model.set_classes(list(BACKPACK_VERIFY_PROMPTS))
        try:
            for candidate in backpacks:
                x1, y1, x2, y2 = candidate.box
                pad_x = max(int((x2 - x1) * 0.20), 4)
                pad_y = max(int((y2 - y1) * 0.15), 4)
                crop = frame[
                    max(0, y1 - pad_y):min(frame_height, y2 + pad_y),
                    max(0, x1 - pad_x):min(frame_width, x2 + pad_x),
                ]
                if crop.size == 0:
                    continue
                result = self.custom_model.predict(
                    source=crop,
                    conf=0.005,
                    imgsz=640,
                    iou=0.50,
                    max_det=20,
                    verbose=False,
                )
                verified = self._parse(
                    result,
                    self.custom_model,
                    crop.shape[:2],
                    detector_source="yolo_world",
                )
                backpack_score = max(
                    (item.confidence for item in verified if item.name == "backpack"),
                    default=0.0,
                )
                clothing_score = max(
                    (item.confidence for item in verified if item.name == "hanging clothes"),
                    default=0.0,
                )
                if self._clothing_evidence_wins(backpack_score, clothing_score):
                    rejected_ids.add(id(candidate))
        finally:
            self.custom_model.set_classes(list(CUSTOM_PROMPTS))
        return [item for item in detections if id(item) not in rejected_ids]

    @staticmethod
    def _clothing_evidence_wins(backpack_score: float, clothing_score: float) -> bool:
        """Require meaningful and clearly stronger evidence before suppression."""
        return clothing_score >= 0.015 and clothing_score >= max(
            backpack_score * 1.5,
            backpack_score + 0.005,
        )

    @staticmethod
    def _custom_tiles(frame: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
        """Return overlapping tiles for small open-vocabulary objects."""
        height, width = frame.shape[:2]
        if width < int(height * 1.15):
            return [(frame, 0, 0)]
        tile_width = max(2, int(round(width * 0.62)))
        right_start = width - tile_width
        return [(frame[:, :tile_width], 0, 0), (frame[:, right_start:], right_start, 0)]

    @staticmethod
    def _offset_detections(
        detections: list[ObjectDetection], offset_x: int, offset_y: int
    ) -> list[ObjectDetection]:
        return [
            ObjectDetection(
                (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                item.name,
                item.confidence,
                item.tracking_id,
                item.detector_source,
            )
            for item in detections
            for x1, y1, x2, y2 in (item.box,)
        ]

    @staticmethod
    def _focus_candidates(
        detections: list[ObjectDetection],
    ) -> list[tuple[ObjectDetection, str]]:
        """Return plausible COCO confusions worth a targeted second look."""
        candidates: list[tuple[ObjectDetection, str]] = []
        for item in detections:
            x1, y1, x2, y2 = item.box
            width, height = max(x2 - x1, 1), max(y2 - y1, 1)
            elongation = max(width, height) / min(width, height)
            if item.name in CUSTOM_OVERRIDES["bottle"] and height > width and elongation >= 1.40:
                candidates.append((item, "bottle"))
            elif item.name in CUSTOM_OVERRIDES["fan"] and elongation <= 2.20:
                candidates.append((item, "fan"))
        return candidates

    def reset_tracking(self) -> None:
        """Reset persistent ByteTrack state before a new camera/video source."""
        with self.lock:
            for model in (self.model, self.custom_model, self.daily_model, self.key_model):
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
    def _postprocess_custom(
        detections: list[ObjectDetection], frame_shape: tuple[int, int]
    ) -> list[ObjectDetection]:
        """Reject implausible shapes from the low-threshold targeted World pass."""
        frame_height, frame_width = frame_shape
        frame_area = max(frame_width * frame_height, 1)
        kept: list[ObjectDetection] = []
        for item in detections:
            x1, y1, x2, y2 = item.box
            width, height = max(x2 - x1, 1), max(y2 - y1, 1)
            elongation = max(width, height) / min(width, height)
            area_ratio = (width * height) / frame_area

            # Two- and three-percent World predictions are too weak to draw
            # over confident YOLO11 objects. Accessory prompts retain their
            # lower thresholds because they cover classes missing from COCO.
            if item.name in {"bottle", "fan"} and item.confidence < 0.05:
                continue

            if item.name in {"earbuds", "headphones"} and item.confidence < 0.08:
                continue

            if item.name == "key" and item.confidence < 0.08:
                continue

            # The reflective-flask prompt is intentionally broad. Retaining
            # only portrait, bottle-like regions removes its common person and
            # room-sized false positives.
            if item.name == "bottle" and (
                height <= width or elongation < 1.40 or area_ratio > 0.15
            ):
                continue

            # Ceiling fans span both axes; long thin room edges do not.
            if item.name == "fan" and (elongation > 2.20 or area_ratio < 0.01):
                continue

            # Weak predictions around extremely thin frame/UI edges are a
            # recurring false positive for the open-vocabulary pen prompt.
            # Strong model evidence remains eligible regardless of shape.
            if item.name == "pen" and item.confidence < 0.08 and elongation > 8.0:
                continue
            kept.append(item)
        return kept

    @staticmethod
    def _fuse(standard: list[ObjectDetection], custom: list[ObjectDetection]) -> list[ObjectDetection]:
        """Class-aware fusion for the optional open-vocabulary pass."""
        custom_kept: list[ObjectDetection] = []
        for candidate in sorted(custom, key=lambda item: item.confidence, reverse=True):
            if any(
                (
                    candidate.name == other.name
                    and (
                        ObjectDetector._iou(candidate.box, other.box) >= 0.50
                        or ObjectDetector._overlap(candidate.box, other.box) >= 0.65
                    )
                )
                or ObjectDetector._iou(candidate.box, other.box) >= 0.75
                for other in custom_kept
            ):
                continue
            custom_kept.append(candidate)

        # Prefer strong standard evidence to a weak open-vocabulary guess on
        # the same region. In particular, do not let a 2% "fan" cover an 89%
        # phone while still allowing strong World evidence to correct COCO.
        custom_kept = [candidate for candidate in custom_kept if not any(
            candidate.name in {"bottle", "fan"}
            and standard_item.name in CUSTOM_OVERRIDES.get(candidate.name, set())
            and standard_item.confidence >= 0.65
            and candidate.confidence < 0.15
            and ObjectDetector._overlap(standard_item.box, candidate.box) >= 0.65
            for standard_item in standard
        )]

        # A weak open-vocabulary pen score on a static window/frame edge can
        # persist across frames and defeat temporal confirmation.  Keep weak
        # WORLD pen candidates only when they overlap a detected person (the
        # live-demo use case is a hand-held pen). Strong WORLD evidence and
        # the separately trained daily-object model remain eligible without
        # this contextual requirement.
        people = [item for item in standard if item.name == "person"]
        custom_kept = [candidate for candidate in custom_kept if not (
            candidate.name == "pen"
            and candidate.detector_source == "yolo_world"
            and candidate.confidence < 0.08
            and not any(
                ObjectDetector._overlap(candidate.box, person.box) >= 0.20
                for person in people
            )
        )]

        # A long keychain/lanyard is often interpreted as a bottle or pen. A
        # confident nested key prediction is more specific, so suppress those
        # broader conflicting regions before bottle/fan priority is applied.
        keys = [
            item for item in custom_kept
            if item.name == "key" and item.confidence >= 0.08
        ]
        custom_kept = [candidate for candidate in custom_kept if (
            candidate.name == "key"
            or candidate.name not in {"bottle", "earbuds", "headphones", "pen"}
            or not any(
                ObjectDetector._overlap(candidate.box, key.box) >= 0.25
                for key in keys
            )
        )]

        # The supervised daily-object model was trained with an explicit
        # watch class.  Its watch prediction is therefore more specific than
        # a YOLO-World bottle guess on the same physical object.  Resolve this
        # conflict before bottle/fan regions suppress other custom detections.
        specialist_watches = [
            item for item in custom_kept
            if item.name == "watch"
            and item.detector_source == "daily_yolo"
            and item.confidence >= DAILY_THRESHOLDS["watch"]
        ]
        custom_kept = [candidate for candidate in custom_kept if not (
            candidate.name == "bottle"
            and candidate.detector_source == "yolo_world"
            and any(
                ObjectDetector._overlap(candidate.box, watch.box) >= 0.50
                for watch in specialist_watches
            )
        )]

        targeted_regions = [item for item in custom_kept if item.name in {"bottle", "fan"}]
        custom_kept = [item for item in custom_kept if (
            item.name in {"bottle", "fan"}
            or not any(ObjectDetector._overlap(item.box, target.box) >= 0.65 for target in targeted_regions)
        )]

        standard_kept = list(standard)
        for candidate in custom_kept:
            overridden_names = CUSTOM_OVERRIDES.get(candidate.name, set())
            if overridden_names:
                standard_kept = [item for item in standard_kept if not (
                    item.name in overridden_names
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
            elif item.detector_source == "daily_yolo":
                color = (210, 90, 255)
            elif item.detector_source == "key_yolo":
                color = (255, 130, 70)
            elif item.detector_source == "objects365":
                color = (180, 150, 40)
            else:
                color = (46, 204, 113) if item.name == "person" else (0, 174, 255)
            suffix = f" #{item.tracking_id}" if item.tracking_id is not None else ""
            source = (
                " · WORLD" if item.detector_source == "yolo_world"
                else " · DAILY" if item.detector_source == "daily_yolo"
                else " · KEY" if item.detector_source == "key_yolo"
                else " · 365" if item.detector_source == "objects365"
                else ""
            )
            label = f"{item.name.title()}{suffix} - {item.confidence:.0%}{source}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_right = min(frame.shape[1], x1 + max(120, len(label) * 8))
            cv2.rectangle(frame, (x1, max(0, y1 - 26)), (label_right, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, max(14, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .52, (10, 15, 25), 1, cv2.LINE_AA)
        return frame
