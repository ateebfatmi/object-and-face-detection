from __future__ import annotations

from collections import Counter
from datetime import datetime
from time import perf_counter

import cv2
import numpy as np

from .config import SCREENSHOTS_DIR, UNKNOWN_FACES_DIR, DetectionSettings, ensure_directories
from .face_detector import FaceDetector
from .face_recognizer import FaceMatch, FaceRecognizer
from .object_detector import ObjectDetection, ObjectDetector
from .video_utils import Cooldown, resize_for_inference


class VisionProcessor:
    def __init__(self, object_detector: ObjectDetector | None, face_detector: FaceDetector | None,
                 face_recognizer: FaceRecognizer | None) -> None:
        self.objects = object_detector
        self.faces = face_detector
        self.recognizer = face_recognizer
        self.unknown_cooldown = Cooldown(12)
        self.evidence_cooldown = Cooldown(10)

    def reset_tracking(self) -> None:
        if self.objects:
            self.objects.reset_tracking()

    def process(self, frame: np.ndarray, settings: DetectionSettings,
                evidence_classes: set[str] | None = None) -> tuple[np.ndarray, dict, list[dict]]:
        started_at = perf_counter()
        ensure_directories()
        work, _ = resize_for_inference(frame, settings.max_width)
        annotated = work.copy()
        object_items: list[ObjectDetection] = []
        face_rows: list[tuple] = []
        logs: list[dict] = []
        if settings.object_detection and self.objects:
            object_items = self.objects.detect(
                work,
                settings.confidence,
                settings.tracking,
                iou_threshold=settings.iou_threshold,
                custom_confidence=settings.custom_confidence,
                max_detections=settings.max_detections,
            )
            self.objects.draw(annotated, object_items)
            for item in object_items:
                logs.append({"object_type": "object", "object_name": item.name, "confidence": item.confidence,
                             "tracking_id": item.tracking_id, "recognized_person": None,
                             "recognition_confidence": None, "image_path": None,
                             "detector_source": item.detector_source})
        if settings.face_detection and self.faces:
            for face in self.faces.detect(work):
                match = FaceMatch("Face", 0, False)
                if settings.face_recognition and self.recognizer:
                    match = self.recognizer.match(self.recognizer.feature(work, face), settings.face_threshold)
                x1, y1, x2, y2 = face.box
                color = (60, 210, 120) if match.known else (55, 80, 245)
                label = f"{match.name} - {match.similarity:.0%}" if settings.face_recognition else "Face"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
                saved_path = None
                if not match.known and settings.save_unknown_faces and self.unknown_cooldown.ready("unknown"):
                    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    saved_path = UNKNOWN_FACES_DIR / f"{stamp}_unknown.jpg"
                    cv2.imwrite(str(saved_path), work[y1:y2, x1:x2])
                face_rows.append((face, match))
                logs.append({"object_type": "face", "object_name": "person", "confidence": face.confidence,
                             "tracking_id": None, "recognized_person": match.name if settings.face_recognition else None,
                             "recognition_confidence": match.similarity if settings.face_recognition else None,
                             "image_path": str(saved_path) if saved_path else None,
                             "detector_source": "yunet"})
        triggers = set(evidence_classes or ())
        present = {x.name for x in object_items}
        if any(not m.known for _, m in face_rows): present.add("unknown_person")
        if settings.capture_screenshots:
            for trigger in triggers & present:
                if self.evidence_cooldown.ready(trigger):
                    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    path = SCREENSHOTS_DIR / f"{stamp}_{trigger}.jpg"
                    cv2.imwrite(str(path), annotated)
                    for row in logs:
                        if row["object_name"] == trigger or (trigger == "unknown_person" and row["recognized_person"] == "Unknown"):
                            row["image_path"] = str(path)
        counts = Counter(x.name for x in object_items)
        processing_ms = (perf_counter() - started_at) * 1000
        stats = {"objects": len(object_items), "faces": len(face_rows),
                 "known": sum(m.known for _, m in face_rows), "unknown": sum(not m.known for _, m in face_rows),
                 "fps": 1000 / max(processing_ms, .001), "processing_ms": processing_ms,
                 "classes": dict(counts)}
        return annotated, stats, logs
