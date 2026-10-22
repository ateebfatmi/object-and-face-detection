from __future__ import annotations

import urllib.request
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from .config import YUNET_MODEL, YUNET_URL, ensure_directories


@dataclass(slots=True)
class FaceDetection:
    raw: np.ndarray
    box: tuple[int, int, int, int]
    confidence: float


def ensure_model(path, url: str) -> None:
    ensure_directories()
    if not path.exists():
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            raise RuntimeError(f"Could not download face model. Check your connection: {exc}") from exc


class FaceDetector:
    def __init__(self) -> None:
        ensure_model(YUNET_MODEL, YUNET_URL)
        self.detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), 0.75, 0.3, 5000)
        self.lock = threading.RLock()

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        height, width = image.shape[:2]
        with self.lock:
            self.detector.setInputSize((width, height))
            _, faces = self.detector.detect(image)
        output: list[FaceDetection] = []
        if faces is None:
            return output
        for face in faces:
            raw_x, raw_y, box_width, box_height = map(int, face[:4])
            x1, y1 = max(0, raw_x), max(0, raw_y)
            x2 = min(width, raw_x + max(box_width, 0))
            y2 = min(height, raw_y + max(box_height, 0))
            if x2 > x1 and y2 > y1:
                output.append(FaceDetection(face, (x1, y1, x2, y2), float(face[-1])))
        return output

