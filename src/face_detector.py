from __future__ import annotations

import urllib.request
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

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        output: list[FaceDetection] = []
        if faces is None:
            return output
        for face in faces:
            x, y, w, h = map(int, face[:4])
            x, y = max(0, x), max(0, y)
            output.append(FaceDetection(face, (x, y, min(width, x + w), min(height, y + h)), float(face[-1])))
        return output

