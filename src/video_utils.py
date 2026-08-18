from __future__ import annotations

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a valid image.")
    return image


def resize_for_inference(frame: np.ndarray, max_width: int = 960) -> tuple[np.ndarray, float]:
    if frame.shape[1] <= max_width:
        return frame, 1.0
    scale = max_width / frame.shape[1]
    return cv2.resize(frame, None, fx=scale, fy=scale), scale


class Cooldown:
    def __init__(self, seconds: float = 10.0) -> None:
        self.seconds = seconds
        self.events: dict[str, float] = {}

    def ready(self, key: str) -> bool:
        now = time.monotonic()
        if now - self.events.get(key, 0) < self.seconds:
            return False
        self.events[key] = now
        return True


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]

