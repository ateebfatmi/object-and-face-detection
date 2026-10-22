from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import KNOWN_FACES_DIR, SFACE_MODEL, SFACE_URL, ensure_directories
from .face_detector import FaceDetection, FaceDetector, ensure_model


@dataclass(slots=True)
class FaceMatch:
    name: str
    similarity: float
    known: bool


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "person"


class FaceRecognizer:
    def __init__(self) -> None:
        ensure_model(SFACE_MODEL, SFACE_URL)
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
        self.embeddings: list[tuple[str, np.ndarray]] = []
        self.lock = threading.RLock()
        self.reload()

    def reload(self) -> None:
        with self.lock:
            self.embeddings.clear()
            ensure_directories()
            for file in KNOWN_FACES_DIR.glob("*/embedding.npy"):
                try:
                    embedding = np.load(file).astype(np.float32).reshape(-1)
                    if embedding.size and np.isfinite(embedding).all():
                        self.embeddings.append((file.parent.name, embedding))
                except (OSError, ValueError):
                    continue

    def feature(self, image: np.ndarray, face: FaceDetection) -> np.ndarray:
        with self.lock:
            aligned = self.recognizer.alignCrop(image, face.raw)
            return self.recognizer.feature(aligned).flatten().astype(np.float32)

    def match(self, feature: np.ndarray, threshold: float = 0.45) -> FaceMatch:
        best_name, best_score = "Unknown", -1.0
        with self.lock:
            for name, known in self.embeddings:
                if known.shape != feature.shape:
                    continue
                score = float(self.recognizer.match(feature, known, cv2.FaceRecognizerSF_FR_COSINE))
                if score > best_score:
                    best_name, best_score = name, score
        known = best_score >= threshold
        return FaceMatch(best_name.replace("_", " ").title() if known else "Unknown", max(0.0, best_score), known)

    def register(self, name: str, images: list[np.ndarray], detector: FaceDetector) -> tuple[Path, int]:
        slug = slugify(name)
        person_dir = KNOWN_FACES_DIR / slug
        person_dir.mkdir(parents=True, exist_ok=True)
        features: list[np.ndarray] = []
        for index, image in enumerate(images, 1):
            faces = detector.detect(image)
            if len(faces) != 1:
                continue
            features.append(self.feature(image, faces[0]))
            cv2.imwrite(str(person_dir / f"face_{index}.jpg"), image)
        if not features:
            if not any(person_dir.iterdir()):
                person_dir.rmdir()
            raise ValueError("Each accepted registration image must contain exactly one clear face.")
        embedding = np.mean(np.stack(features), axis=0).astype(np.float32)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
        np.save(person_dir / "embedding.npy", embedding)
        self.reload()
        return person_dir, len(features)

