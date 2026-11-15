from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .database import RegisteredPerson
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer


@dataclass(slots=True)
class AttendanceMatch:
    person_id: int
    name: str
    identifier: str
    similarity: float
    box: tuple[int, int, int, int]


@dataclass(slots=True)
class AttendanceAnalysis:
    annotated: np.ndarray
    present: list[AttendanceMatch]
    unknown_faces: int
    detected_faces: int


@dataclass(frozen=True, slots=True)
class FaceProfileIssue:
    person_id: int
    name: str
    identifier: str
    reason: str


def _profile_embedding(person: RegisteredPerson) -> tuple[np.ndarray | None, str | None]:
    if not person.embedding_path:
        return None, "No embedding is linked"
    path = Path(person.embedding_path)
    if not path.is_file():
        return None, "Embedding file is missing"
    try:
        feature = np.load(path, allow_pickle=False).astype(np.float32).reshape(-1)
    except (OSError, ValueError):
        return None, "Embedding file is unreadable"
    if not feature.size or not np.isfinite(feature).all():
        return None, "Embedding data is invalid"
    return feature, None


def validate_face_profiles(people: Iterable[RegisteredPerson]) -> list[FaceProfileIssue]:
    issues: list[FaceProfileIssue] = []
    for person in people:
        _feature, reason = _profile_embedding(person)
        if reason:
            issues.append(FaceProfileIssue(
                person_id=person.id,
                name=person.name,
                identifier=person.person_identifier,
                reason=reason,
            ))
    return issues


def _load_candidates(people: Iterable[RegisteredPerson]) -> list[tuple[RegisteredPerson, np.ndarray]]:
    candidates: list[tuple[RegisteredPerson, np.ndarray]] = []
    for person in people:
        feature, _reason = _profile_embedding(person)
        if feature is not None:
            candidates.append((person, feature))
    return candidates


def analyze_attendance(
    image: np.ndarray,
    people: Iterable[RegisteredPerson],
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    threshold: float = 0.45,
) -> AttendanceAnalysis:
    """Recognize registered people in an image and keep each student only once."""
    annotated = image.copy()
    candidates = _load_candidates(people)
    strongest: dict[int, AttendanceMatch] = {}
    unknown = 0
    faces = detector.detect(image)

    for face in faces:
        feature = recognizer.feature(image, face)
        best_person: RegisteredPerson | None = None
        best_score = -1.0
        for person, known in candidates:
            if known.shape != feature.shape:
                continue
            score = recognizer.similarity(feature, known)
            if score > best_score:
                best_person, best_score = person, score

        x1, y1, x2, y2 = face.box
        if best_person is not None and best_score >= threshold:
            match = AttendanceMatch(
                best_person.id,
                best_person.name,
                best_person.person_identifier,
                best_score,
                face.box,
            )
            current = strongest.get(best_person.id)
            if current is None or match.similarity > current.similarity:
                strongest[best_person.id] = match
            color = (98, 183, 178)
            label = f"{best_person.name} · {best_score:.0%}"
        else:
            unknown += 1
            color = (55, 80, 245)
            label = "Unknown"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    return AttendanceAnalysis(
        annotated=annotated,
        present=sorted(strongest.values(), key=lambda item: item.name.casefold()),
        unknown_faces=unknown,
        detected_faces=len(faces),
    )
