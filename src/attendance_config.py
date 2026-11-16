from __future__ import annotations

from dataclasses import dataclass
import os

from .config import DetectionSettings


@dataclass(frozen=True, slots=True)
class AttendanceMode:
    key: str
    label: str
    database_value: str
    icon: str


FULL_CLASS_MODE = AttendanceMode("full_class", "Full class photo", "Full class photo", "◫")
INDIVIDUAL_MODE = AttendanceMode("individual", "One by one", "One by one", "◎")
ATTENDANCE_MODES = (FULL_CLASS_MODE, INDIVIDUAL_MODE)


def _environment_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


def _environment_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class AttendanceSettings:
    threshold_min: float
    threshold_max: float
    threshold_default: float
    threshold_step: float
    history_limit: int
    image_extensions: tuple[str, ...]
    evidence_extension: str

    @classmethod
    def from_environment(cls) -> "AttendanceSettings":
        detector_defaults = DetectionSettings()
        threshold_min = 0.30
        threshold_max = 0.80
        configured_threshold = _environment_float(
            "VISIONGUARD_FACE_MATCH_THRESHOLD",
            detector_defaults.face_threshold,
        )
        extensions = tuple(
            dict.fromkeys(
                item.strip().lower().lstrip(".")
                for item in os.getenv("VISIONGUARD_ATTENDANCE_IMAGE_TYPES", "jpg,jpeg,png,webp").split(",")
                if item.strip()
            )
        )
        if not extensions:
            extensions = ("jpg", "jpeg", "png", "webp")
        evidence_extension = os.getenv("VISIONGUARD_ATTENDANCE_EVIDENCE_TYPE", "jpg").strip().lower().lstrip(".")
        if evidence_extension not in extensions:
            evidence_extension = extensions[0]
        return cls(
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_default=min(threshold_max, max(threshold_min, configured_threshold)),
            threshold_step=0.01,
            history_limit=min(500, max(1, _environment_int("VISIONGUARD_ATTENDANCE_HISTORY_LIMIT", 100))),
            image_extensions=extensions,
            evidence_extension=evidence_extension,
        )
