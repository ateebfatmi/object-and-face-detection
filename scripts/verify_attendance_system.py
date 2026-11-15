from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attendance import analyze_attendance, validate_face_profiles
from src.database import Database
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer


def _reference_image(person) -> np.ndarray | None:
    images = sorted(Path(person.face_directory).glob("face_*.jpg"))
    return cv2.imread(str(images[0])) if images else None


def _face_tile(image: np.ndarray, detector: FaceDetector, size: int = 360) -> np.ndarray | None:
    faces = detector.detect(image)
    if len(faces) != 1:
        return None
    x1, y1, x2, y2 = faces[0].box
    padding = int(max(x2 - x1, y2 - y1) * 0.65)
    crop = image[
        max(0, y1 - padding):min(image.shape[0], y2 + padding),
        max(0, x1 - padding):min(image.shape[1], x2 + padding),
    ]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def verify(threshold: float = 0.45) -> dict:
    database = Database()
    people = database.people()
    failures: list[str] = []
    issues = validate_face_profiles(people)
    for issue in issues:
        failures.append(f"{issue.identifier}: {issue.reason}")

    ready_people = [person for person in people if person.id not in {issue.person_id for issue in issues}]
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    individual_scores: dict[str, float] = {}
    tiles: list[np.ndarray] = []

    for person in ready_people:
        image = _reference_image(person)
        if image is None:
            failures.append(f"{person.person_identifier}: no reference image")
            continue
        result = analyze_attendance(image, people, detector, recognizer, threshold)
        own_match = next((item for item in result.present if item.person_id == person.id), None)
        if result.detected_faces != 1 or own_match is None or result.unknown_faces:
            failures.append(f"{person.person_identifier}: individual recognition failed")
            continue
        individual_scores[person.person_identifier] = own_match.similarity
        tile = _face_tile(image, detector)
        if tile is None:
            failures.append(f"{person.person_identifier}: class-photo tile preparation failed")
        else:
            tiles.append(tile)

    group_matches: list[str] = []
    duplicate_suppressed = False
    group_scores: dict[str, float] = {}
    if tiles:
        group_tiles = tiles + [tiles[0].copy()]
        columns = 2
        rows = ceil(len(group_tiles) / columns)
        canvas = np.full((rows * 380, columns * 380, 3), 32, dtype=np.uint8)
        for index, tile in enumerate(group_tiles):
            row, column = divmod(index, columns)
            y, x = 20 + row * 380, 20 + column * 380
            canvas[y:y + 360, x:x + 360] = tile
        group = analyze_attendance(canvas, people, detector, recognizer, threshold)
        group_matches = [item.identifier for item in group.present]
        group_scores = {item.identifier: item.similarity for item in group.present}
        expected = {person.person_identifier for person in ready_people}
        if set(group_matches) != expected or group.unknown_faces:
            failures.append("full-class recognition did not match every ready profile")
        duplicate_suppressed = group.detected_faces == len(group_tiles) and len(group.present) == len(ready_people)
        if not duplicate_suppressed:
            failures.append("duplicate face suppression failed")
    else:
        failures.append("no usable face tiles were available for full-class verification")

    transaction_checks = False
    with tempfile.TemporaryDirectory() as folder:
        temp_db = Database(Path(folder) / "attendance-verification.db")
        temp_people = {}
        for person in ready_people:
            item = temp_db.add_person(
                person.name,
                person.person_identifier,
                person.email,
                person.face_directory,
                person.embedding_path,
            )
            temp_people[person.person_identifier] = item
        if temp_people:
            selected = {
                temp_people[identifier].id: group_scores.get(identifier)
                for identifier in group_matches
                if identifier in temp_people
            }
            session_id = temp_db.record_class_attendance(
                "Verification class",
                selected,
                0,
                str(Path(folder) / "evidence.jpg"),
                "verification@local",
            )
            records = temp_db.attendance_records_frame(session_id)
            class_ok = len(records) == len(temp_people) and set(records["status"]) == {"Present"}

            one_by_one = temp_db.start_attendance("Verification class", "One by one")
            first = next(iter(temp_people.values()))
            marked = temp_db.mark_attendance_person(one_by_one, first.id, .9, "verification@local")
            duplicate = temp_db.mark_attendance_person(one_by_one, first.id, .9, "verification@local")
            temp_db.finalize_attendance_review(one_by_one, {first.id}, "verification@local")
            reviewed = temp_db.attendance_records_frame(one_by_one)
            review_ok = (
                marked
                and not duplicate
                and len(reviewed) == len(temp_people)
                and int((reviewed["status"] == "Present").sum()) == 1
            )
            transaction_checks = class_ok and review_ok
            if not transaction_checks:
                failures.append("attendance transaction or review workflow failed")

    return {
        "ok": not failures,
        "roster_profiles": len(people),
        "face_ready_profiles": len(ready_people),
        "individual_recognition": {key: round(value, 4) for key, value in individual_scores.items()},
        "group_matches": group_matches,
        "duplicate_suppressed": duplicate_suppressed,
        "transaction_checks": transaction_checks,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the local attendance pipeline end to end.")
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()
    report = verify(args.threshold)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
