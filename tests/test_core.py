from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.database import Database
from src.face_recognizer import slugify
from src.video_utils import Cooldown, decode_image
from src.object_detector import ObjectDetection, ObjectDetector
from src.performance import classification_report, summarize_benchmark


class CoreTests(unittest.TestCase):
    def test_benchmark_summary(self) -> None:
        summary = summarize_benchmark([100, 200, 300])
        self.assertEqual(summary.runs, 3)
        self.assertAlmostEqual(summary.mean_ms, 200)
        self.assertAlmostEqual(summary.fps, 5)

    def test_classification_report(self) -> None:
        import pandas as pd
        matrix, report, summary = classification_report(pd.DataFrame({
            "actual": ["person", "person", "bottle"],
            "predicted": ["person", "bottle", "bottle"],
        }))
        self.assertEqual(int(matrix.to_numpy().sum()), 3)
        self.assertEqual(set(report["class"]), {"bottle", "person"})
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)

    def test_iou_and_duplicate_suppression(self) -> None:
        standard = [ObjectDetection((0, 0, 100, 100), "person", .9, 1)]
        custom = [
            ObjectDetection((1, 1, 99, 99), "person", .2, 2),
            ObjectDetection((10, 10, 30, 60), "pen", .08, 3),
        ]
        kept = ObjectDetector._remove_overlaps(custom, standard)
        self.assertEqual([item.name for item in kept], ["pen"])

    def test_custom_prompt_overlap_keeps_best_label(self) -> None:
        custom = [
            ObjectDetection((10, 10, 100, 100), "scissors", .08, 1),
            ObjectDetection((12, 12, 98, 98), "pen", .03, 2),
        ]
        kept = ObjectDetector._remove_overlaps(custom, [])
        self.assertEqual([(item.name, item.confidence) for item in kept], [("scissors", .08)])

    def test_slugify(self) -> None:
        self.assertEqual(slugify("Syed Ateeb Fatmi"), "syed_ateeb_fatmi")
        self.assertEqual(slugify("---"), "person")

    def test_invalid_image(self) -> None:
        with self.assertRaises(ValueError):
            decode_image(b"not an image")

    def test_cooldown(self) -> None:
        cooldown = Cooldown(60)
        self.assertTrue(cooldown.ready("phone"))
        self.assertFalse(cooldown.ready("phone"))

    def test_database_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            db = Database(Path(folder) / "test.db")
            session_id = db.start_session("image")
            db.log_detections(session_id, [{
                "object_type": "object", "object_name": "bottle", "confidence": .9,
                "tracking_id": None, "recognized_person": None,
                "recognition_confidence": None, "image_path": None,
            }])
            db.finish_session(session_id, {"total_objects": 1})
            self.assertEqual(len(db.detections_frame()), 1)
            self.assertEqual(db.dashboard_stats()["objects"], 1)

    def test_running_session_stats(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            db = Database(Path(folder) / "test.db")
            session_id = db.start_session("live")
            db.update_session_stats(session_id, {"total_objects": 7, "total_faces": 2,
                                                  "known_faces": 1, "unknown_faces": 1})
            row = db.sessions_frame().iloc[0]
            self.assertEqual(int(row.total_objects), 7)
            self.assertEqual(int(row.total_faces), 2)
            self.assertIsNone(row.ended_at)

    def test_dashboard_deduplicates_tracks_and_counts_known_identity(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            db = Database(Path(folder) / "test.db")
            session_id = db.start_session("live")
            object_row = {"object_type": "object", "object_name": "person", "confidence": .9,
                          "tracking_id": 7, "recognized_person": None,
                          "recognition_confidence": None, "image_path": None}
            face_row = {"object_type": "face", "object_name": "person", "confidence": .9,
                        "tracking_id": None, "recognized_person": "Ateeb",
                        "recognition_confidence": .7, "image_path": None}
            db.log_detections(session_id, [object_row, object_row, face_row, face_row])
            stats = db.dashboard_stats()
            self.assertEqual(stats["objects"], 1)
            self.assertEqual(stats["object_events"], 2)
            self.assertEqual(stats["known"], 1)
            self.assertEqual(stats["known_events"], 2)


if __name__ == "__main__":
    unittest.main()
