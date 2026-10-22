from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.database import Database
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer, slugify
from src.object_detector import ACCESSORY_PROMPTS, CUSTOM_PROMPTS, ObjectDetection, ObjectDetector
from src.performance import classification_report, summarize_benchmark
from src.ui import compact_navigation
from src.video_utils import Cooldown, decode_image, resize_for_inference
from scripts.evaluate_object_detection import build_report


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def int(self):
        return FakeTensor(self.values.astype(int))

    def tolist(self):
        return self.values.tolist()


class FakeBoxes:
    def __init__(self, xyxy, classes, confidences, ids=None):
        self.xyxy = FakeTensor(xyxy)
        self.cls = FakeTensor(classes)
        self.conf = FakeTensor(confidences)
        self.id = FakeTensor(ids) if ids is not None else None

    def __len__(self):
        return len(self.conf.values)


class CoreTests(unittest.TestCase):
    def test_compact_navigation_marks_current_page(self) -> None:
        markup = compact_navigation("Project Guide")
        self.assertIn('class="vg-mini-link active"', markup)
        self.assertIn('href="/Project_Guide"', markup)
        self.assertEqual(markup.count('class="vg-mini-link active"'), 1)

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

    def test_detection_evaluation_report(self) -> None:
        box = types.SimpleNamespace(
            ap_class_index=np.array([0, 1]),
            p=np.array([.8, .6]),
            r=np.array([.5, .75]),
            ap50=np.array([.7, .65]),
            ap=np.array([.4, .35]),
            mp=.7,
            mr=.625,
            map50=.675,
            map=.375,
        )
        metrics = types.SimpleNamespace(
            names={0: "person", 1: "bottle"},
            box=box,
            confusion_matrix=types.SimpleNamespace(matrix=np.array([
                [2, 1, 1],
                [0, 1, 0],
                [1, 0, 0],
            ])),
            speed={"preprocess": 1.0, "inference": 20.0, "postprocess": 2.0},
        )
        report = build_report(metrics, "accurate", Path("dataset.yaml"), "test", .35, .5)
        self.assertEqual(report["false_positives"], 2)
        self.assertEqual(report["false_negatives"], 2)
        self.assertEqual(report["inference_fps"], 50.0)
        self.assertAlmostEqual(report["per_class"][0]["f1"], 2 * .8 * .5 / 1.3)

    def test_iou_and_overlap(self) -> None:
        self.assertAlmostEqual(ObjectDetector._iou((0, 0, 100, 100), (50, 0, 150, 100)), 1 / 3)
        self.assertAlmostEqual(ObjectDetector._overlap((0, 0, 100, 100), (10, 10, 20, 20)), 1.0)

    def test_fusion_deduplicates_custom_and_prefers_model_predicted_pen(self) -> None:
        standard = [
            ObjectDetection((0, 0, 20, 100), "scissors", .70, 1, "yolo11"),
            ObjectDetection((200, 200, 400, 500), "person", .90, 2, "yolo11"),
        ]
        custom = [
            ObjectDetection((1, 1, 19, 99), "pen", .12, 1_000_001, "yolo_world"),
            ObjectDetection((2, 2, 18, 98), "pen", .08, 1_000_002, "yolo_world"),
            ObjectDetection((5, 20, 15, 50), "pen", .07, 1_000_003, "yolo_world"),
        ]
        fused = ObjectDetector._fuse(standard, custom)
        self.assertEqual([(item.name, item.detector_source) for item in fused], [
            ("person", "yolo11"), ("pen", "yolo_world")
        ])

    def test_fusion_prefers_targeted_bottle_and_fan_over_coco_confusions(self) -> None:
        standard = [
            ObjectDetection((10, 10, 80, 180), "remote", .79, 54, "yolo11"),
            ObjectDetection((200, 10, 500, 300), "airplane", .47, 55, "yolo11"),
            ObjectDetection((600, 10, 750, 300), "person", .90, 56, "yolo11"),
        ]
        custom = [
            ObjectDetection((12, 12, 78, 178), "bottle", .20, 1_000_001, "yolo_world"),
            ObjectDetection((205, 15, 495, 295), "fan", .18, 1_000_002, "yolo_world"),
            ObjectDetection((30, 40, 45, 150), "pen", .10, 1_000_003, "yolo_world"),
        ]
        fused = ObjectDetector._fuse(standard, custom)
        self.assertEqual([item.name for item in fused], ["person", "bottle", "fan"])

    def test_world_prompts_cover_requested_objects(self) -> None:
        self.assertIn("bottle", CUSTOM_PROMPTS)
        self.assertIn("stainless steel bottle", CUSTOM_PROMPTS)
        self.assertIn("electric ceiling fan", CUSTOM_PROMPTS)
        self.assertEqual(ACCESSORY_PROMPTS, ("long thin pen", "earbuds", "headphones"))

    def test_custom_postprocess_keeps_bottle_and_fan_shapes(self) -> None:
        detections = [
            ObjectDetection((100, 100, 300, 800), "bottle", .04, detector_source="yolo_world"),
            ObjectDetection((400, 100, 900, 550), "fan", .03, detector_source="yolo_world"),
            ObjectDetection((0, 0, 1500, 900), "bottle", .06, detector_source="yolo_world"),
            ObjectDetection((0, 0, 1000, 100), "fan", .04, detector_source="yolo_world"),
        ]
        kept = ObjectDetector._postprocess_custom(detections, (1080, 1920))
        self.assertEqual([item.name for item in kept], ["bottle", "fan"])

    def test_custom_postprocess_rejects_weak_edge_like_pen(self) -> None:
        detections = [
            ObjectDetection((10, 10, 150, 320), "pen", .07, detector_source="yolo_world"),
            ObjectDetection((300, 400, 500, 415), "pen", .053, detector_source="yolo_world"),
            ObjectDetection((300, 500, 500, 515), "pen", .12, detector_source="yolo_world"),
        ]
        kept = ObjectDetector._postprocess_custom(detections, (720, 1280))
        self.assertEqual([item.confidence for item in kept], [.07, .12])

    def test_focus_candidates_only_select_plausible_coco_confusions(self) -> None:
        detections = [
            ObjectDetection((10, 10, 110, 410), "cell phone", .75),
            ObjectDetection((200, 10, 600, 360), "airplane", .47),
            ObjectDetection((700, 10, 1000, 310), "person", .90),
            ObjectDetection((1100, 10, 1400, 160), "remote", .80),
        ]
        focused = ObjectDetector._focus_candidates(detections)
        self.assertEqual([(item.name, target) for item, target in focused], [
            ("cell phone", "bottle"),
            ("airplane", "fan"),
        ])

    def test_parse_clamps_boxes_and_records_source(self) -> None:
        result = types.SimpleNamespace(boxes=FakeBoxes(
            [[-4.2, 5.4, 120.1, 90.8], [20, 20, 20, 40]], [0, 1], [.8, .7], [3, 4]
        ))
        model = types.SimpleNamespace(names={0: "person", 1: "bicycle"})
        parsed = ObjectDetector._parse([result], model, (80, 100), detector_source="yolo11")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].box, (0, 5, 100, 80))
        self.assertEqual((parsed[0].name, parsed[0].tracking_id, parsed[0].detector_source),
                         ("person", 3, "yolo11"))

    def test_parse_normalizes_world_prompt_and_offsets_track(self) -> None:
        result = types.SimpleNamespace(boxes=FakeBoxes([[1, 2, 10, 20]], [0], [.12], [5]))
        model = types.SimpleNamespace(names={0: "long thin pen"})
        parsed = ObjectDetector._parse(
            [result], model, (100, 100), track_offset=1_000_000, detector_source="yolo_world"
        )
        self.assertEqual((parsed[0].name, parsed[0].tracking_id, parsed[0].detector_source),
                         ("pen", 1_000_005, "yolo_world"))

    def test_custom_tiles_cover_wide_frame_and_restore_coordinates(self) -> None:
        frame = np.zeros((600, 1000, 3), dtype=np.uint8)
        tiles = ObjectDetector._custom_tiles(frame)
        self.assertEqual([(tile.shape[1], x, y) for tile, x, y in tiles],
                         [(620, 0, 0), (620, 380, 0)])
        shifted = ObjectDetector._offset_detections(
            [ObjectDetection((10, 20, 30, 40), "pen", .6, None, "yolo_world")],
            380,
            0,
        )
        self.assertEqual(shifted[0].box, (390, 20, 410, 40))

    def test_standard_postprocess_relabels_wrist_clock_as_watch(self) -> None:
        detections = [
            ObjectDetection((650, 360, 800, 610), "clock", .59, 7, "yolo11"),
            ObjectDetection((900, 100, 1100, 300), "clock", .80, 8, "yolo11"),
        ]
        kept = ObjectDetector._postprocess_standard(detections, (1080, 1920))
        self.assertEqual(
            [(item.name, item.tracking_id, item.detector_source) for item in kept],
            [("watch", 7, "yolo11"), ("clock", 8, "yolo11")],
        )

    def test_standard_postprocess_rejects_weak_square_phone_box(self) -> None:
        detections = [
            ObjectDetection((0, 100, 300, 500), "cell phone", .58, 1),
            ObjectDetection((400, 100, 580, 500), "cell phone", .58, 2),
        ]
        kept = ObjectDetector._postprocess_standard(detections, (1080, 1920))
        self.assertEqual([item.tracking_id for item in kept], [2])

    def test_standard_postprocess_removes_phone_overlapping_watch(self) -> None:
        detections = [
            ObjectDetection((100, 100, 420, 520), "cell phone", .58, 4),
            ObjectDetection((250, 300, 410, 540), "clock", .59, 7),
            ObjectDetection((800, 100, 950, 430), "cell phone", .80, 9),
        ]
        kept = ObjectDetector._postprocess_standard(detections, (1080, 1920))
        self.assertEqual(
            [(item.name, item.tracking_id) for item in kept],
            [("watch", 7), ("cell phone", 9)],
        )

    def test_frame_validation(self) -> None:
        ObjectDetector._validate_frame(np.zeros((20, 30, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            ObjectDetector._validate_frame(np.zeros((20, 30), dtype=np.uint8))
        with self.assertRaises(ValueError):
            ObjectDetector._validate_frame(np.zeros((20, 30, 3), dtype=np.float32))

    def test_missing_model_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                ObjectDetector._require_file(Path(folder) / "missing.pt", "test weights")

    def test_model_initialization_is_strict_and_configures_world_prompts(self) -> None:
        loaded: list[str] = []

        class FakeModel:
            task = "detect"
            names = {index: "person" if index == 0 else f"class-{index}" for index in range(80)}

            def __init__(self, path):
                loaded.append(str(path))

            def set_classes(self, prompts):
                self.names = dict(enumerate(prompts))

        module = types.ModuleType("ultralytics")
        module.YOLO = FakeModel
        module.YOLOWorld = FakeModel
        with patch.dict(sys.modules, {"ultralytics": module}):
            fast = ObjectDetector("fast")
            hybrid = ObjectDetector("hybrid")
            self.assertEqual(fast.inference_size, 640)
            self.assertEqual(hybrid.inference_size, 960)
            self.assertEqual(tuple(hybrid.custom_model.names.values()), CUSTOM_PROMPTS)
            with self.assertRaises(ValueError):
                ObjectDetector("typo")
        self.assertTrue(any(path.endswith("yolo11n.pt") for path in loaded))
        self.assertTrue(any(path.endswith("yolov8s-worldv2.pt") for path in loaded))

    def test_tracking_reset_resets_each_model_tracker(self) -> None:
        class Tracker:
            resets = 0

            def reset(self):
                self.resets += 1

        standard_tracker, custom_tracker = Tracker(), Tracker()
        detector = ObjectDetector.__new__(ObjectDetector)
        detector.lock = threading.RLock()
        detector.model = types.SimpleNamespace(
            predictor=types.SimpleNamespace(trackers=[standard_tracker], vid_path=["old"])
        )
        detector.custom_model = types.SimpleNamespace(
            predictor=types.SimpleNamespace(trackers=[custom_tracker], vid_path=["old"])
        )
        detector.reset_tracking()
        self.assertEqual((standard_tracker.resets, custom_tracker.resets), (1, 1))
        self.assertEqual(detector.model.predictor.vid_path, [None])

    def test_resize_preserves_aspect_ratio(self) -> None:
        image = np.zeros((600, 1200, 3), dtype=np.uint8)
        resized, scale = resize_for_inference(image, 900)
        self.assertEqual(resized.shape[:2], (450, 900))
        self.assertEqual(scale, .75)

    def test_slugify(self) -> None:
        self.assertEqual(slugify("Syed Ateeb Fatmi"), "syed_ateeb_fatmi")
        self.assertEqual(slugify("---"), "person")

    def test_face_detector_clamps_mocked_yunet_box(self) -> None:
        class Detector:
            def setInputSize(self, size):
                self.size = size

            def detect(self, image):
                return None, np.array([[-5, -4, 20, 15] + [0] * 10 + [.9]], dtype=np.float32)

        detector = FaceDetector.__new__(FaceDetector)
        detector.detector = Detector()
        detector.lock = threading.RLock()
        faces = detector.detect(np.zeros((20, 30, 3), dtype=np.uint8))
        self.assertEqual(faces[0].box, (0, 0, 15, 11))

    def test_face_match_respects_similarity_threshold(self) -> None:
        class Recognizer:
            @staticmethod
            def match(feature, known, metric):
                return float(np.dot(feature, known))

        recognizer = FaceRecognizer.__new__(FaceRecognizer)
        recognizer.recognizer = Recognizer()
        recognizer.embeddings = [("known_person", np.array([1.0, 0.0], dtype=np.float32))]
        recognizer.lock = threading.RLock()
        rejected = recognizer.match(np.array([.4, .6], dtype=np.float32), threshold=.45)
        accepted = recognizer.match(np.array([.8, .2], dtype=np.float32), threshold=.45)
        self.assertEqual(rejected.name, "Unknown")
        self.assertEqual(accepted.name, "Known Person")

    def test_invalid_image(self) -> None:
        with self.assertRaises(ValueError):
            decode_image(b"not an image")

    def test_cooldown(self) -> None:
        cooldown = Cooldown(60)
        self.assertTrue(cooldown.ready("phone"))
        self.assertFalse(cooldown.ready("phone"))

    def test_database_lifecycle_and_detector_source(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            db = Database(Path(folder) / "test.db")
            session_id = db.start_session("image")
            db.log_detections(session_id, [{
                "object_type": "object", "object_name": "bottle", "confidence": .9,
                "tracking_id": None, "recognized_person": None,
                "recognition_confidence": None, "image_path": None,
                "detector_source": "yolo11",
            }])
            db.finish_session(session_id, {"total_objects": 1})
            frame = db.detections_frame()
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0].detector_source, "yolo11")
            self.assertEqual(db.dashboard_stats()["objects"], 1)

    def test_database_migrates_existing_detection_table(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.db"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE detections (id INTEGER PRIMARY KEY)")
            Database(path)
            with sqlite3.connect(path) as connection:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(detections)")}
            self.assertIn("detector_source", columns)

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
                          "recognition_confidence": None, "image_path": None,
                          "detector_source": "yolo11"}
            face_row = {"object_type": "face", "object_name": "person", "confidence": .9,
                        "tracking_id": None, "recognized_person": "Ateeb",
                        "recognition_confidence": .7, "image_path": None,
                        "detector_source": "yunet"}
            db.log_detections(session_id, [object_row, object_row, face_row, face_row])
            stats = db.dashboard_stats()
            self.assertEqual(stats["objects"], 1)
            self.assertEqual(stats["object_events"], 2)
            self.assertEqual(stats["known"], 1)
            self.assertEqual(stats["known_events"], 2)


if __name__ == "__main__":
    unittest.main()
