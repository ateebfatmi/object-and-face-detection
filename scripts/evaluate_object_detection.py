#!/usr/bin/env python3
"""Evaluate VisionGuard YOLO11 weights on a labeled YOLO-format dataset."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import YOLO11N_MODEL, YOLO11S_MODEL


MODEL_PATHS = {"fast": YOLO11N_MODEL, "accurate": YOLO11S_MODEL}
IMAGE_SIZES = {"fast": 640, "accurate": 960}


def build_report(metrics, mode: str, data: Path, split: str, confidence: float, iou: float) -> dict:
    names = {int(key): value for key, value in metrics.names.items()}
    class_ids = np.asarray(metrics.box.ap_class_index, dtype=int)
    precision = np.asarray(metrics.box.p, dtype=float)
    recall = np.asarray(metrics.box.r, dtype=float)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    ap50 = np.asarray(metrics.box.ap50, dtype=float)
    ap = np.asarray(metrics.box.ap, dtype=float)
    matrix = np.asarray(metrics.confusion_matrix.matrix, dtype=float)
    class_count = len(names)

    per_class = []
    for metric_index, class_id in enumerate(class_ids):
        true_positive = int(matrix[class_id, class_id]) if matrix.size else 0
        false_positive = int(matrix[class_id, :].sum() - matrix[class_id, class_id]) if matrix.size else 0
        false_negative = int(matrix[:, class_id].sum() - matrix[class_id, class_id]) if matrix.size else 0
        per_class.append({
            "class_id": int(class_id),
            "class": names[int(class_id)],
            "precision": float(precision[metric_index]),
            "recall": float(recall[metric_index]),
            "f1": float(f1[metric_index]),
            "map50": float(ap50[metric_index]),
            "map50_95": float(ap[metric_index]),
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
        })

    speed = {key: float(value) for key, value in metrics.speed.items()}
    inference_ms = speed.get("inference", 0.0)
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "weights": str(MODEL_PATHS[mode]),
        "dataset": str(data.resolve()),
        "split": split,
        "image_size": IMAGE_SIZES[mode],
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "false_positives": int(sum(row["false_positives"] for row in per_class)),
        "false_negatives": int(sum(row["false_negatives"] for row in per_class)),
        "latency_ms_per_image": speed,
        "inference_fps": 1000.0 / inference_ms if inference_ms > 0 else None,
        "evaluated_classes": len(class_ids),
        "configured_classes": class_count,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="YOLO dataset YAML with a held-out val/test split")
    parser.add_argument("--mode", choices=sorted(MODEL_PATHS), default="accurate")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. cpu, 0, or mps")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results.json"))
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Dataset YAML does not exist: {args.data}")
    weights = MODEL_PATHS[args.mode]
    if not weights.is_file():
        parser.error(f"Model weights do not exist: {weights}")
    if not 0 < args.conf <= 1 or not 0 < args.iou <= 1:
        parser.error("--conf and --iou must be in (0, 1]")

    model = YOLO(str(weights))
    kwargs = {
        "data": str(args.data),
        "split": args.split,
        "imgsz": IMAGE_SIZES[args.mode],
        "conf": args.conf,
        "iou": args.iou,
        "plots": True,
        "verbose": False,
    }
    if args.device:
        kwargs["device"] = args.device
    metrics = model.val(**kwargs)
    report = build_report(metrics, args.mode, args.data, args.split, args.conf, args.iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
