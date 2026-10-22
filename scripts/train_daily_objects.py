#!/usr/bin/env python3
"""Fine-tune YOLO11n on the prepared daily-object dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/openimages_daily_objects/dataset.yaml")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--freeze", type=int)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/daily_objects.pt")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"dataset YAML does not exist: {args.data}")
    if not args.model.is_file():
        parser.error(f"base weights do not exist: {args.model}")
    if (
        args.epochs < 1
        or args.imgsz < 320
        or args.batch < 1
        or args.lr0 <= 0
        or args.warmup_epochs < 0
    ):
        parser.error("training values must be positive; imgsz must be at least 320")

    model = YOLO(str(args.model))
    result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        freeze=args.freeze,
        warmup_epochs=args.warmup_epochs,
        patience=10,
        seed=42,
        deterministic=True,
        project=str(PROJECT_ROOT / "runs/daily_objects"),
        name="yolo11n_openimages",
        exist_ok=True,
        plots=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"training completed without best weights: {best}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    print(f"best weights copied to {args.output}")


if __name__ == "__main__":
    main()
