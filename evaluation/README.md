# Accuracy evaluation

No labeled held-out dataset is currently included, so precision, recall, F1, mAP, false-positive counts, and false-negative counts are **not evaluated**. Runtime output or a few webcam screenshots cannot replace ground truth.

## Object detection dataset

Use standard YOLO detection layout:

```text
visionguard-evaluation/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Each image needs a matching text file containing `class x_center y_center width height`, normalized to 0–1. Keep the test split fixed and separate from any fine-tuning data. Copy `dataset.example.yaml` to `dataset.yaml`, use absolute paths, and list every class ID present in the labels.

Recommended test matrix:

- single and multiple objects;
- small, distant, overlapping, and partially occluded objects;
- bright, low-light, and uneven-light scenes;
- visually similar COCO classes such as phone/scissors;
- moving objects sampled from live video;
- unsupported objects such as pens in a separate YOLO-World evaluation set.

Evaluate the two standard modes:

```bash
.venv/bin/python scripts/evaluate_object_detection.py \
  --data evaluation/dataset.yaml --mode fast --split test \
  --output evaluation/fast-results.json

.venv/bin/python scripts/evaluate_object_detection.py \
  --data evaluation/dataset.yaml --mode accurate --split test \
  --output evaluation/accurate-results.json
```

The reports contain precision, recall, F1, mAP@0.50, mAP@0.50:0.95, per-class metrics, false positives, false negatives, latency, and inference FPS. Compare revisions only with the same dataset split and settings.

## Face evaluation

Evaluate YuNet separately with face bounding-box annotations and IoU-based precision/recall. Evaluate SFace separately with consented positive and negative identity pairs, reporting false-accept and false-reject rates across similarity thresholds. Do not mix face identity similarity with YOLO confidence. No suitable consented face benchmark is bundled, so face metrics are also **not evaluated**.
