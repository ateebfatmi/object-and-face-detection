# VisionGuard technical audit

Audit date: 2026-10-22

## Repository map

- Entry point and navigation: `app.py`; Streamlit pages live in `pages/`.
- Shared pipeline: `src/processor.py`; model loading and session-scoped caching: `src/ui.py`.
- Object detection/tracking: `src/object_detector.py` plus the two `models/bytetrack*.yaml` configurations.
- Face detection/recognition: `src/face_detector.py` and `src/face_recognizer.py`.
- Media decoding/scaling: `src/video_utils.py`; live WebRTC callback: `pages/1_Live_Detection.py`.
- SQLite/SQLAlchemy persistence: `src/database.py`; analytics/history: `pages/0_Dashboard.py`, `pages/5_Analytics.py`, and `pages/6_Detection_History.py`.
- Runtime and evaluation: `requirements.txt`, `pages/7_Model_Performance.py`, and `scripts/evaluate_object_detection.py`.

## Confirmed findings

- `models/yolo11n.pt` and `models/yolo11s.pt` load as 80-class COCO detectors. They cannot natively detect pens, earbuds, headphones, or wristwatches.
- `models/yolov8s-worldv2.pt` loads through `YOLOWorld`; it is experimental and its prompt scores are not calibrated to YOLO11 confidence scores.
- Previous geometry-only rules renamed YOLO11 phone/scissors/clock predictions to unsupported pen/watch classes. This could make a demonstration look correct but was not model evidence and has been removed.
- An unknown mode previously fell through to YOLO11s. Mode and weight validation are now explicit, with no silent fallback.
- Parsed boxes were not explicitly clamped or checked for zero area. They are now validated against the inference frame.
- The YOLO11 and YOLO-World sources were not stored in history. A backward-compatible `detector_source` column now distinguishes `yolo11`, `yolo_world`, and `yunet`; existing rows are retained and marked `legacy_object` or `legacy_face` because their original model cannot be reconstructed.
- ByteTrack state lived on a globally cached mutable detector. Detector caches are now scoped by browser session and trackers reset for each new camera/video source.
- IoU suppression was fixed in code and unavailable in the UI. Confidence and IoU are now independently configurable.
- No labeled detection or face benchmark exists, so accuracy and mAP improvement claims cannot be made.

## Suspected issues requiring labeled tests

- The best confidence, IoU, and input-size settings vary by camera, lighting, and object scale.
- YOLO-World prompt quality for `bottle`, `stainless steel bottle`, `electric ceiling fan`, `long thin pen`, `earbuds`, and `headphones` requires a representative labeled image set. Wide frames are evaluated as two overlapping vertical tiles so narrow objects retain more pixels; tile detections are fused in full-frame coordinates.
- Face-recognition threshold `0.45` requires genuine/impostor validation using consented data from the target camera domain.
- Frame skipping can reduce temporal recall and cause unannotated frames in exported videos; stride 1 remains the accuracy default.

## Verified local environment

- Python 3.12.14
- Ultralytics 8.4.121
- PyTorch 2.13.0
- OpenCV 4.14.0
- NumPy 2.5.2
- CUDA unavailable; MPS unavailable in the installed PyTorch build, so validation ran on CPU.

Verified checkpoint SHA-256 values:

- YOLO11n: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
- YOLO11s: `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`
- YOLO-World v2: `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`
- YuNet: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- SFace: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`

## Performance bottlenecks

- This environment is CPU-only. Accurate still-image TTA and the Experimental mode's second 1280px inference pass are intentionally slower.
- Live inference runs synchronously inside the WebRTC callback. A lock protects each session's mutable model/tracker state, so concurrent calls serialize rather than corrupt tracks.
- Face detection and recognition share cached model objects; locks now protect their mutable OpenCV state.
- Live database writes remain sampled at one per second, while uploaded-video writes are batched in groups of 100.

## Final configuration

| Mode | Primary model | Size | Still-image TTA | Additional detector |
|---|---|---:|---|---|
| Fast | YOLO11n COCO | 640 | No | None |
| Accurate | YOLO11s COCO | 960 | Yes | None |
| Extended | YOLO11s COCO | 960 | No while tracking | YOLO-World v2 on overlapping 960px tiles for fans, bottles, and accessories |

Defaults are YOLO11 confidence `0.35`, NMS IoU `0.50`, and YOLO-World confidence `0.02`. These are operating defaults, not validated optimums.

## Priority plan completed

1. Removed fabricated standard-detector labels and separated unsupported categories into YOLO-World.
2. Added strict model/mode/input validation and bounded box parsing.
3. Added session-scoped tracker state and explicit resets.
4. Added detector provenance without deleting existing database records.
5. Added reproducible YOLO-format evaluation tooling and dataset documentation.
6. Expanded mocked regression coverage and retained real-model smoke tests.
