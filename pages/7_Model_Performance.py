from __future__ import annotations

import platform
import time
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MODELS_DIR, DetectionSettings
from src.performance import classification_report, summarize_benchmark
from src.ui import get_processor, hero, metric_cards, setup_page, show_processing_error
from src.video_utils import decode_image

setup_page("Model Performance", "📊")
hero("Model Performance", "Measure runtime speed and evaluate labeled classification results without invented scores")

st.subheader("System and model inventory")
model_rows = []
for filename, role in [
    ("yolo11n.pt", "Fast COCO object detector"),
    ("yolo11s.pt", "Accurate COCO object detector"),
    ("yolov8s-worldv2.pt", "Open-vocabulary hybrid detector"),
    ("face_detection_yunet_2023mar.onnx", "YuNet face detector"),
    ("face_recognition_sface_2021dec.onnx", "SFace face recognizer"),
]:
    path = MODELS_DIR / filename
    model_rows.append({"model": filename, "role": role, "available": "Yes" if path.exists() else "No",
                       "size (MB)": round(path.stat().st_size / 1_048_576, 1) if path.exists() else None})
left, right = st.columns([1.6, 1])
left.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
right.markdown(f"""
**Runtime environment**

- Operating system: `{platform.system()} {platform.release()}`
- Processor: `{platform.machine()}`
- Python: `{platform.python_version()}`
- OpenCV: `{cv2.__version__}`
""")

st.subheader("Local inference benchmark")
st.caption("Upload a representative image. The first pass warms the model; reported values use only measured runs after warm-up.")
controls, uploader = st.columns([1, 1.5])
with controls:
    model_label = st.selectbox("Object model", ["Fast - YOLO11n", "Accurate - YOLO11s", "Hybrid - YOLO11s + YOLO-World"])
    include_faces = st.toggle("Include face pipeline", True)
    runs = st.slider("Measured runs", 1, 10, 3)
    confidence = st.slider("Confidence threshold", .1, .95, .25, .05)
with uploader:
    upload = st.file_uploader("Benchmark image", type=["jpg", "jpeg", "png", "webp"])

if upload and st.button("Run benchmark", type="primary"):
    try:
        image = decode_image(upload.getvalue())
        mode = "hybrid" if model_label.startswith("Hybrid") else "fast" if model_label.startswith("Fast") else "accurate"
        processor = get_processor(True, include_faces, mode)
        settings = DetectionSettings(object_detection=True, face_detection=include_faces,
                                     face_recognition=include_faces, tracking=False,
                                     confidence=confidence, max_width=1440 if mode == "hybrid" else 960)
        with st.spinner("Warming up models and measuring inference…"):
            processor.process(image, settings)
            durations, output, last_stats = [], image, {}
            for _ in range(runs):
                started = time.perf_counter()
                output, last_stats, _ = processor.process(image, settings)
                durations.append((time.perf_counter() - started) * 1000)
        summary = summarize_benchmark(durations)
        metric_cards([
            ("Mean latency", f"{summary.mean_ms:.0f} ms", "◷"),
            ("95th percentile", f"{summary.p95_ms:.0f} ms", "↗"),
            ("Throughput", f"{summary.fps:.1f} FPS", "▶"),
            ("Objects", last_stats["objects"], "◉"),
            ("Faces", last_stats["faces"], "◎"),
        ])
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption=f"Last benchmark result · {model_label}", use_container_width=True)
        st.dataframe(pd.DataFrame({"run": range(1, runs + 1), "latency_ms": durations}), use_container_width=True, hide_index=True)
    except Exception as exc:
        show_processing_error("Benchmark", exc)

st.subheader("Labeled-result evaluation")
st.write("Upload a CSV with `actual` and `predicted` columns to calculate accuracy, per-class precision, recall, F1, and a confusion matrix.")
template = b"actual,predicted\nperson,person\nbottle,bottle\nphone,person\n"
st.download_button("Download CSV template", template, "visionguard_evaluation_template.csv", "text/csv")
evaluation = st.file_uploader("Evaluation CSV", type=["csv"], key="evaluation-csv")
if evaluation:
    try:
        matrix, report, summary = classification_report(pd.read_csv(evaluation))
        metric_cards([
            ("Accuracy", f"{summary['accuracy']:.1%}", "✓"),
            ("Macro precision", f"{summary['macro_precision']:.1%}", "P"),
            ("Macro recall", f"{summary['macro_recall']:.1%}", "R"),
            ("Macro F1", f"{summary['macro_f1']:.1%}", "F1"),
        ])
        chart, table = st.columns([1.2, 1])
        figure = px.imshow(matrix, text_auto=True, color_continuous_scale="Teal", labels={"x": "Predicted", "y": "Actual"})
        figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        chart.plotly_chart(figure, use_container_width=True)
        display = report.copy()
        for column in ["precision", "recall", "f1"]:
            display[column] = display[column].map(lambda value: f"{value:.1%}")
        table.dataframe(display, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")

with st.expander("How to report mAP correctly"):
    st.markdown("""
    Object-detection mAP cannot be calculated from class names alone. It requires a labeled validation set with
    ground-truth bounding boxes, predicted boxes, confidence scores, and an IoU threshold. Use Ultralytics validation
    on a YOLO-format dataset and report the dataset, split, image count, IoU settings, precision, recall, mAP50, and
    mAP50–95. Until that experiment is performed, label those values as **not evaluated**, not zero.
    """)
