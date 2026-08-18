from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import cv2
import streamlit as st

from src.config import PROCESSED_DIR, DetectionSettings, ensure_directories
from src.ui import get_database, get_processor, hero, metric_cards, setup_page

setup_page("Video Detection", "🎞️")
hero("Video Detection", "Process uploaded footage with persistent ByteTrack identities")
ensure_directories()

with st.sidebar:
    objects = st.toggle("Object Detection", True)
    model_label = st.selectbox("Object model", ["Extended - Pens, headphones + custom classes", "Accurate - YOLO11s", "Fast - YOLO11n"], disabled=not objects, key="video_object_model_v3")
    faces = st.toggle("Face Detection", True)
    recognition = st.toggle("Face Recognition", True, disabled=not faces)
    confidence = st.slider("Confidence Threshold", .1, .95, .2, .05)
    face_threshold = st.slider("Face Recognition Threshold", .25, .8, .45, .01)
    stride = st.slider("Process every Nth frame", 1, 5, 1)

upload = st.file_uploader("Upload MP4, MOV, AVI, or MKV", type=["mp4", "mov", "avi", "mkv"])
if upload and st.button("Process video", type="primary"):
    temp_path = None
    try:
        suffix = Path(upload.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(upload.getvalue()); temp_path = Path(temp.name)
        capture = cv2.VideoCapture(str(temp_path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not decode this video.")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = PROCESSED_DIR / f"processed_{Path(upload.name).stem}.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        object_mode = "extended" if model_label.startswith("Extended") else "fast" if model_label.startswith("Fast") else "accurate"
        processor = get_processor(objects, faces, object_mode)
        settings = DetectionSettings(object_detection=objects, face_detection=faces, face_recognition=recognition,
                                     tracking=True, confidence=confidence, face_threshold=face_threshold, max_width=width)
        session_id = get_database().start_session("video")
        progress, status = st.progress(0), st.empty()
        frame_no, aggregate, batch = 0, Counter(), []
        while True:
            ok, frame = capture.read()
            if not ok: break
            frame_output = frame
            if frame_no % stride == 0:
                frame_output, stats, logs = processor.process(frame, settings)
                aggregate.update({"total_objects": stats["objects"], "total_faces": stats["faces"],
                                  "known_faces": stats["known"], "unknown_faces": stats["unknown"]})
                batch.extend(logs)
                if len(batch) >= 100:
                    get_database().log_detections(session_id, batch); batch.clear()
            writer.write(frame_output)
            frame_no += 1
            if frame_no % 10 == 0:
                progress.progress(min(frame_no / max(total, 1), 1.0)); status.caption(f"Processed {frame_no:,} / {total:,} frames")
        if batch: get_database().log_detections(session_id, batch)
        get_database().finish_session(session_id, dict(aggregate))
        capture.release(); writer.release(); progress.progress(1.0)
        metric_cards([("Objects", aggregate["total_objects"], "◉"), ("Faces", aggregate["total_faces"], "◎"),
                      ("Known", aggregate["known_faces"], "✓"), ("Unknown", aggregate["unknown_faces"], "!")])
        data = output_path.read_bytes()
        st.video(data)
        st.download_button("Download processed video", data, output_path.name, "video/mp4")
    except Exception as exc:
        st.error(f"Video processing failed: {exc}")
    finally:
        if temp_path and temp_path.exists(): temp_path.unlink(missing_ok=True)
