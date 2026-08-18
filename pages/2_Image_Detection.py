from __future__ import annotations

from collections import Counter

import cv2
import streamlit as st

from src.config import DetectionSettings
from src.ui import get_database, get_processor, hero, metric_cards, setup_page
from src.video_utils import decode_image

setup_page("Image Detection", "🖼️")
hero("Image Detection", "Inspect objects, faces, identities, and confidence scores in a single image")

with st.sidebar:
    objects = st.toggle("Object Detection", True)
    model_label = st.selectbox("Object model", ["Extended - Pens, headphones + custom classes", "Accurate - YOLO11s", "Fast - YOLO11n"], disabled=not objects, key="image_object_model_v3")
    faces = st.toggle("Face Detection", True)
    recognition = st.toggle("Face Recognition", True, disabled=not faces)
    confidence = st.slider("Confidence Threshold", .1, .95, .2, .05)
    face_threshold = st.slider("Face Recognition Threshold", .25, .8, .45, .01)

upload = st.file_uploader("Upload JPG, PNG, or WEBP", type=["jpg", "jpeg", "png", "webp"])
if upload:
    try:
        original = decode_image(upload.getvalue())
        object_mode = "extended" if model_label.startswith("Extended") else "fast" if model_label.startswith("Fast") else "accurate"
        processor = get_processor(objects, faces, object_mode)
        settings = DetectionSettings(object_detection=objects, face_detection=faces, face_recognition=recognition,
                                     tracking=False, confidence=confidence, face_threshold=face_threshold)
        session_id = get_database().start_session("image")
        processed, stats, logs = processor.process(original, settings)
        get_database().log_detections(session_id, logs)
        get_database().finish_session(session_id, {"total_objects": stats["objects"], "total_faces": stats["faces"],
                                                    "known_faces": stats["known"], "unknown_faces": stats["unknown"]})
        left, right = st.columns(2)
        left.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        right.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Processed", use_container_width=True)
        metric_cards([("Objects", stats["objects"], "◉"), ("Faces", stats["faces"], "◎"),
                      ("Known", stats["known"], "✓"), ("Unknown", stats["unknown"], "!")])
        if stats["classes"]:
            st.write("Detection summary", stats["classes"])
        ok, encoded = cv2.imencode(".jpg", processed)
        if ok:
            st.download_button("Download processed image", encoded.tobytes(), "visionguard_result.jpg", "image/jpeg")
    except Exception as exc:
        st.error(f"The image could not be processed: {exc}")
