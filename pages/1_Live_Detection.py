from __future__ import annotations

import threading
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from src.config import DetectionSettings
from src.database import Database
from src.ui import get_processor, hero, metric_cards, setup_page, show_processing_error

setup_page("Live Detection", "📹")
hero("Live Detection", "Real-time YOLO tracking and private, on-device face recognition")

with st.sidebar:
    st.header("Detection settings")
    object_on = st.toggle("Object Detection", True)
    model_label = st.selectbox(
        "Object model",
        ["Accurate - Standard objects", "Fast - Lower CPU usage", "Experimental - Pens + accessories"],
        help="Accurate is recommended. Experimental adds a filtered open-vocabulary pass for pens, headphones, and earbuds, but may be less reliable.",
        disabled=not object_on,
        key="object_model_v3",
    )
    face_on = st.toggle("Face Detection", True)
    recognition_on = st.toggle("Face Recognition", True, disabled=not face_on)
    tracking_on = st.toggle("Object Tracking", True, disabled=not object_on)
    save_unknown = st.toggle("Save Unknown Faces", False)
    evidence_on = st.toggle("Capture Screenshots", False)
    confidence = st.slider("Confidence Threshold", .1, .95, .40, .05,
                           help="0.40 prioritizes correct labels. Lower it only when you need more small or partly hidden objects.")
    face_threshold = st.slider("Face Recognition Threshold", .25, .80, .45, .01)
    triggers = set(st.multiselect("Screenshot triggers", ["unknown_person", "cell phone", "laptop", "backpack"], ["unknown_person", "cell phone"]))

settings = DetectionSettings(object_detection=object_on, face_detection=face_on,
                             face_recognition=recognition_on, tracking=tracking_on,
                             save_unknown_faces=save_unknown, capture_screenshots=evidence_on,
                             confidence=confidence, face_threshold=face_threshold,
                             max_width=960 if model_label.startswith("Fast") else 1280)
object_mode = "hybrid" if model_label.startswith("Experimental") else "fast" if model_label.startswith("Fast") else "accurate"

try:
    processor = get_processor(object_on, face_on, object_mode)
except Exception as exc:
    show_processing_error("Detection model loading", exc)
    st.stop()

class VideoProcessor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stats = {"objects": 0, "faces": 0, "known": 0, "unknown": 0, "fps": 0,
                      "processing_ms": 0, "classes": {}}
        self.totals = {"total_objects": 0, "total_faces": 0, "known_faces": 0, "unknown_faces": 0}
        self.last = time.perf_counter()
        self.last_log = 0.0
        self.database = Database()
        self.session_id = self.database.start_session("live")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        output, stats, logs = processor.process(image, settings, triggers)
        now = time.perf_counter()
        stats["fps"] = 1 / max(now - self.last, .001)
        self.last = now
        with self.lock:
            self.stats = stats
            self.totals["total_objects"] += stats["objects"]
            self.totals["total_faces"] += stats["faces"]
            self.totals["known_faces"] += stats["known"]
            self.totals["unknown_faces"] += stats["unknown"]
            totals_snapshot = self.totals.copy()
        # Persist at most once per second: enough for analytics without one DB write per frame.
        if logs and now - self.last_log >= 1.0:
            try:
                self.database.log_detections(self.session_id, logs)
                self.database.update_session_stats(self.session_id, totals_snapshot)
                self.last_log = now
            except Exception:
                # A transient logging problem must never interrupt the camera stream.
                pass
        return av.VideoFrame.from_ndarray(output, format="bgr24")

ctx = webrtc_streamer(
    key="visionguard-live", mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor, async_processing=True,
)

@st.fragment(run_every=0.75)
def render_live_metrics() -> None:
    if ctx.state.playing and ctx.video_processor:
        with ctx.video_processor.lock:
            stats = ctx.video_processor.stats.copy()
        metric_cards([
            ("Objects", stats["objects"], "◉"), ("Faces", stats["faces"], "◎"),
            ("Known", stats["known"], "✓"), ("Unknown", stats["unknown"], "!"),
            ("Stream FPS", f"{stats['fps']:.1f}", "↗"),
            ("Inference", f"{stats['processing_ms']:.0f} ms", "◷"),
        ])
        st.caption("Live analytics refresh every 0.75 seconds; database events are sampled once per second.")
    else:
        st.info("Click START above and allow camera access. Use STOP to end the live stream.")

render_live_metrics()

st.caption("Live frames are processed locally by the Streamlit server. Browser camera access requires localhost or HTTPS.")
