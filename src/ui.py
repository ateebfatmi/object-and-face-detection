from __future__ import annotations

import uuid

import streamlit as st

from .config import ROOT
from .database import Database
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .object_detector import ObjectDetector
from .processor import VisionProcessor
from .retention import apply_attendance_retention
from .security import DeploymentSettings, enforce_production_access, show_identity_controls


OBJECT_DETECTOR_REVISION = "supervised-key-specialist-v14"
def setup_page(title: str, icon: str = "🛡️") -> None:
    favicon = ROOT / "assets" / "favicon.svg"
    st.set_page_config(page_title=f"{title} · VisionGuard AI", page_icon=str(favicon))
    deployment = DeploymentSettings.from_environment()
    enforce_production_access(deployment)
    if deployment.production and deployment.retention_days is not None:
        apply_attendance_retention(get_database(), deployment.retention_days)
    show_identity_controls()


@st.cache_resource
def get_database() -> Database:
    return Database()


@st.cache_resource(show_spinner="Loading object-detection model…")
def get_object_detector(
    mode: str = "accurate",
    revision: str = OBJECT_DETECTOR_REVISION,
    session_key: str = "standalone",
) -> ObjectDetector:
    # ``revision`` and ``session_key`` are cache-key fields. The latter keeps
    # mutable ByteTrack state isolated between browser sessions.
    _ = revision, session_key
    return ObjectDetector(mode)


@st.cache_resource(show_spinner="Loading local face models…")
def get_face_detector() -> FaceDetector:
    return FaceDetector()


@st.cache_resource(show_spinner="Loading face embeddings…")
def get_face_recognizer() -> FaceRecognizer:
    return FaceRecognizer()


def get_processor(objects: bool = True, faces: bool = True, object_mode: str = "accurate") -> VisionProcessor:
    if "visionguard_detector_session" not in st.session_state:
        st.session_state.visionguard_detector_session = uuid.uuid4().hex
    session_key = str(st.session_state.visionguard_detector_session)
    detector = get_object_detector(object_mode, OBJECT_DETECTOR_REVISION, session_key) if objects else None
    return VisionProcessor(detector, get_face_detector() if faces else None,
                           get_face_recognizer() if faces else None)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">VISIONGUARD · INTELLIGENCE STUDIO</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_cards(items: list[tuple[str, str | int, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, icon) in zip(cols, items):
        col.markdown(f'<div class="metric-card"><span>{icon}</span><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)


def show_processing_error(context: str, exc: Exception) -> None:
    """Render a useful recovery hint without exposing a wall of technical output."""
    message = str(exc).strip() or exc.__class__.__name__
    st.error(f"{context} failed: {message}")
    lowered = message.lower()
    if "download" in lowered or "connection" in lowered or "url" in lowered:
        st.info("Check the internet connection once so the model files can be downloaded, then retry.")
    elif "decode" in lowered or "codec" in lowered or "video" in lowered:
        st.info("Try an H.264 MP4 file, or re-export the media with a standard codec.")
    elif "memory" in lowered or "allocate" in lowered:
        st.info("Select the Fast model, reduce the input resolution, or close other memory-heavy applications.")
    else:
        st.caption("Try the Fast model first. If the problem continues, restart the app and confirm that all model files exist in the models folder.")
