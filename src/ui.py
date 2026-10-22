from __future__ import annotations

import streamlit as st

from .config import ROOT
from .database import Database
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .object_detector import ObjectDetector
from .processor import VisionProcessor


OBJECT_DETECTOR_REVISION = "pen-refinement-v2"


COMPACT_NAVIGATION = (
    ("OVR", (("Dashboard", "/", "dashboard"),)),
    ("DET", (
        ("Live Detection", "/Live_Detection", "videocam"),
        ("Image Detection", "/Image_Detection", "image_search"),
        ("Video Detection", "/Video_Detection", "movie"),
    )),
    ("DATA", (
        ("Face Registration", "/Face_Registration", "person_add"),
        ("Analytics", "/Analytics", "monitoring"),
        ("Detection History", "/Detection_History", "history"),
    )),
    ("PRO", (
        ("Model Performance", "/Model_Performance", "speed"),
        ("Project Guide", "/Project_Guide", "school"),
    )),
)


def compact_navigation(active_title: str) -> str:
    sections: list[str] = []
    for section, items in COMPACT_NAVIGATION:
        links = []
        for title, href, icon in items:
            active = " active" if title == active_title else ""
            links.append(
                f'<a class="vg-mini-link{active}" href="{href}" target="_self" title="{title}" '
                f'aria-label="{title}"><span class="material-symbols-rounded">{icon}</span></a>'
            )
        sections.append(f'<div class="vg-mini-section"><div class="vg-mini-heading">{section}</div>{"".join(links)}</div>')
    return f'<div class="vg-mini-nav" aria-label="Compact navigation">{"".join(sections)}</div>'


def setup_page(title: str, icon: str = "🛡️") -> None:
    favicon = ROOT / "assets" / "favicon.svg"
    st.set_page_config(page_title=f"{title} · VisionGuard AI", page_icon=str(favicon))
    css = (ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.html(compact_navigation(title))
    logo = ROOT / "assets" / "logo.svg"
    if logo.exists():
        st.logo(str(logo), size="large")


@st.cache_resource
def get_database() -> Database:
    return Database()


@st.cache_resource(show_spinner="Loading object-detection model…")
def get_object_detector(mode: str = "accurate", revision: str = OBJECT_DETECTOR_REVISION) -> ObjectDetector:
    # ``revision`` is part of Streamlit's cache key. Bump it whenever detector
    # post-processing changes so a live app cannot retain stale model logic.
    _ = revision
    return ObjectDetector(mode)


@st.cache_resource(show_spinner="Loading local face models…")
def get_face_detector() -> FaceDetector:
    return FaceDetector()


@st.cache_resource(show_spinner="Loading face embeddings…")
def get_face_recognizer() -> FaceRecognizer:
    return FaceRecognizer()


def get_processor(objects: bool = True, faces: bool = True, object_mode: str = "accurate") -> VisionProcessor:
    return VisionProcessor(get_object_detector(object_mode, OBJECT_DETECTOR_REVISION) if objects else None, get_face_detector() if faces else None,
                           get_face_recognizer() if faces else None)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">VISIONGUARD AI</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


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
