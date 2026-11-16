from __future__ import annotations

from html import escape
import uuid

import streamlit as st

from .config import ROOT
from .database import Database
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .navigation import NAVIGATION_SECTIONS
from .object_detector import ObjectDetector
from .processor import VisionProcessor
from .retention import apply_attendance_retention
from .security import DeploymentSettings, enforce_production_access, show_identity_controls


OBJECT_DETECTOR_REVISION = "supervised-key-specialist-v14"
STYLE_REVISION = "navigation-shell-v1"


def expanded_navigation(active_title: str) -> str:
    sections: list[str] = []
    for section, items in NAVIGATION_SECTIONS:
        section_id = section.lower().replace("&", "and").replace(" ", "-")
        links = []
        for item in items:
            active = " active" if item.title == active_title else ""
            current = ' aria-current="page"' if active else ""
            links.append(
                f'<a class="vg-side-link{active}" href="{item.href}" target="_self"{current}>'
                f'<span class="material-symbols-rounded" aria-hidden="true">{item.icon}</span>'
                f'<span>{escape(item.title)}</span></a>'
            )
        sections.append(
            f'<section class="vg-side-section" aria-labelledby="vg-nav-{section_id}">'
            f'<h2 id="vg-nav-{section_id}">{escape(section)}</h2>{"".join(links)}</section>'
        )
    return f'<nav class="vg-side-nav" aria-label="Primary navigation">{"".join(sections)}</nav>'


def compact_navigation(active_title: str) -> str:
    sections: list[str] = []
    for section, items in NAVIGATION_SECTIONS:
        links = []
        for item in items:
            active = " active" if item.title == active_title else ""
            current = ' aria-current="page"' if active else ""
            links.append(
                f'<a class="vg-mini-link{active}" href="{item.href}" target="_self" title="{escape(item.title)}" '
                f'aria-label="{escape(item.title)}"{current}>'
                f'<span class="material-symbols-rounded" aria-hidden="true">{item.icon}</span></a>'
            )
        sections.append(f'<div class="vg-mini-section" role="group" aria-label="{escape(section)}">{"".join(links)}</div>')
    return f'<nav class="vg-mini-nav" aria-label="Primary navigation">{"".join(sections)}</nav>'


def setup_page(title: str, icon: str = "🛡️") -> None:
    favicon = ROOT / "assets" / "favicon.svg"
    st.set_page_config(page_title=f"{title} · VisionGuard AI", page_icon=str(favicon))
    deployment = DeploymentSettings.from_environment()
    enforce_production_access(deployment)
    if deployment.production and deployment.retention_days is not None:
        apply_attendance_retention(get_database(), deployment.retention_days)
    css = (ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f'<style data-visionguard-style="{STYLE_REVISION}">{css}</style>', unsafe_allow_html=True)
    st.html(compact_navigation(title))
    logo = ROOT / "assets" / "logo.svg"
    if logo.exists():
        st.logo(str(logo), size="large")
    with st.sidebar:
        st.html(expanded_navigation(title))
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
