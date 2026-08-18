from __future__ import annotations

import shutil

import pandas as pd
import streamlit as st

from src.config import UNKNOWN_FACES_DIR
from src.ui import get_database, hero, setup_page

setup_page("Detection History", "🕘")
hero("Detection History", "Filter, audit, and export locally stored detection events")
db = get_database(); df = db.detections_frame()
if df.empty:
    st.info("No detection history is available.")
else:
    df["timestamp"] = pd.to_datetime(df.timestamp)
    min_date, max_date = df.timestamp.dt.date.min(), df.timestamp.dt.date.max()
    date_filter = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    c1, c2, c3, c4, c5 = st.columns(5)
    type_filter = c1.multiselect("Type", sorted(df.object_type.dropna().unique()))
    object_filter = c2.multiselect("Object", sorted(df.object_name.dropna().unique()))
    session_filter = c3.multiselect("Session", sorted(df.session_type.dropna().unique()))
    identity_filter = c4.multiselect("Identity", sorted(df.recognized_person.dropna().unique()))
    face_status = c5.selectbox("Face status", ["All", "Known", "Unknown"])
    filtered = df.copy()
    if isinstance(date_filter, tuple) and len(date_filter) == 2:
        filtered = filtered[filtered.timestamp.dt.date.between(date_filter[0], date_filter[1])]
    if type_filter: filtered = filtered[filtered.object_type.isin(type_filter)]
    if object_filter: filtered = filtered[filtered.object_name.isin(object_filter)]
    if session_filter: filtered = filtered[filtered.session_type.isin(session_filter)]
    if identity_filter: filtered = filtered[filtered.recognized_person.isin(identity_filter)]
    if face_status == "Known": filtered = filtered[(filtered.object_type == "face") & (~filtered.recognized_person.isin([None, "Unknown"]))]
    elif face_status == "Unknown": filtered = filtered[(filtered.object_type == "face") & (filtered.recognized_person == "Unknown")]
    filtered["confidence"] = filtered.confidence.map(lambda x: f"{x:.1%}")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Export filtered CSV", filtered.to_csv(index=False).encode(), "visionguard_history.csv", "text/csv")

st.subheader("Privacy controls")
c1, c2 = st.columns(2)
if c1.button("Delete stored unknown faces"):
    for file in UNKNOWN_FACES_DIR.glob("*"):
        if file.is_file(): file.unlink()
    st.success("Stored unknown-face images were deleted.")
if c2.button("Clear detection history"):
    db.clear_history(); st.success("Detection history was cleared."); st.rerun()
