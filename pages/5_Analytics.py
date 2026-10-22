from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.analytics import object_counts, prepare_activity
from src.ui import get_database, hero, metric_cards, setup_page

setup_page("Analytics", "📊")
hero("Detection Analytics", "Explore detection volume, identity trends, and session performance")
if st.button("Refresh analytics", icon="🔄"):
    st.rerun()
db = get_database(); df = db.detections_frame(); sessions = db.sessions_frame(); stats = db.dashboard_stats()
unique_objects = df[df.object_type == "object"].object_name.nunique() if not df.empty else 0
metric_cards([("Detection Events", len(df), "◉"), ("Face Observations", stats["faces"], "◎"),
              ("Recognized Identities", stats["known"], "✓"), ("Unknown Observations", stats["unknown"], "!"),
              ("Unique Objects", unique_objects, "◇"), ("Sessions", stats["sessions"], "▦")])

if df.empty:
    st.info("Analytics will appear after the first detection session.")
    st.stop()
objects = object_counts(df)
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(objects.head(12), x="count", y="object_name", orientation="h", title="Most Observed Object Classes", color="count", color_continuous_scale="Teal")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis={"categoryorder":"total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.pie(objects.head(10), names="object_name", values="count", hole=.62, title="Detection Distribution")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
activity = prepare_activity(df)
fig = px.line(activity, x="date", y="detections", markers=True, title="Detection Activity Over Time")
fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)
c1, c2 = st.columns(2)
faces = df[df.object_type == "face"].copy()
with c1:
    if not faces.empty:
        faces["status"] = faces.recognized_person.fillna("Unknown").apply(lambda x: "Unknown" if x == "Unknown" else "Known")
        fig = px.pie(faces, names="status", hole=.65, title="Known vs Unknown Faces", color_discrete_sequence=["#62b7b2", "#d7a99e"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(fig, use_container_width=True)
with c2:
    known = faces[~faces.recognized_person.isin([None, "Unknown"])].recognized_person.value_counts().head(10)
    if not known.empty:
        fig = px.bar(x=known.values, y=known.index, orientation="h", title="Top Recognized People")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(fig, use_container_width=True)
st.subheader("Detection sessions")
st.dataframe(sessions, use_container_width=True, hide_index=True)
