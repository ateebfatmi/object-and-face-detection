from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.analytics import object_counts, prepare_activity
from src.ui import get_database, hero, metric_cards, setup_page

setup_page("Dashboard")
db = get_database()
stats = db.dashboard_stats()

hero("VisionGuard AI", "Intelligent Real-Time Face & Object Detection")
metric_cards([
    ("Unique Tracked Objects", f"{stats['objects']:,}", "◉"),
    ("Face Observations", f"{stats['faces']:,}", "◎"),
    ("Recognized Identities", f"{stats['known']:,}", "✓"),
    ("Unknown Observations", f"{stats['unknown']:,}", "!"),
    ("Detection Sessions", f"{stats['sessions']:,}", "▦"),
])

st.subheader("Detection analytics")
df = db.detections_frame()
left, right = st.columns([1.35, 1])
counts = object_counts(df).head(8)
with left:
    if counts.empty:
        st.info("Run a detection session to populate object analytics.")
    else:
        fig = px.bar(counts, x="count", y="object_name", orientation="h", color="count",
                     color_continuous_scale=["#1d4ed8", "#20d7b4"], title="Most observed object classes")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
with right:
    faces = df[df.object_type == "face"] if not df.empty else df
    if faces.empty:
        st.info("Known vs unknown face activity will appear here.")
    else:
        labels = faces.recognized_person.fillna("Unclassified").apply(lambda x: "Unknown" if x == "Unknown" else "Known").value_counts()
        fig = px.pie(values=labels.values, names=labels.index, hole=.68, color_discrete_sequence=["#20d7b4", "#ef5466"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent activity")
if df.empty:
    st.caption("No detections logged yet. Open Live, Image, or Video Detection from the sidebar.")
else:
    recent = df.head(8).copy()
    recent["identity"] = recent.recognized_person.fillna("—")
    recent["confidence"] = recent.confidence.map(lambda x: f"{x:.0%}")
    st.dataframe(recent[["timestamp", "object_type", "object_name", "confidence", "identity"]],
                 use_container_width=True, hide_index=True)
