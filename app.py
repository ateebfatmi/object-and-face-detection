from __future__ import annotations

import streamlit as st

from src.config import ROOT


favicon = ROOT / "assets" / "favicon.svg"
st.set_page_config(
    page_title="VisionGuard AI",
    page_icon=str(favicon),
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    {
        "OVERVIEW": [
            st.Page("pages/0_Dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
        ],
        "DETECTION": [
            st.Page("pages/1_Live_Detection.py", title="Live Detection", icon=":material/videocam:"),
            st.Page("pages/2_Image_Detection.py", title="Image Detection", icon=":material/image_search:"),
            st.Page("pages/3_Video_Detection.py", title="Video Detection", icon=":material/movie:"),
        ],
        "DATA & INSIGHTS": [
            st.Page("pages/4_Face_Registration.py", title="Face Registration", icon=":material/person_add:"),
            st.Page("pages/5_Analytics.py", title="Analytics", icon=":material/monitoring:"),
            st.Page("pages/6_Detection_History.py", title="Detection History", icon=":material/history:"),
        ],
        "PROJECT": [
            st.Page("pages/7_Model_Performance.py", title="Model Performance", icon=":material/speed:"),
            st.Page("pages/8_Project_Guide.py", title="Project Guide", icon=":material/school:"),
        ],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()
