from __future__ import annotations

import streamlit as st

from src.config import ROOT
from src.navigation import NAVIGATION_SECTIONS
from src.security import DeploymentSettings, enforce_production_access


favicon = ROOT / "assets" / "favicon.svg"
st.set_page_config(
    page_title="VisionGuard AI",
    page_icon=str(favicon),
    layout="wide",
    initial_sidebar_state="expanded",
)

deployment = DeploymentSettings.from_environment()
enforce_production_access(deployment)

# Render the shared shell before the selected page. Navigation remains on
# Streamlit's internal router, avoiding full-document reloads and late redraws.
css = (ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
st.markdown('<style data-visionguard-style="navigation-shell-v2">' + css + "</style>", unsafe_allow_html=True)
logo = ROOT / "assets" / "logo.svg"
if logo.exists():
    st.logo(str(logo), size="large")

navigation = st.navigation(
    {
        section: [
            st.Page(
                item.source,
                title=item.title,
                icon=f":material/{item.icon}:",
                default=item.default,
            )
            for item in items
        ]
        for section, items in NAVIGATION_SECTIONS
    },
    position="sidebar",
    expanded=True,
)

navigation.run()
