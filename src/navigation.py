from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NavigationItem:
    """One canonical destination shared by every navigation presentation."""

    title: str
    source: str
    icon: str
    default: bool = False

    @property
    def href(self) -> str:
        if self.default:
            return "/"
        return f"/{Path(self.source).stem.split('_', 1)[-1]}"


NAVIGATION_SECTIONS: tuple[tuple[str, tuple[NavigationItem, ...]], ...] = (
    ("Overview", (NavigationItem("Dashboard", "pages/0_Dashboard.py", "dashboard", default=True),)),
    (
        "Detection",
        (
            NavigationItem("Live Detection", "pages/1_Live_Detection.py", "videocam"),
            NavigationItem("Image Detection", "pages/2_Image_Detection.py", "image_search"),
            NavigationItem("Video Detection", "pages/3_Video_Detection.py", "movie"),
        ),
    ),
    (
        "People & reports",
        (
            NavigationItem("Face Registration", "pages/4_Face_Registration.py", "person_add"),
            NavigationItem("Attendance", "pages/5_Attendance.py", "how_to_reg"),
            NavigationItem("Analytics", "pages/5_Analytics.py", "monitoring"),
            NavigationItem("Detection History", "pages/6_Detection_History.py", "history"),
        ),
    ),
    (
        "System",
        (
            NavigationItem("Model Performance", "pages/7_Model_Performance.py", "speed"),
            NavigationItem("Project Guide", "pages/8_Project_Guide.py", "school"),
        ),
    ),
)
