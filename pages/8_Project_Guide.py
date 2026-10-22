from __future__ import annotations

import streamlit as st

from src.ui import hero, setup_page

setup_page("Project Guide", "🎓")
hero("Project Guide", "A presentation-ready explanation of VisionGuard AI")

st.subheader("Problem and objective")
st.write("VisionGuard AI demonstrates how computer vision can automate object monitoring and local face recognition while producing searchable records and analytics. Its objective is to combine detection, tracking, recognition, storage, and visualization in one privacy-conscious prototype.")

st.subheader("Processing pipeline")
st.markdown("""
1. A webcam, image, or video supplies frames.
2. YOLO detects objects; ByteTrack assigns persistent IDs in video.
3. YuNet detects faces independently of YOLO person boxes.
4. SFace creates an embedding and compares it with locally registered embeddings.
5. OpenCV draws labels; selected evidence and event metadata are stored locally.
6. SQLite data is aggregated into history and Plotly analytics.
""")

st.subheader("Recommended live demonstration")
demo_steps = [
    "Open the dashboard and explain the five headline metrics.",
    "Register one face using a clear front-facing photo.",
    "Start Live Detection and show object labels, track IDs, recognition, FPS, and latency.",
    "Process a prepared image containing multiple objects.",
    "Open Model Performance and run a three-pass benchmark.",
    "Show Detection History and Analytics to demonstrate persistence.",
    "Finish with privacy safeguards, limitations, and future scope.",
]
for number, step in enumerate(demo_steps, 1):
    st.markdown(f"**{number}.** {step}")

st.subheader("Likely viva questions")
questions = {
    "Why use YOLO?": "It performs single-stage detection, offering a practical balance between speed and accuracy for real-time work.",
    "Why is ByteTrack needed?": "Detection identifies objects per frame; ByteTrack associates detections across frames to provide persistent IDs.",
    "How does face recognition work?": "YuNet finds and aligns a face, SFace converts it to an embedding, and cosine similarity is compared with saved embeddings.",
    "Why use SQLite?": "It is lightweight, local, easy to demonstrate, and sufficient for a single-machine academic prototype.",
    "What affects accuracy?": "Lighting, pose, occlusion, image resolution, confidence thresholds, model bias, and the quality of reference images.",
    "What are the ethical concerns?": "Consent, biometric privacy, demographic bias, false matches, retention, and misuse as an access-control system.",
}
for question, answer in questions.items():
    with st.expander(question):
        st.write(answer)

st.subheader("Limitations and future scope")
left, right = st.columns(2)
left.markdown("""
**Current limitations**

- Not a certified security or identity system
- Performance depends on the demonstration hardware
- Recognition quality changes with pose and lighting
- Tracking IDs can change after long occlusion
- Uploaded-video codec support varies by machine
""")
right.markdown("""
**Reasonable future scope**

- Attendance reports and check-in rules
- Multiple IP-camera feeds
- Restricted-zone and occupancy rules
- Liveness detection
- Role-based access and configurable retention
""")

st.info("Presentation tip: keep a short screen recording and two prepared media files as a backup if webcam permissions or lighting fail.")
