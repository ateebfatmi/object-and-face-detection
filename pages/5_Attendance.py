from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.attendance import analyze_attendance
from src.config import ATTENDANCE_DIR, ensure_directories
from src.security import current_actor
from src.ui import get_database, get_face_detector, get_face_recognizer, hero, metric_cards, setup_page
from src.video_utils import decode_image


setup_page("Attendance", "✅")
hero(
    "Class Attendance",
    "Record a full class from one photo, or check students in one by one using their registered faces.",
)

database = get_database()
people = database.people()

with st.sidebar:
    st.subheader("Attendance settings")
    threshold = st.slider(
        "Face match threshold",
        min_value=0.30,
        max_value=0.80,
        value=0.45,
        step=0.01,
        help="Raise this to reduce false matches. Lower it only after testing your classroom camera.",
    )
    st.caption("Photos and face embeddings stay on this device.")

if not people:
    st.warning("Register every student's face before recording attendance.")
    st.page_link("pages/4_Face_Registration.py", label="Open Face Registration", icon="👤")
    st.stop()

metric_cards([
    ("Registered students", len(people), "◎"),
    ("Capture modes", 2, "◫"),
    ("Storage", "Local", "⌂"),
])

group_tab, individual_tab, history_tab = st.tabs(["Full class photo", "One by one", "Attendance history"])


def read_capture(upload, camera):
    source = camera or upload
    return decode_image(source.getvalue()) if source else None


def save_evidence(image, prefix: str) -> str:
    ensure_directories()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    path = ATTENDANCE_DIR / f"{prefix}_{stamp}.jpg"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("The attendance evidence image could not be saved.")
    return str(path)


def show_last_session(session_id: int) -> None:
    session = database.attendance_session(session_id)
    if session is None:
        return
    absent = max(0, session.total_registered - session.total_present)
    st.success(f"Attendance recorded for {session.class_name}.")
    metric_cards([
        ("Present", session.total_present, "✓"),
        ("Absent", absent, "—"),
        ("Unknown faces", session.unknown_faces, "?"),
    ])
    records = database.attendance_records_frame(session_id)
    st.dataframe(
        records[["person_name", "person_identifier", "status", "confidence"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "person_name": "Student",
            "person_identifier": "Student ID",
            "status": "Status",
            "confidence": st.column_config.NumberColumn("Match", format="%.1%%"),
        },
    )


@st.dialog("Cancel attendance session?")
def confirm_session_cancellation(session_id: int, class_name: str) -> None:
    st.write(f"Cancel the unfinished attendance session for **{class_name}**?")
    st.warning("Captured check-ins in this unfinished session will remain in the audit history, but the session cannot be resumed.")
    keep, cancel = st.columns(2)
    if keep.button("Keep session", use_container_width=True):
        st.rerun()
    if cancel.button("Cancel session", type="primary", use_container_width=True):
        database.cancel_attendance(session_id, current_actor())
        if st.session_state.get("individual_attendance_session_id") == session_id:
            st.session_state.pop("individual_attendance_session_id", None)
            st.session_state.pop("individual_attendance_class", None)
        st.rerun()


with group_tab:
    st.subheader("Analyze and review one class photo")
    st.caption("Use a well-lit, front-facing photo, then review the roster before saving. Face matches are suggestions, not final decisions.")
    group_class = st.text_input("Class / section", key="group_attendance_class", placeholder="Example: B.Tech CSE — Section A")
    group_capture_number = int(st.session_state.get("group_capture_number", 0))
    left, right = st.columns(2)
    group_upload = left.file_uploader(
        "Upload a class photo",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"group_attendance_upload_{group_capture_number}",
    )
    group_camera = right.camera_input("Or take the class photo", key=f"group_attendance_camera_{group_capture_number}")
    if st.button("Analyze class photo", type="primary", key="analyze_group_attendance"):
        image = read_capture(group_upload, group_camera)
        if not group_class.strip():
            st.error("Enter the class or section name.")
        elif image is None:
            st.error("Take or upload one class photo first.")
        else:
            try:
                with st.spinner("Recognizing registered students…"):
                    result = analyze_attendance(
                        image,
                        people,
                        get_face_detector(),
                        get_face_recognizer(),
                        threshold,
                    )
                if result.detected_faces == 0:
                    st.error("No faces were found. Move closer, improve the lighting, and try again.")
                else:
                    st.session_state.group_attendance_analysis = {
                        "class_name": group_class.strip(),
                        "annotated": result.annotated,
                        "scores": {item.person_id: item.similarity for item in result.present},
                        "unknown_faces": result.unknown_faces,
                    }
            except Exception as exc:
                st.error(f"The class photo could not be analyzed: {exc}")

    pending = st.session_state.get("group_attendance_analysis")
    if pending:
        st.image(
            cv2.cvtColor(pending["annotated"], cv2.COLOR_BGR2RGB),
            caption="Review result — teal boxes are suggested matches; red boxes are unknown faces.",
            use_container_width=True,
        )
        scores = pending["scores"]
        review_rows = pd.DataFrame([
            {
                "person_id": person.id,
                "Present": person.id in scores,
                "Student": person.name,
                "Student ID": person.person_identifier,
                "Match": scores.get(person.id),
            }
            for person in people
        ])
        st.write("Review attendance")
        reviewed = st.data_editor(
            review_rows,
            use_container_width=True,
            hide_index=True,
            disabled=["person_id", "Student", "Student ID", "Match"],
            column_config={
                "person_id": None,
                "Present": st.column_config.CheckboxColumn("Present", help="Confirm or correct the suggested result."),
                "Match": st.column_config.NumberColumn("Face match", format="%.1%%"),
            },
            key=f"group_review_{group_capture_number}",
        )
        save_col, discard_col = st.columns(2)
        if save_col.button("Save reviewed attendance", type="primary", use_container_width=True):
            selected: dict[int, float | None] = {}
            for row in reviewed.itertuples(index=False):
                if bool(row.Present):
                    person_id = int(row.person_id)
                    selected[person_id] = scores.get(person_id)
            evidence_path = None
            try:
                evidence_path = save_evidence(pending["annotated"], "class")
                session_id = database.record_class_attendance(
                    pending["class_name"],
                    selected,
                    int(pending["unknown_faces"]),
                    evidence_path,
                    current_actor(),
                )
                st.session_state.last_group_attendance = session_id
                st.session_state.pop("group_attendance_analysis", None)
                st.session_state.group_capture_number = group_capture_number + 1
                st.rerun()
            except Exception as exc:
                if evidence_path:
                    Path(evidence_path).unlink(missing_ok=True)
                st.error(f"Attendance could not be saved: {exc}")
        if discard_col.button("Discard analysis", use_container_width=True):
            st.session_state.pop("group_attendance_analysis", None)
            st.session_state.group_capture_number = group_capture_number + 1
            st.rerun()
    if st.session_state.get("last_group_attendance"):
        show_last_session(int(st.session_state.last_group_attendance))


with individual_tab:
    active_id = st.session_state.get("individual_attendance_session_id")
    active_class = st.session_state.get("individual_attendance_class", "")
    st.subheader("Check in one student at a time")

    if active_id is None:
        recoverable = database.open_attendance_sessions()
        if recoverable:
            st.info("An unfinished one-by-one session can be resumed or cancelled.")
            session_labels = {
                f"{item.class_name} · started {item.started_at:%Y-%m-%d %H:%M}": item
                for item in recoverable
                if item.capture_mode == "One by one"
            }
            if session_labels:
                selected_label = st.selectbox("Unfinished session", list(session_labels))
                selected_session = session_labels[selected_label]
                resume_col, cancel_col = st.columns(2)
                if resume_col.button("Resume session", use_container_width=True):
                    st.session_state.individual_attendance_session_id = selected_session.id
                    st.session_state.individual_attendance_class = selected_session.class_name
                    st.session_state.individual_capture_number = 0
                    st.rerun()
                if cancel_col.button("Cancel unfinished session", use_container_width=True):
                    confirm_session_cancellation(selected_session.id, selected_session.class_name)

    individual_class = st.text_input(
        "Class / section",
        value=active_class,
        disabled=active_id is not None,
        key="individual_attendance_class_input",
        placeholder="Example: B.Tech CSE — Section A",
    )

    if active_id is None:
        if st.button("Begin one-by-one attendance", type="primary"):
            if not individual_class.strip():
                st.error("Enter the class or section name.")
            else:
                active_id = database.start_attendance(
                    individual_class.strip(),
                    "One by one",
                    recorded_by=current_actor(),
                )
                st.session_state.pop("individual_last_mark", None)
                st.session_state.individual_attendance_session_id = active_id
                st.session_state.individual_attendance_class = individual_class.strip()
                st.session_state.individual_capture_number = 0
                st.rerun()
    else:
        session = database.attendance_session(int(active_id))
        st.info(f"Session in progress: {session.class_name if session else active_class}")
        capture_number = int(st.session_state.get("individual_capture_number", 0))
        one_upload = st.file_uploader(
            "Upload one student's photo",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"individual_upload_{capture_number}",
        )
        one_camera = st.camera_input(
            "Or take one student's photo",
            key=f"individual_camera_{capture_number}",
        )
        action_col, finish_col = st.columns(2)
        if action_col.button("Mark student present", type="primary", use_container_width=True):
            image = read_capture(one_upload, one_camera)
            if image is None:
                st.error("Take or upload one student's photo first.")
            else:
                try:
                    with st.spinner("Checking the face…"):
                        result = analyze_attendance(
                            image,
                            people,
                            get_face_detector(),
                            get_face_recognizer(),
                            threshold,
                        )
                    if result.detected_faces != 1:
                        st.error("Use a photo containing exactly one clear face.")
                    elif len(result.present) != 1:
                        st.error("This face did not match a registered student. Try again or register the student first.")
                    else:
                        match = result.present[0]
                        if database.mark_attendance_person(
                            int(active_id),
                            match.person_id,
                            match.similarity,
                            current_actor(),
                        ):
                            st.session_state.individual_capture_number = capture_number + 1
                            st.session_state.individual_last_mark = f"{match.name} ({match.identifier})"
                            st.rerun()
                        else:
                            st.warning(f"{match.name} is already marked present in this session.")
                except Exception as exc:
                    st.error(f"The student could not be checked in: {exc}")
        if finish_col.button("Cancel session", use_container_width=True):
            confirm_session_cancellation(int(active_id), session.class_name if session else active_class)

        if st.session_state.get("individual_last_mark"):
            st.success(f"Marked present: {st.session_state.individual_last_mark}")
        current = database.attendance_records_frame(int(active_id))
        present = current[current["status"] == "Present"] if not current.empty else current
        confidence_by_id = dict(zip(present.get("person_identifier", []), present.get("confidence", [])))
        st.caption(f"{len(present)} of {len(people)} students matched so far. Review the roster before finalizing.")
        roster = pd.DataFrame([
            {
                "person_id": person.id,
                "Present": person.person_identifier in confidence_by_id,
                "Student": person.name,
                "Student ID": person.person_identifier,
                "Match": confidence_by_id.get(person.person_identifier),
            }
            for person in people
        ])
        reviewed_roster = st.data_editor(
            roster,
            use_container_width=True,
            hide_index=True,
            disabled=["person_id", "Student", "Student ID", "Match"],
            column_config={
                "person_id": None,
                "Present": st.column_config.CheckboxColumn("Present"),
                "Match": st.column_config.NumberColumn("Face match", format="%.1%%"),
            },
            key=f"individual_review_{active_id}_{capture_number}",
        )
        if st.button("Finalize reviewed attendance", type="primary", use_container_width=True):
            selected_ids = {
                int(row.person_id)
                for row in reviewed_roster.itertuples(index=False)
                if bool(row.Present)
            }
            database.finalize_attendance_review(int(active_id), selected_ids, current_actor())
            st.session_state.last_individual_attendance = int(active_id)
            st.session_state.pop("individual_attendance_session_id", None)
            st.session_state.pop("individual_attendance_class", None)
            st.session_state.pop("individual_capture_number", None)
            st.rerun()

    if active_id is None and st.session_state.get("last_individual_attendance"):
        show_last_session(int(st.session_state.last_individual_attendance))


with history_tab:
    st.subheader("Recorded sessions")
    st.caption("Showing the 100 most recent sessions. Select one session to inspect or export its roster.")
    sessions = database.attendance_sessions_frame()
    if sessions.empty:
        st.info("No attendance sessions have been recorded yet.")
    else:
        st.dataframe(
            sessions[["id", "class_name", "capture_mode", "state", "started_at", "total_registered", "total_present", "unknown_faces"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": "Session",
                "class_name": "Class / section",
                "capture_mode": "Mode",
                "state": "State",
                "started_at": "Recorded at",
                "total_registered": "Registered",
                "total_present": "Present",
                "unknown_faces": "Unknown",
            },
        )
        session_options = {
            f"#{int(row.id)} · {row.class_name} · {row.started_at}": int(row.id)
            for row in sessions.itertuples(index=False)
        }
        selected_history = st.selectbox("Session roster", list(session_options))
        selected_session_id = session_options[selected_history]
        records = database.attendance_records_frame(selected_session_id)
        if not records.empty:
            st.dataframe(
                records[["person_name", "person_identifier", "status", "confidence", "marked_at"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "person_name": "Student",
                    "person_identifier": "Student ID",
                    "status": "Status",
                    "confidence": st.column_config.NumberColumn("Face match", format="%.1%%"),
                    "marked_at": "Marked at",
                },
            )
        st.download_button(
            "Download selected session CSV",
            records.to_csv(index=False).encode("utf-8"),
            f"visionguard_attendance_session_{selected_session_id}.csv",
            "text/csv",
        )
