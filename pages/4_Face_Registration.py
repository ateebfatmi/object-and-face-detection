from __future__ import annotations

import shutil

import cv2
import streamlit as st
from sqlalchemy.exc import IntegrityError

from src.ui import get_database, get_face_detector, get_face_recognizer, hero, setup_page
from src.video_utils import decode_image

setup_page("Face Registration", "👤")
hero("Register Face", "Create a local face profile from up to three clear reference images")

with st.form("registration"):
    c1, c2 = st.columns(2)
    name = c1.text_input("Full Name")
    identifier = c2.text_input("Student / Employee ID")
    email = st.text_input("Email (optional)")
    uploads = st.file_uploader("Reference images — front, slight left, slight right", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    camera = st.camera_input("Or capture a face")
    submitted = st.form_submit_button("Register face", type="primary")

if submitted:
    if not name.strip() or not identifier.strip():
        st.error("Full name and Student / Employee ID are required.")
    elif len(uploads) + (1 if camera else 0) == 0:
        st.error("Provide at least one reference image.")
    elif get_database().person_by_identifier(identifier.strip()) is not None:
        st.error("That Student / Employee ID is already registered.")
    else:
        person_dir = None
        try:
            images = [decode_image(item.getvalue()) for item in uploads[:3]]
            if camera and len(images) < 3: images.append(decode_image(camera.getvalue()))
            recognizer, detector = get_face_recognizer(), get_face_detector()
            person_dir, accepted = recognizer.register(
                name.strip(),
                images,
                detector,
                storage_key=identifier.strip(),
            )
            get_database().add_person(name.strip(), identifier.strip(), email.strip() or None,
                                      str(person_dir), str(person_dir / "embedding.npy"))
            st.success(f"Registered {name.strip()} using {accepted} valid image(s).")
        except IntegrityError:
            if person_dir is not None:
                shutil.rmtree(person_dir, ignore_errors=True)
                get_face_recognizer().reload()
            st.error("That Student / Employee ID is already registered.")
        except Exception as exc:
            if person_dir is not None:
                shutil.rmtree(person_dir, ignore_errors=True)
                get_face_recognizer().reload()
            st.error(f"Registration failed: {exc}")

@st.dialog("Delete face profile?")
def confirm_profile_deletion(person_id: int, person_name: str, identifier: str) -> None:
    st.write(f"Delete **{person_name}** ({identifier})?")
    st.warning("This permanently removes the saved reference photos and face embedding. Existing attendance rows keep their name and ID snapshot.")
    cancel, remove = st.columns(2)
    if cancel.button("Keep profile", use_container_width=True):
        st.rerun()
    if remove.button("Delete profile", type="primary", use_container_width=True):
        removed = get_database().delete_person(person_id)
        if removed:
            shutil.rmtree(removed.face_directory, ignore_errors=True)
            get_face_recognizer().reload()
        st.rerun()


st.subheader("Registered people")
people = get_database().people()
if not people:
    st.info("No face profiles have been registered.")
else:
    for person in people:
        a, b, c = st.columns([2, 2, 1])
        a.write(f"**{person.name}**")
        b.caption(f"ID: {person.person_identifier} · {person.email or 'No email'}")
        if c.button("Delete", key=f"delete-{person.id}"):
            confirm_profile_deletion(person.id, person.name, person.person_identifier)

