from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, create_engine, delete, event, func, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import DATABASE_PATH, ensure_directories


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for VisionGuard database entities."""


class RegisteredPerson(Base):
    __tablename__ = "registered_people"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    person_identifier: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255))
    face_directory: Mapped[str] = mapped_column(String(500))
    embedding_path: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DetectionSession(Base):
    __tablename__ = "detection_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_type: Mapped[str] = mapped_column(String(30), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_objects: Mapped[int] = mapped_column(Integer, default=0)
    total_faces: Mapped[int] = mapped_column(Integer, default=0)
    known_faces: Mapped[int] = mapped_column(Integer, default=0)
    unknown_faces: Mapped[int] = mapped_column(Integer, default=0)
    detections: Mapped[list["Detection"]] = relationship(cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("detection_sessions.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    object_type: Mapped[str] = mapped_column(String(20), index=True)
    object_name: Mapped[str] = mapped_column(String(100), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    tracking_id: Mapped[int | None] = mapped_column(Integer)
    recognized_person: Mapped[str | None] = mapped_column(String(160), index=True)
    recognition_confidence: Mapped[float | None] = mapped_column(Float)
    image_path: Mapped[str | None] = mapped_column(String(500))
    detector_source: Mapped[str | None] = mapped_column(String(40), index=True)


class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    class_name: Mapped[str] = mapped_column(String(160), index=True)
    capture_mode: Mapped[str] = mapped_column(String(30), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_registered: Mapped[int] = mapped_column(Integer, default=0)
    total_present: Mapped[int] = mapped_column(Integer, default=0)
    unknown_faces: Mapped[int] = mapped_column(Integer, default=0)
    evidence_path: Mapped[str | None] = mapped_column(String(500))
    state: Mapped[str] = mapped_column(String(20), default="in_progress", index=True)
    recorded_by: Mapped[str | None] = mapped_column(String(255))
    records: Mapped[list["AttendanceRecord"]] = relationship(cascade="all, delete-orphan")


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    __table_args__ = (UniqueConstraint("session_id", "person_identifier", name="uq_attendance_session_person"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("attendance_sessions.id"), index=True)
    registered_person_id: Mapped[int | None] = mapped_column(ForeignKey("registered_people.id"), index=True)
    person_name: Mapped[str] = mapped_column(String(160), index=True)
    person_identifier: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    marked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    mark_source: Mapped[str] = mapped_column(String(30), default="face_match")
    reviewed_by: Mapped[str | None] = mapped_column(String(255))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Database:
    def __init__(self, path: Path = DATABASE_PATH) -> None:
        ensure_directories()
        self.engine = create_engine(
            f"sqlite:///{path}",
            connect_args={"check_same_thread": False, "timeout": 10},
            pool_pre_ping=True,
        )

        @event.listens_for(self.engine, "connect")
        def _configure_sqlite(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=10000")
            cursor.close()

        Base.metadata.create_all(self.engine)
        self._migrate_schema()
        with self.engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        self.SessionLocal = sessionmaker(self.engine, expire_on_commit=False)

    def _migrate_schema(self) -> None:
        """Apply small backward-compatible SQLite migrations in place."""
        with self.engine.begin() as connection:
            columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(detections)")}
            if "detector_source" not in columns:
                connection.exec_driver_sql("ALTER TABLE detections ADD COLUMN detector_source VARCHAR(40)")
            if "object_type" in columns:
                connection.exec_driver_sql(
                    """UPDATE detections
                       SET detector_source = CASE
                           WHEN object_type = 'face' THEN 'legacy_face'
                           ELSE 'legacy_object'
                       END
                       WHERE detector_source IS NULL"""
                )
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_detections_detector_source ON detections (detector_source)"
            )
            attendance_sessions = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(attendance_sessions)")}
            for name, definition in {
                "state": "VARCHAR(20) DEFAULT 'in_progress'",
                "recorded_by": "VARCHAR(255)",
            }.items():
                if name not in attendance_sessions:
                    connection.exec_driver_sql(f"ALTER TABLE attendance_sessions ADD COLUMN {name} {definition}")
            connection.exec_driver_sql(
                "UPDATE attendance_sessions SET state = CASE WHEN ended_at IS NULL THEN 'in_progress' ELSE 'finalized' END WHERE state IS NULL"
            )
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_attendance_sessions_state ON attendance_sessions (state)"
            )
            attendance_records = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(attendance_records)")}
            for name, definition in {
                "mark_source": "VARCHAR(30) DEFAULT 'face_match'",
                "reviewed_by": "VARCHAR(255)",
                "updated_at": "DATETIME",
            }.items():
                if name not in attendance_records:
                    connection.exec_driver_sql(f"ALTER TABLE attendance_records ADD COLUMN {name} {definition}")
            connection.exec_driver_sql(
                "UPDATE attendance_records SET mark_source = CASE WHEN status = 'Present' THEN 'face_match' ELSE 'unseen' END WHERE mark_source IS NULL"
            )
            connection.exec_driver_sql(
                "UPDATE attendance_records SET updated_at = COALESCE(marked_at, CURRENT_TIMESTAMP) WHERE updated_at IS NULL"
            )

    @contextmanager
    def session(self) -> Iterator[Session]:
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def add_person(self, name: str, identifier: str, email: str | None, face_dir: str, embedding_path: str) -> RegisteredPerson:
        with self.session() as db:
            person = RegisteredPerson(name=name, person_identifier=identifier, email=email or None,
                                      face_directory=face_dir, embedding_path=embedding_path)
            db.add(person)
            db.flush()
            return person

    def people(self) -> list[RegisteredPerson]:
        with self.session() as db:
            return list(db.scalars(select(RegisteredPerson).order_by(RegisteredPerson.name)))

    def person_by_identifier(self, identifier: str) -> RegisteredPerson | None:
        with self.session() as db:
            return db.scalar(select(RegisteredPerson).where(RegisteredPerson.person_identifier == identifier))

    def delete_person(self, person_id: int) -> RegisteredPerson | None:
        with self.session() as db:
            person = db.get(RegisteredPerson, person_id)
            if person:
                db.execute(
                    update(AttendanceRecord)
                    .where(AttendanceRecord.registered_person_id == person.id)
                    .values(registered_person_id=None)
                )
                db.delete(person)
            return person

    def start_session(self, session_type: str) -> int:
        with self.session() as db:
            item = DetectionSession(session_type=session_type)
            db.add(item); db.flush()
            return item.id

    def log_detections(self, session_id: int, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        with self.session() as db:
            db.add_all(Detection(session_id=session_id, **item) for item in items)

    def finish_session(self, session_id: int, stats: dict[str, int]) -> None:
        with self.session() as db:
            item = db.get(DetectionSession, session_id)
            if item:
                item.ended_at = utcnow()
                for key in ("total_objects", "total_faces", "known_faces", "unknown_faces"):
                    setattr(item, key, int(stats.get(key, 0)))

    def update_session_stats(self, session_id: int, stats: dict[str, int]) -> None:
        """Update a running session without marking it as ended."""
        with self.session() as db:
            item = db.get(DetectionSession, session_id)
            if item:
                for key in ("total_objects", "total_faces", "known_faces", "unknown_faces"):
                    setattr(item, key, int(stats.get(key, 0)))

    def detections_frame(self) -> pd.DataFrame:
        query = """SELECT d.id, d.timestamp, d.object_type, d.object_name, d.confidence,
                   d.tracking_id, d.recognized_person, d.recognition_confidence, d.image_path,
                   d.detector_source,
                   s.session_type, d.session_id FROM detections d
                   JOIN detection_sessions s ON s.id=d.session_id ORDER BY d.timestamp DESC"""
        return pd.read_sql(query, self.engine)

    def sessions_frame(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM detection_sessions ORDER BY started_at DESC", self.engine)

    def dashboard_stats(self) -> dict[str, int]:
        with self.session() as db:
            tracked_objects = db.scalar(
                select(func.count(func.distinct(func.printf("%d:%d", Detection.session_id, Detection.tracking_id))))
                .where(Detection.object_type == "object", Detection.tracking_id.is_not(None))
            ) or 0
            untracked_objects = db.scalar(
                select(func.count(Detection.id))
                .where(Detection.object_type == "object", Detection.tracking_id.is_(None))
            ) or 0
            return {
                "objects": tracked_objects + untracked_objects,
                "object_events": db.scalar(select(func.count(Detection.id)).where(Detection.object_type == "object")) or 0,
                "faces": db.scalar(select(func.count(Detection.id)).where(Detection.object_type == "face")) or 0,
                "known": db.scalar(select(func.count(func.distinct(Detection.recognized_person))).where(
                    Detection.object_type == "face",
                    Detection.recognized_person.is_not(None),
                    Detection.recognized_person != "Unknown",
                )) or 0,
                "known_events": db.scalar(select(func.count(Detection.id)).where(
                    Detection.object_type == "face",
                    Detection.recognized_person.is_not(None),
                    Detection.recognized_person != "Unknown",
                )) or 0,
                "unknown": db.scalar(select(func.count(Detection.id)).where(Detection.recognized_person == "Unknown")) or 0,
                "sessions": db.scalar(select(func.count(DetectionSession.id))) or 0,
                "people": db.scalar(select(func.count(RegisteredPerson.id))) or 0,
            }

    def clear_history(self) -> None:
        with self.session() as db:
            db.execute(delete(Detection)); db.execute(delete(DetectionSession))

    def start_attendance(
        self,
        class_name: str,
        capture_mode: str,
        evidence_path: str | None = None,
        recorded_by: str | None = None,
    ) -> int:
        with self.session() as db:
            item = AttendanceSession(
                class_name=class_name,
                capture_mode=capture_mode,
                total_registered=db.scalar(select(func.count(RegisteredPerson.id))) or 0,
                evidence_path=evidence_path,
                state="in_progress",
                recorded_by=recorded_by,
            )
            db.add(item)
            db.flush()
            return item.id

    def mark_attendance_person(
        self,
        session_id: int,
        person_id: int,
        confidence: float,
        reviewed_by: str | None = None,
    ) -> bool:
        """Mark one person present, returning False when already marked in this session."""
        with self.session() as db:
            person = db.get(RegisteredPerson, person_id)
            session = db.get(AttendanceSession, session_id)
            if person is None or session is None or session.state != "in_progress":
                return False
            existing = db.scalar(
                select(AttendanceRecord).where(
                    AttendanceRecord.session_id == session_id,
                    AttendanceRecord.person_identifier == person.person_identifier,
                )
            )
            if existing is not None:
                return False
            db.add(AttendanceRecord(
                session_id=session_id,
                registered_person_id=person.id,
                person_name=person.name,
                person_identifier=person.person_identifier,
                status="Present",
                confidence=float(confidence),
                marked_at=utcnow(),
                mark_source="face_match",
                reviewed_by=reviewed_by,
                updated_at=utcnow(),
            ))
            session.total_present += 1
            return True

    def finish_attendance(self, session_id: int, unknown_faces: int = 0) -> AttendanceSession | None:
        """Finalize a session and add absent rows for every unmarked registered person."""
        with self.session() as db:
            session = db.get(AttendanceSession, session_id)
            if session is None or session.state != "in_progress":
                return session
            marked = set(db.scalars(
                select(AttendanceRecord.person_identifier).where(AttendanceRecord.session_id == session_id)
            ))
            people = list(db.scalars(select(RegisteredPerson).order_by(RegisteredPerson.name)))
            for person in people:
                if person.person_identifier in marked:
                    continue
                db.add(AttendanceRecord(
                    session_id=session_id,
                    registered_person_id=person.id,
                    person_name=person.name,
                    person_identifier=person.person_identifier,
                    status="Absent",
                    confidence=None,
                    marked_at=None,
                    mark_source="unseen",
                    updated_at=utcnow(),
                ))
            session.total_registered = len(people)
            session.total_present = len(marked)
            session.unknown_faces = max(0, int(unknown_faces))
            session.ended_at = utcnow()
            session.state = "finalized"
            return session

    def record_class_attendance(
        self,
        class_name: str,
        present: dict[int, float | None],
        unknown_faces: int,
        evidence_path: str,
        recorded_by: str | None = None,
    ) -> int:
        """Atomically create a reviewed full-class session and its complete roster."""
        with self.session() as db:
            people = list(db.scalars(select(RegisteredPerson).order_by(RegisteredPerson.name)))
            valid_ids = {person.id for person in people}
            selected = {person_id: score for person_id, score in present.items() if person_id in valid_ids}
            session = AttendanceSession(
                class_name=class_name,
                capture_mode="Full class photo",
                started_at=utcnow(),
                ended_at=utcnow(),
                total_registered=len(people),
                total_present=len(selected),
                unknown_faces=max(0, int(unknown_faces)),
                evidence_path=evidence_path,
                state="finalized",
                recorded_by=recorded_by,
            )
            db.add(session)
            db.flush()
            for person in people:
                is_present = person.id in selected
                score = selected.get(person.id)
                db.add(AttendanceRecord(
                    session_id=session.id,
                    registered_person_id=person.id,
                    person_name=person.name,
                    person_identifier=person.person_identifier,
                    status="Present" if is_present else "Absent",
                    confidence=float(score) if score is not None else None,
                    marked_at=utcnow() if is_present else None,
                    mark_source=("face_match" if score is not None else "manual_review") if is_present else "unseen",
                    reviewed_by=recorded_by,
                    updated_at=utcnow(),
                ))
            return session.id

    def finalize_attendance_review(
        self,
        session_id: int,
        present_person_ids: set[int],
        reviewed_by: str | None = None,
        unknown_faces: int = 0,
    ) -> AttendanceSession | None:
        """Finalize an in-progress session from a reviewed roster in one transaction."""
        with self.session() as db:
            session = db.get(AttendanceSession, session_id)
            if session is None or session.state != "in_progress":
                return session
            people = list(db.scalars(select(RegisteredPerson).order_by(RegisteredPerson.name)))
            existing = {
                row.person_identifier: row
                for row in db.scalars(select(AttendanceRecord).where(AttendanceRecord.session_id == session_id))
            }
            valid_ids = {person.id for person in people}
            selected = present_person_ids & valid_ids
            for person in people:
                desired = "Present" if person.id in selected else "Absent"
                row = existing.get(person.person_identifier)
                if row is None:
                    db.add(AttendanceRecord(
                        session_id=session_id,
                        registered_person_id=person.id,
                        person_name=person.name,
                        person_identifier=person.person_identifier,
                        status=desired,
                        confidence=None,
                        marked_at=utcnow() if desired == "Present" else None,
                        mark_source="manual_review" if desired == "Present" else "unseen",
                        reviewed_by=reviewed_by,
                        updated_at=utcnow(),
                    ))
                    continue
                if row.status != desired:
                    row.status = desired
                    row.confidence = row.confidence if desired == "Present" else None
                    row.marked_at = utcnow() if desired == "Present" else None
                    row.mark_source = "manual_review"
                row.reviewed_by = reviewed_by
                row.updated_at = utcnow()
            session.total_registered = len(people)
            session.total_present = len(selected)
            session.unknown_faces = max(0, int(unknown_faces))
            session.ended_at = utcnow()
            session.state = "finalized"
            session.recorded_by = session.recorded_by or reviewed_by
            return session

    def open_attendance_sessions(self) -> list[AttendanceSession]:
        with self.session() as db:
            return list(db.scalars(
                select(AttendanceSession)
                .where(AttendanceSession.state == "in_progress")
                .order_by(AttendanceSession.started_at.desc())
            ))

    def cancel_attendance(self, session_id: int, actor: str | None = None) -> bool:
        with self.session() as db:
            session = db.get(AttendanceSession, session_id)
            if session is None or session.state != "in_progress":
                return False
            session.state = "cancelled"
            session.ended_at = utcnow()
            session.recorded_by = session.recorded_by or actor
            return True

    def purge_expired_attendance(self, cutoff: datetime) -> tuple[int, list[str]]:
        """Delete finalized attendance older than an operator-configured retention cutoff."""
        with self.session() as db:
            sessions = list(db.scalars(
                select(AttendanceSession).where(
                    AttendanceSession.state.in_(("finalized", "cancelled")),
                    AttendanceSession.ended_at.is_not(None),
                    AttendanceSession.ended_at < cutoff,
                )
            ))
            evidence = [item.evidence_path for item in sessions if item.evidence_path]
            for item in sessions:
                db.delete(item)
            return len(sessions), evidence

    def attendance_session(self, session_id: int) -> AttendanceSession | None:
        with self.session() as db:
            return db.get(AttendanceSession, session_id)

    def attendance_records_frame(self, session_id: int | None = None) -> pd.DataFrame:
        where = "WHERE s.id = :session_id" if session_id is not None else ""
        query = f"""SELECT s.id AS session_id, s.class_name, s.capture_mode, s.started_at, s.ended_at,
                    r.person_name, r.person_identifier, r.status, r.confidence, r.marked_at
                    FROM attendance_records r
                    JOIN attendance_sessions s ON s.id = r.session_id
                    {where}
                    ORDER BY s.started_at DESC, r.person_name"""
        params = {"session_id": session_id} if session_id is not None else None
        return pd.read_sql(query, self.engine, params=params)

    def attendance_sessions_frame(self, limit: int = 100) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT * FROM attendance_sessions ORDER BY started_at DESC LIMIT :limit",
            self.engine,
            params={"limit": max(1, min(int(limit), 500))},
        )
