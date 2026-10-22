from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine, delete, func, select
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


class Database:
    def __init__(self, path: Path = DATABASE_PATH) -> None:
        ensure_directories()
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self._migrate_schema()
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

    def delete_person(self, person_id: int) -> RegisteredPerson | None:
        with self.session() as db:
            person = db.get(RegisteredPerson, person_id)
            if person:
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
