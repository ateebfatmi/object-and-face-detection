from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from .config import ATTENDANCE_DIR
from .database import Database, utcnow


def apply_attendance_retention(database: Database, retention_days: int) -> int:
    """Apply an explicitly configured retention period and remove local evidence safely."""
    cutoff = utcnow() - timedelta(days=retention_days)
    removed, evidence_paths = database.purge_expired_attendance(cutoff)
    attendance_root = ATTENDANCE_DIR.resolve()
    for raw_path in evidence_paths:
        path = Path(raw_path).resolve()
        if path.parent == attendance_root:
            path.unlink(missing_ok=True)
    return removed
