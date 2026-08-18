"""Tracking configuration helpers. YOLO ByteTrack integration lives in object_detector."""

from collections import defaultdict


class UniqueTrackCounter:
    def __init__(self) -> None:
        self.seen: dict[str, set[int]] = defaultdict(set)

    def update(self, label: str, tracking_id: int | None) -> bool:
        if tracking_id is None or tracking_id in self.seen[label]:
            return False
        self.seen[label].add(tracking_id)
        return True

