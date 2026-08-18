from __future__ import annotations

import pandas as pd


def prepare_activity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "detections"])
    work = df.copy()
    work["date"] = pd.to_datetime(work["timestamp"]).dt.date
    return work.groupby("date").size().rename("detections").reset_index()


def object_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["object_name", "count"])
    return df[df.object_type == "object"].object_name.value_counts().rename_axis("object_name").reset_index(name="count")

