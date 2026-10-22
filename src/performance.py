from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    runs: int
    mean_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    fps: float


def summarize_benchmark(durations_ms: list[float]) -> BenchmarkSummary:
    if not durations_ms:
        raise ValueError("At least one benchmark duration is required.")
    values = np.asarray(durations_ms, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Benchmark durations must be positive finite numbers.")
    mean_ms = float(values.mean())
    return BenchmarkSummary(
        runs=len(values),
        mean_ms=mean_ms,
        p95_ms=float(np.percentile(values, 95)),
        min_ms=float(values.min()),
        max_ms=float(values.max()),
        fps=1000 / mean_ms,
    )


def classification_report(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    required = {"actual", "predicted"}
    if not required.issubset(frame.columns):
        raise ValueError("CSV must contain actual and predicted columns.")
    clean = frame[["actual", "predicted"]].dropna().astype(str)
    if clean.empty:
        raise ValueError("CSV does not contain any complete label pairs.")

    labels = sorted(set(clean.actual) | set(clean.predicted))
    matrix = pd.crosstab(clean.actual, clean.predicted).reindex(index=labels, columns=labels, fill_value=0)
    rows: list[dict] = []
    for label in labels:
        true_positive = int(matrix.loc[label, label])
        false_positive = int(matrix[label].sum()) - true_positive
        false_negative = int(matrix.loc[label].sum()) - true_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({"class": label, "precision": precision, "recall": recall, "f1": f1,
                     "support": int(matrix.loc[label].sum())})
    report = pd.DataFrame(rows)
    summary = {
        "accuracy": float(np.trace(matrix.to_numpy()) / matrix.to_numpy().sum()),
        "macro_precision": float(report.precision.mean()),
        "macro_recall": float(report.recall.mean()),
        "macro_f1": float(report.f1.mean()),
    }
    return matrix, report, summary
