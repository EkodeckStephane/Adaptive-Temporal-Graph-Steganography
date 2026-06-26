from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

EVENT_COLUMNS = (
    "event_id",
    "source",
    "destination",
    "timestamp",
    "label",
    "sequence_id",
)
TRAJECTORY_COLUMNS = (
    "point_id",
    "sequence_id",
    "entity_id",
    "timestamp",
    "latitude",
    "longitude",
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing canonical columns: {sorted(missing)}")


def validate_events(frame: pd.DataFrame) -> None:
    _require_columns(frame, EVENT_COLUMNS)
    if frame.empty:
        raise ValueError("The event table is empty")
    if frame["event_id"].duplicated().any():
        raise ValueError("event_id must be unique")
    if frame[["source", "destination", "timestamp"]].isna().any().any():
        raise ValueError("source, destination, and timestamp cannot be null")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Events must be sorted chronologically")


def validate_trajectory_points(frame: pd.DataFrame) -> None:
    _require_columns(frame, TRAJECTORY_COLUMNS)
    if frame.empty:
        raise ValueError("The trajectory table is empty")
    if frame["point_id"].duplicated().any():
        raise ValueError("point_id must be unique")
    if frame[list(TRAJECTORY_COLUMNS)].isna().any().any():
        raise ValueError("Canonical trajectory columns cannot be null")
    if not frame["latitude"].between(-90, 90).all():
        raise ValueError("Invalid latitude")
    if not frame["longitude"].between(-180, 180).all():
        raise ValueError("Invalid longitude")
    ordered = frame.sort_values(["sequence_id", "timestamp"], kind="stable")
    if not ordered.index.equals(frame.index):
        raise ValueError("Trajectory points must be ordered by sequence and time")
