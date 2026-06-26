from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GridSpec:
    cell_size_degrees: float
    latitude_origin: float = -90.0
    longitude_origin: float = -180.0

    def __post_init__(self) -> None:
        if self.cell_size_degrees <= 0:
            raise ValueError("cell_size_degrees must be positive")


def cell_id(latitude: float, longitude: float, spec: GridSpec) -> str:
    if not -90 <= latitude <= 90:
        raise ValueError("latitude out of range")
    if not -180 <= longitude <= 180:
        raise ValueError("longitude out of range")
    lat_index = math.floor((latitude - spec.latitude_origin) / spec.cell_size_degrees)
    lon_index = math.floor((longitude - spec.longitude_origin) / spec.cell_size_degrees)
    return f"cell:{lat_index}:{lon_index}"


def trajectory_points_to_cell_events(points: pd.DataFrame, spec: GridSpec) -> pd.DataFrame:
    required = {"sequence_id", "timestamp", "latitude", "longitude", "split"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"Missing trajectory columns: {sorted(missing)}")
    if points.empty:
        return pd.DataFrame(
            columns=["source", "destination", "timestamp", "label", "sequence_id", "split"]
        )
    ordered = points.sort_values(["sequence_id", "timestamp"], kind="stable").copy()
    ordered["cell"] = [
        cell_id(latitude, longitude, spec)
        for latitude, longitude in ordered[["latitude", "longitude"]].itertuples(index=False)
    ]
    ordered["next_cell"] = ordered.groupby("sequence_id", sort=False)["cell"].shift(-1)
    ordered["next_timestamp"] = ordered.groupby("sequence_id", sort=False)["timestamp"].shift(-1)
    events = ordered.loc[ordered["next_cell"].notna()].copy()
    result = pd.DataFrame(
        {
            "source": events["cell"].astype(str),
            "destination": events["next_cell"].astype(str),
            "timestamp": events["next_timestamp"].astype("int64"),
            "label": 0,
            "sequence_id": events["sequence_id"].astype(str),
            "split": events["split"].astype(str),
        }
    )
    return result.sort_values(["timestamp", "sequence_id"], kind="stable").reset_index(drop=True)


def summarize_cell_events(events: pd.DataFrame) -> dict[str, int | float | dict[Hashable, int]]:
    if events.empty:
        raise ValueError("Cannot summarize an empty event table")
    pairs = list(zip(events["source"], events["destination"]))
    return {
        "events": len(events),
        "cells": len(set(events["source"]) | set(events["destination"])),
        "unique_directed_edges": len(set(pairs)),
        "self_loop_fraction": float((events["source"] == events["destination"]).mean()),
        "split_counts": {str(key): int(value) for key, value in events["split"].value_counts().items()},
    }
