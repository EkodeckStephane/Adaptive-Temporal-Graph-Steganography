from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import validate_events, validate_trajectory_points


def read_bipartite_interactions(
    path: Path,
    *,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(
        path,
        header=None,
        skiprows=1,
        usecols=[0, 1, 2, 3],
        names=["source", "destination", "timestamp", "label"],
        chunksize=chunksize,
    ):
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True)
    frame["source"] = "user:" + frame["source"].astype(str)
    frame["destination"] = "item:" + frame["destination"].astype(str)
    frame.insert(0, "event_id", range(len(frame)))
    frame["sequence_id"] = pd.NA
    frame = frame.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
    frame["event_id"] = range(len(frame))
    frame = frame[["event_id", "source", "destination", "timestamp", "label", "sequence_id"]]
    validate_events(frame)
    return frame


def read_tgb_wiki(path: Path, *, chunksize: int = 50_000) -> pd.DataFrame:
    return read_bipartite_interactions(path, chunksize=chunksize)


def read_tdrive_directory(path: Path) -> pd.DataFrame:
    frames = []
    for source in sorted(path.glob("*.txt")):
        frame = pd.read_csv(
            source,
            header=None,
            names=["entity_id", "timestamp", "longitude", "latitude"],
            parse_dates=["timestamp"],
        )
        frame["sequence_id"] = source.stem
        frames.append(frame)
    if not frames:
        raise ValueError(f"No T-Drive .txt files found in {path}")
    result = pd.concat(frames, ignore_index=True)
    return _finalize_points(result)


def read_geolife_plt(path: Path, *, sequence_id: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        skiprows=6,
        header=None,
        usecols=[0, 1, 5, 6],
        names=["latitude", "longitude", "date", "time"],
        dtype={"date": str, "time": str},
    )
    frame["timestamp"] = pd.to_datetime(
        frame.pop("date") + " " + frame.pop("time"),
        format="%Y-%m-%d %H:%M:%S",
    )
    frame["sequence_id"] = sequence_id or path.stem
    frame["entity_id"] = path.parents[1].name if len(path.parents) > 1 else "unknown"
    return _finalize_points(frame)


def read_porto_csv(path: Path, *, limit: int | None = None) -> pd.DataFrame:
    rows = []
    source = pd.read_csv(
        path,
        usecols=["TRIP_ID", "TAXI_ID", "TIMESTAMP", "MISSING_DATA", "POLYLINE"],
        nrows=limit,
    )
    for record in source.itertuples(index=False):
        if bool(record.MISSING_DATA):
            continue
        coordinates = json.loads(record.POLYLINE)
        for offset, (longitude, latitude) in enumerate(coordinates):
            rows.append(
                {
                    "sequence_id": str(record.TRIP_ID),
                    "entity_id": str(record.TAXI_ID),
                    "timestamp": pd.to_datetime(
                        int(record.TIMESTAMP) + 15 * offset,
                        unit="s",
                        utc=True,
                    ).tz_localize(None),
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
    if not rows:
        raise ValueError("No valid Porto trajectory was found")
    return _finalize_points(pd.DataFrame(rows))


def _finalize_points(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[
        ["sequence_id", "entity_id", "timestamp", "latitude", "longitude"]
    ].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True).dt.tz_localize(None)
    result = result.sort_values(["sequence_id", "timestamp"], kind="stable").reset_index(drop=True)
    result.insert(0, "point_id", range(len(result)))
    validate_trajectory_points(result)
    return result
