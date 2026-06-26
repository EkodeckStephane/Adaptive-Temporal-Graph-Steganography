from __future__ import annotations

import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .geolife_stream import GEOLIFE_SCHEMA


def _timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def _cutoffs(counts: Counter[int], train_fraction: float, validation_fraction: float) -> tuple[int, int]:
    total = sum(counts.values())
    train_target = train_fraction * total
    validation_target = (train_fraction + validation_fraction) * total
    daily_counts: Counter[int] = Counter()
    for timestamp, count in counts.items():
        daily_counts[timestamp - timestamp % 86_400] += count
    cumulative = 0
    train_end = validation_end = 0
    for day_start in sorted(daily_counts):
        cumulative += daily_counts[day_start]
        if not train_end and cumulative >= train_target:
            train_end = day_start + 86_399
        if cumulative >= validation_target:
            validation_end = day_start + 86_399
            break
    return train_end, validation_end


def _sessions(
    points: list[tuple[int, float, float]],
    gap_seconds: int,
) -> list[list[tuple[int, float, float]]]:
    if not points:
        return []
    points.sort(key=lambda row: row[0])
    result = [[points[0]]]
    for point in points[1:]:
        previous = result[-1][-1]
        crosses_day = datetime.fromtimestamp(point[0], timezone.utc).date() != datetime.fromtimestamp(
            previous[0], timezone.utc
        ).date()
        if crosses_day or point[0] - previous[0] > gap_seconds:
            result.append([point])
        else:
            result[-1].append(point)
    return result


def stream_tdrive_archives(
    archive_paths: list[Path],
    output_directory: Path,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    gap_seconds: int = 1_200,
    batch_points: int = 200_000,
    coordinate_bounds: tuple[float, float, float, float] = (18.0, 54.0, 73.0, 135.0),
) -> dict[str, Any]:
    timestamp_counts: Counter[int] = Counter()
    sources: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    source_files = 0
    raw_points = 0
    for archive_path in archive_paths:
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".txt"):
                    continue
                source_files += 1
                sources[Path(name).stem].append((archive_path, name))
                with archive.open(name) as handle:
                    for raw_line in handle:
                        columns = raw_line.decode("ascii").strip().split(",")
                        if len(columns) >= 4:
                            timestamp_counts[_timestamp(columns[1])] += 1
                            raw_points += 1
    train_end, validation_end = _cutoffs(
        timestamp_counts, train_fraction, validation_fraction
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary = {
        split: output_directory / f".{split}.parquet.tmp"
        for split in ("train", "validation", "test")
    }
    writers = {
        split: pq.ParquetWriter(
            path,
            GEOLIFE_SCHEMA,
            compression="zstd",
            use_dictionary=["sequence_id", "entity_id", "split"],
        )
        for split, path in temporary.items()
    }
    buffers = {
        split: {field.name: [] for field in GEOLIFE_SCHEMA}
        for split in ("train", "validation", "test")
    }

    def flush(split: str) -> None:
        if buffers[split]["point_id"]:
            writers[split].write_table(
                pa.Table.from_pydict(buffers[split], schema=GEOLIFE_SCHEMA)
            )
            buffers[split] = {field.name: [] for field in GEOLIFE_SCHEMA}

    point_id = 0
    invalid_points = 0
    spatial_outlier_points = 0
    split_points: Counter[str] = Counter()
    split_sessions: Counter[str] = Counter()
    crossing_sessions = 0
    crossing_points = 0
    users: set[str] = set()
    point_counts: list[int] = []
    durations: list[int] = []
    median_intervals: list[float] = []
    bounds = [90.0, -90.0, 180.0, -180.0]
    try:
        for entity_id, locations in sorted(sources.items()):
            points = []
            for archive_path, name in locations:
                with zipfile.ZipFile(archive_path) as archive:
                    with archive.open(name) as handle:
                        for raw_line in handle:
                            columns = raw_line.decode("ascii").strip().split(",")
                            if len(columns) < 4:
                                invalid_points += 1
                                continue
                            try:
                                timestamp = _timestamp(columns[1])
                                longitude = float(columns[2])
                                latitude = float(columns[3])
                            except ValueError:
                                invalid_points += 1
                                continue
                            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                                invalid_points += 1
                                continue
                            lat_min, lat_max, lon_min, lon_max = coordinate_bounds
                            if not (
                                lat_min <= latitude <= lat_max
                                and lon_min <= longitude <= lon_max
                            ):
                                spatial_outlier_points += 1
                                continue
                            points.append((timestamp, latitude, longitude))
            for session_index, session in enumerate(_sessions(points, gap_seconds)):
                start, end = session[0][0], session[-1][0]
                if end <= train_end:
                    split = "train"
                elif start > train_end and end <= validation_end:
                    split = "validation"
                elif start > validation_end:
                    split = "test"
                else:
                    crossing_sessions += 1
                    crossing_points += len(session)
                    continue
                sequence_id = f"{entity_id}:{start}:{session_index}"
                users.add(entity_id)
                split_sessions[split] += 1
                split_points[split] += len(session)
                point_counts.append(len(session))
                if len(session) >= 2:
                    timestamps = np.asarray([point[0] for point in session])
                    durations.append(int(timestamps[-1] - timestamps[0]))
                    median_intervals.append(float(np.median(np.diff(timestamps))))
                for timestamp, latitude, longitude in session:
                    bounds[0] = min(bounds[0], latitude)
                    bounds[1] = max(bounds[1], latitude)
                    bounds[2] = min(bounds[2], longitude)
                    bounds[3] = max(bounds[3], longitude)
                    row = buffers[split]
                    row["point_id"].append(point_id)
                    row["sequence_id"].append(sequence_id)
                    row["entity_id"].append(entity_id)
                    row["timestamp"].append(timestamp)
                    row["latitude"].append(latitude)
                    row["longitude"].append(longitude)
                    row["split"].append(split)
                    point_id += 1
                    if len(row["point_id"]) >= batch_points:
                        flush(split)
        for split in writers:
            flush(split)
    finally:
        for writer in writers.values():
            writer.close()

    for split, temp_path in temporary.items():
        temp_path.replace(output_directory / f"{split}.parquet")
    sessions = sum(split_sessions.values())
    points = sum(split_points.values())
    return {
        "source_files": source_files,
        "unique_taxi_files": len(sources),
        "duplicate_taxi_files_across_archives": source_files - len(sources),
        "article_reported_taxis": 10_357,
        "article_reported_points": 15_000_000,
        "raw_points": raw_points,
        "users": len(users),
        "sessions": sessions,
        "points": points,
        "invalid_points": invalid_points,
        "spatial_outlier_points": spatial_outlier_points,
        "coordinate_bounds": {
            "latitude_min": coordinate_bounds[0],
            "latitude_max": coordinate_bounds[1],
            "longitude_min": coordinate_bounds[2],
            "longitude_max": coordinate_bounds[3],
        },
        "session_gap_seconds": gap_seconds,
        "boundary_crossing_sessions": crossing_sessions,
        "boundary_crossing_points": crossing_points,
        "median_points_per_session": float(median(point_counts)),
        "median_duration_seconds": float(median(durations)),
        "median_sampling_interval_seconds": float(median(median_intervals)),
        "latitude_min": bounds[0],
        "latitude_max": bounds[1],
        "longitude_min": bounds[2],
        "longitude_max": bounds[3],
        "split_sessions": dict(split_sessions),
        "split_points": dict(split_points),
        "train_end": train_end,
        "validation_end": validation_end,
        "actual_train_point_fraction": split_points["train"] / points,
        "actual_validation_point_fraction": split_points["validation"] / points,
        "actual_test_point_fraction": split_points["test"] / points,
    }
