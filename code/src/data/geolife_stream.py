from __future__ import annotations

import math
import zipfile
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GEOLIFE_SCHEMA = pa.schema(
    [
        ("point_id", pa.int64()),
        ("sequence_id", pa.string()),
        ("entity_id", pa.string()),
        ("timestamp", pa.int64()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("split", pa.string()),
    ]
)


def _trajectory_entries(archive: zipfile.ZipFile) -> list[str]:
    return sorted(name for name in archive.namelist() if name.lower().endswith(".plt"))


def _start_timestamp(name: str) -> int:
    stem = Path(name).stem
    return int(np.datetime64(f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}T{stem[8:10]}:{stem[10:12]}:{stem[12:14]}", "s").astype(int))


def _split_map(
    entries: list[str],
    train_fraction: float,
    validation_fraction: float,
) -> tuple[int, int]:
    ordered = sorted(entries, key=lambda name: (_start_timestamp(name), name))
    train_index = max(0, int(math.ceil(train_fraction * len(ordered))) - 1)
    validation_index = max(
        train_index + 1,
        int(math.ceil((train_fraction + validation_fraction) * len(ordered))) - 1,
    )
    train_end = _start_timestamp(ordered[train_index])
    validation_end = _start_timestamp(ordered[min(validation_index, len(ordered) - 1)])
    return train_end, validation_end


def stream_geolife_archive(
    archive_path: Path,
    output_directory: Path,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    batch_points: int = 200_000,
) -> dict[str, Any]:
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

    point_id = 0
    invalid_points = 0
    nonmonotonic_steps = 0
    boundary_crossing_trajectories = 0
    boundary_crossing_points = 0
    empty_trajectories = 0
    point_counts: list[int] = []
    durations: list[int] = []
    median_intervals: list[float] = []
    split_points: Counter[str] = Counter()
    split_trajectories: Counter[str] = Counter()
    users: set[str] = set()
    bounds = [90.0, -90.0, 180.0, -180.0]

    def flush(split: str) -> None:
        if not buffers[split]["point_id"]:
            return
        writers[split].write_table(pa.Table.from_pydict(buffers[split], schema=GEOLIFE_SCHEMA))
        buffers[split] = {field.name: [] for field in GEOLIFE_SCHEMA}

    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = _trajectory_entries(archive)
            train_end, validation_end = _split_map(
                entries, train_fraction, validation_fraction
            )
            for name in entries:
                parts = Path(name).parts
                entity_id = parts[-3]
                sequence_id = f"{entity_id}:{Path(name).stem}"
                timestamps: list[int] = []
                latitudes: list[float] = []
                longitudes: list[float] = []
                previous_timestamp: int | None = None
                with archive.open(name) as handle:
                    for _ in range(6):
                        next(handle, None)
                    for raw_line in handle:
                        columns = raw_line.decode("ascii").strip().split(",")
                        if len(columns) < 7:
                            invalid_points += 1
                            continue
                        try:
                            latitude = float(columns[0])
                            longitude = float(columns[1])
                            timestamp = int(round((float(columns[4]) - 25569.0) * 86400.0))
                        except ValueError:
                            invalid_points += 1
                            continue
                        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                            invalid_points += 1
                            continue
                        if previous_timestamp is not None and timestamp < previous_timestamp:
                            nonmonotonic_steps += 1
                        previous_timestamp = timestamp
                        timestamps.append(timestamp)
                        latitudes.append(latitude)
                        longitudes.append(longitude)
                if not timestamps:
                    empty_trajectories += 1
                    continue
                start, end = timestamps[0], timestamps[-1]
                if end <= train_end:
                    split = "train"
                elif start > train_end and end <= validation_end:
                    split = "validation"
                elif start > validation_end:
                    split = "test"
                else:
                    boundary_crossing_trajectories += 1
                    boundary_crossing_points += len(timestamps)
                    continue

                users.add(entity_id)
                split_trajectories[split] += 1
                split_points[split] += len(timestamps)
                point_counts.append(len(timestamps))
                if len(timestamps) >= 2:
                    durations.append(timestamps[-1] - timestamps[0])
                    median_intervals.append(float(np.median(np.diff(timestamps))))
                for timestamp, latitude, longitude in zip(
                    timestamps, latitudes, longitudes
                ):
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

    output_files = {}
    for split, temp_path in temporary.items():
        final_path = output_directory / f"{split}.parquet"
        temp_path.replace(final_path)
        output_files[split] = str(final_path)

    total_trajectories = sum(split_trajectories.values())
    total_points = sum(split_points.values())
    return {
        "users": len(users),
        "trajectories": total_trajectories,
        "archive_trajectory_files": len(entries),
        "bundled_guide_reported_trajectories": 17_621,
        "archive_file_count_difference_from_guide": len(entries) - 17_621,
        "points": total_points,
        "invalid_points": invalid_points,
        "nonmonotonic_steps": nonmonotonic_steps,
        "empty_trajectories": empty_trajectories,
        "boundary_crossing_trajectories": boundary_crossing_trajectories,
        "boundary_crossing_points": boundary_crossing_points,
        "median_points_per_trajectory": float(median(point_counts)),
        "median_duration_seconds": float(median(durations)),
        "median_sampling_interval_seconds": float(median(median_intervals)),
        "latitude_min": bounds[0],
        "latitude_max": bounds[1],
        "longitude_min": bounds[2],
        "longitude_max": bounds[3],
        "split_trajectories": dict(split_trajectories),
        "split_points": dict(split_points),
        "actual_train_trajectory_fraction": split_trajectories["train"] / total_trajectories,
        "actual_validation_trajectory_fraction": split_trajectories["validation"]
        / total_trajectories,
        "actual_test_trajectory_fraction": split_trajectories["test"] / total_trajectories,
        "train_end": train_end,
        "validation_end": validation_end,
        "output_files": output_files,
    }
