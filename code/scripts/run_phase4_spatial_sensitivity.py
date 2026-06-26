from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _stream_split_points(path: Path, *, max_points: int, batch_size: int) -> pd.DataFrame:
    frames = []
    count = 0
    columns = ["sequence_id", "timestamp", "latitude", "longitude", "split"]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        remaining = max_points - count
        if remaining <= 0:
            break
        frame = batch.to_pandas().head(remaining)
        count += len(frame)
        frames.append(frame)
        if count >= max_points:
            break
    if not frames:
        raise ValueError(f"No points read from {path}")
    return pd.concat(frames, ignore_index=True)


def _load_dataset_points(dataset: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    source_dir = ROOT / dataset["processed_directory"]
    max_points = int(config["max_points_per_split"])
    batch_size = int(config["batch_size"])
    frames = [
        _stream_split_points(source_dir / f"{split}.parquet", max_points=max_points, batch_size=batch_size)
        for split in ("train", "validation", "test")
    ]
    return pd.concat(frames, ignore_index=True)


def _evaluate_dataset(dataset: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    from data.spatial import GridSpec, summarize_cell_events, trajectory_points_to_cell_events

    points = _load_dataset_points(dataset, config)
    rows = []
    grid = config["grid"]
    for cell_size in grid["cell_size_degrees"]:
        spec = GridSpec(
            cell_size_degrees=float(cell_size),
            latitude_origin=float(grid["latitude_origin"]),
            longitude_origin=float(grid["longitude_origin"]),
        )
        events = trajectory_points_to_cell_events(points, spec)
        stats = summarize_cell_events(events)
        rows.append(
            {
                "dataset": dataset["id"],
                "cell_size_degrees": float(cell_size),
                "max_points_per_split": int(config["max_points_per_split"]),
                "events": stats["events"],
                "cells": stats["cells"],
                "unique_directed_edges": stats["unique_directed_edges"],
                "self_loop_fraction": stats["self_loop_fraction"],
                "train_events": stats["split_counts"].get("train", 0),
                "validation_events": stats["split_counts"].get("validation", 0),
                "test_events": stats["split_counts"].get("test", 0),
            }
        )
    return rows


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))
    config = _load_yaml(ROOT / "experiments/real_world/phase4_spatial_sensitivity.yaml")
    rows = [row for dataset in config["datasets"] for row in _evaluate_dataset(dataset, config)]
    summary = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "rows": rows,
    }
    output_json = ROOT / "results/tables/phase4_spatial_sensitivity.json"
    output_csv = ROOT / "results/tables/phase4_spatial_sensitivity.csv"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
