from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
EVENT_SCHEMA = pa.schema(
    [
        ("event_id", pa.int64()),
        ("source", pa.string()),
        ("destination", pa.string()),
        ("timestamp", pa.int64()),
        ("label", pa.int64()),
        ("sequence_id", pa.string()),
        ("split", pa.string()),
    ]
)


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


def _write_dataset(dataset: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    from data.spatial import GridSpec, summarize_cell_events, trajectory_points_to_cell_events

    spec = GridSpec(
        cell_size_degrees=float(config["grid"]["cell_size_degrees"]),
        latitude_origin=float(config["grid"]["latitude_origin"]),
        longitude_origin=float(config["grid"]["longitude_origin"]),
    )
    source_dir = ROOT / dataset["processed_directory"]
    output_dir = ROOT / dataset["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "events.parquet"
    max_points = int(config["max_points_per_split"])
    batch_size = int(config["batch_size"])
    event_id = 0
    summaries = []
    with pq.ParquetWriter(output_path, EVENT_SCHEMA, compression="zstd") as writer:
        for split in ("train", "validation", "test"):
            points = _stream_split_points(source_dir / f"{split}.parquet", max_points=max_points, batch_size=batch_size)
            events = trajectory_points_to_cell_events(points, spec)
            events.insert(0, "event_id", range(event_id, event_id + len(events)))
            event_id += len(events)
            writer.write_table(pa.Table.from_pandas(events, schema=EVENT_SCHEMA, preserve_index=False))
            summaries.append(events)
    combined = pd.concat(summaries, ignore_index=True)
    return {
        "id": dataset["id"],
        "source_directory": dataset["processed_directory"],
        "output_path": str(output_path.relative_to(ROOT)),
        "max_points_per_split": max_points,
        "grid": config["grid"],
        "statistics": summarize_cell_events(combined),
    }


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    config = _load_yaml(ROOT / "experiments/real_world/phase4_spatial_discretization.yaml")
    summary = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "datasets": [_write_dataset(dataset, config) for dataset in config["datasets"]],
    }
    output = ROOT / "results/tables/phase4_spatial_discretization.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
