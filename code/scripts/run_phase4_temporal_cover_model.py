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


def _load_phase3_catalog() -> list[dict[str, Any]]:
    with (ROOT / "results/tables/phase3_dataset_statistics.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["datasets"]


def _event_catalog(config: dict[str, Any]) -> list[dict[str, Any]]:
    extra = config.get("event_datasets", [])
    return [*_load_phase3_catalog(), *extra]


def _read_event_dataset(record: dict[str, Any], *, max_rows_per_split: int | None) -> pd.DataFrame:
    path = ROOT / record["processed_path"]
    columns = ["source", "destination", "timestamp", "split"]
    if max_rows_per_split is None:
        table = pq.read_table(path, columns=columns)
        return table.to_pandas().sort_values(["timestamp"], kind="stable").reset_index(drop=True)

    counts = {"train": 0, "validation": 0, "test": 0}
    frames = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        chunk = batch.to_pandas()
        selected = []
        for split in ("train", "validation", "test"):
            remaining = max_rows_per_split - counts[split]
            if remaining <= 0:
                continue
            subset = chunk.loc[chunk["split"] == split].head(remaining)
            if not subset.empty:
                counts[split] += len(subset)
                selected.append(subset)
        if selected:
            frames.append(pd.concat(selected, ignore_index=True))
        if all(value >= max_rows_per_split for value in counts.values()):
            break
    if not frames:
        raise ValueError(f"No events read from {path}")
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["timestamp"], kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from models.temporal import TemporalBackoffModel, metrics_to_dict

    config = _load_yaml(ROOT / "experiments/real_world/phase4_temporal_cover_model.yaml")
    dataset_ids = set(config["datasets"])
    max_rows_per_split = config.get("max_rows_per_split")
    model_config = config["model"]

    rows = []
    summary: dict[str, Any] = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "model": model_config,
        "max_rows_per_split": max_rows_per_split,
        "datasets": {},
    }
    for record in _event_catalog(config):
        dataset_id = record["id"]
        if dataset_id not in dataset_ids:
            continue
        frame = _read_event_dataset(record, max_rows_per_split=max_rows_per_split)
        model = TemporalBackoffModel(
            prior_strength=float(model_config["prior_strength"]),
            top_k=int(model_config["top_k"]),
        ).fit(frame.loc[frame["split"] == "train"])
        metrics = {
            split: model.evaluate(frame.loc[frame["split"] == split])
            for split in ("train", "validation", "test")
            if not frame.loc[frame["split"] == split].empty
        }
        summary["datasets"][dataset_id] = metrics_to_dict(metrics)
        for split, values in summary["datasets"][dataset_id].items():
            rows.append({"dataset": dataset_id, "split": split, **values})

    output_dir = ROOT / "results/tables"
    json_output = output_dir / "phase4_temporal_cover_model.json"
    csv_output = output_dir / "phase4_temporal_cover_model.csv"
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
