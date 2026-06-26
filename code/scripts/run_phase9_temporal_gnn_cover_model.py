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
        return json.load(handle)["datasets"]


def _event_catalog(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in [*_load_phase3_catalog(), *config.get("event_datasets", [])]}


def _read_event_dataset(record: dict[str, Any], *, max_rows_per_split: int) -> pd.DataFrame:
    columns = ["source", "destination", "timestamp", "split"]
    counts = {"train": 0, "validation": 0, "test": 0}
    frames = []
    parquet = pq.ParquetFile(ROOT / record["processed_path"])
    for batch in parquet.iter_batches(batch_size=10_000, columns=columns):
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
        raise ValueError(f"No rows read for {record['id']}")
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp"], kind="stable").reset_index(drop=True)


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from models.temporal_gnn import metrics_to_dict, train_and_evaluate_temporal_gnn

    config = _load_yaml(ROOT / "experiments/real_world/phase9_temporal_gnn_cover_model.yaml")
    catalog = _event_catalog(config)
    rows = []
    summary: dict[str, Any] = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "training": config["training"],
        "datasets": {},
    }
    for dataset_id in config["datasets"]:
        frame = _read_event_dataset(catalog[dataset_id], max_rows_per_split=int(config["max_rows_per_split"]))
        metrics = train_and_evaluate_temporal_gnn(frame, **config["training"])
        summary["datasets"][dataset_id] = metrics_to_dict(metrics)
        for split, values in summary["datasets"][dataset_id].items():
            rows.append({"dataset": dataset_id, "model": "temporal_graph_encoder", "split": split, **values})

    output_dir = ROOT / "results/tables"
    json_output = output_dir / "phase9_temporal_gnn_cover_model.json"
    csv_output = output_dir / "phase9_temporal_gnn_cover_model.csv"
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
