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
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("phase3_dataset_statistics.json must contain a datasets list")
    return datasets


def _read_event_dataset(record: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / record["processed_path"]
    table = pq.read_table(path, columns=["source", "destination", "timestamp", "split"])
    frame = table.to_pandas()
    return frame.sort_values(["timestamp"], kind="stable").reset_index(drop=True)


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from models.coverage import evaluate_splits, metrics_to_dict

    config = _load_yaml(ROOT / "experiments/real_world/phase4_cover_model.yaml")
    dataset_ids = set(config["datasets"])
    prior_strength = float(config["model"]["prior_strength"])
    calibration_bins = int(config["evaluation"]["calibration_bins"])

    summary: dict[str, Any] = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "model": config["model"],
        "evaluation": config["evaluation"],
        "datasets": {},
    }
    rows = []
    for record in _load_phase3_catalog():
        dataset_id = record["id"]
        if dataset_id not in dataset_ids:
            continue
        if "processed_path" not in record:
            raise ValueError(f"{dataset_id} is not an event dataset with one processed_path")

        metrics = evaluate_splits(
            _read_event_dataset(record),
            prior_strength=prior_strength,
            calibration_bins=calibration_bins,
        )
        summary["datasets"][dataset_id] = metrics_to_dict(metrics)
        for split, values in summary["datasets"][dataset_id].items():
            rows.append({"dataset": dataset_id, "split": split, **values})

    if not rows:
        raise ValueError("No configured Phase 4 datasets were evaluated")

    output_dir = ROOT / "results/tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output = output_dir / "phase4_cover_model_baseline.json"
    csv_output = output_dir / "phase4_cover_model_baseline.csv"
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
