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
        raise ValueError(f"No rows read for {record['id']}")
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp"], kind="stable").reset_index(drop=True)


def _public_eve_records(model: Any, frame: pd.DataFrame, *, split: str, config: Any) -> pd.DataFrame:
    from steganalysis.samples import make_steganalysis_records

    records = make_steganalysis_records(model, frame, split=split, config=config)
    source_timestamp = frame.sort_values(["timestamp"], kind="stable").reset_index(drop=True)
    repeated = source_timestamp.loc[
        source_timestamp.index.repeat(2), ["source", "timestamp"]
    ].reset_index(drop=True)
    records.insert(3, "source", repeated["source"].astype(str))
    records.insert(4, "timestamp", repeated["timestamp"].astype("int64"))
    return records[["split", "pair_id", "label", "source", "action", "timestamp"]]


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from models.temporal import TemporalBackoffModel
    from steganalysis.neural_eve import fit_and_score_neural_eve, metrics_to_dict
    from steganalysis.samples import SampleConfig

    config = _load_yaml(ROOT / "experiments/real_world/phase9_independent_neural_steganalysis.yaml")
    catalog = _event_catalog(config)
    sample_config = SampleConfig(
        max_bits_per_transition=int(config["codec"]["max_bits_per_transition"]),
        seed=int(config["seed"]),
        max_local_total_variation=float(config["codec"]["max_local_total_variation"]),
        max_local_kl_bits=float(config["codec"]["max_local_kl_bits"]),
        min_entropy_bits=float(config["codec"]["min_entropy_bits"]),
        cover_when_unsafe=bool(config["codec"]["cover_when_unsafe"]),
        codec_backend=str(config["codec"].get("codec_backend", "range")),
        min_encoded_probability=float(config["codec"].get("min_encoded_probability", 0.0)),
        max_encoded_surprise_bits=float(config["codec"].get("max_encoded_surprise_bits", "inf")),
        max_encoded_rank_fraction=float(config["codec"].get("max_encoded_rank_fraction", 1.0)),
        require_encoded_top_action=bool(config["codec"].get("require_encoded_top_action", False)),
        require_encoded_self_loop=bool(config["codec"].get("require_encoded_self_loop", False)),
    )
    summary: dict[str, Any] = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "detectors": {},
        "training": config["training"],
    }
    rows = []
    all_records = []
    for dataset_id in config["datasets"]:
        frame = _read_event_dataset(catalog[dataset_id], max_rows_per_split=int(config["max_rows_per_split"]))
        model = TemporalBackoffModel(
            prior_strength=float(config["model"]["prior_strength"]),
            top_k=int(config["model"]["top_k"]),
        ).fit(frame.loc[frame["split"] == "train"])
        validation = _public_eve_records(
            model,
            frame.loc[frame["split"] == "validation"],
            split="validation",
            config=sample_config,
        )
        test = _public_eve_records(model, frame.loc[frame["split"] == "test"], split="test", config=sample_config)
        validation.insert(0, "dataset", dataset_id)
        test.insert(0, "dataset", dataset_id)
        all_records.extend([validation, test])
        summary["detectors"][dataset_id] = {}
        for detector in config["detectors"]:
            metrics = fit_and_score_neural_eve(validation, test, kind=detector, **config["training"])
            values = metrics_to_dict(metrics)
            summary["detectors"][dataset_id][detector] = values
            rows.append({"dataset": dataset_id, "detector": detector, **values})

    output_dir = ROOT / "results/tables"
    (output_dir / "phase9_independent_neural_steganalysis_records.csv").write_text(
        pd.concat(all_records, ignore_index=True).to_csv(index=False),
        encoding="utf-8",
    )
    (output_dir / "phase9_independent_neural_steganalysis.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "phase9_independent_neural_steganalysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
