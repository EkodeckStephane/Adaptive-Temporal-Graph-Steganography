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


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from models.temporal import TemporalBackoffModel
    from steganalysis.adaptive_eve import (
        BOUNDED_WHITE_BOX_FEATURE_COLUMNS,
        fit_and_score_adaptive_eve,
        metrics_to_dict,
    )
    from steganalysis.samples import SampleConfig, make_steganalysis_records

    config = _load_yaml(ROOT / "experiments/real_world/phase9_independent_neural_steganalysis.yaml")
    catalog = _event_catalog(config)
    dataset_id = str(config.get("adaptive_hardening_dataset", "t_drive_cells"))
    frame = _read_event_dataset(catalog[dataset_id], max_rows_per_split=int(config["max_rows_per_split"]))
    model = TemporalBackoffModel(
        prior_strength=float(config["model"]["prior_strength"]),
        top_k=int(config["model"]["top_k"]),
    ).fit(frame.loc[frame["split"] == "train"])

    base_codec = config["codec"]
    candidates = [
        {
            "name": "previous_high_payload",
            "max_bits_per_transition": 2,
            "min_encoded_probability": 0.0,
            "max_encoded_rank_fraction": 1.0,
            "max_encoded_surprise_bits": float("inf"),
        },
        {
            "name": "range_one_bit",
            "max_bits_per_transition": 1,
            "min_encoded_probability": 0.0,
            "max_encoded_rank_fraction": 1.0,
            "max_encoded_surprise_bits": float("inf"),
        },
        {
            "name": "selected_rank_half",
            "max_bits_per_transition": 1,
            "min_encoded_probability": 0.0,
            "max_encoded_rank_fraction": 0.5,
            "max_encoded_surprise_bits": float("inf"),
        },
        {
            "name": "strict_rank_quarter",
            "max_bits_per_transition": 1,
            "min_encoded_probability": 0.0,
            "max_encoded_rank_fraction": 0.25,
            "max_encoded_surprise_bits": float("inf"),
        },
    ]
    rows = []
    for index, candidate in enumerate(candidates):
        sample_config = SampleConfig(
            max_bits_per_transition=int(candidate["max_bits_per_transition"]),
            seed=int(config["seed"]),
            max_local_total_variation=float(base_codec["max_local_total_variation"]),
            max_local_kl_bits=float(base_codec["max_local_kl_bits"]),
            min_entropy_bits=float(base_codec["min_entropy_bits"]),
            cover_when_unsafe=bool(base_codec["cover_when_unsafe"]),
            codec_backend=str(base_codec.get("codec_backend", "range")),
            min_encoded_probability=float(candidate["min_encoded_probability"]),
            max_encoded_surprise_bits=float(candidate["max_encoded_surprise_bits"]),
            max_encoded_rank_fraction=float(candidate["max_encoded_rank_fraction"]),
        )
        validation = make_steganalysis_records(
            model,
            frame.loc[frame["split"] == "validation"],
            split="validation",
            config=sample_config,
        )
        test = make_steganalysis_records(
            model,
            frame.loc[frame["split"] == "test"],
            split="test",
            config=sample_config,
        )
        metrics = fit_and_score_adaptive_eve(
            validation,
            test,
            seed=int(config["seed"]) + index,
            feature_columns=BOUNDED_WHITE_BOX_FEATURE_COLUMNS,
            bootstrap_rounds=100,
        )
        stego_test = test.loc[test["label"] == 1]
        embedded = stego_test["bits_consumed"] > 0
        rows.append(
            {
                "dataset": dataset_id,
                **candidate,
                **metrics_to_dict(metrics),
                "embedding_rate": float(embedded.mean()),
                "bits_per_transition": float(stego_test["bits_consumed"].mean()),
            }
        )
        print(rows[-1])

    output_dir = ROOT / "results/tables"
    output_path = output_dir / "phase10_adaptive_hardening_sweep.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    feasible = [
        row
        for row in rows
        if row["auc_ci_high"] <= 0.60 and row["bits_per_transition"] > 0.0
    ]
    feasible.sort(key=lambda row: (row["bits_per_transition"], -row["auc_ci_high"]), reverse=True)
    (output_dir / "phase10_adaptive_hardening_sweep.json").write_text(
        json.dumps({"dataset": dataset_id, "best_feasible": feasible[:10], "all": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
