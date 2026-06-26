from __future__ import annotations

import csv
import itertools
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _summarize(decisions: list[Any]) -> dict[str, Any]:
    counts = Counter(decision.mode for decision in decisions)
    total = len(decisions)
    return {
        "cases": total,
        "mode_counts": dict(sorted(counts.items())),
        "mean_local_payload_bits": sum(decision.local_payload_bits for decision in decisions) / total,
        "abstention_rate": sum(decision.mode in {"COVER", "PAUSE", "STOP"} for decision in decisions)
        / total,
        "embed_rate": counts["EMBED"] / total,
    }


def _input_matrix(inputs: list[Any]) -> list[list[float]]:
    return [
        [
            item.predictive_entropy,
            item.calibration_uncertainty,
            item.steganalysis_risk,
            item.payload_pressure,
            item.dead_end_risk,
            item.channel_fragility,
        ]
        for item in inputs
    ]


def _mlp_training_labels(inputs: list[Any]) -> list[str]:
    labels = []
    for item in inputs:
        if item.dead_end_risk >= 0.95:
            labels.append("STOP")
        elif (
            item.predictive_entropy >= 0.55
            and item.calibration_uncertainty <= 0.55
            and item.steganalysis_risk <= 0.45
            and item.channel_fragility <= 0.60
        ):
            labels.append("EMBED")
        elif item.payload_pressure >= 0.70 and (
            item.steganalysis_risk >= 0.65 or item.channel_fragility >= 0.65
        ):
            labels.append("PAUSE")
        else:
            labels.append("COVER")
    return labels


def _fit_mlp_policy(inputs: list[Any], *, seed: int) -> Pipeline:
    policy = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(12,),
                    activation="relu",
                    alpha=1e-3,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    policy.fit(_input_matrix(inputs), _mlp_training_labels(inputs))
    return policy


def _mlp_decisions(policy: Pipeline, inputs: list[Any], *, max_bits_per_transition: int) -> list[Any]:
    from controllers.fuzzy import ControlDecision

    labels = policy.predict(_input_matrix(inputs))
    decisions = []
    for label, item in zip(labels, inputs):
        if label == "EMBED":
            bits = max(1, min(max_bits_per_transition, round(item.predictive_entropy * max_bits_per_transition)))
        else:
            bits = 0
        decisions.append(ControlDecision(str(label), int(bits), item.predictive_entropy, float(label != "EMBED")))
    return decisions


def _load_phase3_catalog() -> list[dict[str, Any]]:
    with (ROOT / "results/tables/phase3_dataset_statistics.json").open(encoding="utf-8") as handle:
        return json.load(handle)["datasets"]


def _event_catalog(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    extra = config.get("real_data", {}).get("event_datasets", [])
    return {record["id"]: record for record in [*_load_phase3_catalog(), *extra]}


def _read_event_dataset(
    record: dict[str, Any],
    *,
    max_rows_per_split: int,
) -> pd.DataFrame:
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
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["timestamp"], kind="stable")
        .reset_index(drop=True)
    )


def _real_inputs_from_prediction(prediction: Any, *, top_k: int, payload_pressure: float) -> Any:
    from controllers.fuzzy import ControllerInputs

    entropy_scale = max(1.0, (top_k).bit_length() - 1)
    fragility = 0.55 if prediction.unseen_context else 0.35 if prediction.unseen_destination else 0.05
    return ControllerInputs(
        predictive_entropy=min(1.0, prediction.entropy_bits / entropy_scale),
        calibration_uncertainty=0.50
        if prediction.unseen_context
        else 0.25
        if prediction.unseen_destination
        else 0.10,
        steganalysis_risk=0.50 * prediction.top_probability,
        payload_pressure=payload_pressure,
        dead_end_risk=1.0 if prediction.candidate_count < 2 else 0.0,
        channel_fragility=fragility,
    )


def _evaluate_real_data(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from controllers.fuzzy import FuzzyRateController, fixed_entropy_threshold
    from models.temporal import TemporalBackoffModel

    real_config = config["real_data"]
    model_config = real_config["model"]
    controller = FuzzyRateController(
        max_bits_per_transition=int(config["controller"]["max_bits_per_transition"]),
        stop_threshold=float(config["controller"]["stop_threshold"]),
    )
    synthetic_grid = config["grid"]
    synthetic_keys = list(synthetic_grid)
    synthetic_inputs = [
        __import__("controllers.fuzzy", fromlist=["ControllerInputs"]).ControllerInputs(
            **dict(zip(synthetic_keys, values))
        )
        for values in itertools.product(*(synthetic_grid[key] for key in synthetic_keys))
    ]
    mlp_policy = _fit_mlp_policy(synthetic_inputs, seed=int(config["mlp_baseline"]["seed"]))
    output_rows = []
    summary: dict[str, Any] = {}
    catalog = _event_catalog(config)
    for dataset_id in real_config["datasets"]:
        frame = _read_event_dataset(
            catalog[dataset_id],
            max_rows_per_split=int(real_config["max_rows_per_split"]),
        )
        model = TemporalBackoffModel(
            prior_strength=float(model_config["prior_strength"]),
            top_k=int(model_config["top_k"]),
        ).fit(frame.loc[frame["split"] == "train"])
        summary[dataset_id] = {}
        for split in ("validation", "test"):
            subset = frame.loc[frame["split"] == split]
            if subset.empty:
                continue
            predictions = list(model.iter_predictions(subset))
            inputs = [
                _real_inputs_from_prediction(
                    prediction,
                    top_k=int(model_config["top_k"]),
                    payload_pressure=1.0 - index / max(1, len(predictions) - 1),
                )
                for index, prediction in enumerate(predictions)
            ]
            fuzzy_decisions = [controller.decide(item) for item in inputs]
            fixed_decisions = [
                fixed_entropy_threshold(
                    item,
                    entropy_threshold=float(config["baseline"]["entropy_threshold"]),
                    max_bits_per_transition=int(config["controller"]["max_bits_per_transition"]),
                )
                for item in inputs
            ]
            mlp_decisions = _mlp_decisions(
                mlp_policy,
                inputs,
                max_bits_per_transition=int(config["controller"]["max_bits_per_transition"]),
            )
            summary[dataset_id][split] = {
                "fuzzy": _summarize(fuzzy_decisions),
                "fixed_threshold": _summarize(fixed_decisions),
                "mlp_policy": _summarize(mlp_decisions),
                "mean_predictive_entropy": sum(item.predictive_entropy for item in inputs)
                / len(inputs),
                "mean_channel_fragility": sum(item.channel_fragility for item in inputs) / len(inputs),
            }
            for index, (inputs_row, fuzzy, fixed, mlp) in enumerate(
                zip(inputs, fuzzy_decisions, fixed_decisions, mlp_decisions)
            ):
                output_rows.append(
                    {
                        "dataset": dataset_id,
                        "split": split,
                        "case": index,
                        **vars(inputs_row),
                        "fuzzy_mode": fuzzy.mode,
                        "fuzzy_bits": fuzzy.local_payload_bits,
                        "fixed_mode": fixed.mode,
                        "fixed_bits": fixed.local_payload_bits,
                        "mlp_mode": mlp.mode,
                        "mlp_bits": mlp.local_payload_bits,
                    }
                )
    return summary, output_rows


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from controllers.fuzzy import ControllerInputs, FuzzyRateController, fixed_entropy_threshold

    config = _load_yaml(ROOT / "experiments/real_world/phase6_fuzzy_controller.yaml")
    grid = config["grid"]
    keys = list(grid)
    cases = [
        ControllerInputs(**dict(zip(keys, values)))
        for values in itertools.product(*(grid[key] for key in keys))
    ]

    controller = FuzzyRateController(
        max_bits_per_transition=int(config["controller"]["max_bits_per_transition"]),
        stop_threshold=float(config["controller"]["stop_threshold"]),
    )
    fuzzy_decisions = [controller.decide(case) for case in cases]
    fixed_decisions = [
        fixed_entropy_threshold(
            case,
            entropy_threshold=float(config["baseline"]["entropy_threshold"]),
            max_bits_per_transition=int(config["controller"]["max_bits_per_transition"]),
        )
        for case in cases
    ]
    mlp_policy = _fit_mlp_policy(cases, seed=int(config["mlp_baseline"]["seed"]))
    mlp_decisions = _mlp_decisions(
        mlp_policy,
        cases,
        max_bits_per_transition=int(config["controller"]["max_bits_per_transition"]),
    )
    summary = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "controller": config["controller"],
        "baseline": config["baseline"],
        "fuzzy": _summarize(fuzzy_decisions),
        "fixed_threshold": _summarize(fixed_decisions),
        "mlp_policy": _summarize(mlp_decisions),
    }
    real_summary, real_rows = _evaluate_real_data(config)
    summary["real_data"] = real_summary

    output_dir = ROOT / "results/tables"
    json_output = output_dir / "phase6_fuzzy_controller_baseline.json"
    csv_output = output_dir / "phase6_fuzzy_controller_baseline.csv"
    real_csv_output = output_dir / "phase6_fuzzy_controller_real_data.csv"
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                *keys,
                "fuzzy_mode",
                "fuzzy_bits",
                "fixed_mode",
                "fixed_bits",
                "mlp_mode",
                "mlp_bits",
            ],
        )
        writer.writeheader()
        for index, (case, fuzzy, fixed, mlp) in enumerate(
            zip(cases, fuzzy_decisions, fixed_decisions, mlp_decisions)
        ):
            writer.writerow(
                {
                    "case": index,
                    **vars(case),
                    "fuzzy_mode": fuzzy.mode,
                    "fuzzy_bits": fuzzy.local_payload_bits,
                    "fixed_mode": fixed.mode,
                    "fixed_bits": fixed.local_payload_bits,
                    "mlp_mode": mlp.mode,
                    "mlp_bits": mlp.local_payload_bits,
                }
            )
    with real_csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=real_rows[0].keys())
        writer.writeheader()
        writer.writerows(real_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
