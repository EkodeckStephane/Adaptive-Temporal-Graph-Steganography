from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from steganalysis.detectors import fit_detector, metrics_to_dict, score_detector
    from steganalysis.robustness import apply_ablation, apply_attack
    from steganalysis.samples import feature_matrix

    config = _load_yaml(ROOT / "experiments/real_world/phase8_robustness.yaml")
    records = pd.read_csv(ROOT / config["input_records"])
    summary: dict[str, Any] = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "robustness": {},
        "ablations": {},
    }
    rows = []
    for dataset, dataset_records in records.groupby("dataset", sort=False):
        train = dataset_records.loc[dataset_records["split"] == "validation"]
        test = dataset_records.loc[dataset_records["split"] == "test"]
        x_train, y_train = feature_matrix(train)
        summary["robustness"][dataset] = {}
        summary["ablations"][dataset] = {}
        for detector_name in config["detectors"]:
            detector = fit_detector(detector_name, x_train, y_train, seed=int(config["seed"]))
            summary["robustness"][dataset][detector_name] = {}
            for attack in config["attacks"]:
                attacked = apply_attack(test, attack, seed=int(config["seed"]))
                x_test, y_test = feature_matrix(attacked)
                metrics = metrics_to_dict(score_detector(detector, x_test, y_test))
                summary["robustness"][dataset][detector_name][attack] = metrics
                rows.append(
                    {
                        "section": "robustness",
                        "dataset": dataset,
                        "detector": detector_name,
                        "variant": attack,
                        **metrics,
                    }
                )

            summary["ablations"][dataset][detector_name] = {}
            for ablation in config["ablations"]:
                train_ablation = apply_ablation(train, ablation)
                test_ablation = apply_ablation(test, ablation)
                x_train_ablation, y_train_ablation = feature_matrix(train_ablation)
                x_test_ablation, y_test_ablation = feature_matrix(test_ablation)
                ablation_detector = fit_detector(
                    detector_name,
                    x_train_ablation,
                    y_train_ablation,
                    seed=int(config["seed"]),
                )
                metrics = metrics_to_dict(
                    score_detector(ablation_detector, x_test_ablation, y_test_ablation)
                )
                summary["ablations"][dataset][detector_name][ablation] = metrics
                rows.append(
                    {
                        "section": "ablation",
                        "dataset": dataset,
                        "detector": detector_name,
                        "variant": ablation,
                        **metrics,
                    }
                )

    output_dir = ROOT / "results/tables"
    json_output = output_dir / "phase8_robustness.json"
    csv_output = output_dir / "phase8_robustness.csv"
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
