from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_ACTIONS = {"embed", "cover", "pause", "stop"}
REQUIRED_ABLATIONS = {
    "static_vs_temporal",
    "no_gru",
    "fixed_vs_entropy_coding",
    "fixed_threshold_vs_fuzzy",
    "fuzzy_vs_mlp",
    "no_abstention",
    "no_steganalysis_risk",
}
REQUIRED_FUZZY_INPUTS = {
    "predictive_entropy",
    "calibration_uncertainty",
    "steganalysis_risk",
    "payload_pressure",
    "dead_end_risk",
    "channel_fragility",
}


@dataclass(frozen=True)
class SpecificationSummary:
    level_a_datasets: tuple[str, ...]
    actions: tuple[str, ...]
    ablations: tuple[str, ...]


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _as_string_set(value: Any, field: str) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    return {str(item) for item in value}


def validate_research_specification(root: Path) -> SpecificationSummary:
    config = _load_yaml(root / "code/configs/base.yaml")
    manifest = _load_yaml(root / "datasets/metadata/manifest.yaml")

    stego = config.get("stego")
    fuzzy = config.get("fuzzy")
    evaluation = config.get("evaluation")
    constraints = config.get("constraints")
    if not all(isinstance(item, Mapping) for item in (stego, fuzzy, evaluation, constraints)):
        raise ValueError("stego, fuzzy, evaluation, and constraints must be mappings")

    actions = _as_string_set(stego.get("actions"), "stego.actions")
    if actions != REQUIRED_ACTIONS:
        raise ValueError(f"stego.actions must be exactly {sorted(REQUIRED_ACTIONS)}")

    fuzzy_inputs = _as_string_set(fuzzy.get("inputs"), "fuzzy.inputs")
    missing_inputs = REQUIRED_FUZZY_INPUTS - fuzzy_inputs
    if missing_inputs:
        raise ValueError(f"Missing fuzzy inputs: {sorted(missing_inputs)}")

    ablations = _as_string_set(evaluation.get("ablations"), "evaluation.ablations")
    missing_ablations = REQUIRED_ABLATIONS - ablations
    if missing_ablations:
        raise ValueError(f"Missing required ablations: {sorted(missing_ablations)}")

    if float(constraints.get("max_external_auc", 1.0)) > 0.60:
        raise ValueError("max_external_auc must not exceed 0.60")
    if float(constraints.get("passive_ber", 1.0)) != 0.0:
        raise ValueError("passive_ber must be zero")

    datasets = manifest.get("datasets")
    if not isinstance(datasets, Sequence):
        raise ValueError("datasets must be a sequence")
    level_a = tuple(
        str(dataset["id"])
        for dataset in datasets
        if isinstance(dataset, Mapping)
        and dataset.get("level") == "A"
        and dataset.get("real_world") is True
        and dataset.get("observed_trajectories") is True
    )
    minimum_domains = int(constraints.get("minimum_real_domains", 3))
    if len(level_a) < minimum_domains:
        raise ValueError(
            f"At least {minimum_domains} level-A observed-trajectory datasets are required"
        )

    return SpecificationSummary(
        level_a_datasets=level_a,
        actions=tuple(sorted(actions)),
        ablations=tuple(sorted(ablations)),
    )
