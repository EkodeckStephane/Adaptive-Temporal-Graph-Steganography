from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

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

    from controllers.fuzzy import ControllerInputs, FuzzyRateController, fixed_entropy_threshold
    from controllers.fuzzy_optimizer import default_weights, optimize_fuzzy_weights

    config = _load_yaml(ROOT / "experiments/real_world/phase6_fuzzy_controller.yaml")
    grid = config["grid"]
    keys = list(grid)
    cases = [
        ControllerInputs(**dict(zip(keys, values)))
        for values in itertools.product(*(grid[key] for key in keys))
    ]

    max_bits = int(config["controller"]["max_bits_per_transition"])
    stop_threshold = float(config["controller"]["stop_threshold"])

    print(f"Optimizing fuzzy weights on {len(cases)} synthetic cases...")
    best_weights, best_score = optimize_fuzzy_weights(
        cases,
        max_bits_per_transition=max_bits,
        stop_threshold=stop_threshold,
        risk_lambda=2.0,
        abstention_target=0.90,
        abstention_lambda=1.0,
    )
    print(f"Best objective: {best_score:.4f}")
    print("Best weights:", best_weights)

    # Compare baseline vs optimized on the same grid.
    baseline_controller = FuzzyRateController(
        max_bits_per_transition=max_bits,
        stop_threshold=stop_threshold,
        weights=default_weights(),
    )
    optimized_controller = FuzzyRateController(
        max_bits_per_transition=max_bits,
        stop_threshold=stop_threshold,
        weights=best_weights,
    )
    baseline_decisions = [baseline_controller.decide(case) for case in cases]
    optimized_decisions = [optimized_controller.decide(case) for case in cases]
    fixed_decisions = [
        fixed_entropy_threshold(
            case,
            entropy_threshold=float(config["baseline"]["entropy_threshold"]),
            max_bits_per_transition=max_bits,
        )
        for case in cases
    ]

    def summarize(decisions: list[Any]) -> dict[str, Any]:
        from collections import Counter

        counts = Counter(d.mode for d in decisions)
        total = len(decisions)
        return {
            "cases": total,
            "mode_counts": dict(sorted(counts.items())),
            "mean_local_payload_bits": sum(d.local_payload_bits for d in decisions) / total,
            "abstention_rate": sum(d.mode in {"COVER", "PAUSE", "STOP"} for d in decisions) / total,
            "embed_rate": counts["EMBED"] / total,
        }

    summary = {
        "campaign": "phase6_fuzzy_weight_optimization",
        "date": str(config["date"]),
        "best_weights": best_weights.__dict__,
        "best_objective": best_score,
        "baseline_fuzzy": summarize(baseline_decisions),
        "optimized_fuzzy": summarize(optimized_decisions),
        "fixed_threshold": summarize(fixed_decisions),
    }

    output_path = ROOT / "results/tables/phase6_fuzzy_weight_optimization.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
