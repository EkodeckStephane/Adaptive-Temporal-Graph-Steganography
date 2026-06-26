from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/tables"

REQUIRED = {
    "phase4_spatial_sensitivity.json",
    "phase5_codec_baseline.json",
    "phase6_fuzzy_controller_baseline.json",
    "phase6_fuzzy_weight_optimization.json",
    "phase7_steganalysis.json",
    "phase8_robustness.json",
    "phase9_temporal_gnn_cover_model.json",
    "phase9_independent_neural_steganalysis.json",
    "phase9_adaptive_steganalysis.json",
    "phase9_active_channel_reliability.json",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    missing = REQUIRED - {path.name for path in RESULTS.glob("*.json")}
    if missing:
        print(f"Missing result files: {sorted(missing)}")
        return 1

    errors = []

    # Phase 7: all external AUCs <= 0.60
    phase7 = _load(RESULTS / "phase7_steganalysis.json")
    for dataset, detectors in phase7["detectors"].items():
        for detector, metrics in detectors.items():
            if metrics["auc"] > 0.60:
                errors.append(f"Phase 7 {dataset}/{detector} AUC = {metrics['auc']:.3f} > 0.60")

    # Phase 8: robustness variants <= 0.60
    phase8 = _load(RESULTS / "phase8_robustness.json")
    for dataset, detectors in phase8["robustness"].items():
        for detector, variants in detectors.items():
            for variant, metrics in variants.items():
                if metrics["auc"] > 0.60:
                    errors.append(
                        f"Phase 8 {dataset}/{detector}/{variant} AUC = {metrics['auc']:.3f} > 0.60"
                    )

    # Phase 9 independent neural Eves <= 0.60
    phase9_neural = _load(RESULTS / "phase9_independent_neural_steganalysis.json")
    max_neural_auc = 0.0
    for dataset, detectors in phase9_neural["detectors"].items():
        for detector, metrics in detectors.items():
            max_neural_auc = max(max_neural_auc, metrics["auc"])
            if metrics["auc"] > 0.60:
                errors.append(f"Phase 9 neural {dataset}/{detector} AUC = {metrics['auc']:.3f} > 0.60")

    # Phase 9 adaptive: report but do not fail (it is expected to exceed threshold)
    phase9_adaptive = _load(RESULTS / "phase9_adaptive_steganalysis.json")
    max_adaptive_auc = 0.0
    for dataset, detectors in phase9_adaptive["detectors"].items():
        for detector, metrics in detectors.items():
            if detector == "adaptive_gradient_boosting":
                max_adaptive_auc = max(max_adaptive_auc, metrics["auc"])

    # Phase 9 active channel: corrected BER = 0.0
    phase9_active = _load(RESULTS / "phase9_active_channel_reliability.json")
    for attack, metrics in phase9_active["attacks"].items():
        if metrics["corrected_ber"] != 0.0:
            errors.append(f"Phase 9 active {attack} corrected BER = {metrics['corrected_ber']} != 0.0")

    if errors:
        print("Verification failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("All Q1 result checks passed.")
    print(f"Max independent neural-Eve AUC: {max_neural_auc:.3f}")
    print(f"Max adaptive detector AUC: {max_adaptive_auc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
