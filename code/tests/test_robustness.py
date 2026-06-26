from __future__ import annotations

import pandas as pd

from steganalysis.robustness import apply_ablation, apply_attack


def records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "action_probability": [0.1, 0.2, 0.3, 0.4],
            "surprise_bits": [3.0, 2.0, 1.0, 0.5],
            "rank_fraction": [1.0, 0.5, 0.25, 0.25],
            "is_top_action": [0, 0, 1, 1],
            "entropy_bits": [1.0, 1.0, 2.0, 2.0],
            "top_probability": [0.5, 0.5, 0.4, 0.4],
            "candidate_count": [4, 4, 4, 4],
            "unseen_context": [0, 1, 0, 1],
            "unseen_destination": [0, 0, 1, 1],
            "same_as_previous": [0, 1, 0, 1],
            "self_loop": [0, 0, 1, 1],
            "log_inter_event_gap": [0.0, 1.0, 2.0, 3.0],
        }
    )


def test_attack_preserves_labels() -> None:
    attacked = apply_attack(records(), "timestamp_noise", seed=5)

    assert attacked["label"].tolist() == records()["label"].tolist()
    assert (attacked["log_inter_event_gap"] >= 0).all()


def test_ablation_zeroes_feature_group() -> None:
    ablated = apply_ablation(records(), "no_probability")

    assert (ablated[["action_probability", "surprise_bits", "rank_fraction", "is_top_action"]] == 0).all().all()
