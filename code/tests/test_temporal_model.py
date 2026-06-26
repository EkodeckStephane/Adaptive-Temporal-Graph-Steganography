from __future__ import annotations

import pandas as pd

from models.temporal import TemporalBackoffModel, evaluate_temporal_splits


def temporal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["u1", "u1", "u1", "u1", "u2", "u2", "u1", "u1"],
            "destination": ["a", "b", "a", "b", "c", "d", "a", "b"],
            "timestamp": list(range(8)),
            "split": ["train", "train", "train", "train", "train", "validation", "test", "test"],
        }
    )


def test_temporal_backoff_conditions_on_previous_destination() -> None:
    model = TemporalBackoffModel(prior_strength=1.0, top_k=4).fit(
        temporal_frame().loc[lambda frame: frame["split"] == "train"]
    )

    after_a = model.candidate_distribution("u1", "a")
    after_b = model.candidate_distribution("u1", "b")

    assert after_a[0].action == "b"
    assert after_b[0].action == "a"


def test_temporal_split_evaluation_reports_metrics() -> None:
    metrics = evaluate_temporal_splits(temporal_frame(), prior_strength=1.0, top_k=4)

    assert set(metrics) == {"train", "validation", "test"}
    assert metrics["test"].rows == 2
    assert metrics["test"].mean_entropy_bits >= 0
