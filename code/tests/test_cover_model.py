from __future__ import annotations

import pandas as pd
import pytest

from models.coverage import SourceDestinationFrequencyModel, evaluate_splits


def event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["u1", "u1", "u1", "u2", "u2", "u3", "u3", "u3"],
            "destination": ["a", "a", "b", "c", "c", "a", "d", "d"],
            "timestamp": list(range(8)),
            "split": [
                "train",
                "train",
                "train",
                "train",
                "validation",
                "validation",
                "test",
                "test",
            ],
        }
    )


def test_frequency_model_assigns_nonzero_probability_to_cold_start() -> None:
    model = SourceDestinationFrequencyModel(prior_strength=2.0).fit(
        event_frame().loc[lambda frame: frame["split"] == "train"]
    )

    assert model.probability("u1", "a") > model.probability("u1", "c")
    assert model.probability("new-user", "a") > 0
    assert model.probability("u1", "new-item") > 0


def test_frequency_model_reports_prediction_metrics() -> None:
    model = SourceDestinationFrequencyModel(prior_strength=2.0).fit(
        event_frame().loc[lambda frame: frame["split"] == "train"]
    )

    metrics = model.evaluate(event_frame().loc[lambda frame: frame["split"] == "validation"])

    assert metrics.rows == 2
    assert metrics.mean_nll_bits > 0
    assert metrics.perplexity > 1
    assert 0 <= metrics.expected_calibration_error <= 1
    assert metrics.unseen_source_fraction == pytest.approx(0.5)


def test_evaluate_splits_uses_train_only() -> None:
    metrics = evaluate_splits(event_frame(), prior_strength=2.0, calibration_bins=5)

    assert set(metrics) == {"train", "validation", "test"}
    assert metrics["test"].unseen_destination_fraction == pytest.approx(1.0)
