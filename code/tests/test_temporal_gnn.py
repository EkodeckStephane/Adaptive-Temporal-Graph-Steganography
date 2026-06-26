from __future__ import annotations

import pandas as pd

from models.temporal_gnn import train_and_evaluate_temporal_gnn


def test_temporal_gnn_cover_model_reports_calibration_metrics() -> None:
    frame = pd.DataFrame(
        [
            ("a", "b", 1, "train"),
            ("a", "c", 2, "train"),
            ("b", "c", 3, "train"),
            ("a", "b", 4, "validation"),
            ("b", "c", 5, "validation"),
            ("a", "c", 6, "test"),
            ("b", "d", 7, "test"),
        ],
        columns=["source", "destination", "timestamp", "split"],
    )

    metrics = train_and_evaluate_temporal_gnn(
        frame,
        embedding_dim=8,
        hidden_dim=12,
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        seed=7,
    )

    assert metrics["test"].rows == 2
    assert 0.0 <= metrics["test"].expected_calibration_error <= 1.0
    assert 0.0 <= metrics["test"].brier_score <= 1.0
