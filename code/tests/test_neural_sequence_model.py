from __future__ import annotations

import pandas as pd

from models.neural_sequence import train_and_evaluate


def test_neural_sequence_gru_and_transformer_run() -> None:
    frame = pd.DataFrame(
        {
            "source": ["u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2"],
            "destination": ["a", "b", "a", "b", "c", "d", "c", "d"],
            "timestamp": list(range(8)),
            "split": ["train", "train", "train", "train", "validation", "validation", "test", "test"],
        }
    )

    for kind in ("gru", "transformer"):
        metrics = train_and_evaluate(
            frame,
            kind=kind,
            context_length=2,
            embedding_dim=8,
            hidden_dim=16,
            epochs=1,
            batch_size=2,
            learning_rate=0.01,
            seed=7,
        )
        assert set(metrics) == {"train", "validation", "test"}
        assert metrics["test"].rows == 2
        assert metrics["test"].mean_nll_bits > 0
