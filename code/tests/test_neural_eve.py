from __future__ import annotations

import pandas as pd

from steganalysis.neural_eve import fit_and_score_neural_eve


def test_neural_eve_uses_public_sequence_fields() -> None:
    validation = pd.DataFrame(
        [
            ("validation", 0, 0, "u1", "a", 1),
            ("validation", 0, 1, "u1", "b", 1),
            ("validation", 1, 0, "u1", "a", 2),
            ("validation", 1, 1, "u1", "b", 2),
        ],
        columns=["split", "pair_id", "label", "source", "action", "timestamp"],
    )
    test = validation.assign(split="test", timestamp=[3, 3, 4, 4])

    metrics = fit_and_score_neural_eve(
        validation,
        test,
        kind="temporal_graph_eve",
        context_length=3,
        embedding_dim=8,
        hidden_dim=12,
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        seed=3,
    )

    assert 0.0 <= metrics.auc <= 1.0
