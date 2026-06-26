from __future__ import annotations

import pandas as pd

from models.temporal import TemporalBackoffModel
from steganalysis.samples import FEATURE_COLUMNS, SampleConfig, feature_matrix, make_steganalysis_records


def test_make_steganalysis_records_pairs_natural_and_stego() -> None:
    frame = pd.DataFrame(
        {
            "source": ["u1", "u1", "u1", "u1"],
            "destination": ["a", "b", "a", "b"],
            "timestamp": [1, 2, 3, 4],
            "split": ["train", "train", "validation", "validation"],
        }
    )
    model = TemporalBackoffModel(prior_strength=1.0, top_k=4).fit(
        frame.loc[frame["split"] == "train"]
    )

    records = make_steganalysis_records(
        model,
        frame.loc[frame["split"] == "validation"],
        split="validation",
        config=SampleConfig(max_bits_per_transition=1, seed=11),
    )
    x, y = feature_matrix(records)

    assert len(records) == 4
    assert set(records["label"]) == {0, 1}
    assert x.shape[1] == len(FEATURE_COLUMNS)
    assert y.tolist().count(1) == 2
