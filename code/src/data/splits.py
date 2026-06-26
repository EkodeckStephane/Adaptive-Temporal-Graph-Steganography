from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalCutoffs:
    train_end: float
    validation_end: float
    train_fraction: float
    validation_fraction: float
    test_fraction: float


def assign_causal_splits(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, TemporalCutoffs]:
    if frame.empty:
        raise ValueError("Cannot split an empty table")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("A positive test fraction is required")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Frame must be sorted chronologically before splitting")

    timestamps = frame["timestamp"].to_numpy()
    train_position = min(len(frame) - 1, max(0, int(np.ceil(train_fraction * len(frame))) - 1))
    validation_position = min(
        len(frame) - 1,
        max(train_position + 1, int(np.ceil((train_fraction + validation_fraction) * len(frame))) - 1),
    )
    train_end = float(timestamps[train_position])
    validation_end = float(timestamps[validation_position])
    if validation_end <= train_end:
        later = timestamps[timestamps > train_end]
        if later.size == 0:
            raise ValueError("At least three temporal regions are required")
        validation_end = float(later[min(len(later) - 1, max(0, len(later) // 2))])

    result = frame.copy()
    result["split"] = np.where(
        result["timestamp"] <= train_end,
        "train",
        np.where(result["timestamp"] <= validation_end, "validation", "test"),
    )
    fractions = result["split"].value_counts(normalize=True)
    if set(fractions.index) != {"train", "validation", "test"}:
        raise ValueError("The timestamps do not permit three non-empty causal splits")
    return result, TemporalCutoffs(
        train_end=train_end,
        validation_end=validation_end,
        train_fraction=float(fractions["train"]),
        validation_fraction=float(fractions["validation"]),
        test_fraction=float(fractions["test"]),
    )
