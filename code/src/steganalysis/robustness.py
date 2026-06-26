from __future__ import annotations

import numpy as np
import pandas as pd

PROBABILITY_COLUMNS = ["action_probability", "surprise_bits", "rank_fraction", "is_top_action"]
ENTROPY_COLUMNS = ["entropy_bits", "top_probability", "candidate_count"]
COLD_START_COLUMNS = ["unseen_context", "unseen_destination"]
TEMPORAL_COLUMNS = ["same_as_previous", "self_loop", "log_inter_event_gap"]


def apply_attack(frame: pd.DataFrame, name: str, *, seed: int) -> pd.DataFrame:
    if name == "none":
        return frame.copy()
    if name == "subsample_80":
        return (
            frame.groupby("label", group_keys=False)
            .sample(frac=0.8, random_state=seed)
            .sort_index()
            .reset_index(drop=True)
        )
    if name == "truncate_50":
        return (
            frame.groupby("label", group_keys=False)
            .head(max(1, int(0.5 * frame["label"].value_counts().min())))
            .reset_index(drop=True)
        )
    if name == "timestamp_noise":
        rng = np.random.default_rng(seed)
        attacked = frame.copy()
        attacked["log_inter_event_gap"] = np.maximum(
            0.0,
            attacked["log_inter_event_gap"].to_numpy(dtype=float)
            + rng.normal(0.0, 0.25, size=len(attacked)),
        )
        return attacked
    raise ValueError(f"Unknown attack: {name}")


def apply_ablation(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    ablated = frame.copy()
    if name == "full":
        return ablated
    if name == "no_probability":
        columns = PROBABILITY_COLUMNS
    elif name == "no_entropy":
        columns = ENTROPY_COLUMNS
    elif name == "no_cold_start":
        columns = COLD_START_COLUMNS
    elif name == "no_temporal":
        columns = TEMPORAL_COLUMNS
    else:
        raise ValueError(f"Unknown ablation: {name}")
    for column in columns:
        ablated[column] = 0.0
    return ablated
