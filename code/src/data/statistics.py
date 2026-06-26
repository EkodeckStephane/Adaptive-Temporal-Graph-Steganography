from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def _entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _weighted_conditional_entropy(frame: pd.DataFrame) -> float:
    total = len(frame)
    entropy = 0.0
    for _, group in frame.groupby("source", sort=False):
        counts = group["destination"].value_counts().to_numpy()
        entropy += len(group) / total * _entropy(counts)
    return entropy


def _burstiness(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    mean = float(np.mean(values))
    std = float(np.std(values))
    if mean + std == 0:
        return 0.0
    return (std - mean) / (std + mean)


def describe_temporal_events(frame: pd.DataFrame) -> dict[str, Any]:
    events = len(frame)
    pairs = list(zip(frame["source"], frame["destination"]))
    unique_pairs = set(pairs)
    nodes = set(frame["source"]) | set(frame["destination"])
    reciprocal = sum((target, source) in unique_pairs for source, target in unique_pairs)
    gaps = np.diff(frame["timestamp"].to_numpy(dtype=float))

    source_burstiness = []
    for _, group in frame.groupby("source", sort=False):
        source_gaps = np.diff(group["timestamp"].to_numpy(dtype=float))
        value = _burstiness(source_gaps)
        if value is not None:
            source_burstiness.append(value)

    split_counts = Counter(frame["split"]) if "split" in frame else Counter()
    train_nodes: set[Any] = set()
    cold_start: dict[str, int] = {}
    if split_counts:
        train = frame[frame["split"] == "train"]
        train_nodes = set(train["source"]) | set(train["destination"])
        for split in ("validation", "test"):
            subset = frame[frame["split"] == split]
            split_nodes = set(subset["source"]) | set(subset["destination"])
            cold_start[split] = len(split_nodes - train_nodes)

    destination_counts = frame["destination"].value_counts().to_numpy()
    conditional_entropy = _weighted_conditional_entropy(frame)
    return {
        "events": events,
        "nodes": len(nodes),
        "sources": int(frame["source"].nunique()),
        "destinations": int(frame["destination"].nunique()),
        "unique_directed_edges": len(unique_pairs),
        "repeated_edge_fraction": 1.0 - len(unique_pairs) / events,
        "exact_duplicate_fraction": float(
            frame.duplicated(["source", "destination", "timestamp"]).mean()
        ),
        "reciprocal_edge_fraction": reciprocal / max(1, len(unique_pairs)),
        "destination_entropy_bits": _entropy(destination_counts),
        "conditional_destination_entropy_bits": conditional_entropy,
        "effective_branching_factor": 2**conditional_entropy,
        "global_inter_event_burstiness": _burstiness(gaps),
        "median_source_burstiness": (
            float(np.median(source_burstiness)) if source_burstiness else None
        ),
        "timestamp_min": float(frame["timestamp"].min()),
        "timestamp_max": float(frame["timestamp"].max()),
        "split_counts": dict(split_counts),
        "cold_start_nodes": cold_start,
        "self_loop_fraction": float((frame["source"] == frame["destination"]).mean()),
        "log_events": math.log10(events),
    }
