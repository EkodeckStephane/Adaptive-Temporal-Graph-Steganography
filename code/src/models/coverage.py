from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

UNKNOWN_DESTINATION = "__unknown_destination__"


@dataclass(frozen=True)
class PredictionMetrics:
    rows: int
    mean_nll_bits: float
    perplexity: float
    top1_accuracy: float
    expected_calibration_error: float
    unseen_source_fraction: float
    unseen_destination_fraction: float


@dataclass(frozen=True)
class TopPrediction:
    destination: Hashable
    probability: float


class SourceDestinationFrequencyModel:
    """Smoothed source-conditioned next-destination baseline."""

    def __init__(self, *, prior_strength: float = 8.0, top_global_candidates: int = 32) -> None:
        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        if top_global_candidates < 1:
            raise ValueError("top_global_candidates must be at least one")
        self.prior_strength = float(prior_strength)
        self.top_global_candidates = int(top_global_candidates)
        self._source_counts: dict[Hashable, Counter[Hashable]] = {}
        self._source_totals: dict[Hashable, int] = {}
        self._global_counts: Counter[Hashable] = Counter()
        self._global_probabilities: dict[Hashable, float] = {}
        self._global_top: list[Hashable] = []
        self._top_cache: dict[Hashable, TopPrediction] = {}

    @property
    def sources(self) -> frozenset[Hashable]:
        return frozenset(self._source_counts)

    @property
    def destinations(self) -> frozenset[Hashable]:
        return frozenset(self._global_counts)

    def fit(self, frame: pd.DataFrame) -> SourceDestinationFrequencyModel:
        _require_columns(frame, ("source", "destination"))
        if frame.empty:
            raise ValueError("Cannot fit a coverage model on an empty frame")

        source_counts: defaultdict[Hashable, Counter[Hashable]] = defaultdict(Counter)
        global_counts: Counter[Hashable] = Counter()
        for source, destination in frame[["source", "destination"]].itertuples(index=False):
            source_counts[source][destination] += 1
            global_counts[destination] += 1

        self._source_counts = dict(source_counts)
        self._source_totals = {source: sum(counts.values()) for source, counts in source_counts.items()}
        self._global_counts = global_counts
        self._global_probabilities = self._make_global_probabilities(global_counts)
        self._global_top = [
            destination for destination, _ in global_counts.most_common(self.top_global_candidates)
        ]
        self._top_cache = {}
        return self

    def probability(self, source: Hashable, destination: Hashable) -> float:
        self._ensure_fitted()
        normalized_destination = destination if destination in self._global_counts else UNKNOWN_DESTINATION
        if source not in self._source_counts:
            return self._global_probabilities[normalized_destination]

        counts = self._source_counts[source]
        total = self._source_totals[source]
        prior = self.prior_strength * self._global_probabilities[normalized_destination]
        empirical = counts.get(normalized_destination, 0)
        return float((empirical + prior) / (total + self.prior_strength))

    def top_prediction(self, source: Hashable) -> TopPrediction:
        self._ensure_fitted()
        if source in self._top_cache:
            return self._top_cache[source]

        if source not in self._source_counts:
            destination = self._global_top[0]
            prediction = TopPrediction(
                destination=destination,
                probability=self._global_probabilities[destination],
            )
            self._top_cache[source] = prediction
            return prediction

        candidates = set(self._source_counts[source]) | set(self._global_top) | {UNKNOWN_DESTINATION}
        destination = max(candidates, key=lambda candidate: self.probability(source, candidate))
        prediction = TopPrediction(destination=destination, probability=self.probability(source, destination))
        self._top_cache[source] = prediction
        return prediction

    def evaluate(self, frame: pd.DataFrame, *, calibration_bins: int = 10) -> PredictionMetrics:
        _require_columns(frame, ("source", "destination"))
        if frame.empty:
            raise ValueError("Cannot evaluate a coverage model on an empty frame")
        if calibration_bins < 1:
            raise ValueError("calibration_bins must be at least one")

        nll_bits = []
        confidences = []
        correct = []
        unseen_sources = 0
        unseen_destinations = 0
        for source, destination in frame[["source", "destination"]].itertuples(index=False):
            if source not in self._source_counts:
                unseen_sources += 1
            if destination not in self._global_counts:
                unseen_destinations += 1

            probability = max(self.probability(source, destination), np.finfo(float).tiny)
            nll_bits.append(-math.log2(probability))

            top = self.top_prediction(source)
            confidences.append(top.probability)
            correct.append(top.destination == destination)

        mean_nll = float(np.mean(nll_bits))
        rows = len(frame)
        return PredictionMetrics(
            rows=rows,
            mean_nll_bits=mean_nll,
            perplexity=float(2**mean_nll),
            top1_accuracy=float(np.mean(correct)),
            expected_calibration_error=_expected_calibration_error(
                np.asarray(confidences, dtype=float),
                np.asarray(correct, dtype=bool),
                bins=calibration_bins,
            ),
            unseen_source_fraction=unseen_sources / rows,
            unseen_destination_fraction=unseen_destinations / rows,
        )

    def _make_global_probabilities(self, counts: Counter[Hashable]) -> dict[Hashable, float]:
        support = len(counts) + 1
        denominator = sum(counts.values()) + self.prior_strength * support
        probabilities = {
            destination: (count + self.prior_strength) / denominator
            for destination, count in counts.items()
        }
        probabilities[UNKNOWN_DESTINATION] = self.prior_strength / denominator
        return probabilities

    def _ensure_fitted(self) -> None:
        if not self._global_counts:
            raise ValueError("The coverage model must be fitted before use")


def evaluate_splits(
    frame: pd.DataFrame,
    *,
    prior_strength: float = 8.0,
    calibration_bins: int = 10,
) -> dict[str, PredictionMetrics]:
    _require_columns(frame, ("source", "destination", "split"))
    train = frame.loc[frame["split"] == "train"]
    model = SourceDestinationFrequencyModel(prior_strength=prior_strength).fit(train)
    return {
        split: model.evaluate(frame.loc[frame["split"] == split], calibration_bins=calibration_bins)
        for split in ("train", "validation", "test")
        if not frame.loc[frame["split"] == split].empty
    }


def metrics_to_dict(metrics: Mapping[str, PredictionMetrics]) -> dict[str, dict[str, Any]]:
    return {split: vars(record) for split, record in metrics.items()}


def _expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        weight = float(mask.mean())
        accuracy = float(correct[mask].mean())
        confidence = float(confidences[mask].mean())
        error += weight * abs(accuracy - confidence)
    return error


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
