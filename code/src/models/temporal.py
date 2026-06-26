from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stego.coding import Candidate

UNKNOWN_DESTINATION = "__unknown_destination__"


@dataclass(frozen=True)
class TemporalPrediction:
    source: Hashable
    previous_destination: Hashable | None
    destination: Hashable
    top_destination: Hashable
    probability: float
    top_probability: float
    entropy_bits: float
    candidate_count: int
    unseen_context: bool
    unseen_destination: bool


@dataclass(frozen=True)
class TemporalMetrics:
    rows: int
    mean_nll_bits: float
    perplexity: float
    top1_accuracy: float
    mean_entropy_bits: float
    unseen_context_fraction: float
    unseen_destination_fraction: float


class TemporalBackoffModel:
    """Causal next-destination model with source/history backoff."""

    def __init__(self, *, prior_strength: float = 8.0, top_k: int = 32) -> None:
        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        if top_k < 2:
            raise ValueError("top_k must be at least two")
        self.prior_strength = float(prior_strength)
        self.top_k = int(top_k)
        self._context_counts: dict[tuple[Hashable, Hashable | None], Counter[Hashable]] = {}
        self._source_counts: dict[Hashable, Counter[Hashable]] = {}
        self._global_counts: Counter[Hashable] = Counter()
        self._global_distribution: dict[Hashable, float] = {}

    @property
    def destinations(self) -> frozenset[Hashable]:
        return frozenset(self._global_counts)

    def has_context(self, source: Hashable, previous_destination: Hashable | None) -> bool:
        return (source, previous_destination) in self._context_counts

    def fit(self, frame: pd.DataFrame) -> TemporalBackoffModel:
        _require_columns(frame, ("source", "destination", "timestamp"))
        if frame.empty:
            raise ValueError("Cannot fit temporal model on an empty frame")
        ordered = frame.sort_values(["timestamp"], kind="stable")
        context_counts: defaultdict[tuple[Hashable, Hashable | None], Counter[Hashable]] = defaultdict(Counter)
        source_counts: defaultdict[Hashable, Counter[Hashable]] = defaultdict(Counter)
        global_counts: Counter[Hashable] = Counter()
        previous_by_source: dict[Hashable, Hashable] = {}
        for source, destination in ordered[["source", "destination"]].itertuples(index=False):
            context = (source, previous_by_source.get(source))
            context_counts[context][destination] += 1
            source_counts[source][destination] += 1
            global_counts[destination] += 1
            previous_by_source[source] = destination

        self._context_counts = dict(context_counts)
        self._source_counts = dict(source_counts)
        self._global_counts = global_counts
        self._global_distribution = _smoothed_distribution(global_counts, global_counts, self.prior_strength)
        return self

    def candidate_distribution(
        self,
        source: Hashable,
        previous_destination: Hashable | None,
    ) -> list[Candidate]:
        self._ensure_fitted()
        context = (source, previous_destination)
        if context in self._context_counts:
            counts = self._context_counts[context]
            base = self._source_counts.get(source, self._global_counts)
        elif source in self._source_counts:
            counts = self._source_counts[source]
            base = self._global_counts
        else:
            counts = self._global_counts
            base = self._global_counts

        support = {
            item for item, _ in counts.most_common(self.top_k)
        } | {
            item for item, _ in base.most_common(self.top_k)
        }
        distribution = _smoothed_distribution(
            counts,
            base,
            self.prior_strength,
            support=support,
        )
        ranked = sorted(distribution.items(), key=lambda item: (-item[1], repr(item[0])))[: self.top_k]
        mass = sum(probability for _, probability in ranked)
        return [Candidate(action, probability / mass) for action, probability in ranked]

    def probability(
        self,
        source: Hashable,
        previous_destination: Hashable | None,
        destination: Hashable,
    ) -> float:
        distribution = self.candidate_distribution(source, previous_destination)
        lookup = {candidate.action: candidate.probability for candidate in distribution}
        return lookup.get(destination, min(candidate.probability for candidate in distribution) * 0.5)

    def iter_predictions(self, frame: pd.DataFrame) -> Iterable[TemporalPrediction]:
        _require_columns(frame, ("source", "destination", "timestamp"))
        previous_by_source: dict[Hashable, Hashable] = {}
        for source, destination in frame.sort_values(["timestamp"], kind="stable")[
            ["source", "destination"]
        ].itertuples(index=False):
            previous = previous_by_source.get(source)
            candidates = self.candidate_distribution(source, previous)
            probabilities = {candidate.action: candidate.probability for candidate in candidates}
            top = candidates[0]
            probability = probabilities.get(destination, min(probabilities.values()) * 0.5)
            yield TemporalPrediction(
                source=source,
                previous_destination=previous,
                destination=destination,
                top_destination=top.action,
                probability=probability,
                top_probability=top.probability,
                entropy_bits=_entropy([candidate.probability for candidate in candidates]),
                candidate_count=len(candidates),
                unseen_context=(source, previous) not in self._context_counts,
                unseen_destination=destination not in self._global_counts,
            )
            previous_by_source[source] = destination

    def evaluate(self, frame: pd.DataFrame) -> TemporalMetrics:
        predictions = list(self.iter_predictions(frame))
        if not predictions:
            raise ValueError("Cannot evaluate temporal model on an empty frame")
        nll = [-math.log2(max(prediction.probability, np.finfo(float).tiny)) for prediction in predictions]
        top1 = [prediction.top_destination == prediction.destination for prediction in predictions]
        mean_nll = float(np.mean(nll))
        return TemporalMetrics(
            rows=len(predictions),
            mean_nll_bits=mean_nll,
            perplexity=float(2**mean_nll),
            top1_accuracy=float(np.mean(top1)),
            mean_entropy_bits=float(np.mean([prediction.entropy_bits for prediction in predictions])),
            unseen_context_fraction=float(np.mean([prediction.unseen_context for prediction in predictions])),
            unseen_destination_fraction=float(
                np.mean([prediction.unseen_destination for prediction in predictions])
            ),
        )

    def _ensure_fitted(self) -> None:
        if not self._global_counts:
            raise ValueError("The temporal model must be fitted before use")


def evaluate_temporal_splits(
    frame: pd.DataFrame,
    *,
    prior_strength: float = 8.0,
    top_k: int = 32,
) -> dict[str, TemporalMetrics]:
    _require_columns(frame, ("source", "destination", "timestamp", "split"))
    model = TemporalBackoffModel(prior_strength=prior_strength, top_k=top_k).fit(
        frame.loc[frame["split"] == "train"]
    )
    return {
        split: model.evaluate(frame.loc[frame["split"] == split])
        for split in ("train", "validation", "test")
        if not frame.loc[frame["split"] == split].empty
    }


def metrics_to_dict(metrics: Mapping[str, TemporalMetrics]) -> dict[str, dict[str, Any]]:
    return {split: vars(record) for split, record in metrics.items()}


def _smoothed_distribution(
    counts: Counter[Hashable],
    base_counts: Counter[Hashable],
    prior_strength: float,
    *,
    support: set[Hashable] | None = None,
) -> dict[Hashable, float]:
    support = support or (set(counts) | set(base_counts))
    base_total = sum(base_counts.values())
    count_total = sum(counts.values())
    if base_total == 0 or count_total == 0:
        raise ValueError("counts must contain positive mass")
    distribution = {}
    denominator = count_total + prior_strength
    for item in support:
        base_probability = base_counts[item] / base_total
        distribution[item] = (counts[item] + prior_strength * base_probability) / denominator
    return distribution


def _entropy(probabilities: Iterable[float]) -> float:
    values = np.asarray(list(probabilities), dtype=float)
    values = values[values > 0]
    return float(-(values * np.log2(values)).sum())


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
