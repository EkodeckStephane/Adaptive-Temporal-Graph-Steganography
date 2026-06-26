from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.temporal import TemporalBackoffModel
from stego.coding import Candidate, encode_next_action, encode_next_action_range

FEATURE_COLUMNS = [
    "action_probability",
    "surprise_bits",
    "rank_fraction",
    "is_top_action",
    "entropy_bits",
    "top_probability",
    "candidate_count",
    "unseen_context",
    "unseen_destination",
    "same_as_previous",
    "self_loop",
    "log_inter_event_gap",
]

WHITE_BOX_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS,
    "encoder_capacity_bits",
    "encoder_tv_bound",
    "encoder_kl_bound",
    "embedding_feasible",
]

ORACLE_LEAKAGE_COLUMNS = [
    "bits_consumed",
    "local_total_variation",
    "local_kl_bits",
]


@dataclass(frozen=True)
class SampleConfig:
    max_bits_per_transition: int
    seed: int
    max_local_total_variation: float = 0.10
    max_local_kl_bits: float = 0.10
    min_entropy_bits: float = 1.0
    cover_when_unsafe: bool = True
    codec_backend: str = "range"
    min_encoded_probability: float = 0.0
    max_encoded_surprise_bits: float = float("inf")
    max_encoded_rank_fraction: float = 1.0
    require_encoded_top_action: bool = False
    require_encoded_self_loop: bool = False


def make_steganalysis_records(
    model: TemporalBackoffModel,
    frame: pd.DataFrame,
    *,
    split: str,
    config: SampleConfig,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(config.seed + _stable_offset(split))
    previous_by_source: dict[Hashable, Hashable] = {}
    previous_timestamp_by_source: dict[Hashable, float] = {}
    ordered = frame.sort_values(["timestamp"], kind="stable")
    for index, record in enumerate(ordered[["source", "destination", "timestamp"]].itertuples(index=False)):
        source, natural_destination, timestamp = record
        previous = previous_by_source.get(source)
        candidates = model.candidate_distribution(source, previous)
        bits = rng.integers(0, 2, size=config.max_bits_per_transition).tolist()
        encoded = _encode(bits, candidates, config=config)
        entropy = _entropy([candidate.probability for candidate in candidates])
        encoded_features = _action_position_features(
            source=source,
            action=encoded.action,
            candidates=candidates,
        )
        can_embed = (
            encoded.bits_consumed > 0
            and entropy >= config.min_entropy_bits
            and encoded.local_total_variation <= config.max_local_total_variation
            and encoded.local_kl_bits <= config.max_local_kl_bits
            and encoded_features["action_probability"] >= config.min_encoded_probability
            and encoded_features["surprise_bits"] <= config.max_encoded_surprise_bits
            and encoded_features["rank_fraction"] <= config.max_encoded_rank_fraction
            and (
                not config.require_encoded_top_action
                or bool(encoded_features["is_top_action"])
            )
            and (
                not config.require_encoded_self_loop
                or bool(encoded_features["self_loop"])
            )
        )
        stego_action = encoded.action if can_embed else natural_destination
        stego_bits = encoded.bits_consumed if can_embed else 0
        stego_tv = encoded.local_total_variation if can_embed else 0.0
        stego_kl = encoded.local_kl_bits if can_embed else 0.0
        stego_mode = "EMBED" if can_embed else "COVER"
        if not can_embed and not config.cover_when_unsafe:
            stego_action = encoded.action
            stego_bits = encoded.bits_consumed
            stego_tv = encoded.local_total_variation
            stego_kl = encoded.local_kl_bits
            stego_mode = "FORCED_EMBED"
        gap = (
            float(timestamp) - float(previous_timestamp_by_source[source])
            if source in previous_timestamp_by_source
            else 0.0
        )
        rows.append(
            {
                "split": split,
                "pair_id": index,
                "label": 0,
                "action": str(natural_destination),
                "stego_mode": "NATURAL",
                **_features(
                    source=source,
                    action=natural_destination,
                    previous=previous,
                    candidates=candidates,
                    gap=gap,
                    bits_consumed=0,
                    local_total_variation=0.0,
                    local_kl_bits=0.0,
                    encoder_capacity_bits=encoded.bits_consumed,
                    encoder_tv_bound=encoded.local_total_variation,
                    encoder_kl_bound=encoded.local_kl_bits,
                    embedding_feasible=can_embed,
                    training_destination_seen=natural_destination in model.destinations,
                    context_seen=model.has_context(source, previous),
                ),
            }
        )
        rows.append(
            {
                "split": split,
                "pair_id": index,
                "label": 1,
                "action": str(stego_action),
                "stego_mode": stego_mode,
                **_features(
                    source=source,
                    action=stego_action,
                    previous=previous,
                    candidates=candidates,
                    gap=gap,
                    bits_consumed=stego_bits,
                    local_total_variation=stego_tv,
                    local_kl_bits=stego_kl,
                    encoder_capacity_bits=encoded.bits_consumed,
                    encoder_tv_bound=encoded.local_total_variation,
                    encoder_kl_bound=encoded.local_kl_bits,
                    embedding_feasible=can_embed,
                    training_destination_seen=stego_action in model.destinations,
                    context_seen=model.has_context(source, previous),
                ),
            }
        )
        previous_by_source[source] = natural_destination
        previous_timestamp_by_source[source] = timestamp
    return pd.DataFrame(rows)


def feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return frame[FEATURE_COLUMNS].to_numpy(dtype=float), frame["label"].to_numpy(dtype=int)


def _features(
    *,
    source: Hashable,
    action: Hashable,
    previous: Hashable | None,
    candidates: Sequence[Candidate],
    gap: float,
    bits_consumed: int,
    local_total_variation: float,
    local_kl_bits: float,
    encoder_capacity_bits: int,
    encoder_tv_bound: float,
    encoder_kl_bound: float,
    embedding_feasible: bool,
    training_destination_seen: bool,
    context_seen: bool,
) -> dict[str, float | int]:
    position = _action_position_features(source=source, action=action, candidates=candidates)
    probability = position["action_probability"]
    rank_fraction = position["rank_fraction"]
    is_top_action = position["is_top_action"]
    self_loop = position["self_loop"]
    entropy = _entropy([candidate.probability for candidate in candidates])
    return {
        "action_probability": probability,
        "surprise_bits": position["surprise_bits"],
        "rank_fraction": rank_fraction,
        "is_top_action": is_top_action,
        "entropy_bits": entropy,
        "top_probability": candidates[0].probability,
        "candidate_count": len(candidates),
        "unseen_context": int(not context_seen),
        "unseen_destination": int(not training_destination_seen),
        "same_as_previous": int(previous == action),
        "self_loop": self_loop,
        "log_inter_event_gap": math.log1p(max(0.0, gap)),
        "encoder_capacity_bits": encoder_capacity_bits,
        "encoder_tv_bound": encoder_tv_bound,
        "encoder_kl_bound": encoder_kl_bound,
        "embedding_feasible": int(embedding_feasible),
        "bits_consumed": bits_consumed,
        "local_total_variation": local_total_variation,
        "local_kl_bits": local_kl_bits,
    }


def _action_position_features(
    *,
    source: Hashable,
    action: Hashable,
    candidates: Sequence[Candidate],
) -> dict[str, float | int]:
    probabilities = {candidate.action: candidate.probability for candidate in candidates}
    ranked_actions = [candidate.action for candidate in candidates]
    probability = probabilities.get(action, min(probabilities.values()) * 0.5)
    rank = ranked_actions.index(action) + 1 if action in ranked_actions else len(ranked_actions) + 1
    return {
        "action_probability": probability,
        "surprise_bits": -math.log2(max(probability, np.finfo(float).tiny)),
        "rank_fraction": rank / max(1, len(ranked_actions)),
        "is_top_action": int(rank == 1),
        "self_loop": int(source == action),
    }


def _encode(bits: Sequence[int], candidates: Sequence[Candidate], *, config: SampleConfig):
    if config.codec_backend == "range":
        return encode_next_action_range(bits, candidates, max_bits=config.max_bits_per_transition)
    if config.codec_backend == "quantized":
        return encode_next_action(bits, candidates, max_bits=config.max_bits_per_transition)
    raise ValueError(f"Unknown codec backend: {config.codec_backend}")


def _entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0]
    return float(-(values * np.log2(values)).sum())


def _stable_offset(value: str) -> int:
    return sum(ord(character) for character in value)
