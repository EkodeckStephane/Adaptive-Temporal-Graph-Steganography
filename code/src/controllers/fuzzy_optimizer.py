from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from controllers.fuzzy import ControllerInputs, FuzzyRateController, FuzzyWeights


def _weights_from_vector(vector: Sequence[float]) -> FuzzyWeights:
    return FuzzyWeights(
        opportunity_entropy_weight=float(vector[0]),
        opportunity_payload_weight=float(vector[1]),
        cover_entropy_weight=float(vector[2]),
        pause_risk_weight=float(vector[3]),
        stop_dead_end_weight=float(vector[4]),
        abstention_cover_weight=float(vector[5]),
        abstention_pause_weight=float(vector[6]),
        abstention_stop_weight=float(vector[7]),
    )


def _vector_from_weights(weights: FuzzyWeights) -> np.ndarray:
    return np.array(
        [
            weights.opportunity_entropy_weight,
            weights.opportunity_payload_weight,
            weights.cover_entropy_weight,
            weights.pause_risk_weight,
            weights.stop_dead_end_weight,
            weights.abstention_cover_weight,
            weights.abstention_pause_weight,
            weights.abstention_stop_weight,
        ],
        dtype=float,
    )


def _risk_penalty(inputs: ControllerInputs) -> float:
    """Proxy penalty for embedding in risky contexts."""
    return (
        0.4 * inputs.steganalysis_risk
        + 0.3 * inputs.channel_fragility
        + 0.2 * inputs.calibration_uncertainty
        + 0.1 * inputs.dead_end_risk
    )


def evaluate_weights(
    vector: Sequence[float],
    cases: Sequence[ControllerInputs],
    *,
    max_bits_per_transition: int,
    stop_threshold: float,
    risk_lambda: float = 2.0,
    abstention_target: float = 0.85,
    abstention_lambda: float = 1.0,
) -> float:
    """Scalar objective for fuzzy-weight optimization.

    Maximizes payload while penalizing risky embeddings and deviations from a
    target abstention rate. Higher values are better.
    """
    weights = _weights_from_vector(vector)
    controller = FuzzyRateController(
        max_bits_per_transition=max_bits_per_transition,
        stop_threshold=stop_threshold,
        weights=weights,
    )
    decisions = [controller.decide(case) for case in cases]
    total = len(decisions)
    if total == 0:
        return 0.0

    mean_payload = sum(d.local_payload_bits for d in decisions) / total
    risk_penalty = sum(
        d.local_payload_bits * _risk_penalty(case)
        for d, case in zip(decisions, cases)
    ) / total
    abstention_rate = sum(d.mode in {"COVER", "PAUSE", "STOP"} for d in decisions) / total
    abstention_penalty = abs(abstention_rate - abstention_target)

    return mean_payload - risk_lambda * risk_penalty - abstention_lambda * abstention_penalty


def optimize_fuzzy_weights(
    cases: Sequence[ControllerInputs],
    *,
    max_bits_per_transition: int = 4,
    stop_threshold: float = 0.92,
    risk_lambda: float = 2.0,
    abstention_target: float = 0.85,
    abstention_lambda: float = 1.0,
    seed: int = 20260626,
) -> tuple[FuzzyWeights, float]:
    """Optimize Takagi--Sugeno consequence weights on a validation grid.

    Uses differential evolution with bounds [0, 1] for each weight. Returns the
    best weights and the corresponding objective value.
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        raise ImportError("scipy is required for fuzzy-weight optimization") from exc

    bounds = [(0.0, 1.0) for _ in range(8)]
    result = differential_evolution(
        lambda vector: -evaluate_weights(
            vector,
            cases,
            max_bits_per_transition=max_bits_per_transition,
            stop_threshold=stop_threshold,
            risk_lambda=risk_lambda,
            abstention_target=abstention_target,
            abstention_lambda=abstention_lambda,
        ),
        bounds,
        maxiter=100,
        seed=seed,
        polish=True,
    )
    best_weights = _weights_from_vector(result.x)
    return best_weights, -result.fun


def default_weights() -> FuzzyWeights:
    """Return the hand-tuned baseline weights."""
    return FuzzyWeights()
