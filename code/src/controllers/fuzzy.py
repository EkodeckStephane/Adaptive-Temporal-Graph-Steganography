from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["EMBED", "COVER", "PAUSE", "STOP"]


@dataclass(frozen=True)
class ControllerInputs:
    predictive_entropy: float
    calibration_uncertainty: float
    steganalysis_risk: float
    payload_pressure: float
    dead_end_risk: float
    channel_fragility: float


@dataclass(frozen=True)
class ControlDecision:
    mode: Mode
    local_payload_bits: int
    rate_score: float
    abstention_score: float


@dataclass(frozen=True)
class FuzzyWeights:
    """Tunable Takagi--Sugeno consequence weights for the fuzzy controller.

    All coefficients are in [0, 1] and govern how strongly each fuzzy rule
    contributes to the opportunity, cover, pause and stop signals.
    """

    opportunity_entropy_weight: float = 1.0
    opportunity_payload_weight: float = 0.65
    cover_entropy_weight: float = 1.0
    pause_risk_weight: float = 1.0
    stop_dead_end_weight: float = 1.0
    abstention_cover_weight: float = 0.45
    abstention_pause_weight: float = 0.40
    abstention_stop_weight: float = 0.65


class FuzzyRateController:
    def __init__(
        self,
        *,
        max_bits_per_transition: int = 4,
        stop_threshold: float = 0.92,
        weights: FuzzyWeights | None = None,
    ) -> None:
        if max_bits_per_transition < 1:
            raise ValueError("max_bits_per_transition must be at least one")
        if not 0 < stop_threshold <= 1:
            raise ValueError("stop_threshold must be in (0, 1]")
        self.max_bits_per_transition = max_bits_per_transition
        self.stop_threshold = stop_threshold
        self.weights = weights or FuzzyWeights()

    def decide(self, inputs: ControllerInputs) -> ControlDecision:
        values = _normalized(inputs)
        entropy_high = _ramp_up(values.predictive_entropy, 0.35, 0.85)
        entropy_low = _ramp_down(values.predictive_entropy, 0.15, 0.55)
        payload_high = _ramp_up(values.payload_pressure, 0.30, 0.85)
        risk_high = _ramp_up(values.steganalysis_risk, 0.35, 0.90)
        uncertainty_high = _ramp_up(values.calibration_uncertainty, 0.25, 0.85)
        fragility_high = _ramp_up(values.channel_fragility, 0.30, 0.85)
        dead_end_high = _ramp_up(values.dead_end_risk, 0.45, 0.95)

        w = self.weights
        risk_block = max(risk_high, uncertainty_high, fragility_high)
        opportunity = (
            entropy_high
            * (1.0 - risk_block)
            * (
                (1.0 - w.opportunity_payload_weight)
                + w.opportunity_payload_weight * payload_high
            )
        )
        cover_weight = max(entropy_low, risk_block) * (1.0 - dead_end_high)
        pause_weight = max(risk_block * payload_high, fragility_high) * (1.0 - dead_end_high)
        stop_weight = max(dead_end_high, risk_block * (1.0 - payload_high))

        abstention_score = _clip(
            w.abstention_cover_weight * cover_weight
            + w.abstention_pause_weight * pause_weight
            + w.abstention_stop_weight * stop_weight
        )
        rate_score = _clip(opportunity)
        bits = int(round(rate_score * self.max_bits_per_transition))
        if bits < 1 and rate_score > 0.20:
            bits = 1

        if stop_weight >= self.stop_threshold:
            return ControlDecision("STOP", 0, rate_score, abstention_score)
        if pause_weight > max(opportunity, cover_weight):
            return ControlDecision("PAUSE", 0, rate_score, abstention_score)
        if bits >= 1 and opportunity >= max(cover_weight, pause_weight):
            return ControlDecision("EMBED", bits, rate_score, abstention_score)
        return ControlDecision("COVER", 0, rate_score, abstention_score)


def fixed_entropy_threshold(
    inputs: ControllerInputs,
    *,
    entropy_threshold: float,
    max_bits_per_transition: int,
) -> ControlDecision:
    values = _normalized(inputs)
    if values.dead_end_risk >= 0.95:
        return ControlDecision("STOP", 0, 0.0, 1.0)
    if values.predictive_entropy < entropy_threshold:
        return ControlDecision("COVER", 0, values.predictive_entropy, 1.0 - values.predictive_entropy)
    bits = max(1, int(round(values.predictive_entropy * max_bits_per_transition)))
    return ControlDecision("EMBED", min(bits, max_bits_per_transition), values.predictive_entropy, 0.0)


def _normalized(inputs: ControllerInputs) -> ControllerInputs:
    return ControllerInputs(
        predictive_entropy=_clip(inputs.predictive_entropy),
        calibration_uncertainty=_clip(inputs.calibration_uncertainty),
        steganalysis_risk=_clip(inputs.steganalysis_risk),
        payload_pressure=_clip(inputs.payload_pressure),
        dead_end_risk=_clip(inputs.dead_end_risk),
        channel_fragility=_clip(inputs.channel_fragility),
    )


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _ramp_up(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def _ramp_down(value: float, low: float, high: float) -> float:
    return 1.0 - _ramp_up(value, low, high)
