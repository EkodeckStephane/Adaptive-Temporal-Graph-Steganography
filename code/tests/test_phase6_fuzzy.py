from __future__ import annotations

from controllers.fuzzy import ControllerInputs, FuzzyRateController, fixed_entropy_threshold


def test_fuzzy_controller_embeds_when_entropy_is_high_and_risk_is_low() -> None:
    controller = FuzzyRateController(max_bits_per_transition=4)

    decision = controller.decide(
        ControllerInputs(
            predictive_entropy=0.9,
            calibration_uncertainty=0.05,
            steganalysis_risk=0.05,
            payload_pressure=0.9,
            dead_end_risk=0.0,
            channel_fragility=0.05,
        )
    )

    assert decision.mode == "EMBED"
    assert decision.local_payload_bits >= 1


def test_fuzzy_controller_abstains_under_high_risk() -> None:
    controller = FuzzyRateController(max_bits_per_transition=4)

    decision = controller.decide(
        ControllerInputs(
            predictive_entropy=0.9,
            calibration_uncertainty=0.8,
            steganalysis_risk=0.9,
            payload_pressure=0.7,
            dead_end_risk=0.1,
            channel_fragility=0.8,
        )
    )

    assert decision.mode in {"COVER", "PAUSE"}
    assert decision.local_payload_bits == 0


def test_fuzzy_controller_stops_at_dead_end() -> None:
    controller = FuzzyRateController(max_bits_per_transition=4)

    decision = controller.decide(
        ControllerInputs(
            predictive_entropy=0.4,
            calibration_uncertainty=0.1,
            steganalysis_risk=0.1,
            payload_pressure=0.0,
            dead_end_risk=1.0,
            channel_fragility=0.1,
        )
    )

    assert decision.mode == "STOP"


def test_fixed_threshold_baseline_is_deterministic() -> None:
    inputs = ControllerInputs(0.7, 0.0, 0.0, 0.5, 0.0, 0.0)

    first = fixed_entropy_threshold(inputs, entropy_threshold=0.5, max_bits_per_transition=4)
    second = fixed_entropy_threshold(inputs, entropy_threshold=0.5, max_bits_per_transition=4)

    assert first == second
    assert first.mode == "EMBED"
