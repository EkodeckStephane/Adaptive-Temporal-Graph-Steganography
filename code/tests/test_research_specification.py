from pathlib import Path

from stego.specification import REQUIRED_ACTIONS, validate_research_specification


def test_research_specification_is_complete() -> None:
    root = Path(__file__).resolve().parents[2]
    summary = validate_research_specification(root)

    assert set(summary.actions) == REQUIRED_ACTIONS
    assert len(summary.level_a_datasets) >= 3
    assert {"porto_taxi", "t_drive", "geolife"} <= set(summary.level_a_datasets)
