from __future__ import annotations

import pandas as pd

from data.spatial import GridSpec, cell_id, summarize_cell_events, trajectory_points_to_cell_events


def test_cell_id_is_deterministic() -> None:
    spec = GridSpec(cell_size_degrees=0.01)

    assert cell_id(0.0, 0.0, spec) == cell_id(0.004, 0.004, spec)
    assert cell_id(0.0, 0.0, spec) != cell_id(0.02, 0.0, spec)


def test_trajectory_points_to_cell_events_keeps_sequence_boundaries() -> None:
    points = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s2", "s2"],
            "timestamp": [1, 2, 1, 2],
            "latitude": [0.0, 0.02, 1.0, 1.02],
            "longitude": [0.0, 0.0, 1.0, 1.0],
            "split": ["train", "train", "train", "train"],
        }
    )

    events = trajectory_points_to_cell_events(points, GridSpec(cell_size_degrees=0.01))

    assert len(events) == 2
    assert set(events["sequence_id"]) == {"s1", "s2"}
    assert summarize_cell_events(events)["events"] == 2
