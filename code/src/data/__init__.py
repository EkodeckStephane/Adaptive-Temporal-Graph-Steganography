"""Dataset loading, validation, temporal splitting, and provenance."""

from .adapters import (
    read_bipartite_interactions,
    read_geolife_plt,
    read_porto_csv,
    read_tdrive_directory,
    read_tgb_wiki,
)
from .schema import (
    EVENT_COLUMNS,
    TRAJECTORY_COLUMNS,
    validate_events,
    validate_trajectory_points,
)
from .geolife_stream import stream_geolife_archive
from .splits import TemporalCutoffs, assign_causal_splits
from .statistics import describe_temporal_events
from .tdrive_stream import stream_tdrive_archives

__all__ = [
    "EVENT_COLUMNS",
    "TRAJECTORY_COLUMNS",
    "TemporalCutoffs",
    "assign_causal_splits",
    "describe_temporal_events",
    "read_geolife_plt",
    "read_bipartite_interactions",
    "read_porto_csv",
    "read_tdrive_directory",
    "read_tgb_wiki",
    "stream_geolife_archive",
    "stream_tdrive_archives",
    "validate_events",
    "validate_trajectory_points",
]
