from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import pyarrow.parquet as pq

from data.adapters import read_geolife_plt, read_porto_csv, read_tdrive_directory
from data.geolife_stream import stream_geolife_archive
from data.schema import validate_events
from data.splits import assign_causal_splits
from data.statistics import describe_temporal_events


def test_causal_split_keeps_timestamp_ties_together() -> None:
    frame = pd.DataFrame(
        {
            "event_id": range(10),
            "source": [0, 0, 1, 1, 2, 2, 0, 1, 2, 3],
            "destination": [1, 2, 2, 3, 3, 4, 4, 4, 5, 5],
            "timestamp": [0, 1, 2, 3, 4, 5, 6, 6, 7, 8],
            "label": 0,
            "sequence_id": pd.NA,
        }
    )
    validate_events(frame)
    split, cutoffs = assign_causal_splits(frame, train_fraction=0.6, validation_fraction=0.2)

    assert split.groupby("timestamp")["split"].nunique().max() == 1
    assert split.loc[split["split"] == "train", "timestamp"].max() <= cutoffs.train_end
    assert split.loc[split["split"] == "test", "timestamp"].min() > cutoffs.validation_end


def test_statistics_report_temporal_structure() -> None:
    frame = pd.DataFrame(
        {
            "event_id": range(8),
            "source": [0, 0, 0, 1, 1, 2, 2, 3],
            "destination": [1, 1, 2, 0, 2, 0, 3, 2],
            "timestamp": range(8),
            "label": 0,
            "sequence_id": pd.NA,
        }
    )
    split, _ = assign_causal_splits(frame, train_fraction=0.5, validation_fraction=0.25)
    stats = describe_temporal_events(split)
    assert stats["events"] == 8
    assert stats["repeated_edge_fraction"] > 0
    assert stats["conditional_destination_entropy_bits"] >= 0
    assert set(stats["split_counts"]) == {"train", "validation", "test"}


def test_tdrive_adapter(tmp_path: Path) -> None:
    (tmp_path / "1.txt").write_text(
        "1,2008-02-02 15:36:08,116.51172,39.92123\n"
        "1,2008-02-02 15:36:23,116.51135,39.93883\n",
        encoding="utf-8",
    )
    points = read_tdrive_directory(tmp_path)
    assert list(points["sequence_id"].unique()) == ["1"]
    assert len(points) == 2


def test_geolife_adapter(tmp_path: Path) -> None:
    source = tmp_path / "000" / "Trajectory" / "20081023025304.plt"
    source.parent.mkdir(parents=True)
    source.write_text(
        "Geolife trajectory\nWGS 84\nAltitude is in Feet\nReserved 3\n0,2,255,My Track,0,0,2,8421376\n0\n"
        "39.984702,116.318417,0,492,39744.1201851852,2008-10-23,02:53:04\n",
        encoding="utf-8",
    )
    points = read_geolife_plt(source)
    assert len(points) == 1
    assert points.iloc[0]["entity_id"] == "000"


def test_porto_adapter(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    source.write_text(
        'TRIP_ID,TAXI_ID,TIMESTAMP,MISSING_DATA,POLYLINE\n'
        'trip-1,42,1400000000,False,"[[-8.6,41.1],[-8.61,41.11]]"\n',
        encoding="utf-8",
    )
    points = read_porto_csv(source)
    assert len(points) == 2
    assert (points["timestamp"].diff().dropna().dt.total_seconds() == 15).all()


def test_geolife_stream_split_keeps_trajectories_whole(tmp_path: Path) -> None:
    archive_path = tmp_path / "geolife.zip"
    header = "a\nb\nc\nd\ne\nf\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for index, (stamp, excel_day) in enumerate(
            (
                ("20080101000000", 39448.0),
                ("20090101000000", 39814.0),
                ("20100101000000", 40179.0),
                ("20110101000000", 40544.0),
                ("20120101000000", 40909.0),
                ("20130101000000", 41275.0),
            )
        ):
            body = (
                header
                + f"39.{index},116.{index},0,10,{excel_day},2008-01-01,00:00:00\n"
                + f"39.{index},116.{index},0,10,{excel_day + 0.0001},2008-01-01,00:00:09\n"
            )
            archive.writestr(
                f"Geolife Trajectories 1.3/Data/{index:03d}/Trajectory/{stamp}.plt",
                body,
            )
    output = tmp_path / "processed"
    stats = stream_geolife_archive(
        archive_path,
        output,
        train_fraction=1 / 3,
        validation_fraction=1 / 3,
        batch_points=2,
    )
    assert stats["split_trajectories"] == {"train": 1, "validation": 1, "test": 2}
    assert stats["boundary_crossing_trajectories"] == 2
    for split in ("train", "validation", "test"):
        frame = pq.read_table(output / f"{split}.parquet").to_pandas()
        assert frame["sequence_id"].nunique() >= 1
        assert frame["split"].unique().tolist() == [split]
