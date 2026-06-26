from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))

from data.adapters import read_bipartite_interactions, read_tgb_wiki  # noqa: E402
from data.geolife_stream import stream_geolife_archive  # noqa: E402
from data.schema import validate_events  # noqa: E402
from data.splits import assign_causal_splits  # noqa: E402
from data.statistics import describe_temporal_events  # noqa: E402
from data.tdrive_stream import stream_tdrive_archives  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def process_tgb_wiki(entry: dict[str, Any], split_config: dict[str, Any]) -> dict[str, Any]:
    archive = ROOT / entry["raw_archive"]
    source = ROOT / entry["raw_events"]
    output = ROOT / entry["processed"]
    if not archive.exists() or not source.exists():
        raise FileNotFoundError("Run the official TGB download before preprocessing")
    actual_checksum = sha256(archive)
    if actual_checksum.lower() != entry["archive_sha256"].lower():
        raise ValueError(
            f"Archive checksum mismatch: expected {entry['archive_sha256']}, got {actual_checksum}"
        )

    frame = read_tgb_wiki(source)
    frame, cutoffs = assign_causal_splits(
        frame,
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
    )
    validate_events(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")

    for split in ("train", "validation", "test"):
        frame.loc[frame["split"] == split].to_parquet(
            output.with_name(f"{split}.parquet"),
            index=False,
            compression="zstd",
        )

    statistics = describe_temporal_events(frame)
    processed_checksum = sha256(output)
    return {
        "id": entry["id"],
        "version": entry["version"],
        "level": entry["level"],
        "license": entry["license"],
        "observed_trajectories": entry["observed_trajectories"],
        "source_url": entry["source_url"],
        "raw_archive": entry["raw_archive"],
        "raw_archive_bytes": archive.stat().st_size,
        "raw_sha256": actual_checksum,
        "processed_path": entry["processed"],
        "processed_bytes": output.stat().st_size,
        "processed_sha256": processed_checksum,
        "canonical_columns": list(frame.columns),
        "cutoffs": {
            "train_end": cutoffs.train_end,
            "validation_end": cutoffs.validation_end,
            "actual_train_fraction": cutoffs.train_fraction,
            "actual_validation_fraction": cutoffs.validation_fraction,
            "actual_test_fraction": cutoffs.test_fraction,
            "tie_policy": split_config["tie_policy"],
        },
        "statistics": statistics,
    }


def process_geolife(entry: dict[str, Any], split_config: dict[str, Any]) -> dict[str, Any]:
    archive = ROOT / entry["raw_archive"]
    output = ROOT / entry["processed"]
    actual_checksum = sha256(archive)
    if actual_checksum.lower() != entry["archive_sha256"].lower():
        raise ValueError(
            f"Archive checksum mismatch: expected {entry['archive_sha256']}, got {actual_checksum}"
        )
    statistics = stream_geolife_archive(
        archive,
        output,
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
    )
    files = {
        split: {
            "path": (output / f"{split}.parquet").relative_to(ROOT).as_posix(),
            "bytes": (output / f"{split}.parquet").stat().st_size,
            "sha256": sha256(output / f"{split}.parquet"),
        }
        for split in ("train", "validation", "test")
    }
    statistics.pop("output_files")
    return {
        "id": entry["id"],
        "version": entry["version"],
        "level": entry["level"],
        "license": entry["license"],
        "observed_trajectories": entry["observed_trajectories"],
        "source_url": entry["source_url"],
        "raw_archive": entry["raw_archive"],
        "raw_archive_bytes": archive.stat().st_size,
        "raw_sha256": actual_checksum,
        "processed_path": entry["processed"],
        "processed_files": files,
        "canonical_columns": [
            "point_id",
            "sequence_id",
            "entity_id",
            "timestamp",
            "latitude",
            "longitude",
            "split",
        ],
        "cutoffs": {
            "train_end": statistics["train_end"],
            "validation_end": statistics["validation_end"],
            "tie_policy": "whole_trajectory_same_split",
        },
        "statistics": statistics,
    }


def process_tdrive(entry: dict[str, Any], split_config: dict[str, Any]) -> dict[str, Any]:
    raw_directory = ROOT / entry["raw_directory"]
    archives = sorted(raw_directory.glob("*.zip"))
    if not archives:
        raise FileNotFoundError("No official T-Drive archive was found")
    archive_records = [
        {
            "path": archive.relative_to(ROOT).as_posix(),
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        }
        for archive in archives
    ]
    output = ROOT / entry["processed"]
    statistics = stream_tdrive_archives(
        archives,
        output,
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
        gap_seconds=int(entry["session_gap_seconds"]),
        coordinate_bounds=tuple(float(value) for value in entry["coordinate_bounds"]),
    )
    files = {
        split: {
            "path": (output / f"{split}.parquet").relative_to(ROOT).as_posix(),
            "bytes": (output / f"{split}.parquet").stat().st_size,
            "sha256": sha256(output / f"{split}.parquet"),
        }
        for split in ("train", "validation", "test")
    }
    return {
        "id": entry["id"],
        "version": entry["version"],
        "level": entry["level"],
        "license": entry["license"],
        "observed_trajectories": entry["observed_trajectories"],
        "source_url": entry["source_url"],
        "raw_archives": archive_records,
        "processed_path": entry["processed"],
        "processed_files": files,
        "canonical_columns": [
            "point_id",
            "sequence_id",
            "entity_id",
            "timestamp",
            "latitude",
            "longitude",
            "split",
        ],
        "cutoffs": {
            "train_end": statistics["train_end"],
            "validation_end": statistics["validation_end"],
            "tie_policy": "whole_session_same_split_with_boundary_quarantine",
        },
        "statistics": statistics,
    }


def process_jodie(entry: dict[str, Any], split_config: dict[str, Any]) -> dict[str, Any]:
    source = ROOT / entry["raw_events"]
    actual_checksum = sha256(source)
    if actual_checksum.lower() != entry["raw_sha256"].lower():
        raise ValueError(
            f"Source checksum mismatch: expected {entry['raw_sha256']}, got {actual_checksum}"
        )
    frame = read_bipartite_interactions(source)
    frame, cutoffs = assign_causal_splits(
        frame,
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
    )
    output = ROOT / entry["processed"]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")
    for split in ("train", "validation", "test"):
        frame.loc[frame["split"] == split].to_parquet(
            output.with_name(f"{split}.parquet"),
            index=False,
            compression="zstd",
        )
    return {
        "id": entry["id"],
        "version": entry["version"],
        "level": entry["level"],
        "license": entry["license"],
        "observed_trajectories": entry["observed_trajectories"],
        "source_url": entry["source_url"],
        "raw_archive": entry["raw_events"],
        "raw_archive_bytes": source.stat().st_size,
        "raw_sha256": actual_checksum,
        "processed_path": entry["processed"],
        "processed_bytes": output.stat().st_size,
        "processed_sha256": sha256(output),
        "canonical_columns": list(frame.columns),
        "cutoffs": {
            "train_end": cutoffs.train_end,
            "validation_end": cutoffs.validation_end,
            "actual_train_fraction": cutoffs.train_fraction,
            "actual_validation_fraction": cutoffs.validation_fraction,
            "actual_test_fraction": cutoffs.test_fraction,
            "tie_policy": split_config["tie_policy"],
        },
        "statistics": describe_temporal_events(frame),
    }


def write_provenance(dataset: dict[str, Any], path: Path, date: str) -> None:
    if dataset["id"] == "geolife":
        scope_notes = [
            "Microsoft Research License Agreement restricts use to non-commercial purposes.",
            "The archive contains 18,670 PLT files while the bundled guide reports 17,621 trajectories.",
            "Spatial graph discretization is fitted in downstream train-only preprocessing.",
        ]
    elif dataset["id"] in {"tgbl-wiki", "mooc", "lastfm"}:
        scope_notes = [
            "Level-B interaction stream used as an interaction-event benchmark.",
            "Node namespaces separate users from Wikipedia pages.",
            "The canonical table focuses on source, destination, timestamp, and split-ready metadata; edge features remain available in the raw archive.",
        ]
    else:
        scope_notes = [
            "Redistribution terms are tracked from the Microsoft Research source page.",
            "Sessions are operationally segmented at day changes or gaps above 20 minutes.",
            "Spatial graph discretization is fitted in downstream train-only preprocessing.",
        ]
    provenance = {
        "dataset": dataset["id"],
        "version": dataset["version"],
        "retrieved_on": date,
        "source_url": dataset["source_url"],
        "license": dataset["license"],
        "level": dataset["level"],
        "observed_trajectories": dataset["observed_trajectories"],
        "processed": {"path": dataset["processed_path"],
                      "script": "code/scripts/run_phase3_data_pipeline.py",
                      "columns": dataset["canonical_columns"]},
        "split": dataset["cutoffs"],
        "statistics": dataset["statistics"],
        "scope_notes": scope_notes,
    }
    if "raw_archives" in dataset:
        provenance["raw"] = {"archives": dataset["raw_archives"]}
    else:
        provenance["raw"] = {
            "path": dataset["raw_archive"],
            "bytes": dataset["raw_archive_bytes"],
            "sha256": dataset["raw_sha256"],
        }
    if "processed_files" in dataset:
        provenance["processed"]["files"] = dataset["processed_files"]
    else:
        provenance["processed"]["bytes"] = dataset["processed_bytes"]
        provenance["processed"]["sha256"] = dataset["processed_sha256"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(provenance, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/real_world/phase3_data_pipeline.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/tables/phase3_dataset_statistics.json",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets = []
    for entry in config["datasets"]:
        if entry["adapter"] == "tgb_wiki":
            datasets.append(process_tgb_wiki(entry, config["splits"]))
        elif entry["adapter"] == "geolife_archive":
            datasets.append(process_geolife(entry, config["splits"]))
        elif entry["adapter"] == "tdrive_archives":
            datasets.append(process_tdrive(entry, config["splits"]))
        elif entry["adapter"] == "jodie_csv":
            datasets.append(process_jodie(entry, config["splits"]))
        else:
            raise ValueError(f"Unsupported phase-3 adapter: {entry['adapter']}")
    result = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for dataset in datasets:
        write_provenance(
            dataset,
            ROOT / f"datasets/metadata/{dataset['id'].replace('-', '_')}_{dataset['version'].replace('.', '_')}.yaml",
            result["date"],
        )
    pd.json_normalize(
        [
            {
                "dataset": item["id"],
                "version": item["version"],
                **item["statistics"],
                **item["cutoffs"],
            }
            for item in datasets
        ]
    ).to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
