from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sequence_dataset(dataset: dict) -> dict:
    identifiers = {}
    ranges = {}
    rows = {}
    for split, record in dataset["processed_files"].items():
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}")
        table = pq.read_table(path, columns=["sequence_id", "timestamp", "split"])
        if pc.unique(table["split"]).to_pylist() != [split]:
            raise ValueError(f"Invalid split labels in {path}")
        identifiers[split] = set(pc.unique(table["sequence_id"]).to_pylist())
        ranges[split] = (
            pc.min(table["timestamp"]).as_py(),
            pc.max(table["timestamp"]).as_py(),
        )
        rows[split] = len(table)

    if identifiers["train"] & identifiers["validation"]:
        raise ValueError("Train/validation sequence leakage")
    if identifiers["train"] & identifiers["test"]:
        raise ValueError("Train/test sequence leakage")
    if identifiers["validation"] & identifiers["test"]:
        raise ValueError("Validation/test sequence leakage")
    if not (
        ranges["train"][1]
        < ranges["validation"][0]
        <= ranges["validation"][1]
        < ranges["test"][0]
    ):
        raise ValueError(f"Non-causal temporal ranges: {ranges}")
    return {
        "rows": rows,
        "ranges": ranges,
        "sequence_counts": {key: len(value) for key, value in identifiers.items()},
    }


def main() -> None:
    results = json.loads(
        (ROOT / "results/tables/phase3_dataset_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    validation = {}
    for dataset in results["datasets"]:
        if "processed_files" in dataset:
            validation[dataset["id"]] = validate_sequence_dataset(dataset)
        else:
            path = ROOT / dataset["processed_path"]
            if sha256(path) != dataset["processed_sha256"]:
                raise ValueError(f"Checksum mismatch for {path}")
            frame = pq.read_table(path, columns=["timestamp", "split"])
            timestamps = frame["timestamp"].to_numpy()
            if not np.all(np.diff(timestamps) >= 0):
                raise ValueError("TGB events are not sorted chronologically")
            validation[dataset["id"]] = {"rows": len(frame)}
    output = ROOT / "results/tables/phase3_validation.json"
    output.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
