from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from stego.active_channel import ActiveChannelConfig, attack_packets, decode_packets, encode_packets

    config = _load_yaml(ROOT / "experiments/real_world/phase9_active_channel_reliability.yaml")
    rng = np.random.default_rng(int(config["seed"]))
    payload = [int(value) for value in rng.integers(0, 2, size=int(config["payload_bits"]))]
    channel_config = ActiveChannelConfig(**config["code"])
    encoded = encode_packets(payload, channel_config)
    rows = []
    summary: dict[str, Any] = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "payload_bits": config["payload_bits"],
        "code": config["code"],
        "attacks": {},
    }
    for attack in config["attacks"]:
        attacked = attack_packets(encoded, seed=int(config["seed"]) + int(attack["seed_offset"]), **attack["params"])
        metrics = decode_packets(
            attacked,
            payload_bits=payload,
            config=channel_config,
            deletion_rate=float(attack["params"]["deletion_rate"]),
        )
        values = vars(metrics)
        summary["attacks"][attack["name"]] = {
            "params": attack["params"],
            **values,
            "meets_corrected_ber_target": values["corrected_ber"] <= float(config["corrected_ber_target"]),
        }
        rows.append({"attack": attack["name"], **attack["params"], **summary["attacks"][attack["name"]]})
    output_dir = ROOT / "results/tables"
    (output_dir / "phase9_active_channel_reliability.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "phase9_active_channel_reliability.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "attack",
            "deletion_rate",
            "insertion_rate",
            "truncate_fraction",
            "reorder_window",
            "payload_bits",
            "packets_sent",
            "packets_received",
            "recovered_bits",
            "corrected_ber",
            "recovered_fraction",
            "authentication_success",
            "theoretical_block_failure_bound",
            "meets_corrected_ber_target",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
