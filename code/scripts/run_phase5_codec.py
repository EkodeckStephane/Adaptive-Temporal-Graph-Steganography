from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

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

    from stego.coding import (
        Candidate,
        bits_to_bytes,
        bytes_to_bits,
        decode_trace,
        encode_trace,
        protect_message,
        recover_message,
        repetition3_decode,
        repetition3_encode,
    )

    config = _load_yaml(ROOT / "experiments/real_world/phase5_codec.yaml")
    message = str(config["message"]).encode("utf-8")
    key = b"phase5-deterministic-key"
    protected = protect_message(
        message,
        key,
        associated_data=config["campaign"].encode("utf-8"),
        nonce=bytes.fromhex(config["coder"]["deterministic_nonce_hex"]),
    )
    protected_bits = bytes_to_bits(protected.ciphertext)
    coded_bits = repetition3_encode(protected_bits)
    candidates = [
        Candidate(action, float(probability))
        for action, probability in config["candidate_distribution"]["actions"]
    ]
    max_bits = int(config["coder"]["max_bits_per_transition"])
    required_steps = (len(coded_bits) + max_bits - 1) // max_bits
    candidate_steps = [
        candidates[step % len(candidates) :] + candidates[: step % len(candidates)]
        for step in range(required_steps)
    ]

    actions, widths, tv_distortion, kl_distortion = encode_trace(
        coded_bits,
        candidate_steps,
        max_bits_per_transition=max_bits,
    )
    decoded_coded_bits = decode_trace(actions, widths, candidate_steps)
    decoded_bits = repetition3_decode(decoded_coded_bits[: len(coded_bits)])
    recovered_ciphertext = bits_to_bytes(decoded_bits)
    recovered = recover_message(
        protected.__class__(
            ciphertext=recovered_ciphertext,
            tag=protected.tag,
            nonce=protected.nonce,
            length=protected.length,
            crc32=protected.crc32,
        ),
        key,
        associated_data=config["campaign"].encode("utf-8"),
    )
    bit_errors = sum(left != right for left, right in zip(protected_bits, decoded_bits))
    corrected_bit_errors = sum(
        left != right for left, right in zip(bytes_to_bits(message), bytes_to_bits(recovered))
    )
    summary = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "coder": config["coder"],
        "message_bytes": len(message),
        "protected_bits": len(protected_bits),
        "coded_bits": len(coded_bits),
        "transitions": len(actions),
        "unique_candidate_orders": len({tuple(candidate.action for candidate in step) for step in candidate_steps}),
        "useful_bits_per_transition": len(protected_bits) / len(actions),
        "attempted_bits_per_transition": len(coded_bits) / len(actions),
        "local_total_variation_sum": tv_distortion,
        "local_kl_bits_sum": kl_distortion,
        "mean_local_total_variation": tv_distortion / len(actions),
        "mean_local_kl_bits": kl_distortion / len(actions),
        "passive_ber": bit_errors / len(protected_bits),
        "corrected_ber": corrected_bit_errors / (8 * len(message)),
        "authentication_success": recovered == message,
        "mode": "EMBED_only_codec_contract",
    }
    output = ROOT / "results/tables/phase5_codec_baseline.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
