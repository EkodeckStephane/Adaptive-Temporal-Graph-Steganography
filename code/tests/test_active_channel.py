from __future__ import annotations

from stego.active_channel import ActiveChannelConfig, attack_packets, decode_packets, encode_packets


def test_active_channel_recovers_under_bounded_deletion() -> None:
    payload = [index % 2 for index in range(128)]
    config = ActiveChannelConfig(block_bits=16, repetitions=7, seed=11)
    encoded = encode_packets(payload, config)
    attacked = attack_packets(
        encoded,
        deletion_rate=0.05,
        insertion_rate=0.02,
        truncate_fraction=1.0,
        reorder_window=8,
        seed=12,
    )
    metrics = decode_packets(attacked, payload_bits=payload, config=config, deletion_rate=0.05)

    assert metrics.corrected_ber == 0.0
    assert metrics.authentication_success
