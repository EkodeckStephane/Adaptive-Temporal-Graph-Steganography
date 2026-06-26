from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActiveChannelConfig:
    block_bits: int
    repetitions: int
    seed: int


@dataclass(frozen=True)
class ReliabilityMetrics:
    payload_bits: int
    packets_sent: int
    packets_received: int
    recovered_bits: int
    corrected_ber: float
    recovered_fraction: float
    authentication_success: bool
    theoretical_block_failure_bound: float


@dataclass(frozen=True)
class Packet:
    index: int
    copy: int
    payload: tuple[int, ...]
    crc32: int


def encode_packets(bits: list[int], config: ActiveChannelConfig) -> list[Packet]:
    rng = np.random.default_rng(config.seed)
    packets = []
    for index, start in enumerate(range(0, len(bits), config.block_bits)):
        payload = tuple(bits[start : start + config.block_bits])
        if len(payload) < config.block_bits:
            payload = (*payload, *([0] * (config.block_bits - len(payload))))
        crc = _crc(payload)
        for copy in range(config.repetitions):
            packets.append(Packet(index=index, copy=copy, payload=payload, crc32=crc))
    order = rng.permutation(len(packets))
    return [packets[int(position)] for position in order]


def attack_packets(
    packets: list[Packet],
    *,
    deletion_rate: float,
    insertion_rate: float,
    truncate_fraction: float,
    reorder_window: int,
    seed: int,
) -> list[Packet]:
    rng = np.random.default_rng(seed)
    kept = [packet for packet in packets if rng.random() >= deletion_rate]
    insertions = []
    for _ in range(int(math.ceil(insertion_rate * len(packets)))):
        payload = tuple(int(value) for value in rng.integers(0, 2, size=len(packets[0].payload)))
        insertions.append(
            Packet(
                index=int(rng.integers(0, max(packet.index for packet in packets) + 1)),
                copy=int(rng.integers(0, 10_000)),
                payload=payload,
                crc32=int(rng.integers(0, 2**31)),
            )
        )
    attacked = kept + insertions
    if reorder_window > 1:
        chunks = [attacked[index : index + reorder_window] for index in range(0, len(attacked), reorder_window)]
        attacked = []
        for chunk in chunks:
            attacked.extend([chunk[int(position)] for position in rng.permutation(len(chunk))])
    retained = max(1, int(math.ceil(truncate_fraction * len(attacked))))
    return attacked[:retained]


def decode_packets(
    packets: list[Packet],
    *,
    payload_bits: list[int],
    config: ActiveChannelConfig,
    deletion_rate: float,
) -> ReliabilityMetrics:
    block_count = math.ceil(len(payload_bits) / config.block_bits)
    votes: dict[int, list[tuple[int, ...]]] = {index: [] for index in range(block_count)}
    for packet in packets:
        if packet.index not in votes:
            continue
        if _crc(packet.payload) != packet.crc32:
            continue
        votes[packet.index].append(packet.payload)
    recovered = []
    for index in range(block_count):
        block_votes = votes[index]
        if not block_votes:
            recovered.extend([0] * config.block_bits)
            continue
        matrix = np.asarray(block_votes, dtype=int)
        recovered.extend((matrix.sum(axis=0) >= (len(block_votes) / 2)).astype(int).tolist())
    recovered = recovered[: len(payload_bits)]
    errors = sum(int(a != b) for a, b in zip(recovered, payload_bits))
    corrected_ber = errors / max(1, len(payload_bits))
    recovered_blocks = sum(1 for value in votes.values() if value)
    return ReliabilityMetrics(
        payload_bits=len(payload_bits),
        packets_sent=block_count * config.repetitions,
        packets_received=len(packets),
        recovered_bits=len(recovered),
        corrected_ber=corrected_ber,
        recovered_fraction=recovered_blocks / block_count,
        authentication_success=errors == 0,
        theoretical_block_failure_bound=block_count * (deletion_rate ** config.repetitions),
    )


def _crc(payload: tuple[int, ...]) -> int:
    packed = int("".join(str(bit) for bit in payload), 2).to_bytes(max(1, math.ceil(len(payload) / 8)), "big")
    return zlib.crc32(packed)
