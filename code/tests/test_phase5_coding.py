from __future__ import annotations

import pytest

from stego.coding import (
    AuthenticationError,
    Candidate,
    bits_to_bytes,
    bytes_to_bits,
    decode_action_bits_range,
    decode_trace,
    encode_next_action,
    encode_next_action_range,
    encode_trace,
    protect_message,
    recover_message,
    reed_solomon_decode,
    reed_solomon_encode,
    repetition3_decode,
    repetition3_encode,
)


def candidate_steps(count: int) -> list[list[Candidate]]:
    return [
        [
            Candidate("a", 0.40),
            Candidate("b", 0.30),
            Candidate("c", 0.20),
            Candidate("d", 0.10),
        ]
        for _ in range(count)
    ]


def test_protection_authenticates_and_recovers_message() -> None:
    protected = protect_message(b"phase-five", b"shared-key", associated_data=b"trace-1")

    assert recover_message(protected, b"shared-key", associated_data=b"trace-1") == b"phase-five"
    tampered = protected.__class__(
        ciphertext=protected.ciphertext[:-1] + bytes([protected.ciphertext[-1] ^ 1]),
        tag=protected.tag,
        nonce=protected.nonce,
        length=protected.length,
        crc32=protected.crc32,
    )
    with pytest.raises(AuthenticationError):
        recover_message(tampered, b"shared-key", associated_data=b"trace-1")


def test_repetition3_corrects_one_bit_per_block() -> None:
    encoded = repetition3_encode([1, 0, 1])
    encoded[1] = 0
    encoded[4] = 1

    assert repetition3_decode(encoded) == [1, 0, 1]


def test_trace_coding_roundtrip_is_exact() -> None:
    bits = bytes_to_bits(b"A")
    steps = candidate_steps(4)
    steps[1] = list(reversed(steps[1]))
    steps[3] = [steps[3][1], steps[3][2], steps[3][3], steps[3][0]]
    actions, widths, tv_distortion, kl_distortion = encode_trace(
        bits,
        steps,
        max_bits_per_transition=2,
    )

    decoded = decode_trace(actions, widths, steps)

    assert bits_to_bytes(decoded) == b"A"
    assert widths == [2, 2, 2, 2]
    assert tv_distortion >= 0
    assert kl_distortion >= 0


def test_range_coder_roundtrip_is_exact() -> None:
    candidates = [
        Candidate("a", 0.50),
        Candidate("b", 0.30),
        Candidate("c", 0.20),
        Candidate("d", 0.10),
    ]
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    encoded = encode_next_action_range(bits, candidates, max_bits=2)
    decoded = decode_action_bits_range(
        encoded.action, candidates, bits_consumed=encoded.bits_consumed
    )

    assert decoded == bits[: encoded.bits_consumed]
    assert encoded.bits_consumed == 2


def test_range_coder_reduces_distortion_for_non_uniform_cover() -> None:
    candidates = [
        Candidate("a", 0.70),
        Candidate("b", 0.20),
        Candidate("c", 0.05),
        Candidate("d", 0.05),
    ]
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    quantized = encode_next_action(bits, candidates, max_bits=2)
    ranged = encode_next_action_range(bits, candidates, max_bits=2)

    assert ranged.local_kl_bits < quantized.local_kl_bits
    assert ranged.local_total_variation < quantized.local_total_variation


def test_reed_solomon_corrects_byte_errors() -> None:
    bits = bytes_to_bits(b"phase-five-codec")
    encoded = reed_solomon_encode(bits, nsym=10)
    # Flip a few bits in a single byte
    corrupted = list(encoded)
    corrupted[3] ^= 1
    corrupted[4] ^= 1
    corrupted[5] ^= 1

    decoded = reed_solomon_decode(corrupted, nsym=10)

    assert bits_to_bytes(decoded) == b"phase-five-codec"
