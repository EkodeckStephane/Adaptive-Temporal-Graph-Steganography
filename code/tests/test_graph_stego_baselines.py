import random

import networkx as nx
import pytest

from baselines.graph_stego import (
    BindCapacityError,
    adabind_encode,
    bind_decode,
    bind_encode,
    bynis_decode,
    bynis_encode,
    edge_type_counts,
)


def balanced_cover() -> list[tuple[int, int]]:
    rng = random.Random(1)
    cover: list[tuple[int, int]] = []
    for _ in range(2):
        cover = [
            (source, target)
            for source in range(12)
            for target in range(12)
            if source != target and rng.random() < 0.35
        ]
    return cover


def test_bind_roundtrip_is_deterministic_and_preserves_edges() -> None:
    cover = balanced_cover()
    message = b"A"
    first = bind_encode(cover, message, seed=1234)
    second = bind_encode(cover, message, seed=1234)

    assert first.edges == second.edges
    assert sorted(first.edges) == sorted(cover)
    assert bind_decode(first.edges, seed=1234) == message


def test_bind_rejects_an_unavailable_edge_type() -> None:
    cover = [(0, 1), (1, 2), (2, 3)]
    with pytest.raises(BindCapacityError):
        bind_encode(cover, b"\xff", seed=7)


def test_adabind_adds_edges_then_roundtrips() -> None:
    cover = [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 3),
        (3, 4),
        (4, 0),
        (4, 2),
        (0, 3),
    ]
    result = adabind_encode(
        cover,
        b"\xa5",
        seed=19,
        max_iterations=200,
        sampled_edges=10,
        extra_target_edges=0,
    )

    assert result.added_edges
    assert len(result.augmented_cover_edges) == len(cover) + len(result.added_edges)
    assert bind_decode(result.edges, seed=19) == b"\xa5"
    assert all(value >= 0 for value in edge_type_counts(result.augmented_cover_edges).values())


def test_bynis_roundtrip_with_decoy_edges() -> None:
    reference = nx.barabasi_albert_graph(30, 2, seed=3)
    message = b"phase-two"
    result = bynis_encode(
        message,
        seed=41,
        reference_graph=reference,
        extra_edges=12,
    )

    assert result.extra_edges == 12
    assert bynis_decode(result.edges, seed=41) == message
