from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import networkx as nx
import numpy as np

Node = Hashable
Edge = tuple[Node, Node]
EDGE_TYPES = ("00", "01", "10", "11")


class BindCapacityError(RuntimeError):
    """Raised when the cover does not contain enough edges of a required type."""


@dataclass(frozen=True)
class BindResult:
    edges: tuple[Edge, ...]
    payload_bits: int
    header_bits: int
    capacity_bits: int
    selected_edges: int


@dataclass(frozen=True)
class AdaBindResult:
    edges: tuple[Edge, ...]
    augmented_cover_edges: tuple[Edge, ...]
    added_edges: tuple[Edge, ...]
    payload_bits: int
    header_bits: int
    iterations: int
    final_l1_distance: int


@dataclass(frozen=True)
class BynisResult:
    edges: tuple[tuple[int, int], ...]
    message_edges: int
    extra_edges: int
    bias: int


def _as_edges(edges: Iterable[Sequence[Node]]) -> list[Edge]:
    result = [(edge[0], edge[1]) for edge in edges]
    if not result:
        raise ValueError("At least one edge is required")
    return result


def _graph(edges: Sequence[Edge], directed: bool) -> nx.Graph:
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from(edges)
    return graph


def lee_bitwidth(n: int) -> int:
    """Return the metadata width used by Lee's HMG implementation."""
    if n <= 0:
        raise ValueError("n must be positive")
    return int(2 ** math.ceil(math.log2(math.log2(2 * n))))


def _bytes_to_bits(message: bytes) -> list[int]:
    return [int(bit) for byte in message for bit in f"{byte:08b}"]


def _uint_bits(value: int, width: int) -> list[int]:
    if value < 0 or value >= 2**width:
        raise ValueError(f"{value} does not fit in {width} bits")
    return [int(bit) for bit in f"{value:0{width}b}"]


def _edge_type(graph: nx.Graph, edge: Edge) -> str:
    return f"{graph.degree(edge[0]) % 2}{graph.degree(edge[1]) % 2}"


def edge_type_counts(edges: Iterable[Sequence[Node]], directed: bool = True) -> dict[str, int]:
    edge_list = _as_edges(edges)
    graph = _graph(edge_list, directed)
    counts = dict.fromkeys(EDGE_TYPES, 0)
    for edge in edge_list:
        counts[_edge_type(graph, edge)] += 1
    return counts


def bind_capacity_bits(edges: Iterable[Sequence[Node]], directed: bool = True) -> int:
    """Lee's balanced lower-bound payload capacity, excluding metadata."""
    return 8 * min(edge_type_counts(edges, directed).values())


def _secret_bits(message: bytes, edge_count: int) -> tuple[list[int], int]:
    header_bits = lee_bitwidth(edge_count)
    bits = _uint_bits(len(message), header_bits) + _bytes_to_bits(message)
    if len(bits) % 2:
        raise ValueError("BIND requires an even-sized metadata and payload bitstream")
    return bits, header_bits


def _inverse_permute(values: Sequence[Edge], seed: int) -> list[Edge]:
    rng = np.random.RandomState(seed)
    permutation = np.arange(len(values))
    rng.shuffle(permutation)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(values))
    return [values[int(index)] for index in inverse]


def bind_encode(
    edges: Iterable[Sequence[Node]],
    message: bytes,
    *,
    seed: int = 1,
    directed: bool = True,
) -> BindResult:
    """Encode bytes by reproducing BIND's degree-parity edge ordering."""
    edge_list = _as_edges(edges)
    graph = _graph(edge_list, directed)
    bits, header_bits = _secret_bits(message, len(edge_list))
    pairs = ["".join(str(bit) for bit in bits[index : index + 2]) for index in range(0, len(bits), 2)]

    pools: dict[str, list[int]] = {edge_type: [] for edge_type in EDGE_TYPES}
    for index, edge in enumerate(edge_list):
        pools[_edge_type(graph, edge)].append(index)

    required = {edge_type: pairs.count(edge_type) for edge_type in EDGE_TYPES}
    missing = {
        edge_type: required[edge_type] - len(pools[edge_type])
        for edge_type in EDGE_TYPES
        if required[edge_type] > len(pools[edge_type])
    }
    if missing:
        raise BindCapacityError(f"Insufficient edge types: {missing}")

    rng = np.random.RandomState(seed)
    for edge_type in EDGE_TYPES:
        rng.shuffle(pools[edge_type])

    consumed = dict.fromkeys(EDGE_TYPES, 0)
    ordered_indices: list[int] = []
    for pair in pairs:
        ordered_indices.append(pools[pair][consumed[pair]])
        consumed[pair] += 1
    for edge_type in EDGE_TYPES:
        ordered_indices.extend(pools[edge_type][consumed[edge_type] :])

    ordered_edges = [edge_list[index] for index in ordered_indices]
    output_rng = np.random.RandomState(seed)
    permutation = np.arange(len(ordered_edges))
    output_rng.shuffle(permutation)
    stego_edges = tuple(ordered_edges[int(index)] for index in permutation)

    return BindResult(
        edges=stego_edges,
        payload_bits=8 * len(message),
        header_bits=header_bits,
        capacity_bits=bind_capacity_bits(edge_list, directed),
        selected_edges=len(pairs),
    )


def bind_decode(
    edges: Iterable[Sequence[Node]],
    *,
    seed: int = 1,
    directed: bool = True,
) -> bytes:
    edge_list = _as_edges(edges)
    graph = _graph(edge_list, directed)
    ordered = _inverse_permute(edge_list, seed)
    header_bits = lee_bitwidth(len(edge_list))
    header_edges = header_bits // 2
    header = "".join(_edge_type(graph, edge) for edge in ordered[:header_edges])
    message_size = int(header, 2)
    payload_edges = 4 * message_size
    payload = "".join(
        _edge_type(graph, edge)
        for edge in ordered[header_edges : header_edges + payload_edges]
    )
    if len(payload) != 8 * message_size:
        raise ValueError("Truncated BIND edge list")
    return bytes(int(payload[index : index + 8], 2) for index in range(0, len(payload), 8))


def _edge_type_vector(edges: Sequence[Edge], directed: bool) -> np.ndarray:
    counts = edge_type_counts(edges, directed)
    return np.array([counts[edge_type] for edge_type in EDGE_TYPES], dtype=np.int64)


def adabind_encode(
    edges: Iterable[Sequence[Node]],
    message: bytes,
    *,
    seed: int = 1,
    directed: bool = True,
    max_iterations: int = 10_000,
    sampled_edges: int = 50,
    extra_target_edges: int = 5,
) -> AdaBindResult:
    """AdaBIND with Lee's GW2N objective and deterministic candidate sampling.

    Unlike the published implementation, the best candidate is reset at every
    iteration. This prevents stale-edge stalls while preserving the L1 target.
    """
    if not directed:
        raise ValueError("AdaBIND GW2N is defined on a directed graph")
    cover = _as_edges(edges)
    current = _edge_type_vector(cover, directed)
    initial = current.copy()

    graph = _graph(cover, directed)
    nodes = list(graph.nodes)
    rng = np.random.RandomState(seed)
    added: list[Edge] = []
    final_distance = 0

    for iteration in range(max_iterations + 1):
        bits, header_bits = _secret_bits(message, len(cover))
        pairs = [
            "".join(map(str, bits[index : index + 2]))
            for index in range(0, len(bits), 2)
        ]
        required = np.array(
            [pairs.count(edge_type) for edge_type in EDGE_TYPES], dtype=np.int64
        )
        target = np.maximum(required, initial) + extra_target_edges
        counts = _edge_type_vector(cover, directed)
        final_distance = int(np.abs(counts - target).sum())
        if np.all(counts >= required):
            encoded = bind_encode(cover, message, seed=seed, directed=directed)
            return AdaBindResult(
                edges=encoded.edges,
                augmented_cover_edges=tuple(cover),
                added_edges=tuple(added),
                payload_bits=encoded.payload_bits,
                header_bits=header_bits,
                iterations=iteration,
                final_l1_distance=final_distance,
            )
        if iteration == max_iterations:
            break

        shuffled = nodes.copy()
        rng.shuffle(shuffled)
        candidate_count = min(len(shuffled) // 2, max(1, sampled_edges // 2))
        candidates = [
            (shuffled[2 * index], shuffled[2 * index + 1])
            for index in range(candidate_count)
            if not graph.has_edge(shuffled[2 * index], shuffled[2 * index + 1])
        ]
        if not candidates:
            candidates = [
                (source, target_node)
                for source in nodes
                for target_node in nodes
                if source != target_node and not graph.has_edge(source, target_node)
            ]
        if not candidates:
            break

        best_edge: Edge | None = None
        best_counts: np.ndarray | None = None
        best_distance: int | None = None
        for candidate in candidates:
            candidate_counts = _edge_type_vector([*cover, candidate], directed)
            distance = int(np.abs(candidate_counts - target).sum())
            if best_distance is None or distance < best_distance:
                best_edge = candidate
                best_counts = candidate_counts
                best_distance = distance

        assert best_edge is not None and best_counts is not None and best_distance is not None
        graph.add_edge(*best_edge)
        cover.append(best_edge)
        added.append(best_edge)
        final_distance = best_distance

    bits, _ = _secret_bits(message, len(cover))
    pairs = ["".join(map(str, bits[index : index + 2])) for index in range(0, len(bits), 2)]
    required = np.array([pairs.count(edge_type) for edge_type in EDGE_TYPES], dtype=np.int64)
    counts = _edge_type_vector(cover, directed)
    deficit = {
        EDGE_TYPES[index]: int(required[index] - counts[index])
        for index in range(4)
        if required[index] > counts[index]
    }
    raise BindCapacityError(
        f"AdaBIND did not reach the required edge counts after {max_iterations} iterations: {deficit}"
    )


def _bytewidth(n: int) -> int:
    width = lee_bitwidth(n) // 8
    if width not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported BYNIS byte width: {width}")
    return width


def _estimate_nodes(n_bytes: int) -> int:
    return int(math.ceil(10 ** round(math.log10(n_bytes))))


def bynis_encode(
    message: bytes,
    *,
    seed: int = 1,
    reference_graph: nx.Graph | None = None,
    extra_edges: int = 0,
    directed: bool = False,
) -> BynisResult:
    """Encode one byte per synthetic edge following BYNIS."""
    if not message:
        raise ValueError("BYNIS requires a non-empty message")
    bytewidth = _bytewidth(len(message))
    fmt = {1: "B", 2: "H", 4: "I", 8: "Q"}[bytewidth]
    data = bytes([bytewidth]) + struct.pack(fmt, len(message)) + message
    node_estimate = _estimate_nodes(len(data))
    bias = max(2 ** math.ceil(math.log2(node_estimate)), 256)

    if reference_graph is None:
        edges_per_node = min(node_estimate - 1, len(data) // node_estimate + 1)
        reference_graph = nx.powerlaw_cluster_graph(
            node_estimate, max(1, edges_per_node), 0.1, seed=seed
        )
    degrees = sorted((degree for _, degree in reference_graph.degree), reverse=True)
    if not degrees:
        raise ValueError("reference_graph must contain nodes")

    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    usage = [0] * len(degrees)
    cursor = 0
    output: list[tuple[int, int]] = []
    for value in data:
        while cursor + 1 < len(degrees) and usage[cursor] >= degrees[cursor]:
            cursor += 1
        node_a = cursor
        node_b = value + bias - cursor
        rename = 1
        while graph.has_edge(node_a, node_b):
            node_b += 256 * rename
            rename += 1
            if rename > 100:
                raise RuntimeError("BYNIS could not create a unique message edge")
        graph.add_edge(node_a, node_b)
        output.append((node_a, node_b))
        usage[cursor] += 1

    rng = np.random.RandomState(seed)
    nodes = list(graph.nodes)
    for _ in range(extra_edges):
        for _attempt in range(10_000):
            edge = (nodes[int(rng.randint(0, len(nodes)))], nodes[int(rng.randint(0, len(nodes)))])
            if edge[0] != edge[1] and not graph.has_edge(*edge):
                graph.add_edge(*edge)
                output.append(edge)
                break
        else:
            raise RuntimeError("BYNIS could not sample a unique extra edge")

    permutation = np.arange(len(output))
    rng = np.random.RandomState(seed)
    rng.shuffle(permutation)
    shuffled = tuple(output[int(index)] for index in permutation)
    return BynisResult(
        edges=shuffled,
        message_edges=len(data),
        extra_edges=extra_edges,
        bias=bias,
    )


def bynis_decode(edges: Iterable[Sequence[int]], *, seed: int = 1) -> bytes:
    edge_list = [(int(edge[0]), int(edge[1])) for edge in edges]
    if not edge_list:
        raise ValueError("At least one edge is required")
    ordered = _inverse_permute(edge_list, seed)
    bytewidth = sum(ordered[0]) % 256
    if bytewidth not in (1, 2, 4, 8):
        raise ValueError(f"Invalid BYNIS byte width: {bytewidth}")
    fmt = {1: "B", 2: "H", 4: "I", 8: "Q"}[bytewidth]
    size_bytes = bytes(sum(edge) % 256 for edge in ordered[1 : 1 + bytewidth])
    message_size = struct.unpack(fmt, size_bytes)[0]
    start = 1 + bytewidth
    message_edges = ordered[start : start + message_size]
    if len(message_edges) != message_size:
        raise ValueError("Truncated BYNIS edge list")
    return bytes(sum(edge) % 256 for edge in message_edges)
