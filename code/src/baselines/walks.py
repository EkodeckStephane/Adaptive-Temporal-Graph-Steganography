from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
import numpy as np

Node = Hashable


def _neighbors(graph: nx.Graph, node: Node) -> list[Node]:
    if graph.is_directed():
        return list(graph.successors(node))
    return list(graph.neighbors(node))


def uniform_random_walk(
    graph: nx.Graph,
    start: Node,
    length: int,
    *,
    seed: int,
) -> list[Node]:
    if start not in graph:
        raise ValueError(f"Unknown start node: {start}")
    if length < 1:
        raise ValueError("length must be at least one")
    rng = np.random.RandomState(seed)
    walk = [start]
    while len(walk) < length:
        neighbors = _neighbors(graph, walk[-1])
        if not neighbors:
            break
        walk.append(neighbors[int(rng.randint(0, len(neighbors)))])
    return walk


def deepwalk_corpus(
    graph: nx.Graph,
    *,
    walk_length: int,
    walks_per_node: int,
    seed: int,
) -> list[list[Node]]:
    """Generate the uniform walk corpus used by DeepWalk."""
    rng = np.random.RandomState(seed)
    nodes = list(graph.nodes)
    corpus: list[list[Node]] = []
    for _ in range(walks_per_node):
        order = nodes.copy()
        rng.shuffle(order)
        for node in order:
            corpus.append(
                uniform_random_walk(
                    graph,
                    node,
                    walk_length,
                    seed=int(rng.randint(0, 2**31 - 1)),
                )
            )
    return corpus


def node2vec_walk(
    graph: nx.Graph,
    start: Node,
    length: int,
    *,
    p: float = 1.0,
    q: float = 1.0,
    seed: int,
) -> list[Node]:
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive")
    if start not in graph:
        raise ValueError(f"Unknown start node: {start}")
    rng = np.random.RandomState(seed)
    walk = [start]
    while len(walk) < length:
        current = walk[-1]
        neighbors = _neighbors(graph, current)
        if not neighbors:
            break
        if len(walk) == 1:
            walk.append(neighbors[int(rng.randint(0, len(neighbors)))])
            continue

        previous = walk[-2]
        weights = []
        for candidate in neighbors:
            if candidate == previous:
                weights.append(1.0 / p)
            elif graph.has_edge(previous, candidate) or graph.has_edge(candidate, previous):
                weights.append(1.0)
            else:
                weights.append(1.0 / q)
        probabilities = np.asarray(weights, dtype=float)
        probabilities /= probabilities.sum()
        walk.append(neighbors[int(rng.choice(len(neighbors), p=probabilities))])
    return walk
