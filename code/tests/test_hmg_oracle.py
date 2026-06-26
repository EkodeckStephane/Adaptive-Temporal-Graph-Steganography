from __future__ import annotations

import csv
import sys
from pathlib import Path

import bitstring
import networkx as nx
import pandas as pd

from baselines.graph_stego import bind_encode


def test_bind_matches_the_official_hmg_edge_order() -> None:
    root = Path(__file__).resolve().parents[2]
    snapshot = root / "literature/external/lee_hmg_snapshot"
    sys.path.insert(0, str(snapshot))
    try:
        from hmg.algorithms.realnet.bind import BIND
        from hmg.engine import GraphEngine

        with (snapshot / "data/test/terrorists-911/edges.csv").open(newline="") as handle:
            rows = csv.reader(handle)
            next(rows)
            edges = [tuple(map(int, row)) for row in rows]

        message = bytes((37 * index + 11) % 256 for index in range(16))
        frame = pd.DataFrame(edges, columns=["Source", "Target"])
        graph_engine = GraphEngine("networkx")
        graph = graph_engine.create_graph(
            nx.from_pandas_edgelist(
                frame,
                source="Source",
                target="Target",
                create_using=nx.DiGraph,
            )
        )
        official, _ = BIND(graph_engine, verbose=0).encode(
            graph,
            frame,
            bitstring.BitArray(message),
            1234,
        )
        reproduced = bind_encode(edges, message, seed=1234)

        official_edges = list(map(tuple, official[["Source", "Target"]].to_numpy()))
        assert official_edges == list(reproduced.edges)
    finally:
        sys.path.remove(str(snapshot))
