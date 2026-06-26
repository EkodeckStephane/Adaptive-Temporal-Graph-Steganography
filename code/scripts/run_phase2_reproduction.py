from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))

from baselines.graph_stego import (  # noqa: E402
    adabind_encode,
    bind_capacity_bits,
    bind_decode,
    bind_encode,
    bynis_decode,
    bynis_encode,
    edge_type_counts,
)
from baselines.walks import deepwalk_corpus, node2vec_walk, uniform_random_walk  # noqa: E402


def load_edges(path: Path) -> list[tuple[int, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        next(rows)
        return [tuple(map(int, row)) for row in rows]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bit_error_rate(expected: bytes, recovered: bytes | None) -> float:
    if recovered is None or len(recovered) != len(expected):
        return 1.0
    errors = sum((left ^ right).bit_count() for left, right in zip(expected, recovered))
    return errors / (8 * len(expected))


def attack_adjacent_swaps(
    edges: tuple[tuple[int, int], ...],
    swaps: int,
    seed: int,
) -> tuple[tuple[int, int], ...]:
    attacked = list(edges)
    rng = np.random.RandomState(seed)
    starts = rng.choice(len(attacked) - 1, size=min(swaps, len(attacked) - 1), replace=False)
    for index in starts:
        attacked[int(index)], attacked[int(index) + 1] = (
            attacked[int(index) + 1],
            attacked[int(index)],
        )
    return tuple(attacked)


def walk_metrics(walk: list[int]) -> dict[str, float | int]:
    return {
        "length": len(walk),
        "unique_nodes": len(set(walk)),
        "unique_ratio": len(set(walk)) / len(walk),
        "immediate_return_rate": (
            sum(walk[index] == walk[index - 2] for index in range(2, len(walk)))
            / max(1, len(walk) - 2)
        ),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    data_path = (
        ROOT
        / "literature/external/lee_hmg_snapshot/data/test/terrorists-911/edges.csv"
    )
    edges = load_edges(data_path)
    directed_graph = nx.DiGraph(edges)
    undirected_graph = directed_graph.to_undirected()

    bind_runs = []
    attack_runs = []
    attack_message = bytes((37 * index + 11) % 256 for index in range(16))
    for message_size in config["messages"]["bind_bytes"]:
        message = bytes((37 * index + 11) % 256 for index in range(message_size))
        encoded = bind_encode(edges, message, seed=seed)
        recovered = bind_decode(encoded.edges, seed=seed)
        bind_runs.append(
            {
                "message_bytes": message_size,
                "payload_bits": encoded.payload_bits,
                "header_bits": encoded.header_bits,
                "payload_bpe": encoded.payload_bits / len(encoded.edges),
                "selected_edge_fraction": encoded.selected_edges / len(encoded.edges),
                "roundtrip_exact": recovered == message,
                "edge_multiset_preserved": sorted(encoded.edges) == sorted(edges),
            }
        )
        if message == attack_message:
            for swap_count in config["attacks"]["adjacent_swap_counts"]:
                bers = []
                exact = []
                failures = 0
                for trial in range(int(config["attacks"]["trials"])):
                    attacked = attack_adjacent_swaps(
                        encoded.edges,
                        int(swap_count),
                        seed + 10_000 * int(swap_count) + trial,
                    )
                    try:
                        attacked_recovery = bind_decode(attacked, seed=seed)
                    except (ValueError, OverflowError):
                        attacked_recovery = None
                        failures += 1
                    bers.append(bit_error_rate(message, attacked_recovery))
                    exact.append(attacked_recovery == message)
                attack_runs.append(
                    {
                        "adjacent_swaps": int(swap_count),
                        "edge_fraction_touched": min(2 * int(swap_count), len(edges))
                        / len(edges),
                        "trials": int(config["attacks"]["trials"]),
                        "mean_ber": float(np.mean(bers)),
                        "std_ber": float(np.std(bers, ddof=1)),
                        "exact_recovery_rate": float(np.mean(exact)),
                        "decode_failure_rate": failures / len(bers),
                    }
                )

    adabind_message = bytes.fromhex(config["messages"]["adabind_hex"])
    ada = adabind_encode(
        edges,
        adabind_message,
        seed=seed,
        max_iterations=int(config["adabind"]["max_iterations"]),
        sampled_edges=int(config["adabind"]["sampled_edges"]),
        extra_target_edges=int(config["adabind"]["extra_target_edges"]),
    )
    ada_recovered = bind_decode(ada.edges, seed=seed)

    bynis_runs = []
    bynis_message = config["messages"]["bynis_text"].encode("utf-8")
    for extra_edges in config["bynis"]["extra_edges"]:
        bynis = bynis_encode(
            bynis_message,
            seed=seed,
            reference_graph=undirected_graph,
            extra_edges=int(extra_edges),
        )
        bynis_runs.append(
            {
                "message_bytes": len(bynis_message),
                "message_edges_with_header": bynis.message_edges,
                "extra_edges": bynis.extra_edges,
                "total_edges": len(bynis.edges),
                "payload_bpe": 8 * len(bynis_message) / len(bynis.edges),
                "roundtrip_exact": bynis_decode(bynis.edges, seed=seed) == bynis_message,
            }
        )

    walk_length = int(config["walks"]["length"])
    uniform = uniform_random_walk(undirected_graph, 0, walk_length, seed=seed)
    node2vec = node2vec_walk(
        undirected_graph,
        0,
        walk_length,
        p=float(config["walks"]["node2vec"]["p"]),
        q=float(config["walks"]["node2vec"]["q"]),
        seed=seed,
    )
    corpus = deepwalk_corpus(
        undirected_graph,
        walk_length=walk_length,
        walks_per_node=int(config["walks"]["walks_per_node"]),
        seed=seed,
    )

    return {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "seed": seed,
        "source": {
            **config["source"],
            "path": str(data_path.relative_to(ROOT)),
            "sha256": sha256(data_path),
            "nodes": directed_graph.number_of_nodes(),
            "edges": len(edges),
            "edge_type_counts": edge_type_counts(edges),
            "bind_balanced_capacity_bits": bind_capacity_bits(edges),
        },
        "bind": bind_runs,
        "adabind": {
            "message_bytes": len(adabind_message),
            "added_edges": len(ada.added_edges),
            "iterations": ada.iterations,
            "final_edges": len(ada.augmented_cover_edges),
            "topology_modified": bool(ada.added_edges),
            "payload_bpe": ada.payload_bits / len(ada.edges),
            "roundtrip_exact": ada_recovered == adabind_message,
            "final_l1_distance": ada.final_l1_distance,
        },
        "bynis": bynis_runs,
        "reordering_attack": attack_runs,
        "walks": {
            "uniform": walk_metrics(uniform),
            "node2vec": walk_metrics(node2vec),
            "deepwalk": {
                "corpus_walks": len(corpus),
                "mean_length": float(np.mean([len(walk) for walk in corpus])),
                "mean_unique_ratio": float(
                    np.mean([len(set(walk)) / len(walk) for walk in corpus])
                ),
            },
        },
    }


def write_csv(result: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for item in result["bind"]:
        rows.append({"method": "BIND", **item})
    rows.append({"method": "AdaBIND", **result["adabind"]})
    for item in result["bynis"]:
        rows.append({"method": "BYNIS", **item})
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/real_world/phase2_lee_reproduction.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/tables/phase2_lee_reproduction.json",
    )
    args = parser.parse_args()
    result = run(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(result, args.output.with_suffix(".csv"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
