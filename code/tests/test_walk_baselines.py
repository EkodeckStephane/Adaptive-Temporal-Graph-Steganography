import networkx as nx

from baselines.walks import deepwalk_corpus, node2vec_walk, uniform_random_walk


def graph_fixture() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4)])
    return graph


def assert_valid_walk(graph: nx.Graph, walk: list[int]) -> None:
    assert all(graph.has_edge(source, target) for source, target in zip(walk, walk[1:]))


def test_uniform_walk_is_seeded_and_valid() -> None:
    graph = graph_fixture()
    first = uniform_random_walk(graph, 0, 12, seed=11)
    second = uniform_random_walk(graph, 0, 12, seed=11)
    assert first == second
    assert_valid_walk(graph, first)


def test_deepwalk_corpus_has_expected_size() -> None:
    graph = graph_fixture()
    corpus = deepwalk_corpus(graph, walk_length=6, walks_per_node=3, seed=23)
    assert len(corpus) == 3 * graph.number_of_nodes()
    assert all(1 <= len(walk) <= 6 for walk in corpus)
    for walk in corpus:
        assert_valid_walk(graph, walk)


def test_node2vec_is_seeded_and_valid() -> None:
    graph = graph_fixture()
    first = node2vec_walk(graph, 0, 20, p=0.5, q=2.0, seed=37)
    second = node2vec_walk(graph, 0, 20, p=0.5, q=2.0, seed=37)
    assert first == second
    assert_valid_walk(graph, first)
