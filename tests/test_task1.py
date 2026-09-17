from collections import Counter

import pydot
import pytest
from networkx import MultiDiGraph, NetworkXError

from project.task1 import create_two_cycles_graph, get_graph_info

TWO_CYCLES_CASES = [
    (2, 3, ("a", "b")),
    (3, 2, ("x", "y")),
    (5, 5, ("first", "second")),
    (1, 1, ("p", "q")),
]


@pytest.mark.parametrize(
    ("graph_name", "nodes_count", "edges_count", "labels"),
    [
        (
            "biomedical",
            341,
            459,
            {
                "type",
                "label",
                "subClassOf",
                "comment",
                "versionInfo",
                "title",
                "language",
                "publisher",
                "description",
                "creator",
            },
        ),
        (
            "generations",
            129,
            273,
            {
                "type",
                "first",
                "rest",
                "onProperty",
                "intersectionOf",
                "equivalentClass",
                "someValuesFrom",
                "hasValue",
                "hasSex",
                "hasChild",
                "hasParent",
                "inverseOf",
                "sameAs",
                "hasSibling",
                "oneOf",
                "range",
                "versionInfo",
            },
        ),
        ("bzip", 632, 556, {"a", "d"}),
        ("wc", 332, 269, {"a", "d"}),
    ],
)
def test_get_graph_info(graph_name, nodes_count, edges_count, labels):
    """``get_graph_info`` returns stats of a graph loaded from the dataset.

    The test loads real graphs from the CFPQ_Data collection over the network,
    so it requires an internet connection.
    """
    info = get_graph_info(graph_name)

    assert info.nodes_count == nodes_count
    assert info.edges_count == edges_count
    assert info.labels == labels


def test_get_graph_info_unknown_graph():
    """``get_graph_info`` raises ``FileNotFoundError`` for an unknown name."""
    with pytest.raises(FileNotFoundError):
        get_graph_info("no such graph")


@pytest.mark.parametrize(
    ("first_cycle_nodes", "second_cycle_nodes", "labels"),
    TWO_CYCLES_CASES,
)
def test_create_two_cycles_graph_structure(
    tmp_path, first_cycle_nodes, second_cycle_nodes, labels
):
    """The built two-cycles graph has the expected size and edge labels."""
    graph = create_two_cycles_graph(
        first_cycle_nodes, second_cycle_nodes, labels, tmp_path / "graph.dot"
    )

    assert isinstance(graph, MultiDiGraph)
    assert graph.number_of_nodes() == first_cycle_nodes + second_cycle_nodes + 1
    assert graph.number_of_edges() == first_cycle_nodes + second_cycle_nodes + 2

    label_counts = Counter(data["label"] for _, _, data in graph.edges(data=True))
    assert label_counts[labels[0]] == first_cycle_nodes + 1
    assert label_counts[labels[1]] == second_cycle_nodes + 1


@pytest.mark.parametrize(
    ("first_cycle_nodes", "second_cycle_nodes", "labels"),
    TWO_CYCLES_CASES,
)
def test_create_two_cycles_graph_saves_dot(
    tmp_path, first_cycle_nodes, second_cycle_nodes, labels
):
    """The graph is saved to the given path as a valid DOT file."""
    path = tmp_path / "graph.dot"

    create_two_cycles_graph(first_cycle_nodes, second_cycle_nodes, labels, path)

    assert path.exists()
    assert path.read_text().strip()

    (loaded_graph,) = pydot.graph_from_dot_file(str(path))

    expected_nodes = first_cycle_nodes + second_cycle_nodes + 1
    expected_edges = first_cycle_nodes + second_cycle_nodes + 2

    assert loaded_graph.get_type() == "digraph"
    assert len(loaded_graph.get_nodes()) == expected_nodes
    assert len(loaded_graph.get_edges()) == expected_edges

    edge_labels = {edge.get_attributes()["label"] for edge in loaded_graph.get_edges()}
    assert edge_labels == set(labels)


@pytest.mark.parametrize(
    ("first_cycle_nodes", "second_cycle_nodes"),
    [(0, 0), (1, 0), (0, 1)],
)
def test_create_two_cycles_graph_zero_sized_cycle(
    tmp_path, first_cycle_nodes, second_cycle_nodes
):
    """A zero-sized cycle is rejected with ``IndexError``."""
    with pytest.raises(IndexError):
        create_two_cycles_graph(
            first_cycle_nodes, second_cycle_nodes, ("a", "b"), tmp_path / "graph.dot"
        )


def test_create_two_cycles_graph_negative_size_preserves_file(tmp_path):
    """A failed build must not overwrite an existing file."""
    path = tmp_path / "graph.dot"
    original_content = "digraph { original; }\n"
    path.write_text(original_content)

    with pytest.raises(NetworkXError):
        create_two_cycles_graph(-1, 2, ("a", "b"), path)

    assert path.read_text() == original_content
