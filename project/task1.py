from dataclasses import dataclass
from pathlib import Path

import cfpq_data
from networkx import MultiDiGraph
from networkx.drawing.nx_pydot import to_pydot


@dataclass(frozen=True)
class GraphInfo:
    """Aggregated information about a graph.

    Attributes:
    nodes_count : int
        The number of nodes in the graph.
    edges_count : int
        The number of edges in the graph.
    labels : set[str]
        The set of distinct labels that occur on the graph edges.
    """

    nodes_count: int
    edges_count: int
    labels: set[str]


def get_graph_info(name: str) -> GraphInfo:
    """Return aggregated information about a graph loaded by name.

    The graph is loaded from the CFPQ_Data dataset by its name.

    Parameters:
    name : str
        The name of the graph from the dataset.

    Returns:
    GraphInfo
        The number of nodes, the number of edges and the set of distinct
        edge labels of the graph.
    """
    graph = cfpq_data.graph_from_csv(cfpq_data.download(name))
    labels = {data["label"] for _, _, data in graph.edges(data=True)}

    return GraphInfo(
        nodes_count=graph.number_of_nodes(),
        edges_count=graph.number_of_edges(),
        labels=labels,
    )


def create_two_cycles_graph(
    first_cycle_nodes: int,
    second_cycle_nodes: int,
    labels: tuple[str, str],
    path: str | Path,
) -> MultiDiGraph:
    """Build a two-cycles graph and save it to a file in the DOT format.

    Parameters:
    first_cycle_nodes : int
        The number of nodes in the first cycle, excluding the common node.
    second_cycle_nodes : int
        The number of nodes in the second cycle, excluding the common node.
    labels : tuple[str, str]
        Two labels used to mark the edges of the first and the second cycle.
    path : str | Path
        The path to the file where the graph is saved in the DOT format.

    Returns:
    graph : MultiDiGraph
        The constructed two-cycles graph.
    """
    graph = cfpq_data.labeled_two_cycles_graph(
        first_cycle_nodes, second_cycle_nodes, labels=labels
    )

    to_pydot(graph).write(str(path))

    return graph
