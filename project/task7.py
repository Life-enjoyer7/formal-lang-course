import networkx as nx
import scipy.sparse as sp
from pyformlang.cfg import CFG

from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """Compute all pairs of nodes connected by a path in the grammar language.

    Reachability is computed with the matrix algorithm: a boolean matrix per
    nonterminal is iteratively updated using boolean matrix products.
    """
    cfg = cfg_to_weak_normal_form(cfg)

    nodes = list(graph.nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    n = len(nodes)

    start_set = set(start_nodes) if start_nodes else set(nodes)
    final_set = set(final_nodes) if final_nodes else set(nodes)

    matrices = {
        variable: sp.lil_matrix((n, n), dtype=bool) for variable in cfg.variables
    }
    binary_productions = []

    for production in cfg.productions:
        body = production.body
        if len(body) == 0:
            for index in range(n):
                matrices[production.head][index, index] = True
        elif len(body) == 1:
            terminal = body[0].value
            for u, v, data in graph.edges(data=True):
                if data["label"] == terminal:
                    matrices[production.head][node_to_index[u], node_to_index[v]] = True
        else:
            binary_productions.append((production.head, body[0], body[1]))

    for variable in cfg.variables:
        matrices[variable] = matrices[variable].tocsr()

    changed = True
    while changed:
        changed = False
        for head, left, right in binary_productions:
            updated = (matrices[head] + matrices[left] @ matrices[right]).astype(bool)
            if (updated != matrices[head]).nnz:
                matrices[head] = updated
                changed = True

    result = set()
    start_symbol_matrix = matrices[cfg.start_symbol].tocoo()
    for i, j in zip(start_symbol_matrix.row, start_symbol_matrix.col):
        if nodes[i] in start_set and nodes[j] in final_set:
            result.add((nodes[i], nodes[j]))

    return result
