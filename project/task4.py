import scipy.sparse as sp
from networkx import MultiDiGraph

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Return pairs of start/final nodes connected by a path in the regex language.

    The reachability is computed with a multiple-source BFS implemented with
    boolean matrix-vector products.
    """
    dfa = regex_to_dfa(regex)
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    dfa_states = list(dfa.states)
    dfa_index = {state: index for index, state in enumerate(dfa_states)}
    dfa_start = dfa_index[dfa.start_state]
    dfa_finals = {dfa_index[state] for state in dfa.final_states}

    delta = {}
    for state, transitions in dfa.to_dict().items():
        for symbol, state_to in transitions.items():
            delta[(dfa_index[state], symbol)] = dfa_index[state_to]

    start_indices = sorted(graph_fa.start_states)
    final_indices = sorted(graph_fa.final_states)
    n_start = len(start_indices)
    n_states = graph_fa.states_count
    n_dfa = len(dfa_states)

    frontier = {
        state: sp.csr_matrix((n_start, n_states), dtype=bool) for state in range(n_dfa)
    }

    initial = sp.lil_matrix((n_start, n_states), dtype=bool)
    for index, start_index in enumerate(start_indices):
        initial[index, start_index] = True
    frontier[dfa_start] = initial.tocsr()

    changed = True
    while changed:
        changed = False
        for (state, symbol), state_to in delta.items():
            graph_matrix = graph_fa.bool_matrices.get(symbol)
            if graph_matrix is None:
                continue
            contribution = (frontier[state] @ graph_matrix).astype(bool)
            if (contribution > frontier[state_to]).nnz:
                frontier[state_to] = (frontier[state_to] + contribution).astype(bool)
                changed = True

    result = set()
    for index, start_index in enumerate(start_indices):
        start_vertex = graph_fa.index_to_state[start_index].value
        for final_index in final_indices:
            final_vertex = graph_fa.index_to_state[final_index].value
            if any(frontier[state][index, final_index] for state in dfa_finals):
                result.add((start_vertex, final_vertex))

    return result
