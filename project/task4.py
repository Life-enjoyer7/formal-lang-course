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

    Reachability is computed with a classical multiple-source BFS: for each
    label the current frontier is advanced by two boolean matrix
    multiplications (the graph adjacency matrix and the DFA transition), and
    only the newly discovered states are propagated further.
    """
    dfa = regex_to_dfa(regex)
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    dfa_states = list(dfa.states)
    dfa_index = {state: index for index, state in enumerate(dfa_states)}
    dfa_start = dfa_index[dfa.start_state]
    dfa_finals = {dfa_index[state] for state in dfa.final_states}

    starts = sorted(graph_fa.start_states)
    finals = sorted(graph_fa.final_states)
    n_start = len(starts)
    n_vertices = graph_fa.states_count
    n_dfa = len(dfa_states)

    # Per-label DFA transition as a permutation over (state, source) rows.
    state_mappings = {}
    for state, transitions in dfa.to_dict().items():
        for symbol, state_to in transitions.items():
            state_mappings.setdefault(symbol, {})[dfa_index[state]] = dfa_index[
                state_to
            ]

    dfa_steps = {}
    for symbol, mapping in state_mappings.items():
        step = sp.lil_matrix((n_dfa * n_start, n_dfa * n_start), dtype=bool)
        for state_from, state_to in mapping.items():
            for source in range(n_start):
                step[state_to * n_start + source, state_from * n_start + source] = True
        dfa_steps[symbol] = step.tocsr()

    # Rows of the frontier are (dfa state, source); columns are graph vertices.
    initial = sp.lil_matrix((n_dfa * n_start, n_vertices), dtype=bool)
    for source, start in enumerate(starts):
        initial[dfa_start * n_start + source, start] = True

    frontier = initial.tocsr()
    discovered = frontier.copy()

    while frontier.nnz:
        next_frontier = sp.csr_matrix((n_dfa * n_start, n_vertices), dtype=bool)
        for symbol, graph_matrix in graph_fa.bool_matrices.items():
            dfa_step = dfa_steps.get(symbol)
            if dfa_step is None:
                continue
            next_frontier = next_frontier + dfa_step @ (frontier @ graph_matrix)

        frontier = (next_frontier > discovered).astype(bool)
        discovered = (discovered + frontier).astype(bool)

    result = set()
    for source, start in enumerate(starts):
        start_vertex = graph_fa.index_to_state[start].value
        for final in finals:
            final_vertex = graph_fa.index_to_state[final].value
            if any(
                discovered[final_state * n_start + source, final]
                for final_state in dfa_finals
            ):
                result.add((start_vertex, final_vertex))

    return result
