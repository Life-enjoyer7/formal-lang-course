import networkx as nx
import scipy.sparse as sp
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import Symbol
from pyformlang.rsa import RecursiveAutomaton


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Build a recursive automaton from a context-free grammar."""
    return RecursiveAutomaton.from_text(
        cfg.to_text(), start_symbol=Symbol(cfg.start_symbol.value)
    )


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Build a recursive automaton from an EBNF description."""
    return RecursiveAutomaton.from_text(ebnf)


def _transitive_closure(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Reflexive transitive closure of a boolean matrix."""
    closure = sp.identity(matrix.shape[0], dtype=bool, format="csr") + matrix
    while True:
        updated = (closure + closure @ matrix).astype(bool)
        if (updated != closure).nnz == 0:
            break
        closure = updated
    return closure


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """Compute pairs of nodes connected by a path in the language of the RSM.

    The reachability is computed with the tensor algorithm: for every box the
    product of the box automaton and the current per-nonterminal matrices is
    built with Kronecker products, and its transitive closure gives the new
    per-nonterminal matrix.
    """
    nodes = list(graph.nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    n = len(nodes)

    start_set = set(start_nodes) if start_nodes else set(nodes)
    final_set = set(final_nodes) if final_nodes else set(nodes)

    adjacency = {}
    for u, v, data in graph.edges(data=True):
        symbol = Symbol(data["label"])
        matrix = adjacency.setdefault(symbol, sp.lil_matrix((n, n), dtype=bool))
        matrix[node_to_index[u], node_to_index[v]] = True
    adjacency = {symbol: matrix.tocsr() for symbol, matrix in adjacency.items()}

    nonterminal_labels = set(rsm.boxes.keys())
    reach = {label: sp.csr_matrix((n, n), dtype=bool) for label in nonterminal_labels}

    changed = True
    while changed:
        changed = False
        for label, box in rsm.boxes.items():
            dfa = box.dfa
            states = list(dfa.states)
            state_to_index = {state: index for index, state in enumerate(states)}
            m = len(states)
            start_index = state_to_index[next(iter(box.start_state))]
            final_indices = {state_to_index[state] for state in box.final_states}

            transitions_by_label = {}
            for state, transitions in dfa.to_dict().items():
                state_from = state_to_index[state]
                for symbol, state_to in transitions.items():
                    state_tos = (
                        {state_to} if not isinstance(state_to, set) else state_to
                    )
                    matrix = transitions_by_label.setdefault(
                        symbol, sp.lil_matrix((m, m), dtype=bool)
                    )
                    for target in state_tos:
                        matrix[state_from, state_to_index[target]] = True
            transitions_by_label = {
                symbol: matrix.tocsr()
                for symbol, matrix in transitions_by_label.items()
            }

            product = sp.csr_matrix((m * n, m * n), dtype=bool)
            for symbol, transition_matrix in transitions_by_label.items():
                label_matrix = (
                    reach[symbol]
                    if symbol in nonterminal_labels
                    else adjacency.get(symbol)
                )
                if label_matrix is None:
                    continue
                product = product + sp.kron(
                    transition_matrix, label_matrix, format="csr"
                )

            closure = _transitive_closure(product)

            updated = sp.csr_matrix((n, n), dtype=bool)
            for final_index in final_indices:
                block = closure[
                    start_index * n : (start_index + 1) * n,
                    final_index * n : (final_index + 1) * n,
                ]
                updated = updated + block
            updated = updated.astype(bool)

            if (updated != reach[label]).nnz:
                reach[label] = updated
                changed = True

    result = set()
    start_matrix = reach[rsm.initial_label].tocoo()
    for i, j in zip(start_matrix.row, start_matrix.col):
        if nodes[i] in start_set and nodes[j] in final_set:
            result.add((nodes[i], nodes[j]))

    return result
