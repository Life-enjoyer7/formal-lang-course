from collections import defaultdict, deque

import networkx as nx
from pyformlang.rsa import RecursiveAutomaton


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """Compute pairs of nodes connected by a path in the language of the RSM.

    The reachability is computed with a GLL-style algorithm: descriptors
    (state, vertex) are processed over a graph-structured stack, where
    nonterminal calls push a stack frame and reaching an exit state pops it.
    """
    start_nodes = set(start_nodes) if start_nodes else set(graph.nodes)
    final_nodes = set(final_nodes) if final_nodes else set(graph.nodes)

    nonterminal_labels = set(rsm.boxes.keys())

    entry = {}
    exits = {}
    terminal_transitions = defaultdict(list)
    nonterminal_transitions = defaultdict(list)

    for label, box in rsm.boxes.items():
        dfa = box.dfa
        entry[label] = (label, next(iter(box.start_state)))
        exits[label] = {(label, state) for state in box.final_states}
        for state, transitions in dfa.to_dict().items():
            for symbol, targets in transitions.items():
                if not isinstance(targets, set):
                    targets = {targets}
                for target in targets:
                    if symbol in nonterminal_labels:
                        nonterminal_transitions[(label, state)].append(
                            (symbol, (label, target))
                        )
                    else:
                        terminal_transitions[(label, state)].append(
                            (symbol.value, (label, target))
                        )

    graph_adj = defaultdict(lambda: defaultdict(set))
    for u, v, data in graph.edges(data=True):
        graph_adj[data["label"]][u].add(v)

    reach = set()
    gss_edges = defaultdict(set)
    popped = defaultdict(set)

    descriptors = deque()
    processed = set()

    start_label = rsm.initial_label
    for vertex in start_nodes:
        descriptors.append((entry[start_label], vertex, (None, vertex)))

    while descriptors:
        state, vertex, gss = descriptors.popleft()
        if (state, vertex, gss) in processed:
            continue
        processed.add((state, vertex, gss))

        for terminal, next_state in terminal_transitions.get(state, ()):
            for next_vertex in graph_adj.get(terminal, {}).get(vertex, ()):
                descriptors.append((next_state, next_vertex, gss))

        for nonterminal, next_state in nonterminal_transitions.get(state, ()):
            call_site = (state, vertex, nonterminal)
            edge = (next_state, gss)
            if edge not in gss_edges[call_site]:
                gss_edges[call_site].add(edge)
                for popped_vertex in popped[call_site]:
                    descriptors.append((next_state, popped_vertex, gss))
            descriptors.append((entry[nonterminal], vertex, call_site))

        label = state[0]
        if state in exits[label]:
            call_vertex = gss[1]
            reach.add((label, call_vertex, vertex))
            for return_state, parent_gss in gss_edges[gss]:
                descriptors.append((return_state, vertex, parent_gss))
            popped[gss].add(vertex)

    result = set()
    for label, u, v in reach:
        if label == start_label and u in start_nodes and v in final_nodes:
            result.add((u, v))

    return result
