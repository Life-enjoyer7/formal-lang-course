from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Build a minimal DFA that accepts the language of a regular expression."""
    return Regex(regex).to_epsilon_nfa().minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: set[int], final_states: set[int]
) -> NondeterministicFiniteAutomaton:
    """Build an NFA from a labeled graph.

    Every edge becomes a transition labelled by the edge's ``"label"``
    attribute. Empty ``start_states`` or ``final_states`` mean all graph
    nodes are start or final states respectively.
    """
    nfa = NondeterministicFiniteAutomaton.from_networkx(graph)

    starts = start_states if start_states else set(graph.nodes)
    finals = final_states if final_states else set(graph.nodes)

    for state in starts:
        nfa.add_start_state(state)
    for state in finals:
        nfa.add_final_state(state)

    return nfa
