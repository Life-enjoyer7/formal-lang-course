from collections import deque

import networkx as nx
from pyformlang.cfg import CFG, Production, Terminal, Variable


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Convert a CFG to the weak Chomsky normal form.

    The result contains only productions of the forms ``A -> epsilon``,
    ``A -> a`` (a single terminal) and ``A -> B C`` (two nonterminals).
    """
    cfg = cfg.eliminate_unit_productions()

    existing_variables = set(cfg.variables)
    counter = 0

    def fresh_variable() -> Variable:
        nonlocal counter
        while True:
            counter += 1
            variable = Variable(f"WNF#{counter}")
            if variable not in existing_variables:
                existing_variables.add(variable)
                return variable

    term_to_var = {}
    productions = set()

    for production in cfg.productions:
        if len(production.body) == 1:
            productions.add(production)
            continue

        body = []
        for symbol in production.body:
            if isinstance(symbol, Terminal):
                variable = term_to_var.get(symbol)
                if variable is None:
                    variable = fresh_variable()
                    term_to_var[symbol] = variable
                    productions.add(Production(variable, [symbol]))
                body.append(variable)
            else:
                body.append(symbol)

        if len(body) > 2:
            head = production.head
            for symbol in body[:-2]:
                variable = fresh_variable()
                productions.add(Production(head, [symbol, variable]))
                head = variable
            productions.add(Production(head, [body[-2], body[-1]]))
        else:
            productions.add(Production(production.head, body))

    return CFG(start_symbol=cfg.start_symbol, productions=productions)


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """Compute all pairs of nodes connected by a path in the grammar language."""
    cfg = cfg_to_weak_normal_form(cfg)
    start_nodes = set(start_nodes) if start_nodes else set(graph.nodes)
    final_nodes = set(final_nodes) if final_nodes else set(graph.nodes)

    epsilon_heads = set()
    terminal_heads = {}
    pair_productions = []

    for production in cfg.productions:
        body = production.body
        if len(body) == 0:
            epsilon_heads.add(production.head)
        elif len(body) == 1:
            terminal_heads.setdefault(body[0].value, set()).add(production.head)
        else:
            pair_productions.append((production.head, body[0], body[1]))

    as_left = {}
    as_right = {}
    for head, left, right in pair_productions:
        as_left.setdefault(left, []).append((head, right))
        as_right.setdefault(right, []).append((head, left))

    reached = set()
    right_index = {}
    left_index = {}

    def add_triple(nonterminal, u, v) -> bool:
        if (nonterminal, u, v) in reached:
            return False
        reached.add((nonterminal, u, v))
        right_index.setdefault(nonterminal, {}).setdefault(u, set()).add(v)
        left_index.setdefault(nonterminal, {}).setdefault(v, set()).add(u)
        return True

    queue = deque()
    for u, v, data in graph.edges(data=True):
        for nonterminal in terminal_heads.get(data["label"], ()):
            if add_triple(nonterminal, u, v):
                queue.append((nonterminal, u, v))
    for node in graph.nodes:
        for nonterminal in epsilon_heads:
            if add_triple(nonterminal, node, node):
                queue.append((nonterminal, node, node))

    while queue:
        nonterminal, u, v = queue.popleft()
        for head, right in as_left.get(nonterminal, ()):
            for w in list(right_index.get(right, {}).get(v, ())):
                if add_triple(head, u, w):
                    queue.append((head, u, w))
        for head, left in as_right.get(nonterminal, ()):
            for t in list(left_index.get(left, {}).get(u, ())):
                if add_triple(head, t, v):
                    queue.append((head, t, v))

    result = set()
    for nonterminal, u, v in reached:
        if nonterminal == cfg.start_symbol and u in start_nodes and v in final_nodes:
            result.add((u, v))
    return result
