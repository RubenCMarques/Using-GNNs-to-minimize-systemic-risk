"""
Classical Centrality Algorithms

This module implements classical network centrality measures:
- Degree Centrality: Number of connections (in/out)
- Betweenness Centrality: Bridge Position
- Closeness Centrality: Distance to Others
- Eigenvector Centrality: Influence from neighbors
- Weighted Degree: Total exposure strength
- DebtRank: Systemic Impact
"""

import networkx as nx
import pandas as pd


def _build_graph(edges):
    """Build directed graph from edges DataFrame."""
    return nx.from_pandas_edgelist(
        edges, 'Sourceid', 'Targetid', 'Weights', create_using=nx.DiGraph()
    )


def degree_centrality(edges, nodes):
    """
    Compute degree centrality for each bank (in, out, and total).

    Degree centrality measures a node's influence or importance by counting its direct connections (edges) in a network.
    For directed graphs:
    - in-degree  = number of banks lending to this bank
    - out-degree = number of banks this bank lends to

    Banks with more connections = higher centrality.

    Args:
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' column

    Returns:
        DataFrame: bank_id, degree_centrality_in, degree_centrality_out, degree_centrality_total
    """
    G = _build_graph(edges)
    in_centrality = nx.in_degree_centrality(G)
    out_centrality = nx.out_degree_centrality(G)

    return pd.DataFrame([
        {
            'bank_id': bank,
            'degree_centrality_in': in_centrality.get(bank, 0.0),
            'degree_centrality_out': out_centrality.get(bank, 0.0),
            'degree_centrality_total': in_centrality.get(bank, 0.0) + out_centrality.get(bank, 0.0),
        }
        for bank in nodes['index']
    ])


def betweenness_centrality(edges, nodes):
    """
    Compute betweenness centrality for each bank.

    Betweenness centrality measures a node's importance in a network by counting how often it lies on the shortest paths
    between other pairs of nodes.

    Banks that act as bridges between others = higher centrality.

    Edge weights (exposures) are inverted before computing shortest paths because NetworkX interprets
    weights as distances: higher exposure → shorter effective distance → more likely bridge.

    Args:
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' column

    Returns:
        DataFrame: bank_id, betweenness_centrality
    """
    G = _build_graph(edges)
    for u, v, d in G.edges(data=True):
        d['inv_weight'] = 1.0 / d['Weights'] if d['Weights'] > 0 else float('inf')
    centrality = nx.betweenness_centrality(G, weight='inv_weight')

    return pd.DataFrame([
        {'bank_id': bank, 'betweenness_centrality': centrality.get(bank, 0.0)}
        for bank in nodes['index']
    ])


def closeness_centrality(edges, nodes):
    """
    Compute closeness centrality for each bank.

    Banks that can quickly reach others = higher centrality.
    Wasserman-Faust normalization is applied to handle partially disconnected
    interbank networks without inflating scores for isolated nodes.

    Args:
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' column

    Returns:
        DataFrame: bank_id, closeness_centrality
    """
    G = _build_graph(edges)
    centrality = nx.closeness_centrality(G, wf_improved=True)

    return pd.DataFrame([
        {'bank_id': bank, 'closeness_centrality': centrality.get(bank, 0.0)}
        for bank in nodes['index']
    ])


def eigenvector_centrality(edges, nodes, max_iter=100000):
    """
    Compute eigenvector centrality for each bank.

    Eigenvector centrality measures a node's influence based not just on how many connections it has,
    but on how well-connected its neighbors are. A bank connected to other highly connected banks
    scores higher than one connected to peripheral banks.

    Args:
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' column
        max_iter: Maximum iterations for convergence

    Returns:
        DataFrame: bank_id, eigenvector_centrality
    """
    G = _build_graph(edges)
    try:
        centrality = nx.eigenvector_centrality(G, max_iter=max_iter, weight='Weights')
    except nx.PowerIterationFailedConvergence:
        # Graph is disconnected — compute per weakly connected component.
        # eigenvector_centrality_numpy raises AmbiguousSolution on disconnected graphs.
        centrality = {}
        for component in nx.weakly_connected_components(G):
            sub = G.subgraph(component)
            if len(component) == 1:
                centrality[next(iter(component))] = 0.0
            else:
                try:
                    centrality.update(nx.eigenvector_centrality_numpy(sub, weight='Weights'))
                except (nx.AmbiguousSolution, Exception):
                    for node in component:
                        centrality[node] = 0.0

    return pd.DataFrame([
        {'bank_id': bank, 'eigenvector_centrality': centrality.get(bank, 0.0)}
        for bank in nodes['index']
    ])


def weighted_degree(edges, nodes):
    """
    Compute weighted degree (strength) for each bank.

    Weighted degree sums the weights of all edges connected to a node, capturing the total
    exposure volume rather than just the number of connections.

    - weighted_degree_in  = total borrowing exposure
    - weighted_degree_out = total lending exposure

    Args:
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' column

    Returns:
        DataFrame: bank_id, weighted_degree_in, weighted_degree_out, weighted_degree_total
    """
    G = _build_graph(edges)

    in_strength = dict(G.in_degree(weight='Weights'))
    out_strength = dict(G.out_degree(weight='Weights'))

    results = []
    for bank in nodes['index']:
        w_in = in_strength.get(bank, 0.0)
        w_out = out_strength.get(bank, 0.0)

        results.append({
            'bank_id': bank,
            'weighted_degree_in': w_in,
            'weighted_degree_out': w_out,
            'weighted_degree_total': w_in + w_out,
        })

    return pd.DataFrame(results)


def debtrank(edges, nodes):
    """
    Compute DebtRank for each bank.

    DebtRank centrality is a feedback-based metric used to assess the systemic importance of financial institutions by measuring the
    potential distress or loss an institution's failure can spread through the network, beyond just direct, immediate defaults.

    Measures systemic impact: if this bank fails, how much value is lost?

    Args:
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' and 'Equity' columns

    Returns:
        DataFrame: bank_id, debtrank
    """
    equity = {k: float(v) for k, v in nodes.set_index('index')['Equity'].items()}
    # Use only positive equity as the system baseline to avoid distortion from distressed banks
    total_equity = float(sum(v for v in equity.values() if v > 0))
    bank_ids = nodes['index'].tolist()
    bank_set = set(bank_ids)

    # Build reverse index: lenders_to[borrower] = {lender: weight}
    # Avoids O(n²) inner loop during propagation
    lenders_to = {bank: {} for bank in bank_ids}
    for row in edges.itertuples(index=False):
        lender, borrower, weight = row.Sourceid, row.Targetid, float(row.Weights)
        if lender in bank_set and borrower in bank_set:
            lenders_to[borrower][lender] = weight

    results = []

    for initial_bank in bank_ids:
        # State: 0 = healthy, 1 = distressed
        state = {bank: 0.0 for bank in bank_ids}
        state[initial_bank] = 1.0

        active = {initial_bank}
        initial_equity = equity[initial_bank]
        total_loss = initial_equity

        # Propagate for max N rounds
        for _ in range(len(bank_ids)):
            if not active:
                break

            next_active = set()

            for distressed_bank in active:
                if distressed_bank not in lenders_to:
                    continue
                for creditor, exp in lenders_to[distressed_bank].items():
                    if state[creditor] >= 1.0:
                        continue

                    # Impact = exposure / creditor's equity
                    eq = equity[creditor]
                    impact = min(exp / eq, 1.0) if eq > 0 else 1.0
                    new_state = min(state[creditor] + impact, 1.0)

                    if new_state > state[creditor]:
                        total_loss = total_loss + (new_state - state[creditor]) * eq
                        state[creditor] = new_state
                        next_active.add(creditor)

            active = next_active

        # DebtRank = fraction of system value lost (excluding the initial bank's own equity)
        dr = (total_loss - initial_equity) / total_equity if total_equity > 0 else 0.0
        results.append({'bank_id': initial_bank, 'debtrank': dr})

    return pd.DataFrame(results)

def pagerank_centrality(edges, nodes, alpha=0.85, reverse=False):
    """
    Compute PageRank centrality for each bank.

    PageRank measures systemic importance based on recursive influence
    from connected banks, with a damping factor that ensures stability
    in directed and disconnected interbank networks.

    Economic interpretation depends on graph direction:

    If Source -> Target means lender -> borrower:
    - reverse=False:
        High PageRank = important borrower
        (banks that receive funding from important lenders)

    - reverse=True:
        High PageRank = important lender
        (banks whose distress propagates through the system)

    Args:
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' column
        alpha: damping factor (default 0.85)
        reverse: whether to reverse graph direction

    Returns:
        DataFrame: bank_id, pagerank
    """

    # Clean and validate weights
    e = edges.copy()
    e["Weights"] = pd.to_numeric(e["Weights"], errors="coerce").fillna(0.0)
    e = e[e["Weights"] > 0]

    # Build graph
    G = nx.from_pandas_edgelist(
        e,
        "Sourceid",
        "Targetid",
        edge_attr="Weights",
        create_using=nx.DiGraph()
    )

    # Reverse graph if needed
    if reverse:
        G = G.reverse(copy=False)

    # Compute PageRank
    pr = nx.pagerank(G, alpha=alpha, weight="Weights")

    # Ensure all banks appear
    return pd.DataFrame([
        {
            "bank_id": bank,
            "pagerank": float(pr.get(bank, 0.0))
        }
        for bank in nodes["index"]
    ])