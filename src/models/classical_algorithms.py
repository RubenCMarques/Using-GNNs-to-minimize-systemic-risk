"""
Classical Centrality Algorithms

This module implements classical network centrality measures:
- Degree Centrality: Number of connections
- Betweenness Centrality: Bridge Position
- Closeness Centrality: Distance to Others. ----> Evaluate if needed
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
    Compute degree centrality for each bank.
    
    Degree centrality measures a node's influence or importance by counting its direct connections (edges) in a network. 
    
    Banks with more connections = higher centrality.
    
    Args:
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' column
        
    Returns:
        DataFrame: bank_id, degree_centrality
    """
    G = _build_graph(edges)
    centrality = nx.degree_centrality(G)
    
    return pd.DataFrame([
        {'bank_id': bank, 'degree_centrality': centrality.get(bank, 0.0)}
        for bank in nodes['index']
    ])


def betweenness_centrality(edges, nodes):
    """
    Compute betweenness centrality for each bank.
    
    Betweenness centrality measures a node's importance in a network by counting how often it lies on the shortest paths
    between other pairs of nodes
    
    Banks that act as bridges between others = higher centrality.
    
    Args:
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' column
        
    Returns:
        DataFrame: bank_id, betweenness_centrality
    """
    G = _build_graph(edges)
    centrality = nx.betweenness_centrality(G, weight='Weights')
    
    return pd.DataFrame([
        {'bank_id': bank, 'betweenness_centrality': centrality.get(bank, 0.0)}
        for bank in nodes['index']
    ])


def closeness_centrality(edges, nodes):
    """
    Compute closeness centrality for each bank.
    
    Banks that can quickly reach others = higher centrality.
    
    Args:
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' column
        
    Returns:
        DataFrame: bank_id, closeness_centrality
    """
    G = _build_graph(edges)
    centrality = nx.closeness_centrality(G)
    
    return pd.DataFrame([
        {'bank_id': bank, 'closeness_centrality': centrality.get(bank, 0.0)}
        for bank in nodes['index']
    ])


def eigenvector_centrality(edges, nodes, max_iter=1000):
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
        centrality = nx.eigenvector_centrality_numpy(G, weight='Weights')

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
    equity = nodes.set_index('index')['Equity'].to_dict()
    total_equity = sum(equity.values())
    bank_ids = nodes['index'].tolist()
    
    # Build exposure matrix: exposure[i][j] = how much i loses if j defaults
    exposure = {bank: {} for bank in bank_ids}
    for _, row in edges.iterrows():
        lender = row['Sourceid']
        borrower = row['Targetid']
        weight = row['Weights']
        if lender in exposure:
            exposure[lender][borrower] = weight
    
    results = []
    
    for initial_bank in bank_ids:
        # State: 0 = healthy, 1 = distressed
        state = {bank: 0.0 for bank in bank_ids}
        state[initial_bank] = 1.0
        
        active = set([initial_bank])
        total_loss = equity[initial_bank]
        
        # Propagate for max N rounds
        for _ in range(len(bank_ids)):
            if not active:
                break
                
            next_active = set()
            
            for distressed_bank in active:
                # Find banks exposed to distressed bank
                for creditor in bank_ids:
                    if state[creditor] >= 1.0:
                        continue
                    
                    exp = exposure.get(creditor, {}).get(distressed_bank, 0)
                    if exp <= 0:
                        continue
                    
                    # Impact = exposure / creditor's equity
                    impact = min(exp / equity[creditor], 1.0) if equity[creditor] > 0 else 1.0
                    new_state = min(state[creditor] + impact, 1.0)
                    
                    if new_state > state[creditor]:
                        total_loss += (new_state - state[creditor]) * equity[creditor]
                        state[creditor] = new_state
                        next_active.add(creditor)
            
            active = next_active
        
        # DebtRank = fraction of system value lost
        dr = (total_loss - equity[initial_bank]) / total_equity if total_equity > 0 else 0
        results.append({'bank_id': initial_bank, 'debtrank': dr})
    
    return pd.DataFrame(results)