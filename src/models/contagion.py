
import numpy as np
import pandas as pd


def simulate_failure(initial_bank, edges, nodes, mechanism="Exposure", alpha=1.0):
    """
    Simulate cascade when one bank fails.
    
    Args:
        initial_bank: ID of bank that fails first
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' and 'Equity' columns
        mechanism: "Exposure" or "liquidity"
        
    Returns:
        dict: failed_banks, total_loss, num_failed, rounds
    """
    equity = nodes.set_index('index')['Equity'].to_dict()
    remaining = equity.copy()

    failed = set([initial_bank])
    newly_failed = {initial_bank}
    rounds = 0

    while newly_failed:
        rounds += 1
        next_failed = set()

        for bank in newly_failed:
            if mechanism == "Exposure":
                affected = edges[edges['Targetid'] == bank]
                affected_col = 'Sourceid'
            else:
                affected = edges[edges['Sourceid'] == bank]
                affected_col = 'Targetid'

            for _, row in affected.iterrows():
                other = row[affected_col]

                if other in failed:
                    continue

                # Direct loss scaled by crisis severity
                loss = alpha * row['Weights']
                remaining[other] -= loss

                if remaining[other] <= 0:
                    next_failed.add(other)

        failed.update(next_failed)
        newly_failed = next_failed

    total_loss = sum(equity[b] for b in failed)

    return {
        "failed_banks": failed,
        "total_loss": total_loss,
        "num_failed": len(failed),
        "rounds": rounds,
    }



def simulate_partial_distress(initial_bank, edges, nodes,
                              alpha=1.0,
                              distress_threshold=1.0,
                              recovery_rate=0.0):
    """
    Simulate gradual distress propagation in an interbank network.

    This model allows partial losses to spread through the system before
    banks fully default. It is inspired by DebtRank but focuses on stress
    accumulation.

    Args:
        initial_bank: ID of the initially shocked bank.
        edges: DataFrame [Sourceid, Targetid, Weights].
               Sourceid = lender, Targetid = borrower.
        nodes: DataFrame with 'index' and 'Equity'.
        alpha: Crisis amplification factor.
        distress_threshold: Level at which a bank defaults.
        recovery_rate: Fraction of exposure recovered after default.

    Returns:
        dict:
            - distress_levels: final stress for each bank.
            - failed_banks: banks that defaulted.
            - total_loss: system equity lost.
            - rounds: number of propagation rounds.
    """

    equity = nodes.set_index('index')['Equity'].to_dict()
    bank_ids = nodes['index'].tolist()

    # Stress level between 0 and >1
    # 1 means full loss of equity
    distress = {b: 0.0 for b in bank_ids}

    # Initial shock
    distress[initial_bank] = 1.0

    active = {initial_bank}
    failed = set()
    rounds = 0

    while active:
        rounds += 1
        next_active = set()

        for bank in active:
            # Banks that lent to this distressed borrower
            affected = edges[edges['Targetid'] == bank]

            for _, row in affected.iterrows():
                creditor = row['Sourceid']

                if creditor in failed:
                    continue

                # Exposure relative to creditor capital
                exposure = row['Weights']
                impact = alpha * exposure / max(equity[creditor], 1e-8)

                # Propagate only incremental distress
                new_distress = distress[creditor] + distress[bank] * impact
                new_distress = min(new_distress, 2.0)  # cap to avoid explosion

                if new_distress > distress[creditor]:
                    distress[creditor] = new_distress
                    next_active.add(creditor)

                # Default condition
                if distress[creditor] >= distress_threshold:
                    failed.add(creditor)

        active = next_active

    # Compute system loss
    total_loss = 0.0
    for b in failed:
        total_loss += (1 - recovery_rate) * equity[b]

    return {
        "distress_levels": distress,
        "failed_banks": failed,
        "total_loss": total_loss,
        "rounds": rounds,
    }





def compute_systemic_importance(edges, nodes, mechanism="Exposure"):
    """
    For each bank: what happens if it fails?
    
    Args:
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' and 'Equity'
        mechanism: "Exposure" or "Liquidity"
        
    Returns:
        DataFrame:
            - bank_id
            - total_loss
            - num_failed
            - is_systemically_important (1 if above median, else 0) ------> To be discussed (In the END)
    """
    results = []
    
    for bank_id in nodes['index']:
        result = simulate_failure(bank_id, edges, nodes, mechanism)
        results.append({
            'bank_id': bank_id,
            'total_loss': result['total_loss'],
            'num_failed': result['num_failed']
        })
    
    df = pd.DataFrame(results)
    
    # Label: systemic if above median loss
    # df['is_systemically_important'] = (df['total_loss'] > df['total_loss'].median()).astype(int)
    df['is_systemically_important'] = (df['total_loss'] > df['total_loss'].quantile(0.8)).astype(int)
    
    return df


"""
Question: How should we define systemic importance?
Options:
- Above median total loss
- Above 75th percentile
- Fixed threshold (ex: more than 5 banks fail)
- Loss above threshold (ex: > 1M€)
"""