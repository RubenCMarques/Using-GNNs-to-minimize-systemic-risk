
import numpy as np
import pandas as pd


def simulate_failure(
    initial_bank,
    edges,
    nodes,
    mechanism="Exposure",
    alpha=1.0,
    spread_without_default=True,
    initial_loss_mode="random",
    initial_loss_frac=1.0,
    initial_loss_min=0.05,
    initial_loss_max=0.60,
    random_state=None,
    track_rounds=False,
):
    """
    Simulate cascade after an initial shock on one bank.

    Args:
        initial_bank: ID of bank that is shocked first
        edges: DataFrame [Sourceid, Targetid, Weights]
        nodes: DataFrame with 'index' and 'Equity' columns
        mechanism: "Exposure" or "liquidity"
        alpha: Loss amplification factor
        spread_without_default: If True, contagion starts even when initial
            bank does not default after the initial shock
        initial_loss_mode: "random" or "fixed"
        initial_loss_frac: Initial equity-loss fraction when mode is "fixed"
        initial_loss_min: Lower bound for random initial equity-loss fraction
        initial_loss_max: Upper bound for random initial equity-loss fraction
        random_state: Optional random seed for reproducibility
        track_rounds: If True, include round_summaries in the returned dict

    Returns:
        dict: failed_banks, total_loss, num_failed, rounds, remaining_equity,
              fail_round, initial_shock_fraction, initial_default.
              If track_rounds=True, also includes round_summaries (list of dicts,
              one per cascade round with keys: round, new_failures_count,
              cum_failures_count, round_equity_depletion, cum_equity_depletion,
              num_active_spreaders).
    """
    equity = nodes.set_index('index')['Equity'].to_dict()
    remaining = equity.copy()

    if initial_loss_mode not in {"random", "fixed"}:
        raise ValueError("initial_loss_mode must be 'random' or 'fixed'")

    if initial_loss_mode == "fixed":
        shock_fraction = float(initial_loss_frac)
    else:
        if initial_loss_min > initial_loss_max:
            raise ValueError("initial_loss_min must be <= initial_loss_max")
        rng = np.random.default_rng(random_state)
        shock_fraction = float(rng.uniform(initial_loss_min, initial_loss_max))

    if shock_fraction < 0:
        raise ValueError("Initial loss fraction must be >= 0")

    # Apply initial equity shock to the selected bank.
    remaining[initial_bank] -= shock_fraction * equity[initial_bank]

    initial_default = remaining[initial_bank] <= 0
    failed = set([initial_bank]) if initial_default else set()
    newly_failed = {initial_bank} if spread_without_default or initial_default else set()
    rounds = 0

    # round 0 = initial shock caused the default (before cascade begins)
    fail_round = {initial_bank: 0} if initial_default else {}

    round_summaries = [] if track_rounds else None
    cum_eq_depletion = 0.0

    while newly_failed:
        rounds += 1
        next_failed = set()
        num_spreaders = len(newly_failed)
        round_eq_depletion = 0.0

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

                if track_rounds:
                    round_eq_depletion += loss

                if remaining[other] <= 0:
                    next_failed.add(other)

        for b in next_failed:
            fail_round[b] = rounds

        failed.update(next_failed)
        newly_failed = next_failed

        if track_rounds:
            cum_eq_depletion += round_eq_depletion
            round_summaries.append({
                "round": rounds,
                "new_failures_count": len(next_failed),
                "cum_failures_count": len(failed),
                "round_equity_depletion": round_eq_depletion,
                "cum_equity_depletion": cum_eq_depletion,
                "num_active_spreaders": num_spreaders,
            })

    total_loss = sum(equity[b] for b in failed)

    result = {
        "initial_shock_fraction": shock_fraction,
        "initial_default": initial_default,
        "failed_banks": failed,
        "fail_round": fail_round,
        "remaining_equity": remaining,
        "total_loss": total_loss,
        "num_failed": len(failed),
        "rounds": rounds,
    }
    if track_rounds:
        result["round_summaries"] = round_summaries
    return result



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
