
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_loader import load_data
from src.data.simulation_store import (
    build_bank_rows,
    build_round_rows,
    build_run_row,
)


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
    Compatibility wrapper around the default-only systemic importance workflow.

    Returns the legacy compact output while delegating the simulations to the
    newer batch pipeline.
    """
    outputs = run_default_contagion_analysis(
        edges=edges,
        nodes=nodes,
        year=-1,
        quarter=-1,
        mechanism=mechanism,
        alpha=1.0,
        importance_quantile=0.80,
        track_rounds=False,
        output_dir=None,
    )
    importance_df = outputs["importance"]
    return importance_df.rename(
        columns={
            "cascade_size": "num_failed",
            "failed_equity_loss": "total_loss",
        }
    )[
        ["bank_id", "total_loss", "num_failed", "is_systemically_important"]
    ]


def build_systemic_importance_summary(
    run_df,
    nodes,
    importance_quantile=0.80,
):
    """
    Rank banks by the damage caused by their own default-contagion simulation.

    Args:
        run_df: DataFrame generated from one run per initial bank.
        nodes: DataFrame with at least 'index' and optionally 'Assets'.
        importance_quantile: Quantile cutoff used to label systemic banks.

    Returns:
        DataFrame with one row per initial bank and ranking metrics.
    """
    summary = run_df.copy()
    summary = summary.rename(
        columns={
            "initial_bank": "bank_id",
            "num_failed": "cascade_size",
        }
    )

    summary["secondary_defaults"] = (summary["cascade_size"] - 1).clip(lower=0)
    summary["causes_cascade"] = (summary["secondary_defaults"] > 0).astype(int)

    if "Assets" in nodes.columns:
        bank_assets = nodes.set_index("index")["Assets"].to_dict()
        total_assets = float(nodes["Assets"].sum())
        summary["initial_bank_assets"] = summary["bank_id"].map(bank_assets).fillna(0.0)
        summary["affected_assets"] = summary["impacted_share"] * total_assets
        summary["affected_assets_share"] = (
            summary["affected_assets"] / total_assets if total_assets > 0 else 0.0
        )
    else:
        summary["initial_bank_assets"] = np.nan
        summary["affected_assets"] = np.nan
        summary["affected_assets_share"] = np.nan

    summary["cascade_rank"] = summary["cascade_size"].rank(
        method="dense", ascending=False
    ).astype(int)
    summary["loss_rank"] = summary["system_equity_depletion"].rank(
        method="dense", ascending=False
    ).astype(int)
    summary["composite_score"] = (
        summary["failed_share"]
        + summary["impacted_share"]
        + (
            summary["system_equity_depletion"]
            / max(float(nodes["Equity"].sum()), 1e-8)
        )
    )
    summary["composite_rank"] = summary["composite_score"].rank(
        method="dense", ascending=False
    ).astype(int)

    cutoff = summary["composite_score"].quantile(importance_quantile)
    summary["is_systemically_important"] = (
        summary["composite_score"] >= cutoff
    ).astype(int)

    ordered_cols = [
        "dataset_id",
        "year",
        "quarter",
        "bank_id",
        "initial_default",
        "cascade_size",
        "secondary_defaults",
        "causes_cascade",
        "rounds",
        "num_impacted",
        "failed_share",
        "impacted_share",
        "failed_equity_loss",
        "system_equity_depletion",
        "avg_loss_per_impacted",
        "max_bank_loss",
        "max_loss_bank_id",
        "initial_bank_assets",
        "affected_assets",
        "affected_assets_share",
        "cascade_rank",
        "loss_rank",
        "composite_score",
        "composite_rank",
        "is_systemically_important",
        "run_id",
    ]
    return summary[ordered_cols].sort_values(
        ["composite_rank", "loss_rank", "bank_id"]
    ).reset_index(drop=True)


def run_default_contagion_analysis(
    edges,
    nodes,
    *,
    year,
    quarter,
    mechanism="Exposure",
    alpha=1.0,
    importance_quantile=0.80,
    track_rounds=True,
    output_dir=None,
):
    """
    Run one default-contagion simulation per bank and optionally write parquet tables.

    Option 1 semantics:
    - each bank is defaulted one at a time,
    - contagion starts only from actual default,
    - outputs contain the essential ranking metrics for systemic importance.
    """
    equity_initial = {
        bank_id: float(value)
        for bank_id, value in nodes.set_index("index")["Equity"].items()
    }
    n_banks = len(nodes)
    n_edges = len(edges)

    run_rows = []
    bank_rows = []
    round_rows = []

    for bank_id in nodes["index"]:
        run_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        result = simulate_failure(
            initial_bank=bank_id,
            edges=edges,
            nodes=nodes,
            mechanism=mechanism,
            alpha=alpha,
            spread_without_default=False,
            initial_loss_mode="fixed",
            initial_loss_frac=1.0,
            track_rounds=track_rounds,
        )
        runtime_ms = (time.perf_counter() - t0) * 1_000

        run_rows.append(
            build_run_row(
                run_id=run_id,
                result=result,
                equity_initial=equity_initial,
                n_banks=n_banks,
                n_edges=n_edges,
                initial_bank=bank_id,
                mechanism=mechanism,
                alpha=alpha,
                spread_without_default=False,
                initial_loss_mode="fixed",
                year=year,
                quarter=quarter,
                runtime_ms=runtime_ms,
            )
        )
        bank_rows.extend(
            build_bank_rows(
                run_id=run_id,
                result=result,
                equity_initial=equity_initial,
                initial_bank=bank_id,
            )
        )
        round_rows.extend(build_round_rows(run_id=run_id, result=result))

    runs_df = pd.DataFrame(run_rows)
    bank_state_df = pd.DataFrame(bank_rows)
    round_summary_df = pd.DataFrame(round_rows)
    importance_df = build_systemic_importance_summary(
        runs_df,
        nodes,
        importance_quantile=importance_quantile,
    )

    outputs = {
        "runs": runs_df,
        "bank_state": bank_state_df,
        "round_summary": round_summary_df,
        "importance": importance_df,
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        table_dirs = {
            "runs": output_path / "sim_runs",
            "bank_state": output_path / "sim_bank_state",
            "round_summary": output_path / "sim_round_summary",
            "importance": output_path / "systemic_importance",
        }
        for table_dir in table_dirs.values():
            table_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "runs": table_dirs["runs"] / f"sim_runs_{year}Q{quarter}.parquet",
            "bank_state": table_dirs["bank_state"] / f"sim_bank_state_{year}Q{quarter}.parquet",
            "round_summary": table_dirs["round_summary"] / f"sim_round_summary_{year}Q{quarter}.parquet",
            "importance": table_dirs["importance"] / f"systemic_importance_{year}Q{quarter}.parquet",
        }
        for key, df in outputs.items():
            df.to_parquet(file_map[key], index=False)
        outputs["files"] = {key: str(path) for key, path in file_map.items()}

    return outputs


def run_default_contagion_analysis_for_quarter(
    year,
    quarter,
    *,
    data_path=None,
    output_dir=None,
    mechanism="Exposure",
    alpha=1.0,
    importance_quantile=0.80,
    track_rounds=True,
):
    """
    Load one quarter, run the default-only contagion analysis, and export parquet.
    """
    edges, nodes = load_data(year, quarter, data_path=data_path)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data"

    return run_default_contagion_analysis(
        edges=edges,
        nodes=nodes,
        year=year,
        quarter=quarter,
        mechanism=mechanism,
        alpha=alpha,
        importance_quantile=importance_quantile,
        track_rounds=track_rounds,
        output_dir=output_dir,
    )


"""
Question: How should we define systemic importance?
Options:
- Above median total loss
- Above 75th percentile
- Fixed threshold (ex: more than 5 banks fail)
- Loss above threshold (ex: > 1M€)
"""
