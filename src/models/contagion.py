
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


def _get_asset_column(nodes):
    """Return the balance-sheet asset column used in the node table."""
    if "Total_assets" in nodes.columns:
        return "Total_assets"
    if "Assets" in nodes.columns:
        return "Assets"
    return None


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
        nodes: DataFrame with 'index', 'Equity', and preferably 'Total_assets' columns
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
              remaining_assets, fail_round, initial_shock_fraction, initial_default.
              If track_rounds=True, also includes round_summaries (list of dicts,
              one per cascade round with keys: round, new_failures_count,
              cum_failures_count, round_equity_depletion, cum_equity_depletion,
              num_active_spreaders).
    """
    equity = nodes.set_index('index')['Equity'].astype(float).to_dict()
    remaining_equity = equity.copy()

    asset_col = _get_asset_column(nodes)
    remaining_assets = None
    if asset_col is not None:
        assets = nodes.set_index("index")[asset_col].astype(float).to_dict()
        remaining_assets = assets.copy()

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

    # Apply the initial loss to assets and absorb it through equity.
    initial_loss = shock_fraction * equity[initial_bank]
    remaining_equity[initial_bank] -= initial_loss
    if remaining_assets is not None:
        remaining_assets[initial_bank] -= initial_loss

    initial_default = remaining_equity[initial_bank] <= 0
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

                # Loss given default on the exposure.
                loss = alpha * row['Weights']
                remaining_equity[other] -= loss
                if remaining_assets is not None:
                    remaining_assets[other] -= loss

                if track_rounds:
                    round_eq_depletion += loss

                if remaining_equity[other] <= 0:
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
        "remaining_equity": remaining_equity,
        "remaining_assets": remaining_assets,
        "total_loss": total_loss,
        "num_failed": len(failed),
        "rounds": rounds,
    }
    if track_rounds:
        result["round_summaries"] = round_summaries
    return result





def build_systemic_importance_summary(
    run_df,
    nodes,
):
    """
    Rank banks by the damage caused by their own default-contagion simulation.

    Args:
        run_df: DataFrame generated from one run per initial bank.
        nodes: DataFrame with at least 'index' and optionally 'Total_assets'.

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
    summary["systemic_risk_label"] = summary["cascade_size"]

    asset_col = _get_asset_column(nodes)
    if asset_col is not None:
        bank_assets = nodes.set_index("index")[asset_col].to_dict()
        total_assets = float(nodes[asset_col].sum())
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
    summary["impact_rank"] = summary["num_impacted"].rank(
        method="dense", ascending=False
    ).astype(int)

    ordered_cols = [
        "dataset_id",
        "year",
        "quarter",
        "bank_id",
        "initial_default",
        "systemic_risk_label",
        "cascade_size",
        "secondary_defaults",
        "causes_cascade",
        "rounds",
        "num_impacted",
        "failed_share",
        "impacted_share",
        "failed_equity_loss",
        "system_equity_depletion",
        "system_asset_depletion",
        "avg_loss_per_impacted",
        "max_bank_loss",
        "max_loss_bank_id",
        "initial_bank_assets",
        "affected_assets",
        "affected_assets_share",
        "cascade_rank",
        "impact_rank",
        "loss_rank",
        "run_id",
    ]
    return summary[ordered_cols].sort_values(
        ["systemic_risk_label", "num_impacted", "system_equity_depletion", "bank_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_target_table(importance_df):
    """
    Return the minimal regression target table for one quarter.

    Columns:
    - bank_id
    - systemic_risk_label
    """
    return (
        importance_df[["bank_id", "systemic_risk_label"]]
        .sort_values("bank_id")
        .reset_index(drop=True)
        .copy()
    )


def run_default_contagion_analysis(
    edges,
    nodes,
    *,
    year,
    quarter,
    mechanism="Exposure",
    alpha=1.0,
    track_rounds=True,
    output_dir=None,
):
    """
    Run one default-contagion simulation per bank and optionally write parquet tables.

    Option 1 semantics:
    - each bank is defaulted one at a time,
    - contagion starts only from actual default,
    - outputs contain the essential ranking metrics for regression targets.
    """
    equity_initial = {
        bank_id: float(value)
        for bank_id, value in nodes.set_index("index")["Equity"].items()
    }
    asset_col = _get_asset_column(nodes)
    asset_initial = None
    if asset_col is not None:
        asset_initial = {
            bank_id: float(value)
            for bank_id, value in nodes.set_index("index")[asset_col].items()
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
                asset_initial=asset_initial,
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
                asset_initial=asset_initial,
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
    )
    target_df = build_target_table(importance_df)

    outputs = {
        "runs": runs_df,
        "bank_state": bank_state_df,
        "round_summary": round_summary_df,
        "importance": importance_df,
        "target": target_df,
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        dataset_targets_dir = Path(__file__).resolve().parent.parent.parent / "src" / "datasets" / "dataset_1" / "targets"
        table_dirs = {
            "runs": output_path / "sim_runs",
            "bank_state": output_path / "sim_bank_state",
            "round_summary": output_path / "sim_round_summary",
            "importance": output_path / "systemic_importance",
            "target": dataset_targets_dir,
        }
        for table_dir in table_dirs.values():
            table_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "runs": table_dirs["runs"] / f"sim_runs_{year}Q{quarter}.parquet",
            "bank_state": table_dirs["bank_state"] / f"sim_bank_state_{year}Q{quarter}.parquet",
            "round_summary": table_dirs["round_summary"] / f"sim_round_summary_{year}Q{quarter}.parquet",
            "importance": table_dirs["importance"] / f"systemic_importance_{year}Q{quarter}.parquet",
            "target": table_dirs["target"] / f"target_{year}Q{quarter}.csv",
        }
        for key, df in outputs.items():
            if key == "target":
                df.to_csv(file_map[key], index=False)
            else:
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
        track_rounds=track_rounds,
        output_dir=output_dir,
    )
