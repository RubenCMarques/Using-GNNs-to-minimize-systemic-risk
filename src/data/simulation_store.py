"""
Helpers for converting simulate_failure results into the normalized table schema.

Tables produced
---------------
Table 1 – runs (1 row per simulation run)
    build_run_row(...)

Table 2 – bank_end_state_sparse (1 row per impacted / initial / failed bank)
    build_bank_rows(...)

Table 3 – round_summary (1 row per cascade round per run; only when
           simulate_failure is called with track_rounds=True)
    build_round_rows(...)

Usage in a simulation loop
---------------------------
    equity_initial = nodes.set_index("index")["Equity"].to_dict()
    n_banks = len(nodes)
    n_edges = len(edges)

    run_rows, bank_rows, round_rows = [], [], []

    for bank_id, shock_frac in zip(sampled_banks, shock_fracs):
        run_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        result = simulate_failure(
            int(bank_id), edges, nodes,
            initial_loss_mode="fixed",
            initial_loss_frac=float(shock_frac),
            track_rounds=True,   # omit for speed; Table 3 only needs it
        )

        runtime_ms = (time.perf_counter() - t0) * 1_000

        run_rows.append(build_run_row(
            run_id, result, equity_initial, n_banks, n_edges,
            initial_bank=int(bank_id), mechanism="Exposure",
            alpha=1.0, spread_without_default=True,
            initial_loss_mode="fixed", year=year, quarter=q,
            runtime_ms=runtime_ms,
        ))
        bank_rows.extend(build_bank_rows(run_id, result, equity_initial, int(bank_id)))
        round_rows.extend(build_round_rows(run_id, result))
"""

import uuid
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Table 1 – runs
# ---------------------------------------------------------------------------

def build_run_row(
    run_id,
    result,
    equity_initial,
    n_banks,
    n_edges,
    initial_bank,
    mechanism,
    alpha,
    spread_without_default,
    initial_loss_mode,
    year,
    quarter,
    random_state=None,
    runtime_ms=None,
    timestamp_utc=None,
):
    """Return a single dict representing one row of the *runs* table (Table 1).

    Parameters
    ----------
    run_id : str
        Unique identifier for this simulation run (typically a UUID).
    result : dict
        Return value of ``simulate_failure``.
    equity_initial : dict
        Mapping {bank_id: initial_equity} precomputed once per quarter from
        ``nodes.set_index("index")["Equity"].to_dict()``.
    n_banks, n_edges : int
        Network size (precomputed per quarter).
    initial_bank : int / str
        ID of the initially shocked bank.
    mechanism : str
        "Exposure" or "Liquidity".
    alpha : float
        Loss amplification factor used in the run.
    spread_without_default : bool
    initial_loss_mode : str
        "fixed" or "random".
    year, quarter : int
    random_state : optional
        Seed used (if any).
    runtime_ms : float, optional
        Wall-clock time of the ``simulate_failure`` call in milliseconds.
    timestamp_utc : str, optional
        ISO-8601 timestamp.  Defaults to *now* when omitted.
    """
    remaining = result["remaining_equity"]

    # Per-bank loss (clamped at 0 — equity cannot go negative in accounting)
    losses = {b: max(0.0, equity_initial[b] - remaining[b]) for b in equity_initial}

    num_impacted = sum(1 for v in losses.values() if v > 0)
    system_equity_depletion = sum(losses.values())
    avg_loss_per_impacted = (
        system_equity_depletion / num_impacted if num_impacted > 0 else 0.0
    )

    pos_losses = {b: v for b, v in losses.items() if v > 0}
    if pos_losses:
        max_loss_bank_id = max(pos_losses, key=pos_losses.get)
        max_bank_loss = pos_losses[max_loss_bank_id]
    else:
        max_loss_bank_id = None
        max_bank_loss = 0.0

    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc).isoformat()

    return {
        # --- IDs / context ---
        "run_id": run_id,
        "dataset_id": f"{year}Q{quarter}",
        "year": year,
        "quarter": quarter,
        "mechanism": mechanism,
        "initial_bank": initial_bank,
        # --- Shock / params ---
        "initial_loss_mode": initial_loss_mode,
        "shock_fraction": result["initial_shock_fraction"],
        "alpha": alpha,
        "spread_without_default": spread_without_default,
        "random_state": random_state,
        # --- Cascade outcomes ---
        "initial_default": result["initial_default"],
        "rounds": result["rounds"],
        "num_failed": result["num_failed"],
        "num_impacted": num_impacted,
        "failed_share": result["num_failed"] / n_banks,
        "impacted_share": num_impacted / n_banks,
        # --- Loss metrics ---
        "failed_equity_loss": result["total_loss"],
        "system_equity_depletion": system_equity_depletion,
        "avg_loss_per_impacted": avg_loss_per_impacted,
        "max_bank_loss": max_bank_loss,
        "max_loss_bank_id": max_loss_bank_id,
        # --- Bookkeeping ---
        "n_banks": n_banks,
        "n_edges": n_edges,
        "runtime_ms": runtime_ms,
        "timestamp_utc": timestamp_utc,
    }


# ---------------------------------------------------------------------------
# Table 2 – bank_end_state_sparse
# ---------------------------------------------------------------------------

def build_bank_rows(run_id, result, equity_initial, initial_bank, eps=1e-10):
    """Return a list of dicts, one per bank that was impacted / failed / initial.

    Omits banks whose equity was not touched by the simulation and that are not
    the initially shocked bank.  This keeps the table sparse without losing
    any information about affected counterparties.

    Parameters
    ----------
    run_id : str
    result : dict
        Return value of ``simulate_failure``.
    equity_initial : dict
        Precomputed {bank_id: initial_equity} for the quarter.
    initial_bank : int / str
    eps : float
        Small constant to avoid division by zero in loss_frac_of_initial.
    """
    remaining = result["remaining_equity"]
    failed_banks = result["failed_banks"]
    fail_round_map = result["fail_round"]

    rows = []
    for bank_id, eq_init in equity_initial.items():
        eq_final = remaining[bank_id]
        eq_loss = max(0.0, eq_init - eq_final)
        is_impacted = eq_loss > 0
        is_initial = bank_id == initial_bank
        is_failed = bank_id in failed_banks

        if not (is_impacted or is_initial or is_failed):
            continue

        rows.append({
            "run_id": run_id,
            "bank_id": bank_id,
            "equity_initial": eq_init,
            "equity_final": eq_final,
            "equity_delta": eq_final - eq_init,
            "equity_loss": eq_loss,
            "failed": is_failed,
            "fail_round": fail_round_map.get(bank_id),
            "loss_frac_of_initial": eq_loss / max(eq_init, eps),
            "is_initial_bank": is_initial,
        })

    return rows


# ---------------------------------------------------------------------------
# Table 3 – round_summary
# ---------------------------------------------------------------------------

def build_round_rows(run_id, result):
    """Return a list of dicts, one per cascade round (from track_rounds=True).

    Returns an empty list when ``result`` does not contain ``round_summaries``
    (i.e. when ``simulate_failure`` was called without ``track_rounds=True``).
    """
    round_summaries = result.get("round_summaries")
    if not round_summaries:
        return []
    return [{"run_id": run_id, **rs} for rs in round_summaries]
