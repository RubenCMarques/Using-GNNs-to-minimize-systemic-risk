"""Utilities for quarter-by-quarter graph embedding experiments.

Temporal coherence is maintained by threading each quarter's trained model
state dict into the next quarter's initialisation (warm-starting). This means
embeddings evolve smoothly across time rather than being randomised anew each
quarter, making cross-quarter pooled datasets meaningful.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.data_loader import load_data
from src.models.embeddings import GNNConfig, Node2VecConfig, extract_embeddings_for_period


TARGET_COLUMNS = {"rank_next_quarter", "srisk_ratio", "srisk_value", "systemic_risk_label"}


def _iter_quarters(years=None, quarters=(1, 2, 3, 4)):
    if years is None:
        years = range(2016, 2024)
    for year in years:
        for quarter in quarters:
            yield year, quarter


def _load_targets(year, quarter, target_col="systemic_risk_label", target_dir=None):
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent / "datasets" / "targets"

    target_path = Path(target_dir) / f"target_{year}Q{quarter}.csv"
    if not target_path.exists():
        return pd.DataFrame(columns=["bank_id", "year", "quarter", "period", target_col])

    target_df = pd.read_csv(target_path)
    if target_col not in target_df.columns:
        return pd.DataFrame(columns=["bank_id", "year", "quarter", "period", target_col])

    return (
        target_df[["bank_id", target_col]]
        .assign(year=year, quarter=quarter, period=f"{year}Q{quarter}")
        .dropna(subset=[target_col])
        .reset_index(drop=True)
    )


def _load_raw_features(year, quarter):
    _, nodes = load_data(year, quarter)
    feature_cols = [
        col for col in nodes.columns
        if col not in TARGET_COLUMNS
        and col != "index"
        and pd.api.types.is_numeric_dtype(nodes[col])
    ]
    return (
        nodes[["index"] + feature_cols]
        .rename(columns={"index": "bank_id"})
        .assign(year=year, quarter=quarter, period=f"{year}Q{quarter}")
    )


def _build_quarter_dataset(
    year,
    quarter,
    config,
    target_col,
    include_raw_features,
    target_dir,
    init_state_dict=None,
):
    """Build one quarter's dataset and return it alongside the trained model state.

    Args:
        init_state_dict: state dict from the previous quarter's model.  When
            supplied the embedding model is warm-started, ensuring temporal
            coherence across quarters.

    Returns:
        (df, state_dict) — df is empty if no targets are available.
    """
    embeddings, state_dict = extract_embeddings_for_period(
        year, quarter, config=config, init_state_dict=init_state_dict
    )
    targets = _load_targets(year, quarter, target_col=target_col, target_dir=target_dir)

    if targets.empty:
        return pd.DataFrame(), state_dict

    df = embeddings.merge(targets, on=["bank_id", "year", "quarter", "period"], how="inner")

    if include_raw_features:
        raw_features = _load_raw_features(year, quarter)
        df = df.merge(raw_features, on=["bank_id", "year", "quarter", "period"], how="left")

    return df, state_dict


def build_pooled_dataset(
    config: GNNConfig | Node2VecConfig,
    years=None,
    quarters=(1, 2, 3, 4),
    target_col="systemic_risk_label",
    include_raw_features=True,
    target_dir=None,
    output_path=None,
):
    """Build the full pooled dataset from quarter-specific embeddings.

    Each quarter is warm-started from the previous quarter's model weights so
    that the embedding space is temporally coherent — dimension k means the
    same thing in 2016Q1 and 2016Q2.
    """
    frames = []
    prev_state_dict = None  # first quarter trains from a random init

    for year, quarter in _iter_quarters(years=years, quarters=quarters):
        df, prev_state_dict = _build_quarter_dataset(
            year,
            quarter,
            config,
            target_col,
            include_raw_features,
            target_dir,
            init_state_dict=prev_state_dict,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    pooled = pd.concat(frames, ignore_index=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            pooled.to_csv(output_path, index=False)
        else:
            pooled.to_parquet(output_path, index=False)

    return pooled
