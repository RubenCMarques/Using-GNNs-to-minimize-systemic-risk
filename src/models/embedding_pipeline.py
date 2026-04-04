"""Utilities for quarter-by-quarter graph embedding experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data.data_loader import load_data
from src.models.embeddings import GNNConfig, Node2VecConfig, extract_embeddings_for_period


TARGET_COLUMNS = {"rank_next_quarter", "srisk_ratio", "srisk_value", "systemic_risk_label"}


@dataclass(frozen=True, order=True)
class QuarterKey:
    year: int
    quarter: int

    @property
    def period(self) -> str:
        return f"{self.year}Q{self.quarter}"


def iter_quarters(years=None, quarters=(1, 2, 3, 4)):
    """Yield quarter keys in chronological order."""
    if years is None:
        years = range(2016, 2024)

    for year in years:
        for quarter in quarters:
            yield QuarterKey(year=year, quarter=quarter)


def load_targets_for_period(year, quarter, target_col="systemic_risk_label", target_dir=None):
    """Load the supervised target table for one quarter."""
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent.parent / "datasets" / "targets"

    target_path = Path(target_dir) / f"target_{year}Q{quarter}.csv"
    if not target_path.exists():
        return pd.DataFrame(columns=["bank_id", "year", "quarter", "period", target_col])

    target_df = pd.read_csv(target_path)
    if target_col not in target_df.columns:
        return pd.DataFrame(columns=["bank_id", "year", "quarter", "period", target_col])

    target_df = (
        target_df[["bank_id", target_col]]
        .assign(
            year=year,
            quarter=quarter,
            period=f"{year}Q{quarter}",
        )
        .dropna(subset=[target_col])
        .reset_index(drop=True)
    )
    return target_df


def load_raw_features_for_period(year, quarter):
    """Load numeric bank features for one quarter, excluding labels."""
    _, nodes = load_data(year, quarter)

    feature_cols = [
        col
        for col in nodes.columns
        if col not in TARGET_COLUMNS
        and col != "index"
        and pd.api.types.is_numeric_dtype(nodes[col])
    ]

    return (
        nodes[["index"] + feature_cols]
        .rename(columns={"index": "bank_id"})
        .assign(
            year=year,
            quarter=quarter,
            period=f"{year}Q{quarter}",
        )
    )


def build_quarter_dataset(
    year,
    quarter,
    config: GNNConfig | Node2VecConfig,
    target_col="systemic_risk_label",
    include_raw_features=True,
    target_dir=None,
):
    """Build one quarter of the pooled supervised dataset."""
    embeddings = extract_embeddings_for_period(
        year,
        quarter,
        config=config,
    )
    targets = load_targets_for_period(year, quarter, target_col=target_col, target_dir=target_dir)

    if targets.empty:
        return pd.DataFrame()

    df = embeddings.merge(
        targets,
        on=["bank_id", "year", "quarter", "period"],
        how="inner",
    )

    if include_raw_features:
        raw_features = load_raw_features_for_period(year, quarter)
        df = df.merge(
            raw_features,
            on=["bank_id", "year", "quarter", "period"],
            how="left",
        )

    return df


def build_pooled_dataset(
    config: GNNConfig | Node2VecConfig,
    years=None,
    quarters=(1, 2, 3, 4),
    target_col="systemic_risk_label",
    include_raw_features=True,
    target_dir=None,
    output_path=None,
):
    """Build the full pooled dataset from quarter-specific embeddings."""
    frames = []
    for key in iter_quarters(years=years, quarters=quarters):
        quarter_df = build_quarter_dataset(
            key.year,
            key.quarter,
            config=config,
            target_col=target_col,
            include_raw_features=include_raw_features,
            target_dir=target_dir,
        )
        if not quarter_df.empty:
            frames.append(quarter_df)

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


def chronological_split(
    df,
    train_end=(2021, 4),
    val_end=(2022, 4),
    test_end=(2023, 3),
):
    """Split a pooled dataset into train/validation/test by quarter."""
    keyed = df.copy()
    keyed["_quarter_key"] = keyed["year"] * 10 + keyed["quarter"]

    train_key = train_end[0] * 10 + train_end[1]
    val_key = val_end[0] * 10 + val_end[1]
    test_key = test_end[0] * 10 + test_end[1]

    train_df = keyed[keyed["_quarter_key"] <= train_key].drop(columns="_quarter_key")
    val_df = keyed[(keyed["_quarter_key"] > train_key) & (keyed["_quarter_key"] <= val_key)].drop(columns="_quarter_key")
    test_df = keyed[(keyed["_quarter_key"] > val_key) & (keyed["_quarter_key"] <= test_key)].drop(columns="_quarter_key")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def default_feature_columns(df, target_col="systemic_risk_label"):
    """Select model input columns from the pooled table."""
    exclude = {"bank_id", "year", "quarter", "period", target_col}
    return [col for col in df.columns if col not in exclude]


def train_baseline_regressor(
    train_df,
    val_df,
    test_df,
    target_col="systemic_risk_label",
    feature_cols=None,
    random_state=42,
):
    """Fit a simple non-temporal regressor on the pooled quarter dataset."""
    if feature_cols is None:
        feature_cols = default_feature_columns(train_df, target_col=target_col)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(train_df[feature_cols], train_df[target_col])

    results = {}
    for split_name, split_df in {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }.items():
        if split_df.empty:
            results[split_name] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
            }
            continue

        predictions = model.predict(split_df[feature_cols])
        truth = split_df[target_col]
        results[split_name] = {
            "mae": mean_absolute_error(truth, predictions),
            "rmse": mean_squared_error(truth, predictions) ** 0.5,
            "r2": r2_score(truth, predictions),
        }

    return model, feature_cols, results
