"""Shared ML training utilities for regression experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASSICAL_FEATURE_CANDIDATES = [
    "degree_centrality_in",
    "degree_centrality_out",
    "degree_centrality_total",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "weighted_degree_in",
    "weighted_degree_out",
    "weighted_degree_total",
    "pagerank",
    "debtrank",
]


def iter_quarters(years=range(2016, 2024), quarters=(1, 2, 3, 4)):
    for year in years:
        for quarter in quarters:
            yield year, quarter


def load_classical_ml_dataset(
    project_root,
    target_col="systemic_risk_label",
    feature_dir=None,
    target_dir=None,
):
    project_root = Path(project_root)
    if feature_dir is None:
        feature_dir = project_root / "src" / "data" / "classical_features"
    if target_dir is None:
        target_dir = project_root / "src" /"datasets" / "targets"

    frames = []
    for year, quarter in iter_quarters():
        feature_path = Path(feature_dir) / f"classical_features_{year}Q{quarter}.parquet"
        target_path = Path(target_dir) / f"target_{year}Q{quarter}.csv"
        if not feature_path.exists() or not target_path.exists():
            continue

        features = pd.read_parquet(feature_path)
        target = pd.read_csv(target_path)

        features = features.copy()
        features["year"] = year
        features["quarter"] = quarter
        features["period"] = f"{year}Q{quarter}"
        frames.append(features.merge(target, on="bank_id", how="inner"))

    if not frames:
        raise ValueError("No classical feature files were merged.")

    df = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in CLASSICAL_FEATURE_CANDIDATES if c in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    return df, feature_cols


def load_gnn_ml_dataset(
    project_root,
    target_col="log_systemic_risk_label",
    dataset_path=None,
    filename="graphsage_srisk_dataset.parquet",
):
    project_root = Path(project_root)
    if dataset_path is None:
        dataset_path = project_root / "src" / "data" /"embeddings" / filename

    df = pd.read_parquet(dataset_path)
    feature_cols = [c for c in df.columns if c.startswith("emb_")]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    return df, feature_cols


def chronological_split(
    df,
    train_end=(2021, 4),
    val_end=(2022, 4),
    test_end=(2023, 4),
):
    keyed = df.copy()
    keyed["_quarter_key"] = keyed["year"] * 10 + keyed["quarter"]

    train_key = train_end[0] * 10 + train_end[1]
    val_key = val_end[0] * 10 + val_end[1]
    test_key = test_end[0] * 10 + test_end[1]

    train_df = keyed[keyed["_quarter_key"] <= train_key].drop(columns="_quarter_key")
    val_df = keyed[(keyed["_quarter_key"] > train_key) & (keyed["_quarter_key"] <= val_key)].drop(columns="_quarter_key")
    test_df = keyed[(keyed["_quarter_key"] > val_key) & (keyed["_quarter_key"] <= test_key)].drop(columns="_quarter_key")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def evaluate_regressor(model, X, y):
    pred = model.predict(X)
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
        "r2": r2_score(y, pred),
    }


def make_log_regression_model(regressor, scale_features=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "model",
            TransformedTargetRegressor(
                regressor=regressor,
                func=np.log1p,
                inverse_func=np.expm1,
            ),
        )
    )
    return Pipeline(steps)


@dataclass
class MLTrainAndStore:
    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str = "systemic_risk_label"
    train_end: tuple[int, int] = (2021, 4)
    val_end: tuple[int, int] = (2022, 4)
    test_end: tuple[int, int] = (2023, 3)
    models: dict[str, object] = field(default_factory=dict)
    results_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=[
        "model",
        "train_mae", "validation_mae", "test_mae",
        "train_rmse", "validation_rmse", "test_rmse",
        "train_r2", "validation_r2", "test_r2",
    ]))

    def __post_init__(self):
        self.train_df, self.val_df, self.test_df = chronological_split(
            self.df,
            train_end=self.train_end,
            val_end=self.val_end,
            test_end=self.test_end,
        )

    def train_and_store(self, model, name):
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df[self.target_col]
        X_val = self.val_df[self.feature_cols]
        y_val = self.val_df[self.target_col]
        X_test = self.test_df[self.feature_cols]
        y_test = self.test_df[self.target_col]

        model.fit(X_train, y_train)
        self.models[name] = model

        row = {"model": name}
        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            metrics = evaluate_regressor(model, X_split, y_split)
            row[f"{split_name}_mae"] = metrics["mae"]
            row[f"{split_name}_rmse"] = metrics["rmse"]
            row[f"{split_name}_r2"] = metrics["r2"]

        self.results_df = pd.concat(
            [self.results_df, pd.DataFrame([row])],
            ignore_index=True,
        )
        self.results_df = (
            self.results_df
            .sort_values(["validation_rmse", "validation_mae"], ascending=[True, True])
            .reset_index(drop=True)
        )
        return model

    def train_many(self, models: dict[str, object]):
        for name, model in models.items():
            self.train_and_store(model=model, name=name)
        return self.results()

    def results(self):
        return self.results_df.copy()

    def best_model_name(self):
        if self.results_df.empty:
            raise ValueError("No models have been trained.")
        return self.results_df.iloc[0]["model"]

    def best_model(self):
        return self.models[self.best_model_name()]

    def predict_test(self, model_name=None):
        if model_name is None:
            model_name = self.best_model_name()
        model = self.models[model_name]
        pred_df = self.test_df[
            ["bank_id", "year", "quarter", "period", self.target_col]
        ].copy()
        pred_df["prediction"] = model.predict(self.test_df[self.feature_cols])
        pred_df["abs_error"] = (pred_df[self.target_col] - pred_df["prediction"]).abs()
        return pred_df.sort_values("abs_error", ascending=False).reset_index(drop=True)
