"""Shared ML training utilities for regression experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def load_model(path):
    """Load a previously saved model pipeline from disk."""
    return joblib.load(path)


CLASSICAL_FEATURE_CANDIDATES = [
    "degree_centrality_total",
    "weighted_degree_total",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "pagerank",
    "debtrank",
]


def load_classical_dataset(
    project_root,
    target_col="systemic_risk_label",
    feature_dir=None,
    target_dir=None,
):
    project_root = Path(project_root)
    if feature_dir is None:
        feature_dir = project_root / "src" / "data" / "classical_features"
    if target_dir is None:
        target_dir = project_root / "src" / "datasets" / "targets"

    frames = []
    for year in range(2016, 2024):
        for quarter in range(1, 5):
            feature_path = Path(feature_dir) / f"classical_features_{year}Q{quarter}.parquet"
            target_path = Path(target_dir) / f"target_{year}Q{quarter}.csv"
            if not feature_path.exists() or not target_path.exists():
                continue

            features = pd.read_parquet(feature_path).copy()
            features["year"] = year
            features["quarter"] = quarter
            features["period"] = f"{year}Q{quarter}"
            frames.append(features.merge(pd.read_csv(target_path), on="bank_id", how="inner"))

    if not frames:
        raise ValueError("No classical feature files were merged.")

    df = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in CLASSICAL_FEATURE_CANDIDATES if c in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    return df, feature_cols


def load_gnn_dataset(
    project_root,
    target_col="log_systemic_risk_label",
    dataset_path=None,
    filename="graphsage_srisk_dataset.parquet",
):
    project_root = Path(project_root)
    if dataset_path is None:
        dataset_path = project_root / "src" / "data" / "embeddings" / filename

    df = pd.read_parquet(dataset_path)
    feature_cols = [c for c in df.columns if c.startswith("emb_")]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    return df, feature_cols


def time_split(
    df,
    train_end=(2021, 4),
    val_end=(2022, 4),
    test_end=(2023, 4),
):
    keyed = df.copy()
    keyed["_key"] = keyed["year"] * 10 + keyed["quarter"]

    train_key = train_end[0] * 10 + train_end[1]
    val_key = val_end[0] * 10 + val_end[1]
    test_key = test_end[0] * 10 + test_end[1]

    train_df = keyed[keyed["_key"] <= train_key].drop(columns="_key")
    val_df = keyed[(keyed["_key"] > train_key) & (keyed["_key"] <= val_key)].drop(columns="_key")
    test_df = keyed[(keyed["_key"] > val_key) & (keyed["_key"] <= test_key)].drop(columns="_key")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_pipeline(regressor):
    steps = [("imputer", SimpleImputer(strategy="median")), ("model", regressor)]
    return Pipeline(steps)


@dataclass
class ModelTrainer:
    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str = "systemic_risk_label"
    train_end: tuple[int, int] = (2021, 4)
    val_end: tuple[int, int] = (2022, 4)
    test_end: tuple[int, int] = (2023, 3)
    models: dict[str, object] = field(default_factory=dict)
    best_params: dict[str, dict] = field(default_factory=dict)
    results_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=[
        "model",
        "train_mae", "validation_mae", "test_mae",
        "train_rmse", "validation_rmse", "test_rmse",
        "train_r2", "validation_r2", "test_r2",
        "train_top1_mae", "validation_top1_mae",
        "train_top1_rmse", "validation_top1_rmse",
    ]))

    def __post_init__(self):
        self.train_df, self.val_df, self.test_df = time_split(
            self.df,
            train_end=self.train_end,
            val_end=self.val_end,
            test_end=self.test_end,
        )
        self._top1_threshold = self.df[self.target_col].quantile(0.99)

    def _score(self, model, X, y):
        y_arr = np.asarray(y)
        pred = model.predict(X)
        scores = {
            "mae": mean_absolute_error(y_arr, pred),
            "rmse": mean_squared_error(y_arr, pred) ** 0.5,
            "r2": r2_score(y_arr, pred),
        }
        mask = y_arr >= self._top1_threshold
        if mask.sum() > 0:
            scores["top1_mae"]  = mean_absolute_error(y_arr[mask], pred[mask])
            scores["top1_rmse"] = mean_squared_error(y_arr[mask], pred[mask]) ** 0.5
        else:
            scores["top1_mae"] = scores["top1_rmse"] = float("nan")
        return scores

    def train(self, model, name):
        import copy
        model = copy.deepcopy(model)
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df[self.target_col]

        model.fit(X_train, y_train)
        self.models[name] = model

        row = {"model": name}
        for split_name, split_df in [("train", self.train_df), ("validation", self.val_df), ("test", self.test_df)]:
            metrics = self._score(model, split_df[self.feature_cols], split_df[self.target_col])
            row[f"{split_name}_mae"]  = metrics["mae"]
            row[f"{split_name}_rmse"] = metrics["rmse"]
            row[f"{split_name}_r2"]   = metrics["r2"]
            if split_name != "test":
                row[f"{split_name}_top1_mae"]  = metrics["top1_mae"]
                row[f"{split_name}_top1_rmse"] = metrics["top1_rmse"]

        self.results_df = (
            pd.concat([self.results_df, pd.DataFrame([row])], ignore_index=True)
            .sort_values(["validation_rmse", "validation_mae"])
            .reset_index(drop=True)
        )
        return model

    def train_all(self, models: dict[str, object]):
        for name, model in models.items():
            self.train(model=model, name=name)
        return self.leaderboard()

    def leaderboard(self):
        return self.results_df.copy()

    def save_model(self, name, save_dir):
        """Save a fitted model to disk. Call explicitly when you want to persist it."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{name.replace(' ', '_')}.joblib"
        joblib.dump(self.models[name], path)
        print(f"Saved '{name}' → {path}")
        return path

    def test_predictions(self, model_name):
        model = self.models[model_name]
        pred_df = self.test_df[["bank_id", "year", "quarter", "period", self.target_col]].copy()
        pred_df["prediction"] = model.predict(self.test_df[self.feature_cols])
        pred_df["abs_error"] = (pred_df[self.target_col] - pred_df["prediction"]).abs()
        return pred_df.sort_values("abs_error", ascending=False).reset_index(drop=True)
