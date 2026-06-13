"""Shared ML training utilities for regression experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
        feature_dir = project_root / "src" / "data" / "classical_features" / "dataset_1"
    if target_dir is None:
        target_dir = project_root / "src" / "datasets" / "dataset_1" / "targets"

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
        # Route to the segregated embeddings layout based on the filename:
        #   node2vec_* -> network_based, otherwise feature_based
        #   *dataset2* -> dataset_2,    otherwise dataset_1
        name = str(filename)
        ds = "dataset_2" if "dataset2" in name else "dataset_1"
        sub = "network_based" if name.startswith("node2vec") else "feature_based"
        dataset_path = project_root / "src" / "data" / "embeddings" / ds / sub / filename

    df = pd.read_parquet(dataset_path)
    feature_cols = [c for c in df.columns if c.startswith("emb_")]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    return df, feature_cols


def top1_bank_cohort(train_df, target_col, id_col="bank_id", q=0.99):
    """Top-(1-q) banks by mean target on the supplied rows.

    Returns the set of bank ids whose mean ``target_col`` on ``train_df`` is at or
    above the ``q``-quantile of bank-level means. Pass training rows only to avoid
    leaking val/test labels into the cohort definition.
    """
    if id_col not in train_df.columns:
        raise KeyError(f"'{id_col}' missing from train_df; cannot build bank cohort.")
    per_bank = train_df.groupby(id_col)[target_col].mean()
    if per_bank.empty:
        return set()
    threshold = per_bank.quantile(q)
    return set(per_bank[per_bank >= threshold].index)


def top1_metrics(df, y_pred, cohort, target_col, id_col="bank_id"):
    """MAE and RMSE on rows whose ``id_col`` is in ``cohort``."""
    if id_col not in df.columns or not cohort:
        return {"mae": float("nan"), "rmse": float("nan")}
    mask = df[id_col].isin(cohort).to_numpy()
    if not mask.any():
        return {"mae": float("nan"), "rmse": float("nan")}
    y_true = np.asarray(df.loc[mask, target_col])
    y_pred_arr = np.asarray(y_pred)[mask]
    return {
        "mae": mean_absolute_error(y_true, y_pred_arr),
        "rmse": mean_squared_error(y_true, y_pred_arr) ** 0.5,
    }


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


def _make_stratify_bins(df, target_col, max_bins=10):
    """Quantile bins of the target for stratified splitting (or None if infeasible)."""
    y = df[target_col]
    max_bins = min(max_bins, y.nunique())
    for n_bins in range(max_bins, 1, -1):
        try:
            bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        counts = pd.Series(bins).value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            return bins
    return None


def random_split(df, target_col, test_size=0.15, n_bins=10, random_state=42):
    """Stratified random train/val/test split for static (non-temporal) datasets.

    Mirrors the split used in the combined threshold notebooks: a ``test_size``
    hold-out, then an equal-sized validation slice from the remainder, both
    stratified on quantile bins of the target.
    """
    strat = _make_stratify_bins(df, target_col, n_bins)
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True, stratify=strat,
    )
    val_strat = _make_stratify_bins(train_val_df, target_col, n_bins)
    train_df, val_df = train_test_split(
        train_val_df, test_size=test_size / (1 - test_size),
        random_state=random_state, shuffle=True, stratify=val_strat,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


@dataclass
class ModelTrainer:
    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str = "systemic_risk_label"
    split: str = "temporal"  # "temporal" (Dataset 1) or "random" (static datasets)
    train_end: tuple[int, int] = (2021, 4)
    val_end: tuple[int, int] = (2022, 4)
    test_end: tuple[int, int] = (2023, 3)
    test_size: float = 0.15
    stratify_bins: int = 10
    random_state: int = 42
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
        if self.split == "random":
            self.train_df, self.val_df, self.test_df = random_split(
                self.df,
                self.target_col,
                test_size=self.test_size,
                n_bins=self.stratify_bins,
                random_state=self.random_state,
            )
            # Static datasets have one row per bank, so a train-defined top-1%
            # cohort does not overlap the val/test rows — skip top-1% scoring.
            self._top1_bank_ids = set()
        else:
            self.train_df, self.val_df, self.test_df = time_split(
                self.df,
                train_end=self.train_end,
                val_end=self.val_end,
                test_end=self.test_end,
            )
            self._top1_bank_ids = top1_bank_cohort(self.train_df, self.target_col)

    def _score(self, model, split_df):
        y_arr = np.asarray(split_df[self.target_col])
        pred = model.predict(split_df[self.feature_cols])
        scores = {
            "mae": mean_absolute_error(y_arr, pred),
            "rmse": mean_squared_error(y_arr, pred) ** 0.5,
            "r2": r2_score(y_arr, pred),
        }
        top1 = top1_metrics(split_df, pred, self._top1_bank_ids, self.target_col)
        scores["top1_mae"] = top1["mae"]
        scores["top1_rmse"] = top1["rmse"]
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
            metrics = self._score(model, split_df)
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

    def leaderboard(self, round_decimals: int = 3):
        df = self.results_df.copy()
        if round_decimals is not None:
            metric_cols = [c for c in df.columns if any(k in c for k in ("mae", "rmse", "r2"))]
            # results_df columns are object dtype (inherited from the empty template),
            # so .round() is a no-op unless we cast to float first.
            df[metric_cols] = df[metric_cols].astype(float).round(round_decimals)
        return df

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
