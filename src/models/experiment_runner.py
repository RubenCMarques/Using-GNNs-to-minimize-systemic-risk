"""Reusable regression experiment helpers for notebook workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RegressionExperiment:
    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str = "systemic_risk_label"
    train_end: tuple[int, int] = (2021, 4)
    val_end: tuple[int, int] = (2022, 4)
    test_end: tuple[int, int] = (2023, 3)
    models: dict[str, object] = field(default_factory=dict)
    results_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        self.train_df, self.val_df, self.test_df = chronological_split(
            self.df,
            train_end=self.train_end,
            val_end=self.val_end,
            test_end=self.test_end,
        )

    def fit(self, model, name: str):
        """Fit one model, evaluate all splits, and append one row to the results table."""
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
            for metric_name, metric_value in metrics.items():
                row[f"{split_name}_{metric_name}"] = metric_value

        self.results_df = pd.concat(
            [self.results_df, pd.DataFrame([row])],
            ignore_index=True,
        )
        self.results_df = self.results_df.sort_values(
            ["validation_rmse", "validation_mae"],
            ascending=[True, True],
        ).reset_index(drop=True)

        return model

    def get_results(self):
        return self.results_df.copy()

    def get_model(self, name: str):
        return self.models[name]

    def best_model_name(self):
        if self.results_df.empty:
            raise ValueError("No models have been fitted yet.")
        return self.results_df.iloc[0]["model"]

    def best_model(self):
        return self.get_model(self.best_model_name())

    def predict_test(self, name: str | None = None):
        """Return test predictions merged with metadata for one fitted model."""
        if name is None:
            name = self.best_model_name()

        model = self.get_model(name)
        pred_df = self.test_df[
            ["bank_id", "year", "quarter", "period", self.target_col]
        ].copy()
        pred_df["prediction"] = model.predict(self.test_df[self.feature_cols])
        pred_df["abs_error"] = (
            pred_df[self.target_col] - pred_df["prediction"]
        ).abs()
        return pred_df


def chronological_split(
    df: pd.DataFrame,
    train_end=(2021, 4),
    val_end=(2022, 4),
    test_end=(2023, 3),
):
    keyed = df.copy()
    keyed["_quarter_key"] = keyed["year"] * 10 + keyed["quarter"]

    train_key = train_end[0] * 10 + train_end[1]
    val_key = val_end[0] * 10 + val_end[1]
    test_key = test_end[0] * 10 + test_end[1]

    train_df = keyed[keyed["_quarter_key"] <= train_key].drop(columns="_quarter_key")
    val_df = keyed[(keyed["_quarter_key"] > train_key) & (keyed["_quarter_key"] <= val_key)].drop(columns="_quarter_key")
    test_df = keyed[(keyed["_quarter_key"] > val_key) & (keyed["_quarter_key"] <= test_key)].drop(columns="_quarter_key")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def evaluate_regressor(model, X, y):
    pred = model.predict(X)
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
        "r2": r2_score(y, pred),
    }


def make_log_regression_model(regressor, scale_features=False):
    """Wrap any sklearn regressor with optional scaling and log-target transform."""
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
