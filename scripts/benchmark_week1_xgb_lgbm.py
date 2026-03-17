from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from chl_forecast.forecasting import (
    DATE_COLUMN,
    _week1_feature_columns,
    build_training_frame,
    load_data,
)


def main() -> None:
    csv = "/Users/a1amador/Downloads/SJL_daily_df.csv"
    out = Path("/Users/a1amador/Documents/Playground/models_week1_xgb_lgbm_search")
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(csv)
    training_frame, feature_columns, _ = build_training_frame(df)
    modeling = (
        training_frame.dropna(subset=["target_week_1"])
        .copy()
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    split = max(int(len(modeling) * 0.8), 1)
    train = modeling.iloc[:split].copy()
    test = modeling.iloc[split:].copy()
    selected = _week1_feature_columns(feature_columns)
    x_train = train[selected]
    y_train = train["target_week_1"]
    x_test = test[selected]
    y_test = test["target_week_1"]

    specs: list[tuple[str, dict[str, float | int]]] = []
    for max_depth in [3, 4, 5]:
        for n_estimators in [300, 600]:
            specs.append(
                (
                    "xgboost",
                    {
                        "model__n_estimators": n_estimators,
                        "model__learning_rate": 0.03,
                        "model__max_depth": max_depth,
                        "model__min_child_weight": 5,
                        "model__subsample": 0.8,
                        "model__colsample_bytree": 0.8,
                        "model__reg_lambda": 1.0,
                    },
                )
            )
    for num_leaves in [15, 31, 63]:
        for n_estimators in [300, 600]:
            specs.append(
                (
                    "lightgbm",
                    {
                        "model__n_estimators": n_estimators,
                        "model__learning_rate": 0.03,
                        "model__num_leaves": num_leaves,
                        "model__max_depth": -1,
                        "model__min_child_samples": 40,
                        "model__subsample": 0.8,
                        "model__colsample_bytree": 0.8,
                        "model__reg_lambda": 1.0,
                        "model__verbosity": -1,
                    },
                )
            )

    results = []
    cv = TimeSeriesSplit(n_splits=3)
    for i, (family, params) in enumerate(specs, start=1):
        if family == "xgboost":
            base = XGBRegressor(
                random_state=42 + i,
                objective="reg:squarederror",
                n_jobs=1,
            )
        else:
            base = LGBMRegressor(
                random_state=42 + i,
                n_jobs=1,
            )

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", base),
            ]
        )
        pipe.set_params(**params)
        estimator = TransformedTargetRegressor(
            regressor=pipe,
            func=np.log1p,
            inverse_func=np.expm1,
        )

        cv_mae = []
        cv_rmse = []
        cv_r2 = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(x_train), start=1):
            estimator.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = estimator.predict(x_train.iloc[val_idx])
            y_val = y_train.iloc[val_idx]
            cv_mae.append(float(mean_absolute_error(y_val, pred)))
            cv_rmse.append(float(math.sqrt(mean_squared_error(y_val, pred))))
            cv_r2.append(float(r2_score(y_val, pred)))

        estimator.fit(x_train, y_train)
        pred = estimator.predict(x_test)
        results.append(
            {
                "family": family,
                "params": params,
                "cv_mae_mean": float(np.mean(cv_mae)),
                "cv_rmse_mean": float(np.mean(cv_rmse)),
                "cv_r2_mean": float(np.mean(cv_r2)),
                "holdout_mae": float(mean_absolute_error(y_test, pred)),
                "holdout_rmse": float(math.sqrt(mean_squared_error(y_test, pred))),
                "holdout_r2": float(r2_score(y_test, pred)),
            }
        )

    results = sorted(
        results,
        key=lambda r: (-r["holdout_r2"], r["holdout_rmse"], r["holdout_mae"]),
    )
    (out / "week1_xgb_lgbm_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results[:8], indent=2))


if __name__ == "__main__":
    main()
