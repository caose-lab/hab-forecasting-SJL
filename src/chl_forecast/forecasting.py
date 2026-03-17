from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline

TARGET_COLUMN = "CHLL_NN_TOTAL"
DATE_COLUMN = "date"
QUALITY_COLUMN = "coverage_percent"
FORECAST_HORIZONS = (1, 2, 3)
MIN_TARGET_OBS_PER_WEEK = 4
MIN_COVERAGE_PERCENT = 40.0
ROLLING_WINDOWS = (7, 14, 28)
LAG_STEPS = (1, 7, 14, 28)
PERMUTATION_IMPORTANCE_REPEATS = 0
DEFAULT_TRAIN_HORIZONS = FORECAST_HORIZONS
EXCLUDED_PREDICTOR_COLUMNS = {
    DATE_COLUMN,
    TARGET_COLUMN,
    "temp_max",
    "temp_min",
    "wind_avg",
    "wind_speed_2m",
    "CHL_NN_R1",
    "CHL_NN_R2",
    "CHL_NN_R3",
    "CHL_OC4ME",
}
WEEK1_SMALL_ENV_COLUMNS = (
    "precipitation",
    "air_temperature",
    "water_temperature",
    "water_level",
)
WEEK2_SMALL_ENV_COLUMNS = WEEK1_SMALL_ENV_COLUMNS + ("Watt_per_m2", "AWND", "tidal_range")
WEEK3_PRUNED_EXCLUDED_PREFIXES = (
    "OWC",
    "acdm_443",
    "acdom_443",
    "anw_443",
    "aphy_443",
    "bbp_443",
    "bbp_slope",
    "iop_flags",
    "kd_490",
    "target_delta_",
)
WEEK3_COMPACT_EXCLUDED_PREFIXES = WEEK3_PRUNED_EXCLUDED_PREFIXES + (
    "Oa",
    "A865",
    "ADG443_NN",
    "KD490_M07",
    "PAR",
    "TSM_NN",
    "rho_665",
    "rho_681",
    "rho_709",
    "IWV",
    "total_pixels",
    "valid_pixels",
    "CI_index",
)
QUALITY_GATE_EXEMPT_COLUMNS = {
    DATE_COLUMN,
    QUALITY_COLUMN,
    "precipitation",
    "temp_max",
    "temp_min",
    "wind_avg",
    "wind_speed_2m",
    "air_pressure",
    "air_temperature",
    "water_level",
    "water_temperature",
    "Watt_per_m2",
    "AWND",
    "tidal_range",
}
@dataclass
class ModelArtifact:
    horizon: int
    model: Any
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    conformal_intervals: dict[str, Any]


@dataclass
class BacktestFoldResult:
    fold: int
    train_rows: int
    test_rows: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class Week1FeatureSetSpec:
    name: str
    env_columns: tuple[str, ...]


@dataclass(frozen=True)
class Week1TransformProfile:
    name: str
    include_p95: bool
    include_target_deltas: bool
    include_interactions: bool


@dataclass(frozen=True)
class Week1ModelSpec:
    family: str
    params: dict[str, Any]
    target_transform: str = "log1p"


RISK_LABELS = ("low", "medium", "high")


def _safe_float_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    numeric_columns = [c for c in df.columns if c != DATE_COLUMN]
    df = _safe_float_columns(df, numeric_columns)
    df = _apply_quality_gate(df)
    return df


def _apply_quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    if QUALITY_COLUMN not in df.columns:
        return df

    gated = df.copy()
    low_coverage_mask = gated[QUALITY_COLUMN].lt(MIN_COVERAGE_PERCENT).fillna(False)
    gated_columns = [c for c in gated.columns if c not in QUALITY_GATE_EXEMPT_COLUMNS]
    gated.loc[low_coverage_mask, gated_columns] = np.nan
    return gated


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in EXCLUDED_PREDICTOR_COLUMNS]


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    day_of_year = out[DATE_COLUMN].dt.dayofyear
    out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
    out["day_of_week"] = out[DATE_COLUMN].dt.dayofweek
    out["month"] = out[DATE_COLUMN].dt.month
    out["quarter"] = out[DATE_COLUMN].dt.quarter
    return out


def _days_since_last_observation(series: pd.Series) -> pd.Series:
    idx = np.arange(len(series))
    last_idx = np.where(series.notna(), idx, np.nan)
    last_idx = pd.Series(last_idx, index=series.index).ffill()
    return pd.Series(idx - last_idx, index=series.index, dtype=float)


def _extend_to_prediction_date(df: pd.DataFrame, prediction_date: pd.Timestamp) -> pd.DataFrame:
    max_date = df[DATE_COLUMN].max()
    if prediction_date <= max_date:
        return df

    future_dates = pd.date_range(start=max_date + pd.Timedelta(days=1), end=prediction_date, freq="D")
    future_frame = pd.DataFrame({DATE_COLUMN: future_dates})
    for column in df.columns:
        if column != DATE_COLUMN:
            future_frame[column] = np.nan
    return pd.concat([df, future_frame], ignore_index=True)


def _build_feature_frame(df: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    out = _add_calendar_features(df[[DATE_COLUMN]].copy())
    series_for_features = base_features + [TARGET_COLUMN]
    engineered: dict[str, pd.Series] = {}

    for column in series_for_features:
        observed = df[column].shift(1)
        engineered[f"{column}_is_missing"] = observed.isna().astype(float)
        engineered[f"{column}_raw"] = observed
        for lag in LAG_STEPS:
            engineered[f"{column}_lag_{lag}"] = observed.shift(lag)
        for window in ROLLING_WINDOWS:
            roll = observed.rolling(window=window, min_periods=max(3, window // 2))
            engineered[f"{column}_roll_mean_{window}"] = roll.mean()
            engineered[f"{column}_roll_std_{window}"] = roll.std()
            engineered[f"{column}_roll_p95_{window}"] = roll.quantile(0.95)
            q75 = roll.quantile(0.75)
            q25 = roll.quantile(0.25)
            engineered[f"{column}_roll_iqr_{window}"] = q75 - q25
            engineered[f"{column}_valid_count_{window}"] = observed.notna().rolling(
                window=window, min_periods=1
            ).sum()

        if column == TARGET_COLUMN:
            engineered["target_days_since_last_obs"] = _days_since_last_observation(observed)
            engineered["target_ewm_mean_7"] = observed.ewm(
                halflife=7, min_periods=3, adjust=False
            ).mean()
            engineered["target_ewm_mean_21"] = observed.ewm(
                halflife=21, min_periods=3, adjust=False
            ).mean()
            engineered["target_delta_raw_lag_7"] = observed - observed.shift(7)
            engineered["target_delta_raw_lag_14"] = observed - observed.shift(14)
            engineered["target_delta_raw_lag_28"] = observed - observed.shift(28)

    engineered_frame = pd.DataFrame(engineered, index=df.index)
    if TARGET_COLUMN in df.columns:
        target_observed = df[TARGET_COLUMN].shift(1)
        if "water_temperature" in df.columns:
            water_temp_observed = df["water_temperature"].shift(1)
            engineered_frame["target_level_x_water_temperature"] = (
                target_observed * water_temp_observed
            )
        if "precipitation" in df.columns:
            precipitation_observed = df["precipitation"].shift(1)
            engineered_frame["target_level_x_precipitation"] = (
                target_observed * precipitation_observed
            )
    return pd.concat([out, engineered_frame], axis=1)


def _feature_profile_columns(feature_columns: list[str], profile: str) -> list[str]:
    selected: list[str] = []
    for column in feature_columns:
        is_target = column.startswith("CHLL_NN_TOTAL") or column.startswith("target_")
        is_weather = any(
            column.startswith(prefix)
            for prefix in [
                "precipitation",
                "temp_max",
                "temp_min",
                "wind_avg",
                "wind_speed_2m",
                "air_pressure",
                "air_temperature",
                "water_level",
                "water_temperature",
                "Watt_per_m2",
                "AWND",
            ]
        )
        is_calendar = column.startswith("day_") or column in {"month", "quarter"}
        is_week1_small_env = column.startswith(WEEK1_SMALL_ENV_COLUMNS)
        is_week2_small_env = column.startswith(WEEK2_SMALL_ENV_COLUMNS)

        if profile == "all":
            selected.append(column)
        elif profile == "target_weather_calendar" and (is_target or is_weather or is_calendar):
            selected.append(column)
        elif profile == "target_calendar" and (is_target or is_calendar):
            selected.append(column)
        elif profile == "target_small_env_calendar" and (
            is_target or is_week1_small_env or is_calendar
        ):
            selected.append(column)
        elif profile == "target_small_env_calendar_p95":
            if is_target or is_week1_small_env or is_calendar:
                if "_roll_iqr_" in column:
                    continue
                selected.append(column)
        elif profile == "target_small_env_week2_calendar_p95":
            if is_target or is_week2_small_env or is_calendar:
                if "_roll_iqr_" in column:
                    continue
                selected.append(column)
        elif profile == "all_week3_pruned":
            if any(column.startswith(prefix) for prefix in WEEK3_PRUNED_EXCLUDED_PREFIXES):
                continue
            selected.append(column)
        elif profile == "all_week3_compact":
            if any(column.startswith(prefix) for prefix in WEEK3_COMPACT_EXCLUDED_PREFIXES):
                continue
            selected.append(column)

    if not selected:
        raise ValueError(f"Feature profile '{profile}' produced no columns.")
    return selected


def _future_weekly_average(
    target: pd.Series,
    start_offset: int,
    window: int = 7,
    min_obs: int = MIN_TARGET_OBS_PER_WEEK,
) -> pd.Series:
    values = target.to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    for idx in range(len(values)):
        start = idx + start_offset
        end = start + window
        if end > len(values):
            break
        window_values = values[start:end]
        valid = window_values[~np.isnan(window_values)]
        if valid.size >= min_obs:
            result[idx] = float(valid.mean())
    return pd.Series(result, index=target.index)


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    base_features = _select_feature_columns(df)
    feature_frame = _build_feature_frame(df, base_features)
    for horizon in FORECAST_HORIZONS:
        feature_frame[f"target_week_{horizon}"] = _future_weekly_average(
            df[TARGET_COLUMN], start_offset=(horizon - 1) * 7
        )
    feature_frame = feature_frame.dropna(
        subset=[f"target_week_{horizon}" for horizon in FORECAST_HORIZONS], how="all"
    ).reset_index(drop=True)
    excluded = {DATE_COLUMN} | {f"target_week_{horizon}" for horizon in FORECAST_HORIZONS}
    feature_columns = [c for c in feature_frame.columns if c not in excluded]
    return feature_frame, feature_columns, base_features


def build_inference_frame(
    df: pd.DataFrame, base_features: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    if base_features is None:
        base_features = _select_feature_columns(df)
    feature_frame = _build_feature_frame(df, base_features)
    feature_columns = [c for c in feature_frame.columns if c != DATE_COLUMN]
    return feature_frame, feature_columns


def _make_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _default_hist_model(random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        random_state=random_state,
        early_stopping=False,
    )


def _default_hist_classifier(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=random_state,
        early_stopping=False,
    )


def _build_estimator(
    random_state: int,
    model_params: dict[str, Any],
    target_transform: str = "none",
) -> Any:
    model = _default_hist_model(random_state)
    pipeline = _make_pipeline(model)
    pipeline.set_params(**model_params)
    if target_transform == "log1p":
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return pipeline


def _wrap_regressor(estimator: Any, target_transform: str) -> Any:
    if target_transform == "log1p":
        return TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return estimator


def _build_regression_estimator(
    family: str,
    random_state: int,
    model_params: dict[str, Any],
    target_transform: str = "none",
) -> Any:
    if family == "hist_gradient_boosting":
        estimator = _make_pipeline(_default_hist_model(random_state))
    elif family == "random_forest":
        estimator = _make_pipeline(RandomForestRegressor(random_state=random_state, n_jobs=1))
    elif family == "extra_trees":
        estimator = _make_pipeline(ExtraTreesRegressor(random_state=random_state, n_jobs=1))
    else:
        raise ValueError(f"Unsupported regression family: {family}")

    estimator.set_params(**model_params)
    return _wrap_regressor(estimator, target_transform)


def _default_baseline_strategy(horizon: int) -> dict[str, Any]:
    if horizon == 1:
        return {
            "feature_profile": "target_small_env_calendar_p95",
            "target_transform": "log1p",
            "model_family": "hist_gradient_boosting",
            "model_params": {
                "model__learning_rate": 0.02,
                "model__max_depth": 4,
                "model__max_iter": 2000,
                "model__max_leaf_nodes": 23,
                "model__min_samples_leaf": 90,
                "model__l2_regularization": 1.8,
            },
        }
    if horizon == 2:
        return {
            "feature_profile": "target_small_env_week2_calendar_p95",
            "target_transform": "log1p",
            "model_family": "hist_gradient_boosting",
            "model_params": {
                "model__learning_rate": 0.03,
                "model__max_depth": 4,
                "model__max_iter": 1000,
                "model__max_leaf_nodes": 23,
                "model__min_samples_leaf": 90,
                "model__l2_regularization": 1.5,
            },
        }
    if horizon == 3:
        return {
            "feature_profile": "all_week3_compact",
            "target_transform": "log1p",
            "model_family": "hist_gradient_boosting",
            "model_params": {
                "model__learning_rate": 0.03,
                "model__max_depth": 3,
                "model__max_iter": 1200,
                "model__max_leaf_nodes": 31,
                "model__min_samples_leaf": 120,
                "model__l2_regularization": 2.0,
            },
        }
    return {
        "feature_profile": "all",
        "target_transform": "none",
        "model_family": "hist_gradient_boosting",
        "model_params": {
            "model__learning_rate": 0.05,
            "model__max_depth": 5,
            "model__max_iter": 400,
            "model__max_leaf_nodes": 31,
            "model__min_samples_leaf": 20,
            "model__l2_regularization": 0.1,
        },
    }


def _default_risk_strategy(horizon: int = 1) -> dict[str, Any]:
    if horizon == 1:
        return {
            "feature_profile": "week1_base_no_awnd_tidal",
            "weight_strategy": "inverse",
            "model_family": "hist_gradient_boosting_classifier",
            "model_params": {
                "model__learning_rate": 0.05,
                "model__max_depth": 4,
                "model__max_iter": 300,
                "model__max_leaf_nodes": 63,
                "model__min_samples_leaf": 90,
                "model__l2_regularization": 0.8,
            },
        }
    if horizon == 2:
        return {
            "feature_profile": "week3_pruned",
            "weight_strategy": "inverse_sqrt",
            "model_family": "hist_gradient_boosting_classifier",
            "model_params": {
                "model__learning_rate": 0.03,
                "model__max_depth": 3,
                "model__max_iter": 800,
                "model__max_leaf_nodes": 63,
                "model__min_samples_leaf": 60,
                "model__l2_regularization": 0.8,
            },
        }
    if horizon == 3:
        return {
            "feature_profile": "all_week3_pruned",
            "weight_strategy": "none",
            "model_family": "hist_gradient_boosting_classifier",
            "model_params": {
                "model__learning_rate": 0.03,
                "model__max_depth": 3,
                "model__max_iter": 800,
                "model__max_leaf_nodes": 23,
                "model__min_samples_leaf": 90,
                "model__l2_regularization": 1.5,
            },
        }
    return {
        "feature_profile": "target_small_env_calendar_p95",
        "weight_strategy": "none",
        "model_family": "hist_gradient_boosting_classifier",
        "model_params": {
            "model__learning_rate": 0.03,
            "model__max_depth": 4,
            "model__max_iter": 500,
            "model__max_leaf_nodes": 31,
            "model__min_samples_leaf": 60,
            "model__l2_regularization": 0.8,
        },
    }


def _default_high_risk_strategy(horizon: int = 1) -> dict[str, Any]:
    if horizon == 1:
        return {
            "feature_profile": "week2_classifier",
            "weight_strategy": "inverse_sqrt",
            "threshold_strategy": "recall_focused",
            "model_family": "hist_gradient_boosting_classifier",
            "model_params": {
                "model__learning_rate": 0.02,
                "model__max_depth": 3,
                "model__max_iter": 300,
                "model__max_leaf_nodes": 23,
                "model__min_samples_leaf": 40,
                "model__l2_regularization": 1.2,
            },
        }
    if horizon == 2:
        return {
            "feature_profile": "week2_default",
            "weight_strategy": "inverse",
            "threshold_strategy": "recall_focused",
            "model_family": "hist_gradient_boosting_classifier",
            "model_params": {
                "model__learning_rate": 0.05,
                "model__max_depth": 3,
                "model__max_iter": 300,
                "model__max_leaf_nodes": 63,
                "model__min_samples_leaf": 40,
                "model__l2_regularization": 0.8,
            },
        }
    if horizon == 3:
        return {
            "feature_profile": "week3_pruned",
            "weight_strategy": "none",
            "threshold_strategy": "recall_focused",
            "model_family": "hist_gradient_boosting_classifier",
            "model_params": {
                "model__learning_rate": 0.05,
                "model__max_depth": 4,
                "model__max_iter": 500,
                "model__max_leaf_nodes": 23,
                "model__min_samples_leaf": 60,
                "model__l2_regularization": 2.0,
            },
        }
    strategy = _default_risk_strategy(horizon).copy()
    strategy["weight_strategy"] = "inverse"
    strategy["threshold_strategy"] = "recall_focused"
    return strategy


def _default_sample_weight(n_rows: int, horizon: int) -> np.ndarray | None:
    if horizon not in {2, 3} or n_rows <= 1:
        return None
    ages = np.arange(n_rows, dtype=float)
    if horizon == 3:
        return 0.25 + 0.75 * (ages / ages.max()) ** 1.5
    return 0.35 + 0.65 * (ages / ages.max()) ** 1.5


def _conformal_residual_quantile(residuals: np.ndarray, alpha: float) -> float:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[~np.isnan(residuals)]
    if residuals.size == 0:
        return float("nan")
    sorted_residuals = np.sort(residuals)
    rank = int(np.ceil((sorted_residuals.size + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), sorted_residuals.size - 1)
    return float(sorted_residuals[rank])


def _get_model_search_space(horizon: int, random_state: int) -> list[tuple[str, Pipeline, dict[str, list[Any]]]]:
    spaces: list[tuple[str, Pipeline, dict[str, list[Any]]]] = [
        (
            "hist_gradient_boosting",
            _make_pipeline(_default_hist_model(random_state)),
            {
                "model__learning_rate": [0.02, 0.035, 0.05, 0.08, 0.12],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__max_depth": [None, 3, 5, 7],
                "model__min_samples_leaf": [10, 20, 30, 50],
                "model__l2_regularization": [0.0, 0.05, 0.1, 0.3, 0.6],
                "model__max_iter": [250, 400, 600],
            },
        )
    ]

    if horizon == 1:
        spaces.extend(
            [
                (
                    "random_forest",
                    _make_pipeline(RandomForestRegressor(random_state=random_state, n_jobs=1)),
                    {
                        "model__n_estimators": [300, 500],
                        "model__max_depth": [None, 8, 12, 16],
                        "model__min_samples_leaf": [1, 3, 5, 10],
                        "model__max_features": ["sqrt", 0.3, 0.5, None],
                    },
                ),
                (
                    "extra_trees",
                    _make_pipeline(ExtraTreesRegressor(random_state=random_state, n_jobs=1)),
                    {
                        "model__n_estimators": [300, 500],
                        "model__max_depth": [None, 8, 12, 16],
                        "model__min_samples_leaf": [1, 3, 5, 10],
                        "model__max_features": ["sqrt", 0.3, 0.5, None],
                    },
                ),
            ]
        )

    return spaces


def _search_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
    random_state: int,
    n_iter: int,
) -> tuple[Pipeline, dict[str, Any]]:
    if n_iter <= 1:
        strategy = _default_baseline_strategy(horizon)
        estimator = _build_estimator(
            random_state=random_state,
            model_params=strategy["model_params"],
            target_transform=strategy["target_transform"],
        )
        fit_kwargs = {}
        sample_weight = _default_sample_weight(len(X_train), horizon)
        if sample_weight is not None:
            fit_kwargs["model__sample_weight"] = sample_weight
        estimator.fit(X_train, y_train, **fit_kwargs)
        return estimator, {
            "mode": "fixed_baseline",
            "model_family": strategy["model_family"],
            "feature_profile": strategy["feature_profile"],
            "target_transform": strategy["target_transform"],
            "sample_weighting": "recency_soft" if sample_weight is not None else "none",
            **strategy["model_params"],
        }

    splits = min(4, max(2, len(X_train) // 250))
    cv = TimeSeriesSplit(n_splits=splits)
    best_score = np.inf
    best_pipeline: Any | None = None
    best_params: dict[str, Any] | None = None

    for family_name, pipeline, param_space in _get_model_search_space(horizon, random_state):
        sampler = list(ParameterSampler(param_space, n_iter=n_iter, random_state=random_state))
        for sampled_params in sampler:
            candidate = clone(pipeline)
            candidate.set_params(**sampled_params)
            fold_scores = []
            for train_idx, val_idx in cv.split(X_train):
                candidate.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                preds = candidate.predict(X_train.iloc[val_idx])
                fold_scores.append(mean_absolute_error(y_train.iloc[val_idx], preds))
            mean_score = float(np.mean(fold_scores))
            if mean_score < best_score:
                best_score = mean_score
                best_pipeline = clone(candidate)
                best_params = {"mode": "time_series_search", "model_family": family_name, **sampled_params}

    if best_pipeline is None or best_params is None:
        raise RuntimeError("Model search did not produce a candidate.")

    best_pipeline.fit(X_train, y_train)
    return best_pipeline, best_params


def _fit_horizon_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    random_state: int,
    n_iter: int,
) -> ModelArtifact:
    target_column = f"target_week_{horizon}"
    modeling_frame = frame.dropna(subset=[target_column]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    selected_feature_columns = _horizon_feature_columns(feature_columns, horizon)

    split_index = max(int(len(modeling_frame) * 0.8), 1)
    train_frame = modeling_frame.iloc[:split_index]
    test_frame = modeling_frame.iloc[split_index:]
    if test_frame.empty:
        raise ValueError(f"Not enough samples to create a holdout set for horizon {horizon}.")

    X_train = train_frame[selected_feature_columns]
    y_train = train_frame[target_column]
    X_test = test_frame[selected_feature_columns]
    y_test = test_frame[target_column]

    best_model, best_params = _search_best_model(
        X_train=X_train,
        y_train=y_train,
        horizon=horizon,
        random_state=random_state + horizon,
        n_iter=n_iter,
    )

    predictions = best_model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_start": str(train_frame[DATE_COLUMN].min().date()),
        "train_end": str(train_frame[DATE_COLUMN].max().date()),
        "test_start": str(test_frame[DATE_COLUMN].min().date()),
        "test_end": str(test_frame[DATE_COLUMN].max().date()),
        "best_params": best_params,
        "feature_count": int(len(selected_feature_columns)),
    }

    conformal_intervals: dict[str, Any] = {}
    calibration_split = max(int(len(train_frame) * 0.8), 1)
    calibration_train = train_frame.iloc[:calibration_split]
    calibration_frame = train_frame.iloc[calibration_split:]
    if len(calibration_frame) >= 25:
        conformal_model = clone(best_model)
        fit_kwargs = {}
        sample_weight = _default_sample_weight(len(calibration_train), horizon)
        if sample_weight is not None:
            fit_kwargs["model__sample_weight"] = sample_weight
        conformal_model.fit(
            calibration_train[selected_feature_columns],
            calibration_train[target_column],
            **fit_kwargs,
        )
        calibration_predictions = conformal_model.predict(
            calibration_frame[selected_feature_columns]
        )
        residuals = np.abs(calibration_frame[target_column].to_numpy() - calibration_predictions)
        conformal_intervals = {
            "calibration_rows": int(len(calibration_frame)),
            "calibration_start": str(calibration_frame[DATE_COLUMN].min().date()),
            "calibration_end": str(calibration_frame[DATE_COLUMN].max().date()),
            "alphas": {
                "0.5": _conformal_residual_quantile(residuals, 0.5),
                "0.32": _conformal_residual_quantile(residuals, 0.32),
                "0.2": _conformal_residual_quantile(residuals, 0.2),
                "0.1": _conformal_residual_quantile(residuals, 0.1),
            },
        }
        metrics["conformal_intervals"] = conformal_intervals

    if PERMUTATION_IMPORTANCE_REPEATS > 0:
        importance = permutation_importance(
            best_model,
            X_test,
            y_test,
            n_repeats=PERMUTATION_IMPORTANCE_REPEATS,
            random_state=random_state + horizon,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        importance_df = pd.DataFrame(
            {
                "feature": selected_feature_columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
    else:
        importance_df = pd.DataFrame(
            columns=["feature", "importance_mean", "importance_std"]
        )

    return ModelArtifact(
        horizon=horizon,
        model=best_model,
        metrics=metrics,
        feature_importance=importance_df,
        conformal_intervals=conformal_intervals,
    )


def expanding_window_backtest(
    csv_path: str | Path,
    horizon: int = 1,
    n_folds: int = 5,
    min_train_rows: int = 600,
) -> dict[str, Any]:
    if horizon not in FORECAST_HORIZONS:
        raise ValueError(f"Horizon must be one of {FORECAST_HORIZONS}.")

    df = load_data(csv_path)
    training_frame, feature_columns, _ = build_training_frame(df)
    target_column = f"target_week_{horizon}"
    modeling_frame = training_frame.dropna(subset=[target_column]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    strategy = _default_baseline_strategy(horizon)
    selected_feature_columns = _feature_profile_columns(
        feature_columns, strategy["feature_profile"]
    )

    total_rows = len(modeling_frame)
    if total_rows <= min_train_rows + n_folds:
        raise ValueError("Not enough rows for the requested backtest configuration.")

    available_test_rows = total_rows - min_train_rows
    fold_test_size = available_test_rows // n_folds
    if fold_test_size < 25:
        raise ValueError("Fold test windows are too small; reduce n_folds or min_train_rows.")

    folds: list[BacktestFoldResult] = []
    all_true: list[float] = []
    all_pred: list[float] = []

    for fold in range(n_folds):
        train_end = min_train_rows + fold * fold_test_size
        if fold == n_folds - 1:
            test_end = total_rows
        else:
            test_end = train_end + fold_test_size

        train_frame = modeling_frame.iloc[:train_end]
        test_frame = modeling_frame.iloc[train_end:test_end]
        estimator = _build_estimator(
            random_state=42 + horizon + fold,
            model_params=strategy["model_params"],
            target_transform=strategy["target_transform"],
        )

        X_train = train_frame[selected_feature_columns]
        y_train = train_frame[target_column]
        X_test = test_frame[selected_feature_columns]
        y_test = test_frame[target_column]

        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)

        all_true.extend(y_test.tolist())
        all_pred.extend(predictions.tolist())
        folds.append(
            BacktestFoldResult(
                fold=fold + 1,
                train_rows=int(len(train_frame)),
                test_rows=int(len(test_frame)),
                train_start=str(train_frame[DATE_COLUMN].min().date()),
                train_end=str(train_frame[DATE_COLUMN].max().date()),
                test_start=str(test_frame[DATE_COLUMN].min().date()),
                test_end=str(test_frame[DATE_COLUMN].max().date()),
                mae=float(mean_absolute_error(y_test, predictions)),
                rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
                r2=float(r2_score(y_test, predictions)),
            )
        )

    aggregate = {
        "mae": float(mean_absolute_error(all_true, all_pred)),
        "rmse": float(np.sqrt(mean_squared_error(all_true, all_pred))),
        "r2": float(r2_score(all_true, all_pred)),
    }

    return {
        "horizon": horizon,
        "feature_profile": strategy["feature_profile"],
        "target_transform": strategy["target_transform"],
        "feature_count": len(selected_feature_columns),
        "n_folds": n_folds,
        "min_train_rows": min_train_rows,
        "aggregate_metrics": aggregate,
        "folds": [fold.__dict__ for fold in folds],
    }


def train_and_evaluate(
    csv_path: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    search_iterations: int = 25,
    horizons: tuple[int, ...] = DEFAULT_TRAIN_HORIZONS,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    training_frame, feature_columns, base_features = build_training_frame(df)

    artifacts: dict[int, ModelArtifact] = {}
    metrics: dict[str, Any] = {}
    feature_columns_by_horizon: dict[int, list[str]] = {}

    for horizon in horizons:
        artifact = _fit_horizon_model(
            frame=training_frame,
            feature_columns=feature_columns,
            horizon=horizon,
            random_state=random_state,
            n_iter=search_iterations,
        )
        artifacts[horizon] = artifact
        metrics[f"week_{horizon}"] = artifact.metrics
        feature_columns_by_horizon[horizon] = _horizon_feature_columns(feature_columns, horizon)
        artifact.feature_importance.to_csv(
            output_path / f"feature_importance_week_{horizon}.csv",
            index=False,
        )

    bundle = {
        "models": {horizon: artifacts[horizon].model for horizon in horizons},
        "feature_columns": feature_columns,
        "feature_columns_by_horizon": feature_columns_by_horizon,
        "base_feature_columns": base_features,
        "conformal_intervals_by_horizon": {
            horizon: artifacts[horizon].conformal_intervals for horizon in horizons
        },
        "metadata": {
            "target_column": TARGET_COLUMN,
            "date_column": DATE_COLUMN,
            "horizons": list(horizons),
            "min_target_observations_per_week": MIN_TARGET_OBS_PER_WEEK,
            "min_coverage_percent": MIN_COVERAGE_PERCENT,
            "rolling_windows": list(ROLLING_WINDOWS),
            "lag_steps": list(LAG_STEPS),
            "training_rows": int(len(training_frame)),
        },
    }
    joblib.dump(bundle, output_path / "chl_weekly_forecast_bundle.joblib")

    report = {
        "data_summary": {
            "input_rows": int(len(df)),
            "input_start": str(df[DATE_COLUMN].min().date()),
            "input_end": str(df[DATE_COLUMN].max().date()),
            "target_non_missing": int(df[TARGET_COLUMN].notna().sum()),
        },
        "metrics": metrics,
    }

    with open(output_path / "training_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def _week1_feature_columns(feature_columns: list[str]) -> list[str]:
    base = _feature_profile_columns(feature_columns, "target_small_env_calendar_p95")
    base = [column for column in base if not column.startswith("target_delta_")]
    extras = [
        "target_level_x_water_temperature",
        "target_level_x_precipitation",
    ]
    return base + [column for column in extras if column in feature_columns]


def _is_calendar_feature(column: str) -> bool:
    return column.startswith("day_") or column in {"month", "quarter"}


def _week1_regression_feature_columns(
    feature_columns: list[str],
    env_columns: tuple[str, ...],
    *,
    include_p95: bool,
    include_target_deltas: bool,
    include_interactions: bool,
) -> list[str]:
    selected: list[str] = []
    for column in feature_columns:
        is_target = column.startswith("CHLL_NN_TOTAL") or column.startswith("target_")
        is_env = column.startswith(env_columns)
        is_calendar = _is_calendar_feature(column)
        if not (is_target or is_env or is_calendar):
            continue
        if "_roll_iqr_" in column:
            continue
        if not include_p95 and "_roll_p95_" in column:
            continue
        if not include_target_deltas and column.startswith("target_delta_"):
            continue
        selected.append(column)

    interaction_columns = [
        "target_level_x_water_temperature",
        "target_level_x_precipitation",
    ]
    if include_interactions:
        for column in interaction_columns:
            if column in feature_columns and column not in selected:
                selected.append(column)

    return selected


def _week1_feature_set_candidates() -> list[Week1FeatureSetSpec]:
    base = ("precipitation", "air_temperature", "water_temperature", "water_level")
    return [
        Week1FeatureSetSpec("base", base),
        Week1FeatureSetSpec("base_plus_awnd", base + ("AWND",)),
        Week1FeatureSetSpec("base_plus_radiation", base + ("Watt_per_m2",)),
        Week1FeatureSetSpec("base_plus_tidal_range", base + ("tidal_range",)),
        Week1FeatureSetSpec("base_plus_ci_index", base + ("CI_index",)),
        Week1FeatureSetSpec("base_plus_awnd_radiation", base + ("AWND", "Watt_per_m2")),
        Week1FeatureSetSpec("base_plus_awnd_tidal_range", base + ("AWND", "tidal_range")),
        Week1FeatureSetSpec("base_plus_radiation_tidal_range", base + ("Watt_per_m2", "tidal_range")),
        Week1FeatureSetSpec(
            "swap_water_level_for_tidal_range",
            ("precipitation", "air_temperature", "water_temperature", "tidal_range"),
        ),
    ]


def _week1_transform_profiles() -> list[Week1TransformProfile]:
    return [
        Week1TransformProfile(
            name="baseline_full",
            include_p95=True,
            include_target_deltas=True,
            include_interactions=True,
        ),
        Week1TransformProfile(
            name="no_interactions",
            include_p95=True,
            include_target_deltas=True,
            include_interactions=False,
        ),
        Week1TransformProfile(
            name="no_target_deltas",
            include_p95=True,
            include_target_deltas=False,
            include_interactions=True,
        ),
        Week1TransformProfile(
            name="no_p95",
            include_p95=False,
            include_target_deltas=True,
            include_interactions=True,
        ),
    ]


def _week1_model_specs() -> list[Week1ModelSpec]:
    return [
        Week1ModelSpec(
            family="hist_gradient_boosting",
            params={
                "model__learning_rate": 0.02,
                "model__max_depth": 4,
                "model__max_iter": 1400,
                "model__max_leaf_nodes": 31,
                "model__min_samples_leaf": 90,
                "model__l2_regularization": 1.8,
            },
        ),
        Week1ModelSpec(
            family="hist_gradient_boosting",
            params={
                "model__learning_rate": 0.015,
                "model__max_depth": 4,
                "model__max_iter": 1800,
                "model__max_leaf_nodes": 23,
                "model__min_samples_leaf": 60,
                "model__l2_regularization": 1.2,
            },
        ),
        Week1ModelSpec(
            family="hist_gradient_boosting",
            params={
                "model__learning_rate": 0.03,
                "model__max_depth": 3,
                "model__max_iter": 1200,
                "model__max_leaf_nodes": 23,
                "model__min_samples_leaf": 120,
                "model__l2_regularization": 2.0,
            },
        ),
        Week1ModelSpec(
            family="random_forest",
            target_transform="none",
            params={
                "model__n_estimators": 500,
                "model__max_depth": 12,
                "model__min_samples_leaf": 5,
                "model__max_features": 0.5,
            },
        ),
        Week1ModelSpec(
            family="extra_trees",
            target_transform="none",
            params={
                "model__n_estimators": 500,
                "model__max_depth": 12,
                "model__min_samples_leaf": 5,
                "model__max_features": 0.5,
            },
        ),
    ]


def _week1_focused_model_specs() -> list[Week1ModelSpec]:
    specs: list[Week1ModelSpec] = []
    learning_rates = [0.0125, 0.015, 0.02, 0.025]
    max_iters = [1200, 1600, 2000]
    max_depths = [3, 4]
    max_leaf_nodes = [23, 31]
    min_samples_leaf = [60, 90, 120]
    l2_values = [1.2, 1.8, 2.4]

    for learning_rate in learning_rates:
        for max_iter in max_iters:
            for max_depth in max_depths:
                for max_leaf in max_leaf_nodes:
                    for min_leaf in min_samples_leaf:
                        for l2_value in l2_values:
                            specs.append(
                                Week1ModelSpec(
                                    family="hist_gradient_boosting",
                                    params={
                                        "model__learning_rate": learning_rate,
                                        "model__max_depth": max_depth,
                                        "model__max_iter": max_iter,
                                        "model__max_leaf_nodes": max_leaf,
                                        "model__min_samples_leaf": min_leaf,
                                        "model__l2_regularization": l2_value,
                                    },
                                )
                            )
    return specs


def _horizon_feature_columns(feature_columns: list[str], horizon: int) -> list[str]:
    if horizon == 1:
        return _week1_feature_columns(feature_columns)
    if horizon == 2:
        base = _feature_profile_columns(feature_columns, "target_small_env_week2_calendar_p95")
        extras = [
            "target_level_x_water_temperature",
            "target_level_x_precipitation",
            "target_delta_raw_lag_7",
            "target_delta_raw_lag_14",
            "target_delta_raw_lag_28",
        ]
        return base + [column for column in extras if column in feature_columns]
    return _feature_profile_columns(
        feature_columns, _default_baseline_strategy(horizon)["feature_profile"]
    )


def _week1_risk_feature_columns(feature_columns: list[str]) -> list[str]:
    base = _week1_feature_columns(feature_columns)
    awnd_columns = [column for column in feature_columns if column.startswith("AWND")]
    tidal_columns = [column for column in feature_columns if column.startswith("tidal_range")]
    selected = base + [column for column in awnd_columns if column not in base]
    return selected + [column for column in tidal_columns if column not in selected]


def _week2_classifier_feature_columns(feature_columns: list[str]) -> list[str]:
    base = _feature_profile_columns(feature_columns, "target_small_env_calendar_p95")
    extras = [
        "target_level_x_water_temperature",
        "target_level_x_precipitation",
        "target_delta_raw_lag_7",
        "target_delta_raw_lag_14",
        "target_delta_raw_lag_28",
    ]
    selected = base + [column for column in extras if column in feature_columns]
    awnd_columns = [column for column in feature_columns if column.startswith("AWND")]
    return selected + [column for column in awnd_columns if column not in selected]


def _horizon_risk_feature_columns(feature_columns: list[str], horizon: int) -> list[str]:
    strategy = _default_risk_strategy(horizon)
    profile = strategy.get("feature_profile")
    if profile == "week1_base_no_awnd_tidal":
        return _horizon_feature_columns(feature_columns, 1)
    if horizon == 1:
        return _week1_risk_feature_columns(feature_columns)
    if profile == "week2_classifier" or profile == "target_small_env_calendar_p95":
        return _week2_classifier_feature_columns(feature_columns)
    if profile == "week3_pruned" or horizon == 3:
        return _feature_profile_columns(feature_columns, "all_week3_pruned")
    return _horizon_feature_columns(feature_columns, horizon)


def _horizon_high_risk_feature_columns(feature_columns: list[str], horizon: int) -> list[str]:
    strategy = _default_high_risk_strategy(horizon)
    profile = strategy.get("feature_profile")
    if profile == "week2_classifier":
        return _week2_classifier_feature_columns(feature_columns)
    if profile == "week2_default":
        return _week2_classifier_feature_columns(feature_columns)
    if profile == "week3_pruned" or horizon == 3:
        return _feature_profile_columns(feature_columns, "all_week3_pruned")
    return _horizon_feature_columns(feature_columns, horizon)


def _assign_risk_classes(values: pd.Series, q25: float, q75: float) -> pd.Series:
    labels = np.where(values <= q25, "low", np.where(values < q75, "medium", "high"))
    return pd.Series(labels, index=values.index)


def _resolve_high_threshold(
    train_values: pd.Series,
    high_quantile: float,
    fixed_high_threshold: float | None = None,
) -> tuple[float, str]:
    if fixed_high_threshold is not None:
        return float(fixed_high_threshold), "fixed_value"
    return float(train_values.quantile(high_quantile)), "training_quantile"


def _target_column_for_horizon(horizon: int) -> str:
    return f"target_week_{horizon}"


def train_week1_risk_model(
    csv_path: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    high_quantile: float = 0.75,
    fixed_high_threshold: float | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    training_frame, feature_columns, base_features = build_training_frame(df)
    selected_feature_columns = _horizon_risk_feature_columns(feature_columns, 1)
    modeling_frame = training_frame.dropna(subset=["target_week_1"]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = max(int(len(modeling_frame) * 0.8), 1)
    train_frame = modeling_frame.iloc[:split_index].copy()
    test_frame = modeling_frame.iloc[split_index:].copy()
    if test_frame.empty:
        raise ValueError("Not enough samples to create a holdout set for the week-1 risk model.")

    q25 = float(train_frame["target_week_1"].quantile(0.25))
    q_high, high_threshold_mode = _resolve_high_threshold(
        train_frame["target_week_1"], high_quantile, fixed_high_threshold
    )
    train_labels = _assign_risk_classes(train_frame["target_week_1"], q25, q_high)
    test_labels = _assign_risk_classes(test_frame["target_week_1"], q25, q_high)

    strategy = _default_risk_strategy(1)
    classifier = _make_pipeline(_default_hist_classifier(random_state))
    classifier.set_params(**strategy["model_params"])
    weight_strategy = strategy.get("weight_strategy", "none")
    sample_weight = _classification_sample_weight(train_labels, weight_strategy)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    classifier.fit(train_frame[selected_feature_columns], train_labels, **fit_kwargs)

    predictions = classifier.predict(test_frame[selected_feature_columns])
    probabilities = classifier.predict_proba(test_frame[selected_feature_columns])
    class_order = list(classifier.classes_)
    cm = confusion_matrix(test_labels, predictions, labels=class_order)

    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_start": str(train_frame[DATE_COLUMN].min().date()),
        "train_end": str(train_frame[DATE_COLUMN].max().date()),
        "test_start": str(test_frame[DATE_COLUMN].min().date()),
        "test_end": str(test_frame[DATE_COLUMN].max().date()),
        "feature_count": int(len(selected_feature_columns)),
        "best_params": {
            "mode": "fixed_baseline",
            "feature_profile": strategy["feature_profile"],
            "weight_strategy": weight_strategy,
            **strategy["model_params"],
        },
        "thresholds": {
            "low_upper_q25": q25,
            "high_lower_quantile": q_high,
            "high_quantile": high_quantile,
            "high_threshold_mode": high_threshold_mode,
        },
        "class_order": class_order,
        "confusion_matrix": cm.tolist(),
        "test_class_counts": test_labels.value_counts().reindex(class_order, fill_value=0).to_dict(),
    }

    holdout_predictions = pd.concat(
        [
            test_frame[[DATE_COLUMN, "target_week_1"]].reset_index(drop=True),
            test_labels.rename("actual_risk").reset_index(drop=True),
            pd.Series(predictions, name="predicted_risk"),
            pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in class_order]),
        ],
        axis=1,
    )
    holdout_predictions.to_csv(output_path / "week1_risk_holdout_predictions.csv", index=False)

    bundle = {
        "model": classifier,
        "feature_columns": selected_feature_columns,
        "base_feature_columns": base_features,
        "metadata": {
            "target_column": TARGET_COLUMN,
            "date_column": DATE_COLUMN,
            "horizon": 1,
            "risk_labels": list(RISK_LABELS),
            "thresholds": {
                "low_upper_q25": q25,
                "high_lower_quantile": q_high,
                "high_quantile": high_quantile,
                "high_threshold_mode": high_threshold_mode,
            },
        },
    }
    joblib.dump(bundle, output_path / "week1_risk_model_bundle.joblib")

    report = {
        "data_summary": {
            "input_rows": int(len(df)),
            "input_start": str(df[DATE_COLUMN].min().date()),
            "input_end": str(df[DATE_COLUMN].max().date()),
            "target_non_missing": int(df[TARGET_COLUMN].notna().sum()),
        },
        "metrics": metrics,
    }
    with open(output_path / "week1_risk_training_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def search_week1_regression_experiments(
    csv_path: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    cv_splits: int = 3,
    top_k: int = 5,
    compact: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    training_frame, feature_columns, base_features = build_training_frame(df)
    target_column = "target_week_1"
    modeling_frame = training_frame.dropna(subset=[target_column]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = max(int(len(modeling_frame) * 0.8), 1)
    train_frame = modeling_frame.iloc[:split_index].copy()
    test_frame = modeling_frame.iloc[split_index:].copy()
    if test_frame.empty:
        raise ValueError("Not enough samples to create a holdout set for week-1 search.")

    X_train_full = train_frame
    y_train_full = train_frame[target_column]
    X_test_full = test_frame
    y_test = test_frame[target_column]

    cv = TimeSeriesSplit(n_splits=cv_splits)
    results: list[dict[str, Any]] = []
    top_models: list[tuple[float, dict[str, Any], Any, list[str]]] = []

    feature_sets = _week1_feature_set_candidates()
    transform_profiles = _week1_transform_profiles()
    model_specs = _week1_model_specs()
    if compact:
        feature_sets = feature_sets[:6] + feature_sets[-1:]
        transform_profiles = transform_profiles[:3]
        model_specs = model_specs[:3]

    experiment_id = 0
    for feature_set in feature_sets:
        for transform_profile in transform_profiles:
            selected_feature_columns = _week1_regression_feature_columns(
                feature_columns,
                feature_set.env_columns,
                include_p95=transform_profile.include_p95,
                include_target_deltas=transform_profile.include_target_deltas,
                include_interactions=transform_profile.include_interactions,
            )
            for model_spec in model_specs:
                experiment_id += 1
                cv_mae_scores: list[float] = []
                cv_rmse_scores: list[float] = []
                cv_r2_scores: list[float] = []

                for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_full), start=1):
                    estimator = _build_regression_estimator(
                        family=model_spec.family,
                        random_state=random_state + experiment_id + fold_idx,
                        model_params=model_spec.params,
                        target_transform=model_spec.target_transform,
                    )
                    X_train = X_train_full.iloc[train_idx][selected_feature_columns]
                    y_train = y_train_full.iloc[train_idx]
                    X_val = X_train_full.iloc[val_idx][selected_feature_columns]
                    y_val = y_train_full.iloc[val_idx]
                    estimator.fit(X_train, y_train)
                    val_predictions = estimator.predict(X_val)
                    cv_mae_scores.append(float(mean_absolute_error(y_val, val_predictions)))
                    cv_rmse_scores.append(float(np.sqrt(mean_squared_error(y_val, val_predictions))))
                    cv_r2_scores.append(float(r2_score(y_val, val_predictions)))

                final_estimator = _build_regression_estimator(
                    family=model_spec.family,
                    random_state=random_state + experiment_id,
                    model_params=model_spec.params,
                    target_transform=model_spec.target_transform,
                )
                final_estimator.fit(train_frame[selected_feature_columns], y_train_full)
                test_predictions = final_estimator.predict(X_test_full[selected_feature_columns])

                result = {
                    "experiment_id": experiment_id,
                    "feature_set": feature_set.name,
                    "env_columns": list(feature_set.env_columns),
                    "transform_profile": transform_profile.name,
                    "include_p95": transform_profile.include_p95,
                    "include_target_deltas": transform_profile.include_target_deltas,
                    "include_interactions": transform_profile.include_interactions,
                    "model_family": model_spec.family,
                    "target_transform": model_spec.target_transform,
                    "model_params": json.dumps(model_spec.params, sort_keys=True),
                    "feature_count": len(selected_feature_columns),
                    "cv_mae_mean": float(np.mean(cv_mae_scores)),
                    "cv_rmse_mean": float(np.mean(cv_rmse_scores)),
                    "cv_r2_mean": float(np.mean(cv_r2_scores)),
                    "holdout_mae": float(mean_absolute_error(y_test, test_predictions)),
                    "holdout_rmse": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
                    "holdout_r2": float(r2_score(y_test, test_predictions)),
                }
                results.append(result)
                top_models.append((result["holdout_mae"], result, final_estimator, selected_feature_columns))

    results_df = pd.DataFrame(results).sort_values(
        ["holdout_r2", "holdout_rmse", "holdout_mae"],
        ascending=[False, True, True],
    )
    results_df.to_csv(output_path / "week1_regression_search_results.csv", index=False)

    top_models = sorted(
        top_models,
        key=lambda item: (-item[1]["holdout_r2"], item[1]["holdout_rmse"], item[1]["holdout_mae"]),
    )[:top_k]

    top_summaries: list[dict[str, Any]] = []
    for rank, (_, result, estimator, selected_feature_columns) in enumerate(top_models, start=1):
        bundle = {
            "models": {1: estimator},
            "feature_columns": feature_columns,
            "feature_columns_by_horizon": {1: selected_feature_columns},
            "base_feature_columns": base_features,
            "metadata": {
                "target_column": TARGET_COLUMN,
                "date_column": DATE_COLUMN,
                "horizons": [1],
                "bundle_type": "week1_regression_search_candidate",
                "search_result": result,
            },
        }
        bundle_path = output_path / f"week1_candidate_rank_{rank}.joblib"
        joblib.dump(bundle, bundle_path)
        top_summaries.append(
            {
                "rank": rank,
                "bundle_path": str(bundle_path),
                **result,
            }
        )

    summary = {
        "data_summary": {
            "input_rows": int(len(df)),
            "input_start": str(df[DATE_COLUMN].min().date()),
            "input_end": str(df[DATE_COLUMN].max().date()),
            "target_non_missing": int(df[TARGET_COLUMN].notna().sum()),
            "search_train_rows": int(len(train_frame)),
            "search_test_rows": int(len(test_frame)),
        },
        "search_space": {
            "feature_sets": [spec.name for spec in feature_sets],
            "transform_profiles": [spec.name for spec in transform_profiles],
            "model_count": len(model_specs),
            "total_experiments": len(results),
            "cv_splits": cv_splits,
            "compact": compact,
        },
        "top_results": top_summaries,
    }

    with open(output_path / "week1_regression_search_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def search_week1_regression_focused(
    csv_path: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    cv_splits: int = 4,
    top_k: int = 5,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    training_frame, feature_columns, base_features = build_training_frame(df)
    target_column = "target_week_1"
    modeling_frame = training_frame.dropna(subset=[target_column]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = max(int(len(modeling_frame) * 0.8), 1)
    train_frame = modeling_frame.iloc[:split_index].copy()
    test_frame = modeling_frame.iloc[split_index:].copy()
    if test_frame.empty:
        raise ValueError("Not enough samples to create a holdout set for focused week-1 search.")

    feature_set = Week1FeatureSetSpec(
        name="base",
        env_columns=("precipitation", "air_temperature", "water_temperature", "water_level"),
    )
    transform_profiles = _week1_transform_profiles()
    model_specs = _week1_focused_model_specs()

    cv = TimeSeriesSplit(n_splits=cv_splits)
    y_train_full = train_frame[target_column]
    y_test = test_frame[target_column]
    results: list[dict[str, Any]] = []
    top_models: list[tuple[float, dict[str, Any], Any, list[str]]] = []

    experiment_id = 0
    for transform_profile in transform_profiles:
        selected_feature_columns = _week1_regression_feature_columns(
            feature_columns,
            feature_set.env_columns,
            include_p95=transform_profile.include_p95,
            include_target_deltas=transform_profile.include_target_deltas,
            include_interactions=transform_profile.include_interactions,
        )
        for model_spec in model_specs:
            experiment_id += 1
            cv_mae_scores: list[float] = []
            cv_rmse_scores: list[float] = []
            cv_r2_scores: list[float] = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(train_frame), start=1):
                estimator = _build_regression_estimator(
                    family=model_spec.family,
                    random_state=random_state + experiment_id + fold_idx,
                    model_params=model_spec.params,
                    target_transform=model_spec.target_transform,
                )
                X_train = train_frame.iloc[train_idx][selected_feature_columns]
                y_train = y_train_full.iloc[train_idx]
                X_val = train_frame.iloc[val_idx][selected_feature_columns]
                y_val = y_train_full.iloc[val_idx]
                estimator.fit(X_train, y_train)
                val_predictions = estimator.predict(X_val)
                cv_mae_scores.append(float(mean_absolute_error(y_val, val_predictions)))
                cv_rmse_scores.append(float(np.sqrt(mean_squared_error(y_val, val_predictions))))
                cv_r2_scores.append(float(r2_score(y_val, val_predictions)))

            estimator = _build_regression_estimator(
                family=model_spec.family,
                random_state=random_state + experiment_id,
                model_params=model_spec.params,
                target_transform=model_spec.target_transform,
            )
            estimator.fit(train_frame[selected_feature_columns], y_train_full)
            test_predictions = estimator.predict(test_frame[selected_feature_columns])

            result = {
                "experiment_id": experiment_id,
                "feature_set": feature_set.name,
                "env_columns": list(feature_set.env_columns),
                "transform_profile": transform_profile.name,
                "include_p95": transform_profile.include_p95,
                "include_target_deltas": transform_profile.include_target_deltas,
                "include_interactions": transform_profile.include_interactions,
                "model_family": model_spec.family,
                "target_transform": model_spec.target_transform,
                "model_params": json.dumps(model_spec.params, sort_keys=True),
                "feature_count": len(selected_feature_columns),
                "cv_mae_mean": float(np.mean(cv_mae_scores)),
                "cv_rmse_mean": float(np.mean(cv_rmse_scores)),
                "cv_r2_mean": float(np.mean(cv_r2_scores)),
                "holdout_mae": float(mean_absolute_error(y_test, test_predictions)),
                "holdout_rmse": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
                "holdout_r2": float(r2_score(y_test, test_predictions)),
            }
            results.append(result)
            top_models.append((result["holdout_mae"], result, estimator, selected_feature_columns))

    results_df = pd.DataFrame(results).sort_values(
        ["holdout_r2", "holdout_rmse", "holdout_mae"],
        ascending=[False, True, True],
    )
    results_df.to_csv(output_path / "week1_regression_focused_results.csv", index=False)

    top_models = sorted(
        top_models,
        key=lambda item: (-item[1]["holdout_r2"], item[1]["holdout_rmse"], item[1]["holdout_mae"]),
    )[:top_k]

    top_summaries: list[dict[str, Any]] = []
    for rank, (_, result, estimator, selected_feature_columns) in enumerate(top_models, start=1):
        bundle = {
            "models": {1: estimator},
            "feature_columns": feature_columns,
            "feature_columns_by_horizon": {1: selected_feature_columns},
            "base_feature_columns": base_features,
            "metadata": {
                "target_column": TARGET_COLUMN,
                "date_column": DATE_COLUMN,
                "horizons": [1],
                "bundle_type": "week1_regression_focused_candidate",
                "search_result": result,
            },
        }
        bundle_path = output_path / f"week1_focused_candidate_rank_{rank}.joblib"
        joblib.dump(bundle, bundle_path)
        top_summaries.append({"rank": rank, "bundle_path": str(bundle_path), **result})

    summary = {
        "data_summary": {
            "input_rows": int(len(df)),
            "input_start": str(df[DATE_COLUMN].min().date()),
            "input_end": str(df[DATE_COLUMN].max().date()),
            "target_non_missing": int(df[TARGET_COLUMN].notna().sum()),
            "search_train_rows": int(len(train_frame)),
            "search_test_rows": int(len(test_frame)),
        },
        "search_space": {
            "feature_set": feature_set.name,
            "transform_profiles": [spec.name for spec in transform_profiles],
            "model_count": len(model_specs),
            "total_experiments": len(results),
            "cv_splits": cv_splits,
        },
        "top_results": top_summaries,
    }

    with open(output_path / "week1_regression_focused_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def train_horizon_risk_model(
    csv_path: str | Path,
    output_dir: str | Path,
    horizon: int,
    random_state: int = 42,
    high_quantile: float = 0.75,
    fixed_high_threshold: float | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    training_frame, feature_columns, base_features = build_training_frame(df)
    selected_feature_columns = _horizon_risk_feature_columns(feature_columns, horizon)
    target_column = _target_column_for_horizon(horizon)
    modeling_frame = training_frame.dropna(subset=[target_column]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = max(int(len(modeling_frame) * 0.8), 1)
    train_frame = modeling_frame.iloc[:split_index].copy()
    test_frame = modeling_frame.iloc[split_index:].copy()
    if test_frame.empty:
        raise ValueError(f"Not enough samples to create a holdout set for horizon {horizon} risk model.")

    q25 = float(train_frame[target_column].quantile(0.25))
    q_high, high_threshold_mode = _resolve_high_threshold(
        train_frame[target_column], high_quantile, fixed_high_threshold
    )
    train_labels = _assign_risk_classes(train_frame[target_column], q25, q_high)
    test_labels = _assign_risk_classes(test_frame[target_column], q25, q_high)

    strategy = _default_risk_strategy(horizon)
    classifier = _make_pipeline(_default_hist_classifier(random_state))
    classifier.set_params(**strategy["model_params"])
    weight_strategy = strategy.get("weight_strategy", "none")
    sample_weight = _classification_sample_weight(train_labels, weight_strategy)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    classifier.fit(train_frame[selected_feature_columns], train_labels, **fit_kwargs)

    predictions = classifier.predict(test_frame[selected_feature_columns])
    probabilities = classifier.predict_proba(test_frame[selected_feature_columns])
    class_order = list(classifier.classes_)
    cm = confusion_matrix(test_labels, predictions, labels=class_order)

    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_start": str(train_frame[DATE_COLUMN].min().date()),
        "train_end": str(train_frame[DATE_COLUMN].max().date()),
        "test_start": str(test_frame[DATE_COLUMN].min().date()),
        "test_end": str(test_frame[DATE_COLUMN].max().date()),
        "feature_count": int(len(selected_feature_columns)),
        "best_params": {
            "mode": "fixed_baseline",
            "feature_profile": strategy["feature_profile"],
            "horizon": horizon,
            "weight_strategy": weight_strategy,
            **strategy["model_params"],
        },
        "thresholds": {
            "low_upper_q25": q25,
            "high_lower_quantile": q_high,
            "high_quantile": high_quantile,
            "high_threshold_mode": high_threshold_mode,
        },
        "class_order": class_order,
        "confusion_matrix": cm.tolist(),
        "test_class_counts": test_labels.value_counts().reindex(class_order, fill_value=0).to_dict(),
    }

    holdout_predictions = pd.concat(
        [
            test_frame[[DATE_COLUMN, target_column]].reset_index(drop=True),
            test_labels.rename("actual_risk").reset_index(drop=True),
            pd.Series(predictions, name="predicted_risk"),
            pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in class_order]),
        ],
        axis=1,
    )
    holdout_predictions.to_csv(
        output_path / f"horizon_{horizon}_risk_holdout_predictions.csv", index=False
    )

    bundle = {
        "model": classifier,
        "feature_columns": selected_feature_columns,
        "base_feature_columns": base_features,
        "metadata": {
            "target_column": TARGET_COLUMN,
            "date_column": DATE_COLUMN,
            "horizon": horizon,
            "risk_labels": list(RISK_LABELS),
            "thresholds": {
                "low_upper_q25": q25,
                "high_lower_quantile": q_high,
                "high_quantile": high_quantile,
                "high_threshold_mode": high_threshold_mode,
            },
        },
    }
    joblib.dump(bundle, output_path / f"horizon_{horizon}_risk_model_bundle.joblib")

    report = {
        "data_summary": {
            "input_rows": int(len(df)),
            "input_start": str(df[DATE_COLUMN].min().date()),
            "input_end": str(df[DATE_COLUMN].max().date()),
            "target_non_missing": int(df[TARGET_COLUMN].notna().sum()),
        },
        "metrics": metrics,
    }
    with open(
        output_path / f"horizon_{horizon}_risk_training_report.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    return report


def _inverse_frequency_sample_weight(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts()
    weights = labels.map({label: len(labels) / count for label, count in counts.items()})
    return weights.to_numpy(dtype=float)


def _classification_sample_weight(labels: pd.Series, strategy: str) -> np.ndarray | None:
    if strategy == "none":
        return None
    base = _inverse_frequency_sample_weight(labels)
    if strategy == "inverse":
        return base
    if strategy == "inverse_sqrt":
        return np.sqrt(base)
    if strategy == "inverse_pow_1_5":
        return np.power(base, 1.5)
    raise ValueError(f"Unsupported classification weight strategy: {strategy}")


def train_week1_risk_two_stage_model(
    csv_path: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    high_quantile: float = 0.75,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    training_frame, feature_columns, base_features = build_training_frame(df)
    selected_feature_columns = _week1_feature_columns(feature_columns)
    modeling_frame = training_frame.dropna(subset=["target_week_1"]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = max(int(len(modeling_frame) * 0.8), 1)
    train_frame = modeling_frame.iloc[:split_index].copy()
    test_frame = modeling_frame.iloc[split_index:].copy()
    if test_frame.empty:
        raise ValueError("Not enough samples to create a holdout set for the week-1 risk model.")

    q25 = float(train_frame["target_week_1"].quantile(0.25))
    q_high = float(train_frame["target_week_1"].quantile(high_quantile))
    train_labels = _assign_risk_classes(train_frame["target_week_1"], q25, q_high)
    test_labels = _assign_risk_classes(test_frame["target_week_1"], q25, q_high)

    strategy = _default_risk_strategy(1)
    stage1 = _make_pipeline(_default_hist_classifier(random_state))
    stage1.set_params(**strategy["model_params"])
    stage1_labels = pd.Series(
        np.where(train_labels == "high", "high", "not_high"),
        index=train_labels.index,
    )
    stage1.fit(
        train_frame[selected_feature_columns],
        stage1_labels,
        model__sample_weight=_inverse_frequency_sample_weight(stage1_labels),
    )

    stage2_train_mask = train_labels != "high"
    stage2 = _make_pipeline(_default_hist_classifier(random_state + 1))
    stage2.set_params(**strategy["model_params"])
    stage2_labels = train_labels.loc[stage2_train_mask]
    stage2.fit(
        train_frame.loc[stage2_train_mask, selected_feature_columns],
        stage2_labels,
        model__sample_weight=_inverse_frequency_sample_weight(stage2_labels),
    )

    stage1_test_prob = stage1.predict_proba(test_frame[selected_feature_columns])
    stage1_class_index = {label: idx for idx, label in enumerate(stage1.classes_)}
    high_prob = stage1_test_prob[:, stage1_class_index["high"]]
    stage2_prob = stage2.predict_proba(test_frame[selected_feature_columns])
    stage2_class_index = {label: idx for idx, label in enumerate(stage2.classes_)}
    low_prob = (1.0 - high_prob) * stage2_prob[:, stage2_class_index["low"]]
    medium_prob = (1.0 - high_prob) * stage2_prob[:, stage2_class_index["medium"]]

    probability_frame = pd.DataFrame(
        {
            "prob_low": low_prob,
            "prob_medium": medium_prob,
            "prob_high": high_prob,
        }
    )
    predictions = probability_frame.idxmax(axis=1).str.removeprefix("prob_")
    class_order = ["high", "low", "medium"]
    cm = confusion_matrix(test_labels, predictions, labels=class_order)

    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_start": str(train_frame[DATE_COLUMN].min().date()),
        "train_end": str(train_frame[DATE_COLUMN].max().date()),
        "test_start": str(test_frame[DATE_COLUMN].min().date()),
        "test_end": str(test_frame[DATE_COLUMN].max().date()),
        "feature_count": int(len(selected_feature_columns)),
        "thresholds": {
            "low_upper_q25": q25,
            "high_lower_quantile": q_high,
            "high_quantile": high_quantile,
        },
        "class_order": class_order,
        "confusion_matrix": cm.tolist(),
        "test_class_counts": test_labels.value_counts().reindex(class_order, fill_value=0).to_dict(),
        "stage1_positive_rate": float((predictions == "high").mean()),
    }

    holdout_predictions = pd.concat(
        [
            test_frame[[DATE_COLUMN, "target_week_1"]].reset_index(drop=True),
            test_labels.rename("actual_risk").reset_index(drop=True),
            predictions.rename("predicted_risk").reset_index(drop=True),
            probability_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    holdout_predictions.to_csv(
        output_path / "week1_risk_two_stage_holdout_predictions.csv", index=False
    )

    bundle = {
        "stage1_model": stage1,
        "stage2_model": stage2,
        "feature_columns": selected_feature_columns,
        "base_feature_columns": base_features,
        "metadata": {
            "target_column": TARGET_COLUMN,
            "date_column": DATE_COLUMN,
            "horizon": 1,
            "risk_labels": list(RISK_LABELS),
            "thresholds": {
                "low_upper_q25": q25,
                "high_lower_quantile": q_high,
                "high_quantile": high_quantile,
            },
            "bundle_type": "two_stage_week1_risk",
        },
    }
    joblib.dump(bundle, output_path / "week1_risk_two_stage_bundle.joblib")

    report = {
        "data_summary": {
            "input_rows": int(len(df)),
            "input_start": str(df[DATE_COLUMN].min().date()),
            "input_end": str(df[DATE_COLUMN].max().date()),
            "target_non_missing": int(df[TARGET_COLUMN].notna().sum()),
        },
        "metrics": metrics,
    }
    with open(
        output_path / "week1_risk_two_stage_training_report.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    return report


def _binary_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["high"], average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "high_precision": float(precision[0]),
        "high_recall": float(recall[0]),
        "high_f1": float(f1[0]),
        "roc_auc": float(roc_auc_score((y_true == "high").astype(int), y_prob)),
    }


def _choose_high_threshold(
    probabilities: np.ndarray,
    labels: pd.Series,
    strategy: str = "recall_focused",
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_score = -np.inf
    best_metrics = {
        "balanced_accuracy": 0.0,
        "high_precision": 0.0,
        "high_recall": 0.0,
        "high_f1": 0.0,
    }
    for threshold in np.linspace(0.10, 0.75, 27):
        preds = pd.Series(np.where(probabilities >= threshold, "high", "not_high"))
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=["high"], average=None, zero_division=0
        )
        balanced_acc = balanced_accuracy_score(labels, preds)
        if strategy == "recall_focused":
            score = (2.0 * recall[0]) + f1[0] + (0.25 * precision[0])
        elif strategy == "balanced":
            score = (2.0 * balanced_acc) + f1[0] + (0.5 * recall[0])
        elif strategy == "f1_focused":
            score = (2.0 * f1[0]) + recall[0] + (0.25 * precision[0])
        else:
            raise ValueError(f"Unsupported threshold selection strategy: {strategy}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {
                "balanced_accuracy": float(balanced_acc),
                "high_precision": float(precision[0]),
                "high_recall": float(recall[0]),
                "high_f1": float(f1[0]),
            }
    return best_threshold, best_metrics


def train_week1_high_risk_model(
    csv_path: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    high_quantile: float = 0.75,
    fixed_high_threshold: float | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    training_frame, feature_columns, base_features = build_training_frame(df)
    selected_feature_columns = _week1_feature_columns(feature_columns)
    modeling_frame = training_frame.dropna(subset=["target_week_1"]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = max(int(len(modeling_frame) * 0.8), 1)
    train_frame = modeling_frame.iloc[:split_index].copy()
    test_frame = modeling_frame.iloc[split_index:].copy()
    if test_frame.empty:
        raise ValueError("Not enough samples to create a holdout set for the week-1 high-risk model.")

    q75, high_threshold_mode = _resolve_high_threshold(
        train_frame["target_week_1"], high_quantile, fixed_high_threshold
    )
    train_labels = pd.Series(
        np.where(train_frame["target_week_1"] >= q75, "high", "not_high"),
        index=train_frame.index,
    )
    test_labels = pd.Series(
        np.where(test_frame["target_week_1"] >= q75, "high", "not_high"),
        index=test_frame.index,
    )

    # Chronological validation slice for threshold selection.
    threshold_split = max(int(len(train_frame) * 0.8), 1)
    model_train = train_frame.iloc[:threshold_split]
    model_valid = train_frame.iloc[threshold_split:]
    model_train_labels = train_labels.iloc[:threshold_split]
    model_valid_labels = train_labels.iloc[threshold_split:]

    strategy = _default_high_risk_strategy(1)
    classifier = _make_pipeline(_default_hist_classifier(random_state))
    classifier.set_params(**strategy["model_params"])
    weight_strategy = strategy.get("weight_strategy", "inverse")
    threshold_strategy = strategy.get("threshold_strategy", "recall_focused")
    train_weight = _classification_sample_weight(model_train_labels, weight_strategy)
    fit_kwargs = {}
    if train_weight is not None:
        fit_kwargs["model__sample_weight"] = train_weight
    classifier.fit(
        model_train[selected_feature_columns],
        model_train_labels,
        **fit_kwargs,
    )

    valid_prob = classifier.predict_proba(model_valid[selected_feature_columns])
    valid_index = {label: idx for idx, label in enumerate(classifier.classes_)}
    high_valid_prob = valid_prob[:, valid_index["high"]]
    threshold, threshold_metrics = _choose_high_threshold(
        high_valid_prob, model_valid_labels, threshold_strategy
    )

    # Refit on the full training window before holdout evaluation.
    full_train_weight = _classification_sample_weight(train_labels, weight_strategy)
    refit_kwargs = {}
    if full_train_weight is not None:
        refit_kwargs["model__sample_weight"] = full_train_weight
    classifier.fit(
        train_frame[selected_feature_columns],
        train_labels,
        **refit_kwargs,
    )
    test_prob = classifier.predict_proba(test_frame[selected_feature_columns])
    test_index = {label: idx for idx, label in enumerate(classifier.classes_)}
    high_test_prob = test_prob[:, test_index["high"]]
    test_pred = pd.Series(
        np.where(high_test_prob >= threshold, "high", "not_high"),
        index=test_frame.index,
    )
    cm = confusion_matrix(test_labels, test_pred, labels=["high", "not_high"])
    metrics = _binary_metrics(test_labels, test_pred, high_test_prob)
    metrics.update(
        {
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "train_start": str(train_frame[DATE_COLUMN].min().date()),
            "train_end": str(train_frame[DATE_COLUMN].max().date()),
            "test_start": str(test_frame[DATE_COLUMN].min().date()),
            "test_end": str(test_frame[DATE_COLUMN].max().date()),
            "feature_count": int(len(selected_feature_columns)),
            "high_lower_q75": q75,
            "high_quantile": high_quantile,
            "high_threshold_mode": high_threshold_mode,
            "probability_threshold": threshold,
            "weight_strategy": weight_strategy,
            "threshold_strategy": threshold_strategy,
            "validation_threshold_metrics": threshold_metrics,
            "confusion_matrix": cm.tolist(),
            "class_order": ["high", "not_high"],
            "test_class_counts": test_labels.value_counts().reindex(["high", "not_high"], fill_value=0).to_dict(),
            "predicted_positive_rate": float((test_pred == "high").mean()),
        }
    )

    holdout_predictions = pd.concat(
        [
            test_frame[[DATE_COLUMN, "target_week_1"]].reset_index(drop=True),
            test_labels.rename("actual_high_risk").reset_index(drop=True),
            test_pred.rename("predicted_high_risk").reset_index(drop=True),
            pd.Series(high_test_prob, name="prob_high"),
        ],
        axis=1,
    )
    holdout_predictions.to_csv(
        output_path / "week1_high_risk_holdout_predictions.csv", index=False
    )

    bundle = {
        "model": classifier,
        "feature_columns": selected_feature_columns,
        "base_feature_columns": base_features,
        "metadata": {
            "target_column": TARGET_COLUMN,
            "date_column": DATE_COLUMN,
            "horizon": 1,
            "high_lower_q75": q75,
            "high_quantile": high_quantile,
            "high_threshold_mode": high_threshold_mode,
            "probability_threshold": threshold,
            "weight_strategy": weight_strategy,
            "threshold_strategy": threshold_strategy,
            "bundle_type": "binary_high_risk_week1",
        },
    }
    joblib.dump(bundle, output_path / "week1_high_risk_bundle.joblib")

    report = {
        "data_summary": {
            "input_rows": int(len(df)),
            "input_start": str(df[DATE_COLUMN].min().date()),
            "input_end": str(df[DATE_COLUMN].max().date()),
            "target_non_missing": int(df[TARGET_COLUMN].notna().sum()),
        },
        "metrics": metrics,
    }
    with open(
        output_path / "week1_high_risk_training_report.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    return report


def train_horizon_high_risk_model(
    csv_path: str | Path,
    output_dir: str | Path,
    horizon: int,
    random_state: int = 42,
    high_quantile: float = 0.75,
    fixed_high_threshold: float | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    training_frame, feature_columns, base_features = build_training_frame(df)
    selected_feature_columns = _horizon_high_risk_feature_columns(feature_columns, horizon)
    target_column = _target_column_for_horizon(horizon)
    modeling_frame = training_frame.dropna(subset=[target_column]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = max(int(len(modeling_frame) * 0.8), 1)
    train_frame = modeling_frame.iloc[:split_index].copy()
    test_frame = modeling_frame.iloc[split_index:].copy()
    if test_frame.empty:
        raise ValueError(f"Not enough samples to create a holdout set for horizon {horizon} high-risk model.")

    q_high, high_threshold_mode = _resolve_high_threshold(
        train_frame[target_column], high_quantile, fixed_high_threshold
    )
    train_labels = pd.Series(
        np.where(train_frame[target_column] >= q_high, "high", "not_high"),
        index=train_frame.index,
    )
    test_labels = pd.Series(
        np.where(test_frame[target_column] >= q_high, "high", "not_high"),
        index=test_frame.index,
    )

    threshold_split = max(int(len(train_frame) * 0.8), 1)
    model_train = train_frame.iloc[:threshold_split]
    model_valid = train_frame.iloc[threshold_split:]
    model_train_labels = train_labels.iloc[:threshold_split]
    model_valid_labels = train_labels.iloc[threshold_split:]

    strategy = _default_high_risk_strategy(horizon)
    classifier = _make_pipeline(_default_hist_classifier(random_state))
    classifier.set_params(**strategy["model_params"])
    weight_strategy = strategy.get("weight_strategy", "inverse")
    threshold_strategy = strategy.get("threshold_strategy", "recall_focused")
    train_weight = _classification_sample_weight(model_train_labels, weight_strategy)
    fit_kwargs = {}
    if train_weight is not None:
        fit_kwargs["model__sample_weight"] = train_weight
    classifier.fit(
        model_train[selected_feature_columns],
        model_train_labels,
        **fit_kwargs,
    )

    valid_prob = classifier.predict_proba(model_valid[selected_feature_columns])
    valid_index = {label: idx for idx, label in enumerate(classifier.classes_)}
    high_valid_prob = valid_prob[:, valid_index["high"]]
    threshold, threshold_metrics = _choose_high_threshold(
        high_valid_prob, model_valid_labels, threshold_strategy
    )

    full_train_weight = _classification_sample_weight(train_labels, weight_strategy)
    refit_kwargs = {}
    if full_train_weight is not None:
        refit_kwargs["model__sample_weight"] = full_train_weight
    classifier.fit(
        train_frame[selected_feature_columns],
        train_labels,
        **refit_kwargs,
    )
    test_prob = classifier.predict_proba(test_frame[selected_feature_columns])
    test_index = {label: idx for idx, label in enumerate(classifier.classes_)}
    high_test_prob = test_prob[:, test_index["high"]]
    test_pred = pd.Series(
        np.where(high_test_prob >= threshold, "high", "not_high"),
        index=test_frame.index,
    )
    cm = confusion_matrix(test_labels, test_pred, labels=["high", "not_high"])
    metrics = _binary_metrics(test_labels, test_pred, high_test_prob)
    metrics.update(
        {
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "train_start": str(train_frame[DATE_COLUMN].min().date()),
            "train_end": str(train_frame[DATE_COLUMN].max().date()),
            "test_start": str(test_frame[DATE_COLUMN].min().date()),
            "test_end": str(test_frame[DATE_COLUMN].max().date()),
            "feature_count": int(len(selected_feature_columns)),
            "high_lower_q75": q_high,
            "high_quantile": high_quantile,
            "high_threshold_mode": high_threshold_mode,
            "probability_threshold": threshold,
            "weight_strategy": weight_strategy,
            "threshold_strategy": threshold_strategy,
            "validation_threshold_metrics": threshold_metrics,
            "confusion_matrix": cm.tolist(),
            "class_order": ["high", "not_high"],
            "test_class_counts": test_labels.value_counts().reindex(["high", "not_high"], fill_value=0).to_dict(),
            "predicted_positive_rate": float((test_pred == "high").mean()),
        }
    )

    holdout_predictions = pd.concat(
        [
            test_frame[[DATE_COLUMN, target_column]].reset_index(drop=True),
            test_labels.rename("actual_high_risk").reset_index(drop=True),
            test_pred.rename("predicted_high_risk").reset_index(drop=True),
            pd.Series(high_test_prob, name="prob_high"),
        ],
        axis=1,
    )
    holdout_predictions.to_csv(
        output_path / f"horizon_{horizon}_high_risk_holdout_predictions.csv", index=False
    )

    bundle = {
        "model": classifier,
        "feature_columns": selected_feature_columns,
        "base_feature_columns": base_features,
        "metadata": {
            "target_column": TARGET_COLUMN,
            "date_column": DATE_COLUMN,
            "horizon": horizon,
            "high_lower_q75": q_high,
            "high_quantile": high_quantile,
            "high_threshold_mode": high_threshold_mode,
            "probability_threshold": threshold,
            "weight_strategy": weight_strategy,
            "threshold_strategy": threshold_strategy,
            "bundle_type": f"binary_high_risk_horizon_{horizon}",
        },
    }
    joblib.dump(bundle, output_path / f"horizon_{horizon}_high_risk_bundle.joblib")

    report = {
        "data_summary": {
            "input_rows": int(len(df)),
            "input_start": str(df[DATE_COLUMN].min().date()),
            "input_end": str(df[DATE_COLUMN].max().date()),
            "target_non_missing": int(df[TARGET_COLUMN].notna().sum()),
        },
        "metrics": metrics,
    }
    with open(
        output_path / f"horizon_{horizon}_high_risk_training_report.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    return report


def load_bundle(bundle_path: str | Path) -> dict[str, Any]:
    return joblib.load(bundle_path)


def predict_from_bundle(
    csv_path: str | Path,
    bundle_path: str | Path,
    prediction_date: str | None = None,
) -> pd.DataFrame:
    df = load_data(csv_path)
    bundle = load_bundle(bundle_path)
    if prediction_date is None:
        prediction_timestamp = df[DATE_COLUMN].max() + pd.Timedelta(days=1)
    else:
        prediction_timestamp = pd.Timestamp(prediction_date)
    df = _extend_to_prediction_date(df, prediction_timestamp)
    frame, _ = build_inference_frame(df, base_features=bundle["base_feature_columns"])
    feature_columns = bundle["feature_columns"]
    feature_columns_by_horizon = bundle.get("feature_columns_by_horizon", {})
    conformal_by_horizon = bundle.get("conformal_intervals_by_horizon", {})

    missing_features = [column for column in feature_columns if column not in frame.columns]
    if missing_features:
        raise ValueError(
            "Input data is missing engineered feature columns required by the model: "
            + ", ".join(missing_features[:10])
        )

    candidate_rows = frame.loc[frame[DATE_COLUMN] == prediction_timestamp].copy()
    if candidate_rows.empty:
        raise ValueError(f"No feature row is available for prediction date {prediction_timestamp.date()}.")
    row = candidate_rows.tail(1)

    forecast_date = row[DATE_COLUMN].iloc[0].date().isoformat()
    predictions = {
        "prediction_date": forecast_date,
    }
    for horizon in bundle["metadata"]["horizons"]:
        model_columns = feature_columns_by_horizon.get(horizon, feature_columns)
        value = bundle["models"][horizon].predict(row[model_columns])[0]
        predictions[f"week_{horizon}_ahead_avg"] = float(value)
        conformal = conformal_by_horizon.get(horizon, {})
        for alpha_label, quantile in conformal.get("alphas", {}).items():
            if np.isnan(quantile):
                continue
            coverage = int(round((1.0 - float(alpha_label)) * 100))
            predictions[f"week_{horizon}_ahead_lower_{coverage}"] = float(value - quantile)
            predictions[f"week_{horizon}_ahead_upper_{coverage}"] = float(value + quantile)

    return pd.DataFrame([predictions])


def predict_week1_risk(
    csv_path: str | Path,
    bundle_path: str | Path,
    prediction_date: str | None = None,
) -> pd.DataFrame:
    df = load_data(csv_path)
    bundle = load_bundle(bundle_path)
    if prediction_date is None:
        prediction_timestamp = df[DATE_COLUMN].max() + pd.Timedelta(days=1)
    else:
        prediction_timestamp = pd.Timestamp(prediction_date)
    df = _extend_to_prediction_date(df, prediction_timestamp)
    frame, _ = build_inference_frame(df, base_features=bundle["base_feature_columns"])
    candidate_rows = frame.loc[frame[DATE_COLUMN] == prediction_timestamp].copy()
    if candidate_rows.empty:
        raise ValueError(f"No feature row is available for prediction date {prediction_timestamp.date()}.")
    row = candidate_rows.tail(1)
    probabilities = bundle["model"].predict_proba(row[bundle["feature_columns"]])[0]
    predicted_risk = bundle["model"].predict(row[bundle["feature_columns"]])[0]
    result = {
        "prediction_date": row[DATE_COLUMN].iloc[0].date().isoformat(),
        "predicted_risk": predicted_risk,
    }
    for label, prob in zip(bundle["model"].classes_, probabilities):
        result[f"prob_{label}"] = float(prob)
    result["low_upper_q25"] = float(bundle["metadata"]["thresholds"]["low_upper_q25"])
    result["high_lower_quantile"] = float(
        bundle["metadata"]["thresholds"].get(
            "high_lower_quantile", bundle["metadata"]["thresholds"].get("high_lower_q75")
        )
    )
    result["high_quantile"] = float(bundle["metadata"]["thresholds"].get("high_quantile", 0.75))
    result["high_threshold_mode"] = bundle["metadata"]["thresholds"].get(
        "high_threshold_mode", "training_quantile"
    )
    return pd.DataFrame([result])


def predict_horizon_risk(
    csv_path: str | Path,
    bundle_path: str | Path,
    prediction_date: str | None = None,
) -> pd.DataFrame:
    df = load_data(csv_path)
    bundle = load_bundle(bundle_path)
    if prediction_date is None:
        prediction_timestamp = df[DATE_COLUMN].max() + pd.Timedelta(days=1)
    else:
        prediction_timestamp = pd.Timestamp(prediction_date)
    df = _extend_to_prediction_date(df, prediction_timestamp)
    frame, _ = build_inference_frame(df, base_features=bundle["base_feature_columns"])
    candidate_rows = frame.loc[frame[DATE_COLUMN] == prediction_timestamp].copy()
    if candidate_rows.empty:
        raise ValueError(f"No feature row is available for prediction date {prediction_timestamp.date()}.")
    row = candidate_rows.tail(1)
    probabilities = bundle["model"].predict_proba(row[bundle["feature_columns"]])[0]
    predicted_risk = bundle["model"].predict(row[bundle["feature_columns"]])[0]
    result = {
        "prediction_date": row[DATE_COLUMN].iloc[0].date().isoformat(),
        "horizon": int(bundle["metadata"]["horizon"]),
        "predicted_risk": predicted_risk,
    }
    for label, prob in zip(bundle["model"].classes_, probabilities):
        result[f"prob_{label}"] = float(prob)
    result["low_upper_q25"] = float(bundle["metadata"]["thresholds"]["low_upper_q25"])
    result["high_lower_quantile"] = float(
        bundle["metadata"]["thresholds"].get(
            "high_lower_quantile", bundle["metadata"]["thresholds"].get("high_lower_q75")
        )
    )
    result["high_quantile"] = float(bundle["metadata"]["thresholds"].get("high_quantile", 0.75))
    result["high_threshold_mode"] = bundle["metadata"]["thresholds"].get(
        "high_threshold_mode", "training_quantile"
    )
    return pd.DataFrame([result])


def predict_week1_risk_two_stage(
    csv_path: str | Path,
    bundle_path: str | Path,
    prediction_date: str | None = None,
) -> pd.DataFrame:
    df = load_data(csv_path)
    bundle = load_bundle(bundle_path)
    if prediction_date is None:
        prediction_timestamp = df[DATE_COLUMN].max() + pd.Timedelta(days=1)
    else:
        prediction_timestamp = pd.Timestamp(prediction_date)
    df = _extend_to_prediction_date(df, prediction_timestamp)
    frame, _ = build_inference_frame(df, base_features=bundle["base_feature_columns"])
    candidate_rows = frame.loc[frame[DATE_COLUMN] == prediction_timestamp].copy()
    if candidate_rows.empty:
        raise ValueError(f"No feature row is available for prediction date {prediction_timestamp.date()}.")
    row = candidate_rows.tail(1)
    stage1_prob = bundle["stage1_model"].predict_proba(row[bundle["feature_columns"]])[0]
    stage1_index = {label: idx for idx, label in enumerate(bundle["stage1_model"].classes_)}
    high_prob = float(stage1_prob[stage1_index["high"]])
    stage2_prob = bundle["stage2_model"].predict_proba(row[bundle["feature_columns"]])[0]
    stage2_index = {label: idx for idx, label in enumerate(bundle["stage2_model"].classes_)}
    low_prob = float((1.0 - high_prob) * stage2_prob[stage2_index["low"]])
    medium_prob = float((1.0 - high_prob) * stage2_prob[stage2_index["medium"]])
    probs = {"low": low_prob, "medium": medium_prob, "high": high_prob}
    predicted_risk = max(probs, key=probs.get)
    result = {
        "prediction_date": row[DATE_COLUMN].iloc[0].date().isoformat(),
        "predicted_risk": predicted_risk,
        "prob_low": low_prob,
        "prob_medium": medium_prob,
        "prob_high": high_prob,
        "low_upper_q25": float(bundle["metadata"]["thresholds"]["low_upper_q25"]),
        "high_lower_quantile": float(
            bundle["metadata"]["thresholds"].get(
                "high_lower_quantile", bundle["metadata"]["thresholds"].get("high_lower_q75")
            )
        ),
        "high_quantile": float(bundle["metadata"]["thresholds"].get("high_quantile", 0.75)),
        "high_threshold_mode": bundle["metadata"]["thresholds"].get(
            "high_threshold_mode", "training_quantile"
        ),
    }
    return pd.DataFrame([result])


def predict_week1_high_risk(
    csv_path: str | Path,
    bundle_path: str | Path,
    prediction_date: str | None = None,
) -> pd.DataFrame:
    df = load_data(csv_path)
    bundle = load_bundle(bundle_path)
    if prediction_date is None:
        prediction_timestamp = df[DATE_COLUMN].max() + pd.Timedelta(days=1)
    else:
        prediction_timestamp = pd.Timestamp(prediction_date)
    df = _extend_to_prediction_date(df, prediction_timestamp)
    frame, _ = build_inference_frame(df, base_features=bundle["base_feature_columns"])
    candidate_rows = frame.loc[frame[DATE_COLUMN] == prediction_timestamp].copy()
    if candidate_rows.empty:
        raise ValueError(f"No feature row is available for prediction date {prediction_timestamp.date()}.")
    row = candidate_rows.tail(1)
    probabilities = bundle["model"].predict_proba(row[bundle["feature_columns"]])[0]
    class_index = {label: idx for idx, label in enumerate(bundle["model"].classes_)}
    prob_high = float(probabilities[class_index["high"]])
    threshold = float(bundle["metadata"]["probability_threshold"])
    predicted_label = "high" if prob_high >= threshold else "not_high"
    return pd.DataFrame(
        [
            {
                "prediction_date": row[DATE_COLUMN].iloc[0].date().isoformat(),
                "predicted_high_risk": predicted_label,
                "prob_high": prob_high,
                "probability_threshold": threshold,
                "high_lower_q75": float(bundle["metadata"]["high_lower_q75"]),
                "high_quantile": float(bundle["metadata"].get("high_quantile", 0.75)),
                "high_threshold_mode": bundle["metadata"].get(
                    "high_threshold_mode", "training_quantile"
                ),
            }
        ]
    )


def predict_horizon_high_risk(
    csv_path: str | Path,
    bundle_path: str | Path,
    prediction_date: str | None = None,
) -> pd.DataFrame:
    df = load_data(csv_path)
    bundle = load_bundle(bundle_path)
    if prediction_date is None:
        prediction_timestamp = df[DATE_COLUMN].max() + pd.Timedelta(days=1)
    else:
        prediction_timestamp = pd.Timestamp(prediction_date)
    df = _extend_to_prediction_date(df, prediction_timestamp)
    frame, _ = build_inference_frame(df, base_features=bundle["base_feature_columns"])
    candidate_rows = frame.loc[frame[DATE_COLUMN] == prediction_timestamp].copy()
    if candidate_rows.empty:
        raise ValueError(f"No feature row is available for prediction date {prediction_timestamp.date()}.")
    row = candidate_rows.tail(1)
    probabilities = bundle["model"].predict_proba(row[bundle["feature_columns"]])[0]
    class_index = {label: idx for idx, label in enumerate(bundle["model"].classes_)}
    prob_high = float(probabilities[class_index["high"]])
    threshold = float(bundle["metadata"]["probability_threshold"])
    predicted_label = "high" if prob_high >= threshold else "not_high"
    return pd.DataFrame(
        [
            {
                "prediction_date": row[DATE_COLUMN].iloc[0].date().isoformat(),
                "horizon": int(bundle["metadata"]["horizon"]),
                "predicted_high_risk": predicted_label,
                "prob_high": prob_high,
                "probability_threshold": threshold,
                "high_lower_q75": float(bundle["metadata"]["high_lower_q75"]),
                "high_quantile": float(bundle["metadata"].get("high_quantile", 0.75)),
                "high_threshold_mode": bundle["metadata"].get(
                    "high_threshold_mode", "training_quantile"
                ),
            }
        ]
    )
