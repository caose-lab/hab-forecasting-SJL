from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import ParameterSampler

from chl_forecast.forecasting import (
    DATE_COLUMN,
    _add_calendar_features,
    _assign_risk_classes,
    _classification_sample_weight,
    _default_hist_classifier,
    _feature_profile_columns,
    _horizon_feature_columns,
    _horizon_risk_feature_columns,
    _make_pipeline,
    _resolve_high_threshold,
    _target_column_for_horizon,
    _week1_risk_feature_columns,
    _week2_classifier_feature_columns,
    build_training_frame,
    load_data,
)


FIXED_HIGH_THRESHOLD = 16.41


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a systematic search for 3-class risk classifiers.")
    parser.add_argument("--csv-path", default="/Users/a1amador/Downloads/SJL_daily_df.csv")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--output-dir",
        default="/Users/a1amador/Documents/Playground/models_multiclass_risk_search",
    )
    parser.add_argument("--n-iter", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _dedupe_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _candidate_feature_sets(feature_columns: list[str], horizon: int) -> list[tuple[str, list[str]]]:
    candidates: list[tuple[str, list[str]]]
    if horizon == 1:
        candidates = [
            ("week1_risk_default", _week1_risk_feature_columns(feature_columns)),
            ("week1_no_awnd_tidal", _horizon_feature_columns(feature_columns, 1)),
            ("week2_classifier", _week2_classifier_feature_columns(feature_columns)),
            ("small_env_week2", _feature_profile_columns(feature_columns, "target_small_env_week2_calendar_p95")),
        ]
    elif horizon == 2:
        candidates = [
            ("week2_risk_default", _horizon_risk_feature_columns(feature_columns, 2)),
            ("week2_horizon_default", _horizon_feature_columns(feature_columns, 2)),
            ("week2_small_env", _feature_profile_columns(feature_columns, "target_small_env_week2_calendar_p95")),
            ("week3_pruned", _feature_profile_columns(feature_columns, "all_week3_pruned")),
        ]
    else:
        candidates = [
            ("week3_pruned", _feature_profile_columns(feature_columns, "all_week3_pruned")),
            ("week3_compact", _feature_profile_columns(feature_columns, "all_week3_compact")),
            ("target_weather_calendar", _feature_profile_columns(feature_columns, "target_weather_calendar")),
            ("week2_default", _week2_classifier_feature_columns(feature_columns)),
        ]
    unique: list[tuple[str, list[str]]] = []
    seen: set[tuple[str, ...]] = set()
    for name, cols in candidates:
        key = tuple(_dedupe_columns(cols))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, list(key)))
    return unique


def _parameter_space(horizon: int) -> dict[str, list[object]]:
    if horizon == 3:
        return {
            "model__learning_rate": [0.02, 0.03, 0.05],
            "model__max_depth": [3, 4],
            "model__max_iter": [500, 800, 1200],
            "model__max_leaf_nodes": [23, 31, 63],
            "model__min_samples_leaf": [60, 90, 120],
            "model__l2_regularization": [0.8, 1.5, 2.0],
        }
    return {
        "model__learning_rate": [0.02, 0.03, 0.05],
        "model__max_depth": [3, 4],
        "model__max_iter": [300, 500, 800],
        "model__max_leaf_nodes": [23, 31, 63],
        "model__min_samples_leaf": [40, 60, 90],
        "model__l2_regularization": [0.4, 0.8, 1.2],
    }


def _multiclass_metrics(y_true: pd.Series, y_pred: pd.Series, class_order: list[str]) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    high_recall = float((((y_true == "high") & (y_pred == "high")).sum()) / max((y_true == "high").sum(), 1))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "high_recall": high_recall,
        "confusion_matrix": cm.tolist(),
        "class_order": class_order,
    }


def _run_horizon_search(
    csv_path: str | Path,
    output_dir: Path,
    horizon: int,
    n_iter: int,
    random_state: int,
) -> dict[str, object]:
    df = load_data(csv_path)
    training_frame, feature_columns, _ = build_training_frame(df)
    target_column = _target_column_for_horizon(horizon)
    modeling_frame = training_frame.dropna(subset=[target_column]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = max(int(len(modeling_frame) * 0.8), 1)
    train_frame = modeling_frame.iloc[:split_index].copy()
    test_frame = modeling_frame.iloc[split_index:].copy()
    if test_frame.empty:
        raise ValueError(f"Not enough samples to create a holdout set for horizon {horizon}.")

    q25 = float(train_frame[target_column].quantile(0.25))
    q_high, high_threshold_mode = _resolve_high_threshold(train_frame[target_column], 0.75, FIXED_HIGH_THRESHOLD)
    train_labels = _assign_risk_classes(train_frame[target_column], q25, q_high)
    test_labels = _assign_risk_classes(test_frame[target_column], q25, q_high)

    feature_sets = _candidate_feature_sets(feature_columns, horizon)
    param_candidates = list(ParameterSampler(_parameter_space(horizon), n_iter=n_iter, random_state=random_state + horizon))
    weight_strategies = ["none", "inverse", "inverse_sqrt"]

    results: list[dict[str, object]] = []
    experiment_id = 0
    for feature_name, selected_feature_columns in feature_sets:
        for params in param_candidates:
            for weight_strategy in weight_strategies:
                experiment_id += 1
                classifier = _make_pipeline(_default_hist_classifier(random_state + experiment_id))
                classifier.set_params(**params)
                sample_weight = _classification_sample_weight(train_labels, weight_strategy)
                fit_kwargs = {}
                if sample_weight is not None:
                    fit_kwargs["model__sample_weight"] = sample_weight
                classifier.fit(train_frame[selected_feature_columns], train_labels, **fit_kwargs)
                predictions = classifier.predict(test_frame[selected_feature_columns])
                class_order = list(classifier.classes_)
                metrics = _multiclass_metrics(test_labels, predictions, class_order)
                metrics.update(
                    {
                        "experiment_id": experiment_id,
                        "horizon": horizon,
                        "feature_set": feature_name,
                        "feature_count": len(selected_feature_columns),
                        "weight_strategy": weight_strategy,
                        "high_threshold_mode": high_threshold_mode,
                        "high_threshold_value": float(q_high),
                        "low_threshold_value": float(q25),
                        "params": json.dumps(params, sort_keys=True),
                    }
                )
                results.append(metrics)

    results_df = pd.DataFrame(results).sort_values(
        ["balanced_accuracy", "macro_f1", "high_recall", "accuracy"],
        ascending=[False, False, False, False],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / f"horizon_{horizon}_multiclass_risk_search_results.csv", index=False)
    best = results_df.iloc[0].to_dict()
    summary = {
        "horizon": horizon,
        "fixed_high_threshold": FIXED_HIGH_THRESHOLD,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "feature_sets_tested": [name for name, _ in feature_sets],
        "parameter_samples": len(param_candidates),
        "weight_strategies": weight_strategies,
        "best_result": best,
    }
    with open(output_dir / f"horizon_{horizon}_multiclass_risk_search_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summaries = []
    for horizon in args.horizons:
        horizon_dir = output_dir / f"horizon_{horizon}"
        summaries.append(
            _run_horizon_search(
                csv_path=args.csv_path,
                output_dir=horizon_dir,
                horizon=horizon,
                n_iter=args.n_iter,
                random_state=args.random_state,
            )
        )
    with open(output_dir / "multiclass_risk_search_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)


if __name__ == "__main__":
    main()
