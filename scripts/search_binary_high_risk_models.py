from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import ParameterSampler

from chl_forecast.forecasting import (
    DATE_COLUMN,
    _binary_metrics,
    _build_feature_frame,
    _default_hist_classifier,
    _feature_profile_columns,
    _horizon_feature_columns,
    _inverse_frequency_sample_weight,
    _make_pipeline,
    _resolve_high_threshold,
    _select_feature_columns,
    _target_column_for_horizon,
    _week1_feature_columns,
    _week1_risk_feature_columns,
    _week2_classifier_feature_columns,
    build_training_frame,
    load_data,
)


FIXED_HIGH_THRESHOLD = 16.41


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a systematic search for binary high-risk classifiers."
    )
    parser.add_argument(
        "--csv-path",
        default="/Users/a1amador/Downloads/SJL_daily_df.csv",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/a1amador/Documents/Playground/models_binary_high_risk_search",
    )
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _dedupe_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _candidate_feature_sets(feature_columns: list[str], horizon: int) -> list[tuple[str, list[str]]]:
    candidates: list[tuple[str, list[str]]] = []
    if horizon == 1:
        candidates = [
            ("week1_default", _week1_feature_columns(feature_columns)),
            ("week1_plus_awnd_tidal", _week1_risk_feature_columns(feature_columns)),
            ("week2_classifier", _week2_classifier_feature_columns(feature_columns)),
            ("small_env_week2", _feature_profile_columns(feature_columns, "target_small_env_week2_calendar_p95")),
        ]
    elif horizon == 2:
        candidates = [
            ("week2_default", _week2_classifier_feature_columns(feature_columns)),
            ("horizon_default", _horizon_feature_columns(feature_columns, 2)),
            ("small_env_week2", _feature_profile_columns(feature_columns, "target_small_env_week2_calendar_p95")),
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
        deduped = tuple(_dedupe_columns(cols))
        if deduped in seen:
            continue
        seen.add(deduped)
        unique.append((name, list(deduped)))
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


def _make_sample_weight(labels: pd.Series, strategy: str) -> np.ndarray | None:
    if strategy == "none":
        return None
    counts = labels.value_counts()
    n_classes = max(len(counts), 1)
    base = labels.map({label: len(labels) / (n_classes * count) for label, count in counts.items()}).to_numpy(float)
    if strategy == "inverse":
        return base
    if strategy == "inverse_sqrt":
        return np.sqrt(base)
    if strategy == "inverse_pow_1_5":
        return np.power(base, 1.5)
    raise ValueError(f"Unsupported weight strategy: {strategy}")


def _choose_threshold(probabilities: np.ndarray, labels: pd.Series, strategy: str) -> tuple[float, dict[str, float]]:
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
            raise ValueError(f"Unsupported threshold strategy: {strategy}")
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
            best_metrics = {
                "balanced_accuracy": float(balanced_acc),
                "high_precision": float(precision[0]),
                "high_recall": float(recall[0]),
                "high_f1": float(f1[0]),
            }
    return best_threshold, best_metrics


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

    q_high, high_threshold_mode = _resolve_high_threshold(
        train_frame[target_column], 0.75, FIXED_HIGH_THRESHOLD
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

    feature_sets = _candidate_feature_sets(feature_columns, horizon)
    param_candidates = list(
        ParameterSampler(_parameter_space(horizon), n_iter=n_iter, random_state=random_state + horizon)
    )
    weight_strategies = ["inverse", "inverse_sqrt", "none"]
    threshold_strategies = ["recall_focused", "balanced", "f1_focused"]

    results: list[dict[str, object]] = []
    experiment_id = 0
    for feature_name, selected_feature_columns in feature_sets:
        for params in param_candidates:
            for weight_strategy in weight_strategies:
                for threshold_strategy in threshold_strategies:
                    experiment_id += 1
                    classifier = _make_pipeline(_default_hist_classifier(random_state + experiment_id))
                    classifier.set_params(**params)
                    sample_weight = _make_sample_weight(model_train_labels, weight_strategy)
                    fit_kwargs = {}
                    if sample_weight is not None:
                        fit_kwargs["model__sample_weight"] = sample_weight
                    classifier.fit(
                        model_train[selected_feature_columns],
                        model_train_labels,
                        **fit_kwargs,
                    )

                    valid_prob = classifier.predict_proba(model_valid[selected_feature_columns])
                    valid_index = {label: idx for idx, label in enumerate(classifier.classes_)}
                    high_valid_prob = valid_prob[:, valid_index["high"]]
                    threshold, valid_metrics = _choose_threshold(
                        high_valid_prob, model_valid_labels, threshold_strategy
                    )

                    refit_weight = _make_sample_weight(train_labels, weight_strategy)
                    refit_kwargs = {}
                    if refit_weight is not None:
                        refit_kwargs["model__sample_weight"] = refit_weight
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
                    metrics = _binary_metrics(test_labels, test_pred, high_test_prob)
                    metrics.update(
                        {
                            "experiment_id": experiment_id,
                            "horizon": horizon,
                            "feature_set": feature_name,
                            "feature_count": len(selected_feature_columns),
                            "weight_strategy": weight_strategy,
                            "threshold_strategy": threshold_strategy,
                            "probability_threshold": float(threshold),
                            "validation_balanced_accuracy": valid_metrics["balanced_accuracy"],
                            "validation_high_precision": valid_metrics["high_precision"],
                            "validation_high_recall": valid_metrics["high_recall"],
                            "validation_high_f1": valid_metrics["high_f1"],
                            "high_threshold_mode": high_threshold_mode,
                            "high_threshold_value": float(q_high),
                            "predicted_positive_rate": float((test_pred == "high").mean()),
                            "params": json.dumps(params, sort_keys=True),
                        }
                    )
                    results.append(metrics)

    results_df = pd.DataFrame(results).sort_values(
        ["balanced_accuracy", "high_f1", "high_recall", "roc_auc", "high_precision"],
        ascending=[False, False, False, False, False],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / f"horizon_{horizon}_binary_high_risk_search_results.csv", index=False)

    best = results_df.iloc[0].to_dict()
    summary = {
        "horizon": horizon,
        "fixed_high_threshold": FIXED_HIGH_THRESHOLD,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "feature_sets_tested": [name for name, _ in feature_sets],
        "parameter_samples": len(param_candidates),
        "weight_strategies": weight_strategies,
        "threshold_strategies": threshold_strategies,
        "best_result": best,
    }
    with open(output_dir / f"horizon_{horizon}_binary_high_risk_search_summary.json", "w", encoding="utf-8") as handle:
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
    with open(output_dir / "binary_high_risk_search_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)


if __name__ == "__main__":
    main()
