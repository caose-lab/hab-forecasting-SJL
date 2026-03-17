from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chl_forecast.forecasting import (
    DATE_COLUMN,
    _assign_risk_classes,
    _classification_sample_weight,
    _default_hist_classifier,
    _feature_profile_columns,
    _make_pipeline,
    _resolve_high_threshold,
    _week1_feature_columns,
    _week1_risk_feature_columns,
    _week2_classifier_feature_columns,
    build_training_frame,
    load_data,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


CSV_PATH = "/Users/a1amador/Downloads/SJL_daily_df.csv"
OUTPUT_PATH = Path("/Users/a1amador/Documents/Playground/models_multiclass_risk_newvars")
FIXED_HIGH_THRESHOLD = 16.41


def multiclass_metrics(y_true: pd.Series, y_pred: pd.Series, class_order: list[str]) -> dict[str, object]:
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


def evaluate_horizon(horizon: int) -> pd.DataFrame:
    df = load_data(CSV_PATH)
    training_frame, feature_columns, _ = build_training_frame(df)
    target_col = f"target_week_{horizon}"
    modeling_frame = training_frame.dropna(subset=[target_col]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = max(int(len(modeling_frame) * 0.8), 1)
    train_frame = modeling_frame.iloc[:split_index].copy()
    test_frame = modeling_frame.iloc[split_index:].copy()

    q25 = float(train_frame[target_col].quantile(0.25))
    q_high, _ = _resolve_high_threshold(train_frame[target_col], 0.75, FIXED_HIGH_THRESHOLD)
    train_labels = _assign_risk_classes(train_frame[target_col], q25, q_high)
    test_labels = _assign_risk_classes(test_frame[target_col], q25, q_high)

    if horizon == 1:
        candidates = {
            "base_no_newvars": _week1_feature_columns(feature_columns),
            "plus_awnd_tidal": _week1_risk_feature_columns(feature_columns),
            "plus_awnd_watt_tidal": _feature_profile_columns(feature_columns, "target_small_env_week2_calendar_p95"),
            "week2_classifier_awnd_only": _week2_classifier_feature_columns(feature_columns),
        }
        params = {
            "model__learning_rate": 0.05,
            "model__max_depth": 4,
            "model__max_iter": 300,
            "model__max_leaf_nodes": 63,
            "model__min_samples_leaf": 90,
            "model__l2_regularization": 0.8,
        }
        weight_strategy = "inverse"
    else:
        candidates = {
            "week2_awnd_only": _week2_classifier_feature_columns(feature_columns),
            "week2_awnd_watt_tidal": _feature_profile_columns(feature_columns, "target_small_env_week2_calendar_p95"),
            "week2_base_no_newvars": _week1_feature_columns(feature_columns),
            "week3_pruned_broad": _feature_profile_columns(feature_columns, "all_week3_pruned"),
        }
        params = {
            "model__learning_rate": 0.03,
            "model__max_depth": 3,
            "model__max_iter": 800,
            "model__max_leaf_nodes": 63,
            "model__min_samples_leaf": 60,
            "model__l2_regularization": 0.8,
        }
        weight_strategy = "inverse_sqrt"

    rows: list[dict[str, object]] = []
    for name, selected_feature_columns in candidates.items():
        clf = _make_pipeline(_default_hist_classifier(42 + horizon))
        clf.set_params(**params)
        sample_weight = _classification_sample_weight(train_labels, weight_strategy)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["model__sample_weight"] = sample_weight
        clf.fit(train_frame[selected_feature_columns], train_labels, **fit_kwargs)
        pred = clf.predict(test_frame[selected_feature_columns])
        metrics = multiclass_metrics(test_labels, pred, list(clf.classes_))
        metrics.update({"candidate": name, "feature_count": len(selected_feature_columns)})
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "macro_f1", "high_recall", "accuracy"],
        ascending=[False, False, False, False],
    )


def main() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    results = {}
    for horizon in (1, 2):
        df = evaluate_horizon(horizon)
        df.to_csv(OUTPUT_PATH / f"horizon_{horizon}_newvars_comparison.csv", index=False)
        results[f"horizon_{horizon}"] = df.to_dict(orient="records")
    with open(OUTPUT_PATH / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
