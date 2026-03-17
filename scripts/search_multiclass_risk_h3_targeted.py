from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from chl_forecast.forecasting import (
    DATE_COLUMN,
    _assign_risk_classes,
    _classification_sample_weight,
    _default_hist_classifier,
    _feature_profile_columns,
    _make_pipeline,
    _resolve_high_threshold,
    build_training_frame,
    load_data,
)

CSV_PATH = "/Users/a1amador/Downloads/SJL_daily_df.csv"
OUTPUT_DIR = Path("/Users/a1amador/Documents/Playground/models_multiclass_risk_h3_targeted")
FIXED_HIGH_THRESHOLD = 16.41


def metrics(y_true: pd.Series, y_pred: pd.Series, class_order: list[str]) -> dict[str, object]:
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


def main() -> None:
    df = load_data(CSV_PATH)
    training_frame, feature_columns, _ = build_training_frame(df)
    target_col = "target_week_3"
    modeling_frame = training_frame.dropna(subset=[target_col]).sort_values(DATE_COLUMN).reset_index(drop=True)
    split_index = max(int(len(modeling_frame) * 0.8), 1)
    train_frame = modeling_frame.iloc[:split_index].copy()
    test_frame = modeling_frame.iloc[split_index:].copy()

    q25 = float(train_frame[target_col].quantile(0.25))
    q_high, _ = _resolve_high_threshold(train_frame[target_col], 0.75, FIXED_HIGH_THRESHOLD)
    train_labels = _assign_risk_classes(train_frame[target_col], q25, q_high)
    test_labels = _assign_risk_classes(test_frame[target_col], q25, q_high)

    candidates = [
        (
            "baseline_current",
            _feature_profile_columns(feature_columns, "all_week3_pruned"),
            {
                "model__learning_rate": 0.03,
                "model__max_depth": 3,
                "model__max_iter": 800,
                "model__max_leaf_nodes": 23,
                "model__min_samples_leaf": 90,
                "model__l2_regularization": 1.5,
            },
            "none",
        ),
        (
            "broader_weighted",
            _feature_profile_columns(feature_columns, "all_week3_pruned"),
            {
                "model__learning_rate": 0.03,
                "model__max_depth": 3,
                "model__max_iter": 800,
                "model__max_leaf_nodes": 63,
                "model__min_samples_leaf": 60,
                "model__l2_regularization": 0.8,
            },
            "inverse_sqrt",
        ),
        (
            "compact_week3",
            _feature_profile_columns(feature_columns, "all_week3_compact"),
            {
                "model__learning_rate": 0.03,
                "model__max_depth": 3,
                "model__max_iter": 800,
                "model__max_leaf_nodes": 63,
                "model__min_samples_leaf": 60,
                "model__l2_regularization": 0.8,
            },
            "inverse_sqrt",
        ),
        (
            "target_weather_calendar",
            _feature_profile_columns(feature_columns, "target_weather_calendar"),
            {
                "model__learning_rate": 0.03,
                "model__max_depth": 3,
                "model__max_iter": 800,
                "model__max_leaf_nodes": 63,
                "model__min_samples_leaf": 60,
                "model__l2_regularization": 0.8,
            },
            "inverse_sqrt",
        ),
    ]

    rows = []
    for idx, (name, cols, params, weight_strategy) in enumerate(candidates, start=1):
        clf = _make_pipeline(_default_hist_classifier(100 + idx))
        clf.set_params(**params)
        sample_weight = _classification_sample_weight(train_labels, weight_strategy)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["model__sample_weight"] = sample_weight
        clf.fit(train_frame[cols], train_labels, **fit_kwargs)
        pred = clf.predict(test_frame[cols])
        row = metrics(test_labels, pred, list(clf.classes_))
        row.update(
            {
                "candidate": name,
                "feature_count": len(cols),
                "weight_strategy": weight_strategy,
                "params": json.dumps(params, sort_keys=True),
            }
        )
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "macro_f1", "high_recall", "accuracy"],
        ascending=[False, False, False, False],
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_DIR / "horizon_3_targeted_results.csv", index=False)
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(result_df.to_dict(orient="records"), handle, indent=2)


if __name__ == "__main__":
    main()
