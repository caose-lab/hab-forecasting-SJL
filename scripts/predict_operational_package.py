from __future__ import annotations

import argparse
import json
from pathlib import Path

from chl_forecast.forecasting import (
    predict_from_bundle,
    predict_horizon_high_risk,
    predict_horizon_risk,
)


DEFAULT_WEEK1_REGRESSION = (
    "/Users/a1amador/Documents/Playground/operational_models/week1/regression/chl_weekly_forecast_bundle.joblib"
)
DEFAULT_WEEK1_RISK = (
    "/Users/a1amador/Documents/Playground/operational_models/week1/risk_3class/week1_risk_model_bundle.joblib"
)
DEFAULT_WEEK1_HIGH_RISK = (
    "/Users/a1amador/Documents/Playground/operational_models/week1/high_risk/week1_high_risk_bundle.joblib"
)
DEFAULT_WEEK2_REGRESSION = (
    "/Users/a1amador/Documents/Playground/operational_models/week2/regression/chl_weekly_forecast_bundle.joblib"
)
DEFAULT_WEEK3_REGRESSION = (
    "/Users/a1amador/Documents/Playground/operational_models/week3/regression/chl_weekly_forecast_bundle.joblib"
)
DEFAULT_WEEK3_RISK = (
    "/Users/a1amador/Documents/Playground/operational_models/week3/risk_3class/horizon_3_risk_model_bundle.joblib"
)
DEFAULT_WEEK3_HIGH_RISK = (
    "/Users/a1amador/Documents/Playground/operational_models/week3/high_risk/horizon_3_high_risk_bundle.joblib"
)
DEFAULT_WEEK2_RISK = (
    "/Users/a1amador/Documents/Playground/operational_models/week2/risk_3class/horizon_2_risk_model_bundle.joblib"
)
DEFAULT_WEEK2_HIGH_RISK = (
    "/Users/a1amador/Documents/Playground/operational_models/week2/high_risk/horizon_2_high_risk_bundle.joblib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the combined operational forecast package for week 1 and week 2."
    )
    parser.add_argument("--csv", required=True, help="Path to the latest CSV file.")
    parser.add_argument(
        "--prediction-date",
        default=None,
        help="Optional prediction date. If omitted, predicts for the day after the latest date in the CSV.",
    )
    parser.add_argument("--week1-regression-bundle", default=DEFAULT_WEEK1_REGRESSION)
    parser.add_argument("--week3-regression-bundle", default=DEFAULT_WEEK3_REGRESSION)
    parser.add_argument("--week1-risk-bundle", default=DEFAULT_WEEK1_RISK)
    parser.add_argument("--week1-high-risk-bundle", default=DEFAULT_WEEK1_HIGH_RISK)
    parser.add_argument("--week2-regression-bundle", default=DEFAULT_WEEK2_REGRESSION)
    parser.add_argument("--week2-risk-bundle", default=DEFAULT_WEEK2_RISK)
    parser.add_argument("--week2-high-risk-bundle", default=DEFAULT_WEEK2_HIGH_RISK)
    parser.add_argument("--week3-risk-bundle", default=DEFAULT_WEEK3_RISK)
    parser.add_argument("--week3-high-risk-bundle", default=DEFAULT_WEEK3_HIGH_RISK)
    return parser.parse_args()


def _single_record(df) -> dict:
    return df.to_dict(orient="records")[0]


def _combine_operational_signals(regression: dict, risk_3class: dict, high_risk_alert: dict) -> dict:
    high_threshold = float(risk_3class["high_lower_quantile"])
    point = float(regression["predicted_avg_mg_m3"])
    upper_50 = regression.get("upper_50_mg_m3")
    upper_68 = regression.get("upper_68_mg_m3")
    upper_80 = regression.get("upper_80_mg_m3")
    binary_prob = float(high_risk_alert["prob_high"])
    binary_threshold = float(high_risk_alert["probability_threshold"])
    binary_label = high_risk_alert["predicted_high_risk"]
    multiclass_label = risk_3class["predicted_risk"]
    multiclass_high_prob = float(risk_3class["prob_high"])

    exceeds_point = point >= high_threshold
    exceeds_upper_50 = upper_50 is not None and upper_50 >= high_threshold
    exceeds_upper_68 = upper_68 is not None and upper_68 >= high_threshold
    exceeds_upper_80 = upper_80 is not None and upper_80 >= high_threshold

    evidence_score = 0
    if binary_label == "high":
        evidence_score += 3
    elif binary_prob >= max(binary_threshold * 0.8, binary_threshold - 0.03):
        evidence_score += 1

    if multiclass_label == "high":
        evidence_score += 3
    elif multiclass_label == "medium":
        evidence_score += 1

    if exceeds_point:
        evidence_score += 3
    elif exceeds_upper_50:
        evidence_score += 1

    if exceeds_upper_68:
        evidence_score += 1
    if exceeds_upper_80:
        evidence_score += 1
    if multiclass_high_prob >= 0.30:
        evidence_score += 1

    if binary_label == "high" and (exceeds_point or multiclass_label == "high" or exceeds_upper_68):
        warning_level = "high"
    elif evidence_score >= 4:
        warning_level = "elevated"
    elif multiclass_label == "medium" or exceeds_upper_68:
        warning_level = "moderate"
    else:
        warning_level = "low"

    agreement_count = sum(
        [
            binary_label == "high",
            multiclass_label == "high",
            exceeds_point,
        ]
    )
    if warning_level == "low":
        confidence = "high" if agreement_count == 0 and not exceeds_upper_68 else "moderate"
    elif agreement_count >= 2:
        confidence = "high"
    elif binary_label == "high" or exceeds_upper_68 or multiclass_label == "medium":
        confidence = "moderate"
    else:
        confidence = "low"

    if warning_level == "high":
        if exceeds_point:
            summary = "Elevated conditions expected, and bloom risk is high."
        else:
            summary = "Moderate conditions expected, but elevated bloom risk remains likely."
    elif warning_level == "elevated":
        summary = "Moderate conditions expected, but elevated bloom risk remains possible."
    elif warning_level == "moderate":
        summary = "Moderate conditions expected, with some uncertainty about elevated bloom risk."
    else:
        summary = "Low to moderate conditions expected, and elevated bloom risk appears limited."

    return {
        "warning_level": warning_level,
        "confidence": confidence,
        "summary": summary,
        "bloom_threshold_mg_m3": high_threshold,
        "point_exceeds_bloom_threshold": exceeds_point,
        "upper_50_exceeds_bloom_threshold": exceeds_upper_50,
        "upper_68_exceeds_bloom_threshold": exceeds_upper_68,
        "upper_80_exceeds_bloom_threshold": exceeds_upper_80,
        "binary_alert_supports_high_risk": binary_label == "high",
        "binary_high_risk_probability": binary_prob,
        "three_class_supports_high_risk": multiclass_label == "high",
        "three_class_high_probability": multiclass_high_prob,
        "signal_agreement_count": agreement_count,
        "evidence_score": evidence_score,
    }


def main() -> None:
    args = parse_args()

    week1_reg = _single_record(
        predict_from_bundle(
            csv_path=args.csv,
            bundle_path=args.week1_regression_bundle,
            prediction_date=args.prediction_date,
        )
    )
    week2_reg = _single_record(
        predict_from_bundle(
            csv_path=args.csv,
            bundle_path=args.week2_regression_bundle,
            prediction_date=args.prediction_date,
        )
    )
    week3_reg = _single_record(
        predict_from_bundle(
            csv_path=args.csv,
            bundle_path=args.week3_regression_bundle,
            prediction_date=args.prediction_date,
        )
    )
    week1_risk = _single_record(
        predict_horizon_risk(
            csv_path=args.csv,
            bundle_path=args.week1_risk_bundle,
            prediction_date=args.prediction_date,
        )
    )
    week2_risk = _single_record(
        predict_horizon_risk(
            csv_path=args.csv,
            bundle_path=args.week2_risk_bundle,
            prediction_date=args.prediction_date,
        )
    )
    week3_risk = _single_record(
        predict_horizon_risk(
            csv_path=args.csv,
            bundle_path=args.week3_risk_bundle,
            prediction_date=args.prediction_date,
        )
    )
    week1_high = _single_record(
        predict_horizon_high_risk(
            csv_path=args.csv,
            bundle_path=args.week1_high_risk_bundle,
            prediction_date=args.prediction_date,
        )
    )
    week2_high = _single_record(
        predict_horizon_high_risk(
            csv_path=args.csv,
            bundle_path=args.week2_high_risk_bundle,
            prediction_date=args.prediction_date,
        )
    )
    week3_high = _single_record(
        predict_horizon_high_risk(
            csv_path=args.csv,
            bundle_path=args.week3_high_risk_bundle,
            prediction_date=args.prediction_date,
        )
    )

    output = {
        "prediction_date": week1_reg["prediction_date"],
        "week1": {
            "regression": {
                "predicted_avg_mg_m3": week1_reg["week_1_ahead_avg"],
                "lower_50_mg_m3": week1_reg.get("week_1_ahead_lower_50"),
                "upper_50_mg_m3": week1_reg.get("week_1_ahead_upper_50"),
                "lower_68_mg_m3": week1_reg.get("week_1_ahead_lower_68"),
                "upper_68_mg_m3": week1_reg.get("week_1_ahead_upper_68"),
                "lower_80_mg_m3": week1_reg.get("week_1_ahead_lower_80"),
                "upper_80_mg_m3": week1_reg.get("week_1_ahead_upper_80"),
                "lower_90_mg_m3": week1_reg.get("week_1_ahead_lower_90"),
                "upper_90_mg_m3": week1_reg.get("week_1_ahead_upper_90"),
            },
            "risk_3class": {
                "predicted_risk": week1_risk["predicted_risk"],
                "prob_low": week1_risk["prob_low"],
                "prob_medium": week1_risk["prob_medium"],
                "prob_high": week1_risk["prob_high"],
                "low_upper_q25": week1_risk["low_upper_q25"],
                "high_lower_quantile": week1_risk["high_lower_quantile"],
                "high_quantile": week1_risk["high_quantile"],
                "high_threshold_mode": week1_risk["high_threshold_mode"],
            },
            "high_risk_alert": {
                "predicted_high_risk": week1_high["predicted_high_risk"],
                "prob_high": week1_high["prob_high"],
                "probability_threshold": week1_high["probability_threshold"],
                "high_lower_q75": week1_high["high_lower_q75"],
                "high_quantile": week1_high["high_quantile"],
                "high_threshold_mode": week1_high["high_threshold_mode"],
            },
        },
        "week2": {
            "regression": {
                "predicted_avg_mg_m3": week2_reg["week_2_ahead_avg"],
                "lower_50_mg_m3": week2_reg.get("week_2_ahead_lower_50"),
                "upper_50_mg_m3": week2_reg.get("week_2_ahead_upper_50"),
                "lower_68_mg_m3": week2_reg.get("week_2_ahead_lower_68"),
                "upper_68_mg_m3": week2_reg.get("week_2_ahead_upper_68"),
                "lower_80_mg_m3": week2_reg.get("week_2_ahead_lower_80"),
                "upper_80_mg_m3": week2_reg.get("week_2_ahead_upper_80"),
                "lower_90_mg_m3": week2_reg.get("week_2_ahead_lower_90"),
                "upper_90_mg_m3": week2_reg.get("week_2_ahead_upper_90"),
            },
            "risk_3class": {
                "predicted_risk": week2_risk["predicted_risk"],
                "prob_low": week2_risk["prob_low"],
                "prob_medium": week2_risk["prob_medium"],
                "prob_high": week2_risk["prob_high"],
                "low_upper_q25": week2_risk["low_upper_q25"],
                "high_lower_quantile": week2_risk["high_lower_quantile"],
                "high_quantile": week2_risk["high_quantile"],
                "high_threshold_mode": week2_risk["high_threshold_mode"],
            },
            "high_risk_alert": {
                "predicted_high_risk": week2_high["predicted_high_risk"],
                "prob_high": week2_high["prob_high"],
                "probability_threshold": week2_high["probability_threshold"],
                "high_lower_q75": week2_high["high_lower_q75"],
                "high_quantile": week2_high["high_quantile"],
                "high_threshold_mode": week2_high["high_threshold_mode"],
            },
        },
        "week3": {
            "regression": {
                "predicted_avg_mg_m3": week3_reg["week_3_ahead_avg"],
                "lower_50_mg_m3": week3_reg.get("week_3_ahead_lower_50"),
                "upper_50_mg_m3": week3_reg.get("week_3_ahead_upper_50"),
                "lower_68_mg_m3": week3_reg.get("week_3_ahead_lower_68"),
                "upper_68_mg_m3": week3_reg.get("week_3_ahead_upper_68"),
                "lower_80_mg_m3": week3_reg.get("week_3_ahead_lower_80"),
                "upper_80_mg_m3": week3_reg.get("week_3_ahead_upper_80"),
                "lower_90_mg_m3": week3_reg.get("week_3_ahead_lower_90"),
                "upper_90_mg_m3": week3_reg.get("week_3_ahead_upper_90"),
            },
            "risk_3class": {
                "predicted_risk": week3_risk["predicted_risk"],
                "prob_low": week3_risk["prob_low"],
                "prob_medium": week3_risk["prob_medium"],
                "prob_high": week3_risk["prob_high"],
                "low_upper_q25": week3_risk["low_upper_q25"],
                "high_lower_quantile": week3_risk["high_lower_quantile"],
                "high_quantile": week3_risk["high_quantile"],
                "high_threshold_mode": week3_risk["high_threshold_mode"],
            },
            "high_risk_alert": {
                "predicted_high_risk": week3_high["predicted_high_risk"],
                "prob_high": week3_high["prob_high"],
                "probability_threshold": week3_high["probability_threshold"],
                "high_lower_q75": week3_high["high_lower_q75"],
                "high_quantile": week3_high["high_quantile"],
                "high_threshold_mode": week3_high["high_threshold_mode"],
            },
        },
    }

    for week_key in ("week1", "week2", "week3"):
        output[week_key]["operational_assessment"] = _combine_operational_signals(
            regression=output[week_key]["regression"],
            risk_3class=output[week_key]["risk_3class"],
            high_risk_alert=output[week_key]["high_risk_alert"],
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
