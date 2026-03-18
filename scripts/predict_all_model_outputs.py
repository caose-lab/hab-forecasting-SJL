from __future__ import annotations

import argparse
import json
from pathlib import Path

from chl_forecast.forecasting import (
    predict_from_bundle,
    predict_horizon_high_risk,
    predict_horizon_risk,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = Path("/Users/a1amador/Downloads/SJL_daily_df.csv")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

DEFAULT_WEEK1_REGRESSION = REPO_ROOT / "operational_models/week1/regression/chl_weekly_forecast_bundle.joblib"
DEFAULT_WEEK1_RISK = REPO_ROOT / "operational_models/week1/risk_3class/week1_risk_model_bundle.joblib"
DEFAULT_WEEK1_HIGH_RISK = REPO_ROOT / "operational_models/week1/high_risk/week1_high_risk_bundle.joblib"
DEFAULT_WEEK2_REGRESSION = REPO_ROOT / "operational_models/week2/regression/chl_weekly_forecast_bundle.joblib"
DEFAULT_WEEK2_RISK = REPO_ROOT / "operational_models/week2/risk_3class/horizon_2_risk_model_bundle.joblib"
DEFAULT_WEEK2_HIGH_RISK = REPO_ROOT / "operational_models/week2/high_risk/horizon_2_high_risk_bundle.joblib"
DEFAULT_WEEK3_REGRESSION = REPO_ROOT / "operational_models/week3/regression/chl_weekly_forecast_bundle.joblib"
DEFAULT_WEEK3_RISK = REPO_ROOT / "operational_models/week3/risk_3class/horizon_3_risk_model_bundle.joblib"
DEFAULT_WEEK3_HIGH_RISK = REPO_ROOT / "operational_models/week3/high_risk/horizon_3_high_risk_bundle.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all deployed regression, classification, and combined operational prediction outputs."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the source CSV file.")
    parser.add_argument(
        "--prediction-date",
        default=None,
        help="Optional prediction date. If omitted, predicts for the day after the latest date in the CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to outputs/all_model_outputs_<prediction_date>.json.",
    )
    parser.add_argument("--week1-regression-bundle", type=Path, default=DEFAULT_WEEK1_REGRESSION)
    parser.add_argument("--week1-risk-bundle", type=Path, default=DEFAULT_WEEK1_RISK)
    parser.add_argument("--week1-high-risk-bundle", type=Path, default=DEFAULT_WEEK1_HIGH_RISK)
    parser.add_argument("--week2-regression-bundle", type=Path, default=DEFAULT_WEEK2_REGRESSION)
    parser.add_argument("--week2-risk-bundle", type=Path, default=DEFAULT_WEEK2_RISK)
    parser.add_argument("--week2-high-risk-bundle", type=Path, default=DEFAULT_WEEK2_HIGH_RISK)
    parser.add_argument("--week3-regression-bundle", type=Path, default=DEFAULT_WEEK3_REGRESSION)
    parser.add_argument("--week3-risk-bundle", type=Path, default=DEFAULT_WEEK3_RISK)
    parser.add_argument("--week3-high-risk-bundle", type=Path, default=DEFAULT_WEEK3_HIGH_RISK)
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


def _regression_payload(record: dict, horizon: int) -> dict:
    return {
        "predicted_avg_mg_m3": record[f"week_{horizon}_ahead_avg"],
        "lower_50_mg_m3": record.get(f"week_{horizon}_ahead_lower_50"),
        "upper_50_mg_m3": record.get(f"week_{horizon}_ahead_upper_50"),
        "lower_68_mg_m3": record.get(f"week_{horizon}_ahead_lower_68"),
        "upper_68_mg_m3": record.get(f"week_{horizon}_ahead_upper_68"),
        "lower_80_mg_m3": record.get(f"week_{horizon}_ahead_lower_80"),
        "upper_80_mg_m3": record.get(f"week_{horizon}_ahead_upper_80"),
        "lower_90_mg_m3": record.get(f"week_{horizon}_ahead_lower_90"),
        "upper_90_mg_m3": record.get(f"week_{horizon}_ahead_upper_90"),
    }


def _risk_payload(record: dict) -> dict:
    return {
        "predicted_risk": record["predicted_risk"],
        "prob_low": record["prob_low"],
        "prob_medium": record["prob_medium"],
        "prob_high": record["prob_high"],
        "low_upper_q25": record["low_upper_q25"],
        "high_lower_quantile": record["high_lower_quantile"],
        "high_quantile": record["high_quantile"],
        "high_threshold_mode": record["high_threshold_mode"],
    }


def _high_risk_payload(record: dict) -> dict:
    return {
        "predicted_high_risk": record["predicted_high_risk"],
        "prob_high": record["prob_high"],
        "probability_threshold": record["probability_threshold"],
        "high_lower_q75": record["high_lower_q75"],
        "high_quantile": record["high_quantile"],
        "high_threshold_mode": record["high_threshold_mode"],
    }


def build_output(args: argparse.Namespace) -> dict:
    regression_records = {
        "week1": _single_record(
            predict_from_bundle(
                csv_path=args.csv,
                bundle_path=args.week1_regression_bundle,
                prediction_date=args.prediction_date,
            )
        ),
        "week2": _single_record(
            predict_from_bundle(
                csv_path=args.csv,
                bundle_path=args.week2_regression_bundle,
                prediction_date=args.prediction_date,
            )
        ),
        "week3": _single_record(
            predict_from_bundle(
                csv_path=args.csv,
                bundle_path=args.week3_regression_bundle,
                prediction_date=args.prediction_date,
            )
        ),
    }
    risk_records = {
        "week1": _single_record(
            predict_horizon_risk(
                csv_path=args.csv,
                bundle_path=args.week1_risk_bundle,
                prediction_date=args.prediction_date,
            )
        ),
        "week2": _single_record(
            predict_horizon_risk(
                csv_path=args.csv,
                bundle_path=args.week2_risk_bundle,
                prediction_date=args.prediction_date,
            )
        ),
        "week3": _single_record(
            predict_horizon_risk(
                csv_path=args.csv,
                bundle_path=args.week3_risk_bundle,
                prediction_date=args.prediction_date,
            )
        ),
    }
    high_risk_records = {
        "week1": _single_record(
            predict_horizon_high_risk(
                csv_path=args.csv,
                bundle_path=args.week1_high_risk_bundle,
                prediction_date=args.prediction_date,
            )
        ),
        "week2": _single_record(
            predict_horizon_high_risk(
                csv_path=args.csv,
                bundle_path=args.week2_high_risk_bundle,
                prediction_date=args.prediction_date,
            )
        ),
        "week3": _single_record(
            predict_horizon_high_risk(
                csv_path=args.csv,
                bundle_path=args.week3_high_risk_bundle,
                prediction_date=args.prediction_date,
            )
        ),
    }

    prediction_date = regression_records["week1"]["prediction_date"]
    output = {
        "prediction_date": prediction_date,
        "separate_outputs": {
            "regression": {
                week_key: _regression_payload(record, horizon=index)
                for index, (week_key, record) in enumerate(regression_records.items(), start=1)
            },
            "risk_3class": {
                week_key: _risk_payload(record) for week_key, record in risk_records.items()
            },
            "high_risk_alert": {
                week_key: _high_risk_payload(record) for week_key, record in high_risk_records.items()
            },
        },
        "combined_outputs": {},
    }

    for week_key in ("week1", "week2", "week3"):
        combined = {
            "regression": output["separate_outputs"]["regression"][week_key],
            "risk_3class": output["separate_outputs"]["risk_3class"][week_key],
            "high_risk_alert": output["separate_outputs"]["high_risk_alert"][week_key],
        }
        combined["operational_assessment"] = _combine_operational_signals(
            regression=combined["regression"],
            risk_3class=combined["risk_3class"],
            high_risk_alert=combined["high_risk_alert"],
        )
        output["combined_outputs"][week_key] = combined

    return output


def main() -> None:
    args = parse_args()
    output = build_output(args)
    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"all_model_outputs_{output['prediction_date']}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"prediction_date": output["prediction_date"], "output_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
