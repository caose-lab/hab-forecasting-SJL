from __future__ import annotations

import argparse
import json

from chl_forecast.forecasting import predict_week1_risk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate low/medium/high week-1 risk probabilities."
    )
    parser.add_argument("--csv", required=True, help="Path to the latest CSV file.")
    parser.add_argument(
        "--bundle",
        default="models_week1_risk/week1_risk_model_bundle.joblib",
        help="Path to the trained week-1 risk model bundle.",
    )
    parser.add_argument(
        "--prediction-date",
        default=None,
        help="Optional prediction date. If omitted, predicts for the day after the latest date in the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_week1_risk(
        csv_path=args.csv,
        bundle_path=args.bundle,
        prediction_date=args.prediction_date,
    )
    print(json.dumps(predictions.to_dict(orient="records")[0], indent=2))


if __name__ == "__main__":
    main()
