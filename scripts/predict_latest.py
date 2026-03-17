from __future__ import annotations

import argparse
import json

from chl_forecast.forecasting import predict_from_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 1, 2, and 3 week-ahead CHLL_NN_TOTAL forecasts."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the latest CSV file.",
    )
    parser.add_argument(
        "--bundle",
        default="models/chl_weekly_forecast_bundle.joblib",
        help="Path to the trained model bundle.",
    )
    parser.add_argument(
        "--prediction-date",
        default=None,
        help="Optional prediction date. If omitted, the script predicts for the day after the latest date in the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_from_bundle(
        csv_path=args.csv,
        bundle_path=args.bundle,
        prediction_date=args.prediction_date,
    )
    print(json.dumps(predictions.to_dict(orient="records")[0], indent=2))


if __name__ == "__main__":
    main()
