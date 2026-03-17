from __future__ import annotations

import argparse
import json

from chl_forecast.forecasting import predict_horizon_risk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate horizon-specific low/medium/high risk probabilities.")
    parser.add_argument("--csv", required=True, help="Path to the latest CSV file.")
    parser.add_argument("--bundle", required=True, help="Path to the trained horizon risk model bundle.")
    parser.add_argument("--prediction-date", default=None, help="Optional prediction date.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_horizon_risk(
        csv_path=args.csv,
        bundle_path=args.bundle,
        prediction_date=args.prediction_date,
    )
    print(json.dumps(predictions.to_dict(orient="records")[0], indent=2))


if __name__ == "__main__":
    main()
