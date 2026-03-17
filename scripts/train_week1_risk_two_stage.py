from __future__ import annotations

import argparse
import json
from pathlib import Path

from chl_forecast.forecasting import train_week1_risk_two_stage_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a two-stage week-1 risk classification model."
    )
    parser.add_argument("--csv", required=True, help="Path to the source CSV file.")
    parser.add_argument(
        "--output-dir",
        default="models_week1_risk_two_stage",
        help="Directory where the trained risk model and report will be written.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for model training.",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=0.75,
        help="Upper risk threshold quantile for the high class.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_week1_risk_two_stage_model(
        csv_path=args.csv,
        output_dir=args.output_dir,
        random_state=args.random_state,
        high_quantile=args.high_quantile,
    )
    print(json.dumps(report, indent=2))
    print(
        f"Saved two-stage risk model bundle to {Path(args.output_dir).resolve() / 'week1_risk_two_stage_bundle.joblib'}"
    )


if __name__ == "__main__":
    main()
