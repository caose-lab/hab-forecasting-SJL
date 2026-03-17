from __future__ import annotations

import argparse
import json
from pathlib import Path

from chl_forecast.forecasting import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train direct weekly forecasting models for CHLL_NN_TOTAL."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory where the trained model bundle and reports will be written.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for model search.",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=25,
        help="Number of randomized hyperparameter configurations to test per horizon.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Forecast horizons to train, chosen from 1 2 3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_and_evaluate(
        csv_path=args.csv,
        output_dir=args.output_dir,
        random_state=args.random_state,
        search_iterations=args.search_iterations,
        horizons=tuple(args.horizons),
    )
    print(json.dumps(report, indent=2))
    print(
        f"Saved model bundle to {Path(args.output_dir).resolve() / 'chl_weekly_forecast_bundle.joblib'}"
    )


if __name__ == "__main__":
    main()
