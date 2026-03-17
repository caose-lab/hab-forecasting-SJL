from __future__ import annotations

import argparse
import json

from chl_forecast.forecasting import expanding_window_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an expanding-window backtest for weekly CHLL_NN_TOTAL forecasts."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon to backtest, chosen from 1 2 3.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of expanding-window folds.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=600,
        help="Minimum number of chronological rows in the initial training window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = expanding_window_backtest(
        csv_path=args.csv,
        horizon=args.horizon,
        n_folds=args.folds,
        min_train_rows=args.min_train_rows,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
