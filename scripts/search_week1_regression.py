from __future__ import annotations

import argparse
import json
from pathlib import Path

from chl_forecast.forecasting import search_week1_regression_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a systematic week-1 regression experiment search."
    )
    parser.add_argument("--csv", required=True, help="Path to the source CSV file.")
    parser.add_argument(
        "--output-dir",
        default="models_week1_systematic_search",
        help="Directory where search results and top candidate bundles will be written.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the experiment search.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of expanding-window folds to use on the training split.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top candidate bundles to save.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Run a smaller first-pass search instead of the full experiment matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = search_week1_regression_experiments(
        csv_path=args.csv,
        output_dir=args.output_dir,
        random_state=args.random_state,
        cv_splits=args.cv_splits,
        top_k=args.top_k,
        compact=args.compact,
    )
    print(json.dumps(summary, indent=2))
    print(
        "Saved search results to "
        f"{Path(args.output_dir).resolve() / 'week1_regression_search_results.csv'}"
    )


if __name__ == "__main__":
    main()
