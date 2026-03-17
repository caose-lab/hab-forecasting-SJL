from __future__ import annotations

import argparse
import json
from pathlib import Path

from chl_forecast.forecasting import train_horizon_high_risk_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a horizon-specific binary high-risk alert model.")
    parser.add_argument("--csv", required=True, help="Path to the source CSV file.")
    parser.add_argument("--horizon", type=int, required=True, help="Forecast horizon, e.g. 2.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the trained high-risk model and report will be written.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--high-quantile", type=float, default=0.75, help="High-risk quantile.")
    parser.add_argument(
        "--fixed-high-threshold",
        type=float,
        default=None,
        help="Optional fixed high-risk threshold in mg/m^3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_horizon_high_risk_model(
        csv_path=args.csv,
        output_dir=args.output_dir,
        horizon=args.horizon,
        random_state=args.random_state,
        high_quantile=args.high_quantile,
        fixed_high_threshold=args.fixed_high_threshold,
    )
    print(json.dumps(report, indent=2))
    print(
        f"Saved high-risk model bundle to {Path(args.output_dir).resolve() / f'horizon_{args.horizon}_high_risk_bundle.joblib'}"
    )


if __name__ == "__main__":
    main()
