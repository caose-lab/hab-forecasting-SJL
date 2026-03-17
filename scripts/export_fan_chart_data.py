from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from chl_forecast.forecasting import DATE_COLUMN, build_training_frame, load_bundle, load_data


def _holdout_predictions(csv_path: Path, bundle_path: Path, horizon: int) -> pd.DataFrame:
    df = load_data(csv_path)
    frame, feature_columns, _ = build_training_frame(df)
    target_column = f"target_week_{horizon}"
    modeling = frame.dropna(subset=[target_column]).sort_values(DATE_COLUMN).reset_index(drop=True)
    split_index = max(int(len(modeling) * 0.8), 1)
    holdout = modeling.iloc[split_index:].copy().reset_index(drop=True)

    bundle = load_bundle(bundle_path)
    model_columns = bundle["feature_columns_by_horizon"].get(horizon, bundle["feature_columns"])
    predictions = bundle["models"][horizon].predict(holdout[model_columns])

    out = pd.DataFrame(
        {
            "prediction_date": holdout[DATE_COLUMN],
            "target_window_start": holdout[DATE_COLUMN] + pd.to_timedelta((horizon - 1) * 7, unit="D"),
            "target_window_end": holdout[DATE_COLUMN] + pd.to_timedelta(((horizon - 1) * 7) + 6, unit="D"),
            "pred": predictions,
            "obs": holdout[target_column],
        }
    )
    out["date"] = out["target_window_start"]

    conformal = bundle.get("conformal_intervals_by_horizon", {}).get(horizon, {})
    alpha_to_coverage = {"0.5": 50, "0.32": 68, "0.2": 80, "0.1": 90}
    for alpha_label, coverage in alpha_to_coverage.items():
        quantile = conformal.get("alphas", {}).get(alpha_label)
        if quantile is None:
            continue
        out[f"pi{coverage}_lower"] = out["pred"] - float(quantile)
        out[f"pi{coverage}_upper"] = out["pred"] + float(quantile)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MATLAB-friendly fan chart data for holdout forecasts.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--week1-bundle", type=Path, required=True)
    parser.add_argument("--week2-bundle", type=Path, required=True)
    parser.add_argument("--week3-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {}
    for horizon, bundle_path in [
        (1, args.week1_bundle),
        (2, args.week2_bundle),
        (3, args.week3_bundle),
    ]:
        fan = _holdout_predictions(args.csv, bundle_path, horizon)
        csv_path = args.output_dir / f"week{horizon}_fan_chart_data.csv"
        fan.to_csv(csv_path, index=False)
        outputs[f"week_{horizon}"] = {
            "rows": int(len(fan)),
            "csv": str(csv_path),
            "prediction_start": str(fan["prediction_date"].min().date()),
            "prediction_end": str(fan["prediction_date"].max().date()),
            "verification_start": str(fan["target_window_start"].min().date()),
            "verification_end": str(fan["target_window_end"].max().date()),
            "columns": list(fan.columns),
        }

    summary_path = args.output_dir / "fan_chart_data_summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
