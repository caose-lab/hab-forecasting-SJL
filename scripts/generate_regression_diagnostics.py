from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from chl_forecast.forecasting import (
    DATE_COLUMN,
    TARGET_COLUMN,
    build_training_frame,
    load_bundle,
    load_data,
)


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {"rows": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    yt = y_true.loc[mask]
    yp = y_pred.loc[mask]
    return {
        "rows": int(mask.sum()),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
    }


def _holdout_frame(csv_path: Path, bundle_path: Path, horizon: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = load_data(csv_path)
    training_frame, feature_columns, _ = build_training_frame(df)
    target_column = f"target_week_{horizon}"
    modeling_frame = training_frame.dropna(subset=[target_column]).copy()
    modeling_frame = modeling_frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    split_index = max(int(len(modeling_frame) * 0.8), 1)
    holdout = modeling_frame.iloc[split_index:].copy().reset_index(drop=True)

    bundle = load_bundle(bundle_path)
    model_columns = bundle["feature_columns_by_horizon"].get(horizon, bundle["feature_columns"])
    model = bundle["models"][horizon]
    holdout["model_prediction"] = model.predict(holdout[model_columns])

    holdout["persistence_last_value"] = holdout[f"{TARGET_COLUMN}_raw"]
    holdout["persistence_trailing7_mean"] = holdout[f"{TARGET_COLUMN}_roll_mean_7"]
    return holdout, bundle


def _line_svg(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{dash_attr} />'
    )


def _text_svg(x: float, y: float, text: str, size: int = 14, weight: str = "normal", anchor: str = "start") -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="Helvetica, Arial, sans-serif" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#1f2937">{safe}</text>'
    )


def _circle_svg(cx: float, cy: float, r: float, fill: str, opacity: float) -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" fill-opacity="{opacity:.3f}" />'


def _scatter_panel_svg(
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
    observed: np.ndarray,
    predicted: np.ndarray,
    title: str,
    metrics: dict[str, float],
    axis_limits: tuple[float, float],
) -> str:
    margin_left = 58
    margin_right = 18
    margin_top = 38
    margin_bottom = 44
    plot_x0 = panel_x + margin_left
    plot_y0 = panel_y + margin_top
    plot_w = panel_w - margin_left - margin_right
    plot_h = panel_h - margin_top - margin_bottom

    vmin, vmax = axis_limits

    def sx(v: float) -> float:
        return plot_x0 + (v - vmin) / (vmax - vmin) * plot_w

    def sy(v: float) -> float:
        return plot_y0 + plot_h - (v - vmin) / (vmax - vmin) * plot_h

    parts = [
        f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" fill="#ffffff" stroke="#d1d5db" stroke-width="1" rx="10" ry="10" />',
        _text_svg(panel_x + 18, panel_y + 24, title, size=16, weight="bold"),
        _text_svg(panel_x + 18, panel_y + 44, f'R²={metrics["r2"]:.3f}  RMSE={metrics["rmse"]:.3f}  MAE={metrics["mae"]:.3f}', size=12),
    ]

    # grid and ticks
    ticks = np.linspace(vmin, vmax, 5)
    for tick in ticks:
        x = sx(float(tick))
        y = sy(float(tick))
        parts.append(_line_svg(x, plot_y0, x, plot_y0 + plot_h, "#e5e7eb", 1.0))
        parts.append(_line_svg(plot_x0, y, plot_x0 + plot_w, y, "#e5e7eb", 1.0))
        parts.append(_text_svg(x, plot_y0 + plot_h + 20, f"{tick:.1f}", size=11, anchor="middle"))
        parts.append(_text_svg(plot_x0 - 8, y + 4, f"{tick:.1f}", size=11, anchor="end"))

    # 1:1 line
    parts.append(_line_svg(sx(vmin), sy(vmin), sx(vmax), sy(vmax), "#374151", 1.5, "5 4"))

    # fitted line
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(observed[mask], predicted[mask], 1)
        y1 = slope * vmin + intercept
        y2 = slope * vmax + intercept
        parts.append(_line_svg(sx(vmin), sy(y1), sx(vmax), sy(y2), "#b91c1c", 2.0))

    for xo, yp in zip(observed, predicted):
        if np.isfinite(xo) and np.isfinite(yp):
            parts.append(_circle_svg(sx(float(xo)), sy(float(yp)), 3.2, "#0f766e", 0.42))

    parts.append(_text_svg(panel_x + panel_w / 2, panel_y + panel_h - 10, "Observed weekly mean chlorophyll (mg m^-3)", size=12, anchor="middle"))
    # y-axis label rotated
    parts.append(
        f'<text x="{panel_x + 16:.1f}" y="{panel_y + panel_h / 2:.1f}" font-size="12" '
        'font-family="Helvetica, Arial, sans-serif" fill="#1f2937" '
        f'text-anchor="middle" transform="rotate(-90 {panel_x + 16:.1f} {panel_y + panel_h / 2:.1f})">Predicted weekly mean chlorophyll (mg m^-3)</text>'
    )
    return "\n".join(parts)


def _shared_axis_limits(*frames: pd.DataFrame) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for frame in frames:
        values.append(frame["observed"].to_numpy(dtype=float))
        values.append(frame["predicted"].to_numpy(dtype=float))
    vals = np.concatenate(values)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    pad = 0.05 * (vmax - vmin)
    return vmin - pad, vmax + pad


def _scatter_svg(title: str, left_title: str, right_title: str, left_frame: pd.DataFrame, right_frame: pd.DataFrame) -> str:
    width = 1200
    height = 620
    panel_w = 550
    panel_h = 500
    panel_y = 80
    left_x = 40
    right_x = 610
    axis_limits = _shared_axis_limits(left_frame, right_frame)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        _text_svg(width / 2, 36, title, size=24, weight="bold", anchor="middle"),
        _text_svg(width / 2, 58, "Holdout-period predicted vs observed weekly averages", size=14, anchor="middle"),
        _scatter_panel_svg(
            left_x,
            panel_y,
            panel_w,
            panel_h,
            left_frame["observed"].to_numpy(),
            left_frame["predicted"].to_numpy(),
            left_title,
            _compute_metrics(left_frame["observed"], left_frame["predicted"]),
            axis_limits,
        ),
        _scatter_panel_svg(
            right_x,
            panel_y,
            panel_w,
            panel_h,
            right_frame["observed"].to_numpy(),
            right_frame["predicted"].to_numpy(),
            right_title,
            _compute_metrics(right_frame["observed"], right_frame["predicted"]),
            axis_limits,
        ),
        "</svg>",
    ]
    return "\n".join(pieces)


def _horizon_diagnostics(csv_path: Path, bundle_path: Path, horizon: int, output_dir: Path) -> dict[str, Any]:
    holdout, bundle = _holdout_frame(csv_path, bundle_path, horizon)
    target_column = f"target_week_{horizon}"

    diagnostics = pd.DataFrame(
        {
            "prediction_date": holdout[DATE_COLUMN],
            "observed": holdout[target_column],
            "model_prediction": holdout["model_prediction"],
            "persistence_last_value": holdout["persistence_last_value"],
            "persistence_trailing7_mean": holdout["persistence_trailing7_mean"],
        }
    )
    diagnostics_path = output_dir / f"week{horizon}_holdout_predictions.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    model_metrics = _compute_metrics(diagnostics["observed"], diagnostics["model_prediction"])
    last_value_metrics = _compute_metrics(diagnostics["observed"], diagnostics["persistence_last_value"])
    trailing7_metrics = _compute_metrics(diagnostics["observed"], diagnostics["persistence_trailing7_mean"])

    persistence_mask = diagnostics["observed"].notna() & diagnostics["persistence_last_value"].notna()
    model_plot = pd.DataFrame(
        {
            "observed": diagnostics.loc[persistence_mask, "observed"].reset_index(drop=True),
            "predicted": diagnostics.loc[persistence_mask, "model_prediction"].reset_index(drop=True),
        }
    )
    persistence_plot = pd.DataFrame(
        {
            "observed": diagnostics.loc[persistence_mask, "observed"].reset_index(drop=True),
            "predicted": diagnostics.loc[persistence_mask, "persistence_last_value"].reset_index(drop=True),
        }
    )
    svg = _scatter_svg(
        title=f"Week {horizon} Regression Diagnostics",
        left_title="Model vs observed (matched subset)",
        right_title="Persistence (last observed value) vs observed",
        left_frame=model_plot,
        right_frame=persistence_plot,
    )
    figure_path = output_dir / f"week{horizon}_predicted_vs_observed.svg"
    figure_path.write_text(svg, encoding="utf-8")

    return {
        "bundle_path": str(bundle_path),
        "feature_count": len(bundle["feature_columns_by_horizon"].get(horizon, bundle["feature_columns"])),
        "holdout_rows": int(len(diagnostics)),
        "model": model_metrics,
        "persistence_last_value": last_value_metrics,
        "persistence_trailing7_mean": trailing7_metrics,
        "predictions_csv": str(diagnostics_path),
        "scatter_svg": str(figure_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate regression diagnostics and persistence comparisons.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--week1-bundle", type=Path, required=True)
    parser.add_argument("--week2-bundle", type=Path, required=True)
    parser.add_argument("--week3-bundle", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "week_1": _horizon_diagnostics(args.csv, args.week1_bundle, 1, args.output_dir),
        "week_2": _horizon_diagnostics(args.csv, args.week2_bundle, 2, args.output_dir),
    }
    if args.week3_bundle is not None:
        report["week_3"] = _horizon_diagnostics(args.csv, args.week3_bundle, 3, args.output_dir)
    report_path = args.output_dir / "regression_diagnostics_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
