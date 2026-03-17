from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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


def _units_label_svg(x: float, y: float, *, size: int, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="Helvetica, Arial, sans-serif" '
        f'font-weight="normal" text-anchor="{anchor}" fill="#1f2937">'
        'Observed weekly mean chlorophyll (mg m'
        f'<tspan dy="-{size * 0.35:.1f}" font-size="{size - 3}">-3</tspan>'
        f'<tspan dy="{size * 0.35:.1f}" font-size="{size}">)</tspan>'
        "</text>"
    )


def _rotated_units_label_svg(x: float, y: float, *, size: int) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="Helvetica, Arial, sans-serif" '
        'fill="#1f2937" text-anchor="middle" '
        f'transform="rotate(-90 {x:.1f} {y:.1f})">'
        'Predicted weekly mean chlorophyll (mg m'
        f'<tspan dy="-{size * 0.35:.1f}" font-size="{size - 3}">-3</tspan>'
        f'<tspan dy="{size * 0.35:.1f}" font-size="{size}">)</tspan>'
        "</text>"
    )


def _line_svg(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{dash_attr} />'
    )


def _circle_svg(cx: float, cy: float, r: float, fill: str, opacity: float) -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" fill-opacity="{opacity:.3f}" />'


def _compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed"]
    predicted = frame["predicted"]
    err = observed - predicted
    mae = float(err.abs().mean())
    rmse = float((err.pow(2).mean()) ** 0.5)
    ss_res = float(err.pow(2).sum())
    ss_tot = float(((observed - observed.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _panel_svg(
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
    frame: pd.DataFrame,
    title: str,
    letter: str,
) -> str:
    margin_left = 72
    margin_right = 20
    margin_top = 44
    margin_bottom = 50
    plot_x0 = panel_x + margin_left
    plot_y0 = panel_y + margin_top
    plot_w = panel_w - margin_left - margin_right
    plot_h = panel_h - margin_top - margin_bottom

    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    vmin = 2.0
    vmax = 28.0

    def sx(v: float) -> float:
        return plot_x0 + (v - vmin) / (vmax - vmin) * plot_w

    def sy(v: float) -> float:
        return plot_y0 + plot_h - (v - vmin) / (vmax - vmin) * plot_h

    metrics = _compute_metrics(frame)
    parts = [
        f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" fill="#ffffff" stroke="#9ca3af" stroke-width="1.8" rx="10" ry="10" />',
        _text_svg(panel_x + 16, panel_y + 24, f"{letter})", size=18, weight="bold"),
        _text_svg(panel_x + 48, panel_y + 24, title, size=18, weight="bold"),
        _text_svg(
            panel_x + 84,
            panel_y + 60,
            "Matched-subset metrics:",
            size=14,
            weight="bold",
        ),
        _text_svg(
            panel_x + 84,
            panel_y + 79,
            f'R² = {metrics["r2"]:.3f}   RMSE = {metrics["rmse"]:.3f}   MAE = {metrics["mae"]:.3f}',
            size=15,
            weight="bold",
        ),
    ]

    ticks = [vmin + i * (vmax - vmin) / 4 for i in range(5)]
    for tick in ticks:
        x = sx(float(tick))
        y = sy(float(tick))
        parts.append(_line_svg(x, plot_y0, x, plot_y0 + plot_h, "#e5e7eb", 1.0))
        parts.append(_line_svg(plot_x0, y, plot_x0 + plot_w, y, "#e5e7eb", 1.0))
        parts.append(_text_svg(x, plot_y0 + plot_h + 24, f"{tick:.1f}", size=14, anchor="middle"))
        parts.append(_text_svg(plot_x0 - 12, y + 5, f"{tick:.1f}", size=14, anchor="end"))

    parts.append(_line_svg(sx(vmin), sy(vmin), sx(vmax), sy(vmax), "#374151", 1.4, "5 4"))

    for xo, yp in zip(observed, predicted):
        if pd.notna(xo) and pd.notna(yp):
            parts.append(_circle_svg(sx(float(xo)), sy(float(yp)), 5.0, "#0f766e", 0.42))

    parts.append(_units_label_svg(panel_x + panel_w / 2, panel_y + panel_h - 8, size=17, anchor="middle"))
    parts.append(_rotated_units_label_svg(panel_x + 20, panel_y + panel_h / 2, size=17))
    return "\n".join(parts)


def _paired_frames(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    mask = df["observed"].notna() & df["persistence_trailing7_mean"].notna()
    model = pd.DataFrame(
        {
            "observed": df.loc[mask, "observed"].reset_index(drop=True),
            "predicted": df.loc[mask, "model_prediction"].reset_index(drop=True),
        }
    )
    persistence = pd.DataFrame(
        {
            "observed": df.loc[mask, "observed"].reset_index(drop=True),
            "predicted": df.loc[mask, "persistence_trailing7_mean"].reset_index(drop=True),
        }
    )
    return model, persistence


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a 3x2 regression diagnostics figure.")
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    week1_model, week1_persist = _paired_frames(args.diagnostics_dir / "week1_holdout_predictions.csv")
    week2_model, week2_persist = _paired_frames(args.diagnostics_dir / "week2_holdout_predictions.csv")
    week3_model, week3_persist = _paired_frames(args.diagnostics_dir / "week3_holdout_predictions.csv")

    width = 1280
    height = 1620
    panel_w = 590
    panel_h = 470
    left_x = 36
    right_x = 654
    row_y = [70, 570, 1070]

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc" />',
    ]
    pieces.append(_panel_svg(left_x, row_y[0], panel_w, panel_h, week1_model, "Week 1 model vs observed", "a"))
    pieces.append(_panel_svg(right_x, row_y[0], panel_w, panel_h, week1_persist, "Week 1 trailing-7 persistence vs observed", "b"))
    pieces.append(_panel_svg(left_x, row_y[1], panel_w, panel_h, week2_model, "Week 2 model vs observed", "c"))
    pieces.append(_panel_svg(right_x, row_y[1], panel_w, panel_h, week2_persist, "Week 2 trailing-7 persistence vs observed", "d"))
    pieces.append(_panel_svg(left_x, row_y[2], panel_w, panel_h, week3_model, "Week 3 model vs observed", "e"))
    pieces.append(_panel_svg(right_x, row_y[2], panel_w, panel_h, week3_persist, "Week 3 trailing-7 persistence vs observed", "f"))
    pieces.append("</svg>")

    args.output.write_text("\n".join(pieces), encoding="utf-8")


if __name__ == "__main__":
    main()
