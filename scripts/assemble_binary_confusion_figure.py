from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble binary high-risk confusion matrix figure.")
    parser.add_argument(
        "--week1-report",
        default="/Users/a1amador/Documents/Playground/operational_models/week1/high_risk/week1_high_risk_training_report.json",
    )
    parser.add_argument(
        "--week2-report",
        default="/Users/a1amador/Documents/Playground/operational_models/week2/high_risk/horizon_2_high_risk_training_report.json",
    )
    parser.add_argument(
        "--week3-report",
        default="/Users/a1amador/Documents/Playground/operational_models/week3/high_risk/horizon_3_high_risk_training_report.json",
    )
    parser.add_argument(
        "--output-prefix",
        default="/Users/a1amador/Documents/Playground/diagnostics/classification/binary_confusion_matrices",
    )
    return parser.parse_args()


def load_metrics(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())["metrics"]


def row_normalize(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def make_soft_blue_cmap() -> LinearSegmentedColormap:
    colors = ["#ffffff", "#eef4fb", "#d7e3f3", "#b8cde8", "#7ca9d2", "#2f6fae", "#123d82"]
    return LinearSegmentedColormap.from_list("paper_blues", colors)


def draw_panel(ax_title, ax_mat, ax_footer, metrics: dict, title: str, letter: str, cmap) -> None:
    cm = np.array(metrics["confusion_matrix"], dtype=float)
    cm_norm = row_normalize(cm)

    ax_title.axis("off")
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.text(0.00, 0.88, f"{letter})", fontsize=16, fontweight="bold", ha="left", va="top")
    ax_title.text(0.08, 0.88, title, fontsize=16, fontweight="bold", ha="left", va="top")

    ax_mat.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax_mat.set_xticks([0, 1])
    ax_mat.set_yticks([0, 1])
    ax_mat.set_xticklabels(["High", "Not high"], fontsize=12, fontweight="bold")
    ax_mat.set_yticklabels(["High", "Not high"], fontsize=12, fontweight="bold")
    ax_mat.xaxis.tick_top()
    ax_mat.tick_params(length=0)
    ax_mat.text(
        0.5,
        1.10,
        "Predicted class",
        transform=ax_mat.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )
    ax_mat.set_ylabel("Actual class", fontsize=13, fontweight="bold")
    ax_mat.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax_mat.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax_mat.grid(which="minor", color="#6b7280", linewidth=1.0)
    ax_mat.tick_params(which="minor", bottom=False, left=False)

    for i in range(2):
        for j in range(2):
            val = cm_norm[i, j]
            txt_color = "white" if val >= 0.58 else "#111827"
            ax_mat.text(
                j,
                i - 0.05,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
                color=txt_color,
            )
            ax_mat.text(
                j,
                i + 0.22,
                f"n={int(cm[i, j])}",
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="bold",
                color=txt_color,
            )

    for spine in ax_mat.spines.values():
        spine.set_visible(False)

    ax_footer.axis("off")
    ax_footer.set_xlim(0, 1)
    ax_footer.set_ylim(0, 1)
    ax_footer.text(
        0.02,
        0.92,
        f"Balanced accuracy = {metrics['balanced_accuracy']:.3f}",
        fontsize=13.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax_footer.text(
        0.02,
        0.62,
        f"Precision = {metrics['high_precision']:.3f}",
        fontsize=12.5,
        ha="left",
        va="top",
    )
    ax_footer.text(
        0.02,
        0.38,
        f"Recall = {metrics['high_recall']:.3f}",
        fontsize=12.5,
        ha="left",
        va="top",
    )
    ax_footer.text(
        0.02,
        0.14,
        f"F1 = {metrics['high_f1']:.3f}",
        fontsize=12.5,
        ha="left",
        va="top",
    )

def main() -> None:
    args = parse_args()
    week1 = load_metrics(args.week1_report)
    week2 = load_metrics(args.week2_report)
    week3 = load_metrics(args.week3_report)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmap = make_soft_blue_cmap()

    fig = plt.figure(figsize=(15.5, 7.0), facecolor="white")
    outer = fig.add_gridspec(1, 3, wspace=0.36)

    panels = [
        (week1, "Week 1 binary high-risk alert", "a"),
        (week2, "Week 2 binary high-risk alert", "b"),
        (week3, "Week 3 binary high-risk alert", "c"),
    ]

    for i, (metrics, title, letter) in enumerate(panels):
        sg = outer[i].subgridspec(3, 1, height_ratios=[0.65, 3.0, 1.15], hspace=0.14)
        ax_title = fig.add_subplot(sg[0])
        ax_mat = fig.add_subplot(sg[1])
        ax_footer = fig.add_subplot(sg[2])
        draw_panel(ax_title, ax_mat, ax_footer, metrics, title, letter, cmap)

    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
