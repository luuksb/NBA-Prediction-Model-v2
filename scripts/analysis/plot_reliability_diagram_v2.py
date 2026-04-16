"""plot_reliability_diagram_v2.py — Reliability diagram with uniform binning.

Uses equal-width probability bins so bar height directly shows how many series
fall in each probability range, making the histogram self-explanatory.
Saves results/reliability_diagram_v2.png (v1 with quantile bins is retained).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "final" / "series_dataset.parquet"
MODEL_DIR = ROOT / "results" / "model_selection"
OUT_PATH = ROOT / "results" / "reliability_diagram_v2.png"

WINDOWS = [
    ("full",   "Full (1980–2024, n=659)",   "#4fc3f7"),
    ("modern", "Modern (2000–2024, n=375)", "#ffb74d"),
    ("recent", "Recent (2014–2024, n=165)", "#81c784"),
]

WINDOW_YEARS = {
    "full":   (1980, 2024),
    "modern": (2000, 2024),
    "recent": (2014, 2024),
}

BG_COLOR     = "#0a1929"
PANEL_BG     = "#0d2137"
TEXT_COLOR   = "#e8eaf6"
GRID_COLOR   = "#1e3a5f"
DASHED_COLOR = "#90caf9"

N_BINS = 10
TARGET = "higher_seed_wins"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(window: str) -> dict:
    path = MODEL_DIR / f"chosen_model_{window}.json"
    with open(path) as f:
        return json.load(f)


def predict_proba(spec: dict, df: pd.DataFrame) -> np.ndarray:
    logit = spec["intercept"] + sum(
        coef * df[feat].values
        for feat, coef in spec["coefficients"].items()
    )
    return 1.0 / (1.0 + np.exp(-logit))


def filter_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    lo, hi = WINDOW_YEARS[window]
    return df[(df["year"] >= lo) & (df["year"] <= hi)].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=BG_COLOR)
    fig.suptitle(
        "Reliability Diagram — Model Calibration by Training Window",
        color=TEXT_COLOR,
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for ax, (window, title, color) in zip(axes, WINDOWS):
        spec = load_model(window)
        sub = filter_window(df, window).dropna(subset=spec["features"] + [TARGET])
        y_true = sub[TARGET].astype(int).values
        y_pred = predict_proba(spec, sub)

        # Calibration curve — uniform bins so x-axis is evenly spaced
        frac_pos, mean_pred = calibration_curve(
            y_true, y_pred, n_bins=N_BINS, strategy="uniform"
        )

        # Histogram with the same uniform edges
        bin_edges = np.linspace(0, 1, N_BINS + 1)
        bin_counts, _ = np.histogram(y_pred, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_width = (bin_edges[1] - bin_edges[0]) * 0.85

        # --- Axes styling ---
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        ax.tick_params(colors=TEXT_COLOR, which="both")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.5)

        # Shade under/over-confident regions
        ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.08, color="#ef5350", zorder=0)
        ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.08, color="#5c6bc0", zorder=0)

        ax.text(0.06, 0.88, "Underconfident", color="#ef9a9a", fontsize=8,
                style="italic", transform=ax.transAxes)
        ax.text(0.45, 0.15, "Overconfident", color="#9fa8da", fontsize=8,
                style="italic", transform=ax.transAxes)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle="--", color=DASHED_COLOR,
                linewidth=1.4, label="Perfect calibration", zorder=2)

        # Model calibration line
        ax.plot(mean_pred, frac_pos, marker="o", color=color, linewidth=2,
                markersize=6, label="Model", zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted P(high seed wins)", color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel("Actual win rate", color=TEXT_COLOR, fontsize=9)
        ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight="bold")

        ax.legend(loc="upper left", fontsize=8,
                  facecolor="#0d2a40", edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR)

        # Secondary axis: bar heights = actual series count per 0.1-wide bin
        ax2 = ax.twinx()
        ax2.set_facecolor(PANEL_BG)
        ax2.bar(bin_centers, bin_counts, width=bar_width,
                color=color, alpha=0.22, zorder=1)
        # Scale so tallest bar occupies the bottom 30% of the chart
        ax2.set_ylim(0, bin_counts.max() / 0.30)
        ax2.tick_params(colors=TEXT_COLOR)
        ax2.set_ylabel("Series count", color=TEXT_COLOR, fontsize=9)
        for spine in ax2.spines.values():
            spine.set_edgecolor(GRID_COLOR)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
