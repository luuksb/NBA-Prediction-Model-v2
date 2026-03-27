#!/usr/bin/env python3
"""run_model_selection.py — CLI entry point for Module 2: Model Selection.

Fits logit models on all candidate feature sets × training windows, builds
a ranked leaderboard, compares against the previous run, and (optionally)
locks in the chosen model.

Usage:
    python scripts/run_model_selection.py [--select-top] [--dataset PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path("data/final/series_dataset.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NBA playoff model selection.")
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help=f"Path to series_dataset.parquet (default: {DATASET_PATH}).",
    )
    parser.add_argument(
        "--select-top",
        action="store_true",
        help="Automatically lock in the top-ranked model as chosen_model.json.",
    )
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    from src.model import select

    args = parse_args()
    logger.info("=== Model Selection START ===")

    df = pd.read_parquet(args.dataset)
    logger.info("Loaded dataset: %d rows, %d cols", len(df), len(df.columns))

    leaderboards = select.run_combinatorial_pipeline(df)

    if args.select_top:
        # Flatten leaderboards to find the single best model across all windows/metrics
        import yaml
        from src.model import fit

        all_dfs = [lb for metrics in leaderboards.values() for lb in metrics.values()]
        top_row = max(all_dfs, key=lambda d: d.iloc[0]["mcfadden_r2"]).iloc[0]
        features = top_row["features"]
        window_name = top_row["window"]
        with open(Path("configs/training_windows.yaml")) as f:
            windows_cfg = yaml.safe_load(f)
        window = next(w for w in windows_cfg["windows"] if w["name"] == window_name)
        full_spec = fit.fit_logit(df, features, window["name"],
                                  window["start_year"], window["end_year"])
        path = select.save_chosen_model(full_spec)
        logger.info("Chosen model saved to %s", path)

    logger.info("=== Model Selection DONE ===")


if __name__ == "__main__":
    main()
