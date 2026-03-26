"""benchmark.py — Compare new model leaderboard against a previous run.

Loads the previous leaderboard from results/model_selection/ and produces a
side-by-side comparison that flags improvements and regressions per metric.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/model_selection")
LEADERBOARD_FILE = RESULTS_DIR / "leaderboard.parquet"


def load_previous_leaderboard() -> pd.DataFrame | None:
    """Load the previous leaderboard from disk, if it exists.

    Returns:
        Previous leaderboard DataFrame, or None if no previous run exists.
    """
    if not LEADERBOARD_FILE.exists():
        logger.info("No previous leaderboard found at %s.", LEADERBOARD_FILE)
        return None
    return pd.read_parquet(LEADERBOARD_FILE)


def save_leaderboard(leaderboard: pd.DataFrame) -> None:
    """Persist the current leaderboard to disk.

    Args:
        leaderboard: Leaderboard DataFrame from evaluate.build_leaderboard().
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard.to_parquet(LEADERBOARD_FILE, index=False)
    logger.info("Leaderboard saved to %s", LEADERBOARD_FILE)


def compare_leaderboards(
    new: pd.DataFrame,
    previous: pd.DataFrame | None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Side-by-side comparison of top-N models across runs.

    Args:
        new: Current leaderboard.
        previous: Previous leaderboard (may be None for the first run).
        top_n: How many top models to compare.

    Returns:
        DataFrame with new and previous ranks/scores for each model,
        and a delta_composite_score column.
    """
    metrics = ["mcfadden_r2", "brier_score", "auc_roc", "log_loss", "composite_score"]

    new_top = new.head(top_n).copy()
    new_top["rank_new"] = range(1, len(new_top) + 1)
    new_top["features_key"] = new_top["features"].apply(lambda f: tuple(sorted(f)))

    if previous is None:
        new_top["rank_prev"] = None
        new_top["composite_score_prev"] = None
        new_top["delta_composite_score"] = None
        return new_top

    prev_top = previous.head(top_n).copy()
    prev_top["rank_prev"] = range(1, len(prev_top) + 1)
    prev_top["features_key"] = prev_top["features"].apply(lambda f: tuple(sorted(f)))
    prev_top = prev_top.rename(columns={"composite_score": "composite_score_prev"})

    merged = new_top.merge(
        prev_top[["features_key", "window", "rank_prev", "composite_score_prev"]],
        on=["features_key", "window"],
        how="left",
    )
    merged["delta_composite_score"] = merged["composite_score"] - merged["composite_score_prev"]
    return merged
