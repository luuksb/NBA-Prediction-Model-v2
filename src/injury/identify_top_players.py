"""identify_top_players.py — Identify top-N players per team by configured ranking metric.

Reads ranking weights from configs/features.yaml (top_player_ranking section).
Anti-look-ahead: uses only data available before the target year.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

FEATURES_CONFIG = Path("configs/features.yaml")


def _load_ranking_config() -> dict:
    with open(FEATURES_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return cfg["top_player_ranking"]


def compute_composite_rating(
    players_df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Compute a weighted composite rating for each player row.

    Args:
        players_df: DataFrame with columns for each metric in weights.
        weights: Metric name → weight mapping (need not sum to 1).

    Returns:
        Series of composite ratings aligned to players_df index.
    """
    total_weight = sum(weights.values())
    rating = pd.Series(0.0, index=players_df.index)
    for metric, weight in weights.items():
        if metric in players_df.columns:
            rating += (weight / total_weight) * players_df[metric].fillna(0.0)
        else:
            logger.warning("Metric %r not found in players DataFrame — skipping.", metric)
    return rating


def identify_top_players(
    players_df: pd.DataFrame,
    team_col: str,
    year_col: str,
    target_year: int,
) -> pd.DataFrame:
    """Identify the top-N players per team for the given target year.

    Uses only player data from seasons strictly before target_year to avoid
    look-ahead bias.

    Args:
        players_df: Player-season level DataFrame. Must include team_col,
            year_col, and the metric columns referenced in features.yaml.
        team_col: Column name for team identifier.
        year_col: Column name for season year.
        target_year: The year for which we are predicting (no data from this
            year or later is used).

    Returns:
        DataFrame with top-N rows per team, ranked by composite rating,
        with an added 'composite_rating' column.
    """
    ranking_cfg = _load_ranking_config()
    n_players = ranking_cfg["n_players"]
    weights = ranking_cfg["weights"]

    historical = players_df[players_df[year_col] < target_year].copy()
    historical["composite_rating"] = compute_composite_rating(historical, weights)

    # Most recent season per player per team
    most_recent = (
        historical.sort_values(year_col, ascending=False)
        .groupby([team_col, "player_id"], as_index=False)
        .first()
    )

    top_players = (
        most_recent.sort_values("composite_rating", ascending=False)
        .groupby(team_col, as_index=False)
        .head(n_players)
    )

    logger.info(
        "Identified top-%d players for %d teams (target_year=%d).",
        n_players,
        top_players[team_col].nunique(),
        target_year,
    )
    return top_players
