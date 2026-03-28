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

    Each metric is z-score normalised across all rows before weighting so that
    metrics on different scales (BPM, USG%, MPG) contribute equally when their
    weights are equal.

    Args:
        players_df: DataFrame with columns for each metric in weights.
        weights: Metric name → weight mapping (need not sum to 1).

    Returns:
        Series of composite ratings aligned to players_df index.
    """
    total_weight = sum(weights.values())
    rating = pd.Series(0.0, index=players_df.index)
    for metric, weight in weights.items():
        if metric not in players_df.columns:
            logger.warning("Metric %r not found in players DataFrame — skipping.", metric)
            continue
        col = pd.to_numeric(players_df[metric], errors="coerce").fillna(0.0)
        std = col.std()
        z = (col - col.mean()) / std if std > 0 else pd.Series(0.0, index=players_df.index)
        rating += (weight / total_weight) * z
    return rating


def identify_top_players(
    players_df: pd.DataFrame,
    team_col: str,
    year_col: str,
    target_year: int,
    min_games: int = 15,
    games_col: str = "g",
    min_mpg: float = 20.0,
    mp_col: str = "mp",
) -> pd.DataFrame:
    """Identify the top-N players per team for the given target year.

    Uses only the season immediately preceding target_year (season == target_year - 1)
    so that the roster reflects the current team, not all-time players who once
    wore the uniform.  Minimum-games and minutes-per-game filters remove
    small-sample BPM outliers.

    Args:
        players_df: Player-season level DataFrame. Must include team_col,
            year_col, and the metric columns referenced in features.yaml.
        team_col: Column name for team identifier.
        year_col: Column name for season year.
        target_year: The year for which we are predicting.  Only data from
            season == target_year is used (the regular season that concludes
            before the target-year playoffs begin).
        min_games: Minimum regular-season games required to be eligible
            (default 15).
        games_col: Column in players_df containing games played (default "g").
        min_mpg: Minimum average minutes per game required to be eligible
            (default 20.0).
        mp_col: Column in players_df containing total minutes played (default "mp").

    Returns:
        DataFrame with top-N rows per team, ranked by composite rating,
        with an added 'composite_rating' column.
    """
    ranking_cfg = _load_ranking_config()
    n_players = ranking_cfg["n_players"]
    weights = ranking_cfg["weights"]

    season = target_year
    season_df = players_df[players_df[year_col] == season].copy()

    if games_col in season_df.columns:
        season_df = season_df[season_df[games_col] >= min_games]

    if mp_col in season_df.columns and games_col in season_df.columns:
        season_df = season_df[
            (season_df[mp_col] / season_df[games_col].replace(0, pd.NA)) >= min_mpg
        ]

    season_df["composite_rating"] = compute_composite_rating(season_df, weights)

    top_players = (
        season_df.sort_values("composite_rating", ascending=False)
        .groupby(team_col, as_index=False)
        .head(n_players)
    )

    logger.info(
        "Identified top-%d players for %d teams (season=%d, target_year=%d).",
        n_players,
        top_players[team_col].nunique(),
        season,
        target_year,
    )
    return top_players
