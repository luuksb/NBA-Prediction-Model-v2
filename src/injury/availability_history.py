"""availability_history.py — Compute historical playoff availability rates per player.

Anti-look-ahead: all rates are computed using only data from seasons strictly
before the target year.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_availability_rates(
    games_df: pd.DataFrame,
    player_col: str = "player_id",
    year_col: str = "year",
    games_played_col: str = "games_played",
    games_possible_col: str = "games_possible",
    target_year: int = 2025,
) -> pd.DataFrame:
    """Compute each player's historical playoff availability rate before target_year.

    Availability rate = sum(games_played) / sum(games_possible) across all
    playoff series in years strictly before target_year.

    Args:
        games_df: Player-series level DataFrame with games played and possible.
        player_col: Column identifying the player.
        year_col: Column identifying the season year.
        games_played_col: Games the player actually appeared in.
        games_possible_col: Total games in the series.
        target_year: Exclude data from this year and later.

    Returns:
        DataFrame with columns [player_col, 'availability_rate', 'n_series']
        — one row per player.
    """
    historical = games_df[games_df[year_col] < target_year].copy()

    rates = (
        historical.groupby(player_col, as_index=False)
        .agg(
            total_played=(games_played_col, "sum"),
            total_possible=(games_possible_col, "sum"),
            n_series=(year_col, "count"),
        )
    )

    rates["availability_rate"] = rates["total_played"] / rates["total_possible"].replace(0, pd.NA)
    rates["availability_rate"] = rates["availability_rate"].clip(0.0, 1.0)

    logger.info(
        "Computed availability rates for %d players (target_year=%d).",
        len(rates),
        target_year,
    )
    return rates[[player_col, "availability_rate", "n_series"]]
