"""simulate.py — Monte Carlo simulation of player availability for out-of-sample years.

Draws series-level availability percentages directly from a Beta distribution
parameterised by each player's historical rate. Aggregates to a team-level
weighted availability percentage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

FEATURES_CONFIG = Path("configs/features.yaml")

# Beta distribution concentration parameter — higher = tighter around historical rate.
# Can be overridden per call.
DEFAULT_CONCENTRATION = 20.0


def _beta_params(mean: float, concentration: float) -> tuple[float, float]:
    """Convert a mean and concentration to Beta α, β parameters.

    Args:
        mean: Historical availability rate in (0, 1).
        concentration: Pseudo-sample size; higher values → tighter distribution.

    Returns:
        Tuple (alpha, beta) for scipy/numpy Beta distribution.
    """
    mean = float(np.clip(mean, 0.01, 0.99))
    alpha = mean * concentration
    beta = (1.0 - mean) * concentration
    return alpha, beta


def simulate_player_availability(
    player_id: str,
    historical_rate: float,
    n_draws: int = 1000,
    concentration: float = DEFAULT_CONCENTRATION,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw N simulated availability percentages for a single player.

    Args:
        player_id: Player identifier (used only for logging).
        historical_rate: Historical availability rate in [0, 1].
        n_draws: Number of Monte Carlo draws.
        concentration: Beta distribution concentration.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        1-D array of shape (n_draws,) with simulated rates in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha, beta = _beta_params(historical_rate, concentration)
    draws = rng.beta(alpha, beta, size=n_draws)
    logger.debug("Player %s: rate=%.3f  α=%.2f  β=%.2f", player_id, historical_rate, alpha, beta)
    return draws


def simulate_team_availability(
    top_players: pd.DataFrame,
    availability_rates: pd.DataFrame,
    player_col: str = "player_id",
    rating_col: str = "composite_rating",
    n_draws: int = 1000,
    concentration: float = DEFAULT_CONCENTRATION,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate weighted team availability percentage over N draws.

    Weights each player's availability by their composite rating (normalised).

    Args:
        top_players: DataFrame with top-N players for the team (from identify_top_players).
        availability_rates: DataFrame from availability_history.compute_availability_rates().
        player_col: Column identifying players in both DataFrames.
        rating_col: Column in top_players for weighting.
        n_draws: Number of Monte Carlo draws.
        concentration: Beta distribution concentration.
        rng: Optional numpy random generator.

    Returns:
        1-D array of shape (n_draws,) with team availability percentages.
    """
    if rng is None:
        rng = np.random.default_rng()

    merged = top_players.merge(availability_rates, on=player_col, how="left")
    merged["availability_rate"] = merged["availability_rate"].fillna(
        0.85
    )  # league average fallback
    weights = merged[rating_col].clip(lower=0.0)
    total_weight = weights.sum()
    if total_weight == 0:
        weights = pd.Series(1.0 / len(merged), index=merged.index)
    else:
        weights = weights / total_weight

    team_draws = np.zeros(n_draws)
    for _, player_row in merged.iterrows():
        player_draws = simulate_player_availability(
            player_row[player_col],
            player_row["availability_rate"],
            n_draws=n_draws,
            concentration=concentration,
            rng=rng,
        )
        team_draws += weights[player_row.name] * player_draws

    return team_draws
