"""run_bracket.py — Run 50,000 full bracket simulations for a given year."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from src.simulation.bracket import Bracket, Series, advance_bracket, build_bracket
from src.simulation.simulate_series import simulate_series

logger = logging.getLogger(__name__)

N_SIMULATIONS = 50_000


def run_simulations(
    year: int,
    east_seeds: list[str],
    west_seeds: list[str],
    team_features: pd.DataFrame,
    n_sims: int = N_SIMULATIONS,
    injury_draws: dict[str, np.ndarray] | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Run N full bracket simulations and return per-iteration outcomes.

    Args:
        year: Season year being simulated.
        east_seeds: 8 Eastern Conference team IDs, ordered 1st to 8th.
        west_seeds: 8 Western Conference team IDs, ordered 1st to 8th.
        team_features: DataFrame indexed by team ID with feature columns.
        n_sims: Number of Monte Carlo iterations.
        injury_draws: Optional dict from team_id → availability array (length >= n_sims).
        seed: Optional random seed for reproducibility.

    Returns:
        List of dicts, one per simulation, with keys:
        iteration, champion, finalist_east, finalist_west, round_exits (dict).
    """
    rng = np.random.default_rng(seed)
    outcomes: list[dict] = []

    for i in range(n_sims):
        bracket = build_bracket(year, list(east_seeds), list(west_seeds))
        round_exits: dict[str, int] = {}

        while True:
            current_round = bracket.rounds[-1]
            for series in current_round:
                winner = simulate_series(
                    series.high_seed,
                    series.low_seed,
                    team_features,
                    rng,
                    injury_draws=injury_draws,
                    draw_index=i,
                )
                series.winner = winner
                loser = series.low_seed if winner == series.high_seed else series.high_seed
                round_exits[loser] = series.round_num

            result = advance_bracket(bracket)
            if result is None:
                champion = bracket.rounds[-1][0].winner
                break

        # Conference finalists are the winners of round 3
        finals_series = bracket.rounds[-1][0] if bracket.rounds[-1][0].round_num == 4 else None
        outcomes.append({
            "iteration": i,
            "champion": champion,
            "finalist_east": finals_series.high_seed if finals_series else None,
            "finalist_west": finals_series.low_seed if finals_series else None,
            "round_exits": round_exits,
        })

        if (i + 1) % 10_000 == 0:
            logger.info("Completed %d / %d simulations…", i + 1, n_sims)

    return outcomes
