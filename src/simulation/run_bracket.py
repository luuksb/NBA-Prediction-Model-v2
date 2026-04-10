"""run_bracket.py — Run 50,000 full bracket simulations for a given year."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from src.simulation.bracket import Bracket, Series, advance_bracket, build_bracket
from src.simulation.simulate_series import load_spec, simulate_series

logger = logging.getLogger(__name__)

N_SIMULATIONS = 50_000


def run_simulations(
    year: int,
    east_seeds: list[str],
    west_seeds: list[str],
    team_features: pd.DataFrame,
    window: str = "full",
    n_sims: int = N_SIMULATIONS,
    injury_draws: dict | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Run N full bracket simulations and return per-iteration outcomes.

    Args:
        year: Season year being simulated.
        east_seeds: 8 Eastern Conference team IDs, ordered 1st to 8th.
        west_seeds: 8 Western Conference team IDs, ordered 1st to 8th.
        team_features: DataFrame indexed by team ID with feature columns.
        window: Training window name ('full', 'modern', 'recent').
        n_sims: Number of Monte Carlo iterations.
        injury_draws: Optional dict produced by load_injury_draws() with keys:
            'draws' (ndarray shape n_teams × n_stars × n_rounds × n_sims),
            'team_index' (dict team_id → int), 'player_bpm' (n_teams × n_stars),
            'mean_rates' (n_teams × n_stars).
        seed: Optional random seed for reproducibility.

    Returns:
        List of dicts, one per simulation, with keys:
        iteration, champion, finalist_east, finalist_west, round_exits (dict).
    """
    spec = load_spec(window)
    rng = np.random.default_rng(seed)
    outcomes: list[dict] = []

    # Home-court advantage applies in all normal playoff years; 2020 was a
    # neutral-site bubble. This delta is constant within a given simulated year.
    _NEUTRAL_SITE_YEARS: frozenset[int] = frozenset({2020})
    series_deltas: dict[str, float] = {
        "delta_home_court_advantage": 0.0 if year in _NEUTRAL_SITE_YEARS else 1.0,
    }

    # Extract regular-season wins for cross-conference tiebreaking in Finals.
    # Both conference champions have within-conference seed rank 1; the NBA
    # awards home-court to the team with more regular-season wins.
    # Falls back to n_rtg (×1000) if w is unavailable (rebuild team_season_features
    # with delta_w active to get the exact win count).
    all_teams = list(east_seeds) + list(west_seeds)
    team_wins: dict[str, int] = {}
    tiebreak_col = next((c for c in ("w", "n_rtg") if c in team_features.columns), None)
    if tiebreak_col is not None:
        scale = 1 if tiebreak_col == "w" else 1000
        for team in all_teams:
            if team in team_features.index:
                val = team_features.loc[team, tiebreak_col]
                team_wins[team] = int(round(float(val) * scale))

    for i in range(n_sims):
        bracket = build_bracket(year, list(east_seeds), list(west_seeds), team_wins=team_wins)
        round_exits: dict[str, int] = {}

        while True:
            current_round = bracket.rounds[-1]
            for series in current_round:
                winner = simulate_series(
                    series.high_seed,
                    series.low_seed,
                    team_features,
                    rng,
                    spec=spec,
                    injury_draws=injury_draws,
                    draw_index=i,
                    round_num=series.round_num,
                    series_deltas=series_deltas,
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

        # Count Finals-round injuries for each finalist (if injury draws available)
        finalist_east_injuries: int | None = None
        finalist_west_injuries: int | None = None
        if injury_draws and finals_series:
            draws_arr: np.ndarray = injury_draws["draws"]
            team_index: dict[str, int] = injury_draws["team_index"]
            mean_rates_meta: list[list[float]] = injury_draws["mean_rates"]
            r4 = min(3, draws_arr.shape[2] - 1)  # round 4, 0-indexed
            n_stars = draws_arr.shape[1]
            for team_id, slot in (
                (finals_series.high_seed, "east"),
                (finals_series.low_seed, "west"),
            ):
                if team_id in team_index:
                    t = team_index[team_id]
                    n_injured = sum(
                        1
                        for star_i in range(n_stars)
                        if draws_arr[t, star_i, r4, i] > mean_rates_meta[t][star_i]
                    )
                    if slot == "east":
                        finalist_east_injuries = n_injured
                    else:
                        finalist_west_injuries = n_injured

        outcomes.append(
            {
                "iteration": i,
                "champion": champion,
                "finalist_east": finals_series.high_seed if finals_series else None,
                "finalist_west": finals_series.low_seed if finals_series else None,
                "finalist_east_injuries": finalist_east_injuries,
                "finalist_west_injuries": finalist_west_injuries,
                "round_exits": round_exits,
            }
        )

        if (i + 1) % 10_000 == 0:
            logger.info("Completed %d / %d simulations…", i + 1, n_sims)

    return outcomes
