#!/usr/bin/env python3
"""injury_finals_impact.py — Finals win rates with and without top star player.

For each specified team, finds all Finals appearances in the simulation and
splits them by whether the team's top star (star0) was injured in that round.
Reports win rates under both conditions and the resulting drop.

Usage:
    python scripts/analysis/injury_finals_impact.py [--window WINDOW] [--year YEAR]
    python scripts/analysis/injury_finals_impact.py --teams OKC DEN DET
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

INJURY_SIM_DIR = Path("results/injury_sims")
SIM_DIR = Path("results/simulations")

DEFAULT_TEAMS = ["OKC", "DEN", "DET"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finals win rates with top star injured.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--window", default="full")
    parser.add_argument(
        "--teams",
        nargs="+",
        default=DEFAULT_TEAMS,
        help="Team abbreviations to analyse (default: OKC DEN DET).",
    )
    return parser.parse_args()


def load_inputs(year: int, window: str) -> tuple[np.ndarray, list, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load injury draws, metadata, and simulation iterations.

    Args:
        year: Season year.
        window: Training window name.

    Returns:
        Tuple of (draws, teams, mean_rates, player_bpm, iterations_df).
    """
    draws = np.load(INJURY_SIM_DIR / f"injury_draws_{year}.npy")
    with open(INJURY_SIM_DIR / f"injury_meta_{year}.json") as f:
        meta = json.load(f)
    itr = pd.read_parquet(SIM_DIR / f"{year}_{window}" / "iterations.parquet")
    return (
        draws,
        meta["teams"],
        np.array(meta["mean_rates"]),
        np.array(meta["player_bpm"]),
        itr,
    )


def analyze_finals_star_injury(
    team: str,
    draws: np.ndarray,
    teams: list[str],
    mean_rates: np.ndarray,
    player_bpm: np.ndarray,
    itr: pd.DataFrame,
) -> dict | None:
    """Compute Finals win rates when the team's top star is healthy vs injured.

    A player is considered injured in a given simulation if their pre-drawn
    uniform value exceeds their mean availability rate (draw > mean_rate).
    The Finals correspond to round index 3 (0-indexed).

    Args:
        team: Team abbreviation.
        draws: Array of shape (n_teams, n_stars, n_rounds, n_sims).
        teams: List of team IDs matching axis 0 of draws.
        mean_rates: Array of shape (n_teams, n_stars) with mean availability rates.
        player_bpm: Array of shape (n_teams, n_stars) with raw BPM values.
        itr: Iterations DataFrame with columns: champion, finalist_east, finalist_west.

    Returns:
        Dict with analysis results, or None if the team is not in the draws.
    """
    if team not in teams:
        logger.warning("Team %s not found in injury draws.", team)
        return None

    t = teams.index(team)
    bpm0 = float(player_bpm[t][0])
    mr0 = float(mean_rates[t][0])

    # Star0 injured in Finals (round index 3)
    star0_injured = draws[t, 0, 3, :] > mr0

    # Finals appearances (team appears in either finalist slot)
    in_finals = (itr["finalist_east"] == team) | (itr["finalist_west"] == team)
    won = itr["champion"] == team

    in_finals_injured = in_finals & star0_injured
    in_finals_healthy = in_finals & ~star0_injured

    n_finals = int(in_finals.sum())
    n_fi = int(in_finals_injured.sum())
    n_fh = int(in_finals_healthy.sum())

    if n_finals == 0:
        logger.warning("Team %s never reached the Finals in these simulations.", team)
        return None

    n_wi = int((in_finals_injured & won).sum())
    n_wh = int((in_finals_healthy & won).sum())

    return {
        "team": team,
        "star0_bpm": bpm0,
        "star0_avail": mr0,
        "n_finals": n_finals,
        "n_finals_injured": n_fi,
        "pct_finals_injured": n_fi / n_finals,
        "win_rate_healthy": n_wh / n_fh if n_fh > 0 else float("nan"),
        "win_rate_injured": n_wi / n_fi if n_fi > 0 else float("nan"),
        "win_rate_drop": (n_wh / n_fh - n_wi / n_fi) if (n_fh > 0 and n_fi > 0) else float("nan"),
    }


def main() -> None:
    args = parse_args()
    draws, teams, mean_rates, player_bpm, itr = load_inputs(args.year, args.window)

    print(f"\nFinals win rates with top star injured — {args.year} ({args.window} window)")
    print(
        f"{'Team':<6}  {'BPM':>5}  {'Avail':>6}  {'Finals':>7}  "
        f"{'Finals w/ star injured':>22}  {'Win% (healthy)':>15}  {'Win% (injured)':>15}  {'Drop':>7}"
    )
    print("-" * 92)

    for team in args.teams:
        result = analyze_finals_star_injury(team, draws, teams, mean_rates, player_bpm, itr)
        if result is None:
            continue
        print(
            f"{result['team']:<6}  {result['star0_bpm']:>5.1f}  {result['star0_avail']:>5.0%}  "
            f"{result['n_finals']:>7,}  "
            f"{result['n_finals_injured']:>12,} ({result['pct_finals_injured']:>5.1%})  "
            f"{result['win_rate_healthy']:>14.1%}  "
            f"{result['win_rate_injured']:>14.1%}  "
            f"{result['win_rate_drop']:>+6.1%}"
        )


if __name__ == "__main__":
    main()
