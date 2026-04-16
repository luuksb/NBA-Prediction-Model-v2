#!/usr/bin/env python3
"""injury_scenarios.py — Unlikely injury scenarios and their championship impact.

Scans the pre-drawn injury array for tail scenarios (e.g. all 3 star players
injured in Round 1) and cross-references with the bracket simulation iterations
to show how championship outcomes shift under extreme injury draws.

Usage:
    python scripts/analysis/injury_scenarios.py [--window WINDOW] [--year YEAR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

INJURY_SIM_DIR = Path("results/injury_sims")
SIM_DIR = Path("results/simulations")
BRACKET_SEEDS_CONFIG = Path("configs/bracket_seeds.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unlikely injury scenario analysis.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--window", default="full")
    return parser.parse_args()


def load_inputs(year: int, window: str) -> tuple[np.ndarray, list, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Load injury draws, metadata, and simulation iterations.

    Args:
        year: Season year.
        window: Training window name.

    Returns:
        Tuple of (draws, teams, mean_rates, player_bpm, iterations_df, playoff_teams).
    """
    draws = np.load(INJURY_SIM_DIR / f"injury_draws_{year}.npy")
    with open(INJURY_SIM_DIR / f"injury_meta_{year}.json") as f:
        meta = json.load(f)

    itr = pd.read_parquet(SIM_DIR / f"{year}_{window}" / "iterations.parquet")

    with open(BRACKET_SEEDS_CONFIG) as f:
        seeds = yaml.safe_load(f)
    year_seeds = seeds["bracket_seeds"][year]
    playoff_teams = year_seeds["east"] + year_seeds["west"]

    return (
        draws,
        meta["teams"],
        np.array(meta["mean_rates"]),
        np.array(meta["player_bpm"]),
        itr,
        playoff_teams,
    )


def _in_finals(itr: pd.DataFrame, team: str) -> pd.Series:
    """Boolean mask: iterations where team reached the Finals."""
    return (itr["finalist_east"] == team) | (itr["finalist_west"] == team)


def analyze_all_stars_injured_r1(
    injured: np.ndarray,
    teams: list[str],
    player_bpm: np.ndarray,
    itr: pd.DataFrame,
    playoff_teams: list[str],
) -> None:
    """Print outcomes for simulations where a team had all 3 stars injured in R1.

    Args:
        injured: Boolean array of shape (n_teams, n_stars, n_rounds, n_sims).
        teams: List of team IDs matching axis 0 of injured.
        player_bpm: Array of shape (n_teams, n_stars) with raw BPM values.
        itr: Iterations DataFrame.
        playoff_teams: List of playoff team IDs to consider.
    """
    n_sims = injured.shape[3]
    print("\n=== All-3-stars injured in Round 1 ===")
    print(f"{'Team':<6}  {'Sims':>6}  {'Freq':>7}  {'Own titles':>10}  {'Own win%':>9}  Top beneficiaries")
    print("-" * 75)
    for team in playoff_teams:
        if team not in teams:
            continue
        t = teams.index(team)
        mask = np.all(injured[t, :, 0, :], axis=0)
        n = int(mask.sum())
        if n == 0:
            continue
        own_wins = int((itr.loc[mask, "champion"] == team).sum())
        champ_counts = itr.loc[mask, "champion"].value_counts().head(3)
        beneficiaries = ", ".join(
            f"{tm}({cnt})" for tm, cnt in champ_counts.items() if tm != team
        )[:35]
        print(
            f"{team:<6}  {n:>6,}  {n/n_sims:>6.2%}  "
            f"{own_wins:>10,}  {own_wins/n:>8.1%}  {beneficiaries}"
        )


def analyze_top_star_injury_impact(
    injured: np.ndarray,
    teams: list[str],
    player_bpm: np.ndarray,
    mean_rates: np.ndarray,
    itr: pd.DataFrame,
    playoff_teams: list[str],
) -> None:
    """Print per-team title rate when top star is vs isn't injured in Round 1.

    Args:
        injured: Boolean array of shape (n_teams, n_stars, n_rounds, n_sims).
        teams: List of team IDs matching axis 0 of injured.
        player_bpm: Array of shape (n_teams, n_stars) with raw BPM values.
        mean_rates: Array of shape (n_teams, n_stars) with mean availability rates.
        itr: Iterations DataFrame.
        playoff_teams: List of playoff team IDs to consider.
    """
    n_sims = injured.shape[3]
    print("\n=== Top-star (star0) injury impact on title probability — Round 1 ===")
    print(f"{'Team':<6}  {'BPM':>5}  {'Avail':>6}  {'Injured R1':>10}  {'P(title|healthy)':>17}  {'P(title|injured)':>17}  {'Drop':>7}")
    print("-" * 78)
    for team in playoff_teams:
        if team not in teams:
            continue
        t = teams.index(team)
        bpm0 = player_bpm[t][0]
        mr0 = mean_rates[t][0]
        star0_inj = injured[t, 0, 0, :]
        n_inj = int(star0_inj.sum())
        n_healthy = int((~star0_inj).sum())
        if n_inj == 0:
            continue
        p_title_healthy = (itr.loc[~star0_inj, "champion"] == team).mean()
        p_title_injured = (itr.loc[star0_inj, "champion"] == team).mean()
        drop = p_title_healthy - p_title_injured
        if p_title_healthy < 0.005 and p_title_injured < 0.005:
            continue
        print(
            f"{team:<6}  {bpm0:>5.1f}  {mr0:>5.0%}  "
            f"{n_inj:>10,}  {p_title_healthy:>16.1%}  {p_title_injured:>16.1%}  {drop:>+6.1%}"
        )


def main() -> None:
    args = parse_args()
    draws, teams, mean_rates, player_bpm, itr, playoff_teams = load_inputs(
        args.year, args.window
    )

    # injured[team, star, round, sim] = True if that player is injured in that sim/round
    injured = draws > mean_rates[:, :, np.newaxis, np.newaxis]

    n_sims = draws.shape[3]
    logger.info(
        "Loaded %d teams × %d stars × %d rounds × %d sims",
        *draws.shape,
    )

    analyze_all_stars_injured_r1(injured, teams, player_bpm, itr, playoff_teams)
    analyze_top_star_injury_impact(injured, teams, player_bpm, mean_rates, itr, playoff_teams)


if __name__ == "__main__":
    main()
