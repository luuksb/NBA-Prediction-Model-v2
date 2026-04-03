#!/usr/bin/env python3
"""run_bracket_sim.py — CLI entry point for Module 4: Bracket Simulation.

Runs Monte Carlo bracket iterations for a given year and saves results.

Usage:
    python scripts/run_bracket_sim.py --year 2026 --window modern [--n-sims 50000]
    python scripts/run_bracket_sim.py --year 2024 --window full --no-injury
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
FINAL_DIR = Path("data/final")
INJURY_SIM_DIR = Path("results/injury_sims")
BRACKET_SEEDS_CONFIG = Path("configs/bracket_seeds.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NBA playoff bracket simulation.")
    parser.add_argument("--year", type=int, required=True, help="Season year to simulate.")
    parser.add_argument("--window", required=True, help="Training window name (e.g. 'modern').")
    parser.add_argument("--n-sims", type=int, default=50_000, help="Number of Monte Carlo iterations.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--no-injury", action="store_true", help="Disable injury adjustment.")
    return parser.parse_args()


def load_team_features(year: int) -> pd.DataFrame:
    """Load per-team raw feature values for the given year.

    Reads data/final/team_season_features.parquet and filters to the target year.

    Args:
        year: Season year.

    Returns:
        DataFrame indexed by team abbreviation with all active raw feature columns.

    Raises:
        ValueError: If no rows found for the given year.
    """
    path = FINAL_DIR / "team_season_features.parquet"
    df = pd.read_parquet(path)
    year_df = df[df["year"] == year]
    if year_df.empty:
        raise ValueError(
            f"No team features found for year {year} in {path}. "
            "Re-run the data pipeline or check the year."
        )
    return year_df.set_index("team").drop(columns=["year"], errors="ignore")


def load_bracket_seeds(year: int) -> tuple[list[str], list[str]]:
    """Return east and west seed lists (ordered 1–8) for the given year.

    For historical years with a playoff_series CSV, seeds are inferred from
    Round 1 matchups. For years listed in configs/bracket_seeds.yaml, seeds
    are read from that config file.

    Args:
        year: Season year.

    Returns:
        Tuple of (east_seeds, west_seeds), each a list of 8 team abbreviations
        ordered from 1st seed to 8th seed.

    Raises:
        FileNotFoundError: If neither source provides seeds for the year.
    """
    # Try bracket_seeds.yaml first (takes precedence for manually entered years)
    if BRACKET_SEEDS_CONFIG.exists():
        with open(BRACKET_SEEDS_CONFIG) as f:
            seeds_config = yaml.safe_load(f)
        year_seeds = (seeds_config.get("bracket_seeds") or {}).get(year)
        if year_seeds:
            east = year_seeds["east"]
            west = year_seeds["west"]
            if len(east) != 8 or len(west) != 8:
                raise ValueError(
                    f"bracket_seeds.yaml year {year}: expected 8 seeds per conference, "
                    f"got East={len(east)}, West={len(west)}."
                )
            logger.info("Loaded bracket seeds for %d from %s", year, BRACKET_SEEDS_CONFIG)
            return east, west

    # Fall back to playoff_series CSV for historical years
    csv_path = RAW_DIR / "playoff_series" / f"{year}_nba_api.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No bracket seeds found for year {year}. "
            f"Add an entry to {BRACKET_SEEDS_CONFIG} or place {csv_path}."
        )

    df = pd.read_csv(csv_path)
    r1 = df[df["round_num"] == 1]

    # Try seed columns first
    if pd.notna(r1["seed_high"]).all() and pd.notna(r1["seed_low"]).all():
        def _extract_seeds_from_cols(conf_df: pd.DataFrame) -> list[str]:
            seed_map: dict[int, str] = {}
            for _, row in conf_df.iterrows():
                seed_map[int(row["seed_high"])] = str(row["team_high"])
                seed_map[int(row["seed_low"])] = str(row["team_low"])
            return [seed_map[i] for i in range(1, 9)]

        east_seeds = _extract_seeds_from_cols(r1[r1["conference"].str.lower() == "east"])
        west_seeds = _extract_seeds_from_cols(r1[r1["conference"].str.lower() == "west"])
    else:
        # Fall back to win-based ordering (seed columns absent for older years)
        win_totals = pd.read_csv(RAW_DIR / "Team Summaries.csv")
        win_totals = win_totals[win_totals["lg"] == "NBA"]
        year_wins: dict[str, int] = {
            str(row["abbreviation"]): int(row["w"]) if pd.notna(row["w"]) else 0
            for _, row in win_totals[win_totals["season"] == year].iterrows()
        }

        def _sort_by_wins(teams: list[str]) -> list[str]:
            return sorted(teams, key=lambda t: (-year_wins.get(t, 0), t))

        def _extract_seeds_by_wins(conf_df: pd.DataFrame) -> list[str]:
            teams = (
                [str(r["team_high"]) for _, r in conf_df.iterrows()]
                + [str(r["team_low"]) for _, r in conf_df.iterrows()]
            )
            if len(teams) != 8:
                raise ValueError(
                    f"Expected 8 teams for conference in {year}, got {len(teams)}: {teams}"
                )
            return _sort_by_wins(teams)

        east_seeds = _extract_seeds_by_wins(r1[r1["conference"].str.lower() == "east"])
        west_seeds = _extract_seeds_by_wins(r1[r1["conference"].str.lower() == "west"])

    logger.info("Loaded bracket seeds for %d from %s", year, csv_path)
    return east_seeds, west_seeds


def load_injury_draws(year: int, n_sims: int, seed: int | None) -> dict[str, np.ndarray]:
    """Load injury simulation summaries and pre-generate per-team availability draws.

    Reads results/injury_sims/injury_sims_{year}.parquet (columns: team_id, mean, std, …)
    and samples n_sims availability values per team from N(mean, std) clipped to [0, 1].

    Args:
        year: Season year.
        n_sims: Number of bracket iterations (determines array length).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Dict mapping team_id → np.ndarray of shape (n_sims,).

    Raises:
        FileNotFoundError: If the injury sims file does not exist.
    """
    path = INJURY_SIM_DIR / f"injury_sims_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Injury sims not found: {path}. "
            "Run the injury simulation pipeline first, or use --no-injury."
        )

    inj_df = pd.read_parquet(path)
    # Aggregate across potential series (mean of means / stds per team)
    team_inj = inj_df.groupby("team_id")[["mean", "std"]].mean().reset_index()

    rng = np.random.default_rng(seed)
    injury_draws: dict[str, np.ndarray] = {}
    for _, row in team_inj.iterrows():
        draws = rng.normal(float(row["mean"]), max(float(row["std"]), 0.01), size=n_sims)
        injury_draws[str(row["team_id"])] = np.clip(draws, 0.0, 1.0)

    logger.info(
        "Loaded injury draws for %d teams (year=%d, n_sims=%d)",
        len(injury_draws), year, n_sims,
    )
    return injury_draws


def main() -> None:
    from src.simulation import aggregate, report, run_bracket

    args = parse_args()
    logger.info("=== Bracket Simulation START (year=%d, window=%s) ===", args.year, args.window)

    # Load team features
    logger.info("Loading team features for year %d…", args.year)
    team_features = load_team_features(args.year)
    logger.info("Team features: %d teams, %d features", len(team_features), len(team_features.columns))

    # Load bracket seeds
    logger.info("Loading bracket seeds for year %d…", args.year)
    east_seeds, west_seeds = load_bracket_seeds(args.year)
    logger.info("East: %s", east_seeds)
    logger.info("West: %s", west_seeds)

    # Load injury draws (optional)
    injury_draws: dict[str, np.ndarray] | None = None
    if not args.no_injury:
        try:
            injury_draws = load_injury_draws(args.year, args.n_sims, args.seed)
        except FileNotFoundError as e:
            logger.warning("%s — proceeding without injury adjustment.", e)

    # Run simulations
    all_teams = east_seeds + west_seeds
    outcomes = run_bracket.run_simulations(
        year=args.year,
        east_seeds=east_seeds,
        west_seeds=west_seeds,
        team_features=team_features,
        window=args.window,
        n_sims=args.n_sims,
        seed=args.seed,
        injury_draws=injury_draws,
    )

    aggregated = aggregate.aggregate_outcomes(outcomes, all_teams)
    out_dir = report.save_simulation_report(aggregated, args.year, args.window)
    logger.info("=== Bracket Simulation DONE — results in %s ===", out_dir)


if __name__ == "__main__":
    main()
