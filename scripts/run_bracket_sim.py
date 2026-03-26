#!/usr/bin/env python3
"""run_bracket_sim.py — CLI entry point for Module 4: Bracket Simulation.

Runs 50,000 Monte Carlo bracket iterations for a given year and saves results.

Usage:
    python scripts/run_bracket_sim.py --year 2026 --window modern [--n-sims 50000]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NBA playoff bracket simulation.")
    parser.add_argument("--year", type=int, required=True, help="Season year to simulate.")
    parser.add_argument("--window", required=True, help="Training window name (e.g. 'modern').")
    parser.add_argument("--n-sims", type=int, default=50_000, help="Number of Monte Carlo iterations.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--no-injury", action="store_true", help="Disable injury adjustment.")
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    from src.simulation import aggregate, report, run_bracket

    args = parse_args()
    logger.info("=== Bracket Simulation START (year=%d, window=%s) ===", args.year, args.window)

    # TODO: load team_features from data/final/series_dataset.parquet filtered to args.year
    # TODO: load east_seeds and west_seeds from bracket seeding data for args.year
    # TODO: load injury_draws from results/injury_sims/ if not --no-injury

    logger.warning("Bracket simulation pipeline not yet fully implemented. "
                   "Implement team feature loading and bracket seeding for year %d.", args.year)

    # Example wire-up (replace TODOs above):
    # outcomes = run_bracket.run_simulations(
    #     year=args.year, east_seeds=east_seeds, west_seeds=west_seeds,
    #     team_features=team_features, n_sims=args.n_sims, seed=args.seed,
    #     injury_draws=None if args.no_injury else injury_draws,
    # )
    # aggregated = aggregate.aggregate_outcomes(outcomes, all_teams)
    # report.save_simulation_report(aggregated, args.year, args.window)

    logger.info("=== Bracket Simulation DONE ===")


if __name__ == "__main__":
    main()
