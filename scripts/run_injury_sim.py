#!/usr/bin/env python3
"""run_injury_sim.py — CLI entry point for Module 3: Injury Simulation.

Generates availability distributions for the validation and prediction years.

Usage:
    python scripts/run_injury_sim.py --year 2025 [--n-draws 1000]
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
    parser = argparse.ArgumentParser(description="Run NBA playoff injury simulation.")
    parser.add_argument("--year", type=int, required=True, help="Target year (2025 or 2026).")
    parser.add_argument("--n-draws", type=int, default=1000, help="Monte Carlo draws per player.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== Injury Simulation START (year=%d) ===", args.year)

    # TODO: load players_df and games_df from data/final/ or data/intermediate/
    # from src.injury import identify_top_players, availability_history, simulate, export
    # top_players = identify_top_players.identify_top_players(players_df, ...)
    # rates = availability_history.compute_availability_rates(games_df, target_year=args.year)
    # ... simulate and export

    logger.warning("Injury simulation pipeline not yet fully implemented. "
                   "Implement data loading and wire up src/injury/ modules.")

    logger.info("=== Injury Simulation DONE ===")


if __name__ == "__main__":
    main()
