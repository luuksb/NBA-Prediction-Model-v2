#!/usr/bin/env python3
"""run_data_pipeline.py — CLI entry point for Module 1: Data Pipeline.

Runs all preprocessing steps in order, assembles the final series dataset,
and runs data quality checks.

Usage:
    python scripts/run_data_pipeline.py [--steps STEP1,STEP2] [--skip-dq]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NBA playoff data pipeline.")
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated list of step names to run (default: all).",
    )
    parser.add_argument(
        "--skip-dq",
        action="store_true",
        help="Skip data quality checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== Data Pipeline START ===")

    # Step chain — add new step modules to this list in order
    from src.data import assemble, quality

    # TODO: import and run individual steps from src.data.steps once implemented
    # Example:
    #   from src.data.steps import team_ratings, player_ratings, ...
    #   for step_fn in [team_ratings.run, player_ratings.run, ...]:
    #       df = step_fn(df)
    #       save_intermediate(df, step_fn.__module__)

    logger.info("Assembling final dataset…")
    final_df = assemble.assemble_dataset()
    out_path = assemble.save_final_dataset(final_df)
    logger.info("Final dataset: %s", out_path)

    if not args.skip_dq:
        logger.info("Running data quality checks…")
        report_path = quality.run_quality_checks(final_df)
        logger.info("DQ report: %s", report_path)

    logger.info("=== Data Pipeline DONE ===")


if __name__ == "__main__":
    main()
