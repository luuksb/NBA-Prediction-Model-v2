#!/usr/bin/env python3
"""injury_counterfactual.py — Counterfactual: what if fragile contenders were fully healthy?

Re-runs the 2026 bracket simulation with DET and SAS pinned to full availability
(all star players healthy every round) and compares championship probabilities
against the baseline injury-adjusted run.

Usage:
    python scripts/analysis/injury_counterfactual.py [--window WINDOW] [--n-sims N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.run_bracket_sim import load_bracket_seeds, load_injury_draws, load_team_features
from src.simulation import aggregate
from src.simulation.run_bracket import run_simulations

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

YEAR = 2026
BASELINE_DIR = Path("results/simulations")
TEAMS_TO_PIN = ["DET", "SAS"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Injury counterfactual simulation.")
    parser.add_argument("--window", default="full", help="Training window (default: full).")
    parser.add_argument("--n-sims", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pin_teams_healthy(injury_draws: dict, team_ids: list[str]) -> dict:
    """Return a copy of injury_draws with specified teams pinned to full health.

    Sets all draw values to 0.0 for the given teams so every draw falls below
    any mean_rate, meaning all star players are always counted as healthy.

    Args:
        injury_draws: Dict produced by load_injury_draws().
        team_ids: List of team abbreviations to pin healthy.

    Returns:
        New injury_draws dict with a copied draws array.
    """
    draws_cf = injury_draws["draws"].copy()
    team_index = injury_draws["team_index"]
    for team_id in team_ids:
        if team_id in team_index:
            draws_cf[team_index[team_id], :, :, :] = 0.0
            logger.info("Pinned %s to full health in counterfactual draws.", team_id)
        else:
            logger.warning("Team %s not found in injury draws — skipping.", team_id)
    return {**injury_draws, "draws": draws_cf}


def main() -> None:
    args = parse_args()

    logger.info("Loading inputs for year %d, window=%s…", YEAR, args.window)
    team_features = load_team_features(YEAR)
    east_seeds, west_seeds = load_bracket_seeds(YEAR)
    injury_draws = load_injury_draws(YEAR, args.n_sims, args.seed)

    cf_injury_draws = pin_teams_healthy(injury_draws, TEAMS_TO_PIN)

    logger.info("Running counterfactual simulation (%d iterations)…", args.n_sims)
    outcomes = run_simulations(
        year=YEAR,
        east_seeds=east_seeds,
        west_seeds=west_seeds,
        team_features=team_features,
        window=args.window,
        n_sims=args.n_sims,
        injury_draws=cf_injury_draws,
        seed=args.seed,
    )

    agg = aggregate.aggregate_outcomes(outcomes, east_seeds + west_seeds)
    cf_probs = pd.DataFrame(
        list(agg["championship_prob"].items()), columns=["team", "cf_prob"]
    )

    baseline_path = BASELINE_DIR / f"{YEAR}_{args.window}" / "championship_probs.parquet"
    baseline = pd.read_parquet(baseline_path)

    merged = baseline.merge(cf_probs, on="team")
    merged["delta"] = merged["cf_prob"] - merged["championship_prob"]
    merged = merged.sort_values("cf_prob", ascending=False)

    print(f"\nCounterfactual: {', '.join(TEAMS_TO_PIN)} pinned to full health — {YEAR} ({args.window} window)")
    print(f"{'Team':<6}  {'Baseline':>10}  {'Fully Healthy':>13}  {'Delta':>8}")
    print("-" * 42)
    for _, row in merged.iterrows():
        if row["cf_prob"] > 0.001 or row["championship_prob"] > 0.001:
            marker = " *" if row["team"] in TEAMS_TO_PIN else ""
            print(
                f"{row['team']:<6}  {row['championship_prob']:>9.1%}  "
                f"{row['cf_prob']:>12.1%}  {row['delta']:>+8.1%}{marker}"
            )
    print("\n* = pinned to full health in counterfactual")


if __name__ == "__main__":
    main()
