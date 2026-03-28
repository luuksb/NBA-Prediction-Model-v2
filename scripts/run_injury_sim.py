#!/usr/bin/env python3
"""run_injury_sim.py — CLI entry point for Module 3: Injury Simulation.

Generates availability distributions for the validation and prediction years.

Usage:
    python scripts/run_injury_sim.py --year 2025 [--n-draws 1000] [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.injury import export, identify_top_players, simulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

ADV_CSV = Path("data/raw/Advanced.csv")
AVAIL_PARQUET = Path("data/raw/playoff_availability.parquet")


_SUFFIX_PAT = re.compile(r"_(jr|sr|ii|iii|iv|v)$")


def _normalize_name(name: str) -> str:
    """Normalize a player name to lowercase ASCII underscore format.

    Strips diacritics via NFKD decomposition (ć → c, č → c, etc.) so that
    names like 'Nikola Jokić' and 'Luka Dončić' match their keys in
    playoff_availability.parquet.  Generational suffixes (Jr., III, etc.) are
    stripped so that 'Jimmy Butler' (Advanced.csv) matches 'Jimmy Butler III'
    (PlayerStatisticsMisc.csv) both resolving to 'jimmy_butler'.

    Args:
        name: Display name, e.g. "Nikola Jokić" or "De'Aaron Fox".

    Returns:
        Normalized string, e.g. "nikola_jokic" or "de_aaron_fox".
    """
    ascii_name = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )
    norm = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    return _SUFFIX_PAT.sub("", norm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NBA playoff injury simulation.")
    parser.add_argument("--year", type=int, required=True, help="Target year (2025 or 2026).")
    parser.add_argument("--n-draws", type=int, default=1000, help="Monte Carlo draws per player.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== Injury Simulation START (year=%d) ===", args.year)

    # ── Load data ──────────────────────────────────────────────────────────────
    adv_df = pd.read_csv(ADV_CSV)
    avail_df = pd.read_parquet(AVAIL_PARQUET)

    # ── Normalize player names for joining ────────────────────────────────────
    # Exclude multi-team aggregate rows ("2TM", "3TM", etc.) — use single-team rows only
    adv_df = adv_df[~adv_df["team"].str.match(r"^\dTM$")].copy()
    adv_df["player_name_norm"] = adv_df["player"].apply(_normalize_name)
    adv_df["mpg"] = adv_df["mp"] / adv_df["g"].replace(0, pd.NA)

    availability_rates = avail_df.rename(
        columns={"career_playoff_avail": "availability_rate"}
    ).assign(n_series=1)

    # ── Identify top-3 players per team ───────────────────────────────────────
    top_players = identify_top_players.identify_top_players(
        players_df=adv_df,
        team_col="team",
        year_col="season",
        target_year=args.year,
    )

    # ── Simulate availability per team ────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    sim_records = []

    for team, group in top_players.groupby("team"):
        team_draws = simulate.simulate_team_availability(
            top_players=group,
            availability_rates=availability_rates,
            player_col="player_name_norm",
            rating_col="composite_rating",
            n_draws=args.n_draws,
            rng=rng,
        )
        summary = export.summarise_draws(team_draws)
        sim_records.append({"team_id": team, **summary})
        logger.info("Team %-4s  mean_avail=%.3f  std=%.3f", team, summary["mean"], summary["std"])

    # ── Export ────────────────────────────────────────────────────────────────
    out_path = export.export_injury_sims(sim_records, year=args.year)
    logger.info("Output: %s  (%d teams)", out_path, len(sim_records))
    logger.info("=== Injury Simulation DONE ===")


if __name__ == "__main__":
    main()
