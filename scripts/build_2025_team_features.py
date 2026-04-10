#!/usr/bin/env python3
"""build_2025_team_features.py — Build per-team features for the 2025 validation year.

Computes team_season_features rows for the 16 2025 playoff teams using:
  - 2025 regular-season team stats (team_ratings)
  - Top-3 player BPM with avail=1.0 (full health baseline for injury sim)
  - Playoff experience cumulative through 2024 (anti-look-ahead)
  - Coach experience cumulative through 2024 (if available)

Appends 2025 rows to data/final/team_season_features.parquet.
Injury draws from results/injury_sims/injury_sims_2025.parquet will scale
bpm_avail_sum at simulation time — so full health here is correct.

Usage:
    python scripts/build_2025_team_features.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.steps import (
    team_ratings as tr_mod,
    player_ratings as pr_mod,
    playoff_experience as pe_mod,
    coach_experience as ce_mod,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

YEAR = 2025
PLAYOFF_TEAMS = [
    # East
    "CLE",
    "BOS",
    "NYK",
    "IND",
    "MIL",
    "DET",
    "ORL",
    "MIA",
    # West
    "OKC",
    "HOU",
    "LAL",
    "DEN",
    "LAC",
    "MIN",
    "GSW",
    "MEM",
]
FINAL_DIR = Path("data/final")


def build_team_ratings(year: int) -> pd.DataFrame:
    """Return 2025 regular-season team rating features.

    Args:
        year: Target season year.

    Returns:
        DataFrame with columns: year, team, <all team stat columns>.
    """
    team_stats = tr_mod.build_team_stats([year])
    team_stats = team_stats.rename(columns={"season": "year"})
    logger.info("team_ratings: %d teams for year %d", len(team_stats), year)
    return team_stats


def build_player_features(year: int, teams: list[str]) -> pd.DataFrame:
    """Return bpm_avail_sum and star_flag with avail=1.0 for each team.

    Avail is set to 1.0 (full health) so the injury simulation can scale
    bpm_avail_sum at bracket simulation time. Uses EPM top-5 for star_flag
    (EPM data available from 2002 onward).

    Args:
        year: Target season year.
        teams: List of team abbreviations to include.

    Returns:
        DataFrame with columns: year, team, bpm_avail_sum, per_avail_sum, star_flag.
    """
    player_stats = pr_mod._load_player_stats([year])
    n = pr_mod._load_n_stars()
    top_n = pr_mod._identify_top_n(player_stats, n)

    # Superstar set: EPM top-5 for 2025 (>= EPM_START_SEASON=2002)
    epm_set = pr_mod._load_epm_set([year])
    superstar_set = epm_set

    logger.info(
        "player_features: top-%d ranking complete; %d superstar entries for %d",
        n,
        len(superstar_set),
        year,
    )

    rows: list[dict] = []
    for team in teams:
        team_top = top_n[(top_n["season"] == year) & (top_n["team"] == team)].copy()

        bpm_sum = 0.0
        per_sum = 0.0
        star_flag = 0.0

        for _, row in team_top.iterrows():
            bpm = float(row["bpm"]) if not pd.isna(row["bpm"]) else 0.0
            per = float(row["per"]) if not pd.isna(row["per"]) else 0.0
            bpm_sum += bpm  # avail = 1.0
            per_sum += per

            # star_flag: 1.0 × avail if any top-N player is a superstar
            if star_flag == 0.0 and (year, row["player_norm"]) in superstar_set:
                star_flag = 1.0  # avail = 1.0

        rows.append(
            {
                "year": year,
                "team": team,
                "bpm_avail_sum": bpm_sum,
                "per_avail_sum": per_sum,
                "star_flag": star_flag,
            }
        )
        logger.info(
            "  %-4s  bpm_avail_sum=%.2f  star_flag=%.1f",
            team,
            bpm_sum,
            star_flag,
        )

    return pd.DataFrame(rows)


def build_experience_features(year: int, teams: list[str]) -> pd.DataFrame:
    """Return cumulative playoff experience for each team through year-1.

    Args:
        year: Target season year (experience is cumulative strictly before this year).
        teams: List of team abbreviations to include.

    Returns:
        DataFrame with columns: year, team, playoff_series_wins.
    """
    exp = pe_mod._build_roster_experience_table(year)
    if exp.empty:
        logger.warning("No playoff experience data found; filling zeros.")
        return pd.DataFrame([{"year": year, "team": t, "playoff_series_wins": 0.0} for t in teams])

    exp_year = exp[exp["season"] == year].set_index("team")

    rows: list[dict] = []
    for team in teams:
        wins = float(exp_year.at[team, "series_wins_cum"]) if team in exp_year.index else 0.0
        rows.append({"year": year, "team": team, "playoff_series_wins": wins})
        logger.info("  %-4s  playoff_series_wins=%.0f", team, wins)

    return pd.DataFrame(rows)


def build_coach_features(year: int, teams: list[str]) -> pd.DataFrame:
    """Return cumulative coach series wins for each team's head coach through year-1.

    Args:
        year: Target season year.
        teams: List of team abbreviations to include.

    Returns:
        DataFrame with columns: year, team, coach_series_wins_cum.
    """
    try:
        coach_team = ce_mod._load_coach_team_year()
        coach_record = ce_mod._build_coach_record_table(coach_team, year)
        if coach_record.empty:
            raise ValueError("Empty coach record table")

        cr_year = coach_record[coach_record["season"] == year].set_index("team_abbr")
        rows: list[dict] = []
        for team in teams:
            wins = (
                float(cr_year.at[team, "coach_series_wins_cum"]) if team in cr_year.index else 0.0
            )
            rows.append({"year": year, "team": team, "coach_series_wins_cum": wins})
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("coach_experience: %s — filling zeros.", exc)
        return pd.DataFrame(
            [{"year": year, "team": t, "coach_series_wins_cum": 0.0} for t in teams]
        )


def main() -> None:
    logger.info("=== Build 2025 Team Features START ===")

    # Load existing team_season_features.parquet
    tsf_path = FINAL_DIR / "team_season_features.parquet"
    existing = pd.read_parquet(tsf_path)

    if YEAR in existing["year"].values:
        logger.info("Year %d already present in %s — removing old rows.", YEAR, tsf_path)
        existing = existing[existing["year"] != YEAR]

    # Build feature components
    logger.info("Building team rating features for %d…", YEAR)
    team_ratings_df = build_team_ratings(YEAR)
    # Filter to playoff teams only
    team_ratings_df = team_ratings_df[team_ratings_df["team"].isin(PLAYOFF_TEAMS)].copy()

    logger.info("Building player features (avail=1.0) for %d…", YEAR)
    player_df = build_player_features(YEAR, PLAYOFF_TEAMS)

    logger.info("Building playoff experience features for %d…", YEAR)
    exp_df = build_experience_features(YEAR, PLAYOFF_TEAMS)

    logger.info("Building coach experience features for %d…", YEAR)
    coach_df = build_coach_features(YEAR, PLAYOFF_TEAMS)

    # Merge all components
    features_2025 = team_ratings_df.copy()
    for df in (player_df, exp_df, coach_df):
        features_2025 = features_2025.merge(df, on=["year", "team"], how="left")

    logger.info(
        "2025 features: %d teams × %d columns",
        len(features_2025),
        len(features_2025.columns),
    )

    # Align columns with existing table (fill missing with NaN)
    all_cols = list(existing.columns)
    for col in all_cols:
        if col not in features_2025.columns:
            features_2025[col] = np.nan

    # Keep only columns that exist in the existing file (same schema)
    features_2025 = features_2025[all_cols]

    # Append and save
    updated = pd.concat([existing, features_2025], ignore_index=True)
    updated.to_parquet(tsf_path, index=False)
    logger.info(
        "Saved updated team_season_features.parquet: %d rows (%d years)",
        len(updated),
        updated["year"].nunique(),
    )
    logger.info(
        "2025 teams in file: %s",
        sorted(updated[updated["year"] == YEAR]["team"].tolist()),
    )
    logger.info("=== Build 2025 Team Features DONE ===")


if __name__ == "__main__":
    main()
