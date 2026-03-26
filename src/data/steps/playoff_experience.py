"""playoff_experience.py — Step: historical playoff series wins/played per team.

Reads all playoff_series/{year}_nba_api.csv files (1980-2024) and, for each
series in the input DataFrame, attaches the cumulative playoff experience of
both teams *up to but not including* the current season.

Features produced (four columns added to df):
  playoff_series_wins_high / _low   : total series wins historically
  avg_playoff_series_wins_high / _low : series wins ÷ playoff appearances
  playoff_series_played_high / _low  : total series played historically
  avg_playoff_series_played_high / _low : series played ÷ seasons in league

Input df must have columns: series_id, season, team_high, team_low.
Output df has all original columns plus the eight features above.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PLAYOFF_SERIES_DIR = RAW_DIR / "playoff_series"

# Earliest season in training data
FIRST_SEASON = 1980


def _load_all_series(up_to_season: int) -> pd.DataFrame:
    """Load and concatenate all playoff series CSVs for seasons < up_to_season.

    Args:
        up_to_season: The current season; only seasons strictly before this
            are loaded (anti-look-ahead).

    Returns:
        DataFrame with columns: season, team_high, team_low, higher_seed_wins.
    """
    frames: list[pd.DataFrame] = []
    for path in sorted(PLAYOFF_SERIES_DIR.glob("*_nba_api.csv")):
        year = int(path.stem.split("_")[0])
        if year < FIRST_SEASON or year >= up_to_season:
            continue
        df = pd.read_csv(path, usecols=["season", "team_high", "team_low", "higher_seed_wins"])
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["season", "team_high", "team_low", "higher_seed_wins"])
    return pd.concat(frames, ignore_index=True)


def _build_team_experience_table(max_season: int) -> pd.DataFrame:
    """Build a cumulative experience table for every (team, season) pair.

    For each team and each season Y, compute stats accumulated across all
    seasons in [FIRST_SEASON, Y) — i.e., strictly before Y.

    Args:
        max_season: Highest season to generate rows for (inclusive).

    Returns:
        DataFrame indexed by (season, team_abbr) with columns:
            series_wins, series_played, seasons_in_league.
    """
    all_series = _load_all_series(up_to_season=max_season + 1)
    if all_series.empty:
        return pd.DataFrame()

    # Expand to one row per team per series
    high = all_series[["season", "team_high", "higher_seed_wins"]].copy()
    high.columns = ["season", "team", "won"]
    # high seed wins when higher_seed_wins == 1
    high["won"] = (high["won"] == 1).astype(int)

    low = all_series[["season", "team_low", "higher_seed_wins"]].copy()
    low.columns = ["season", "team", "won"]
    # low seed wins when higher_seed_wins == 0
    low["won"] = (low["won"] == 0).astype(int)

    per_team_series = pd.concat([high, low], ignore_index=True)
    per_team_series["played"] = 1

    # Aggregate to season-level per team
    season_stats = (
        per_team_series.groupby(["season", "team"])
        .agg(series_wins=("won", "sum"), series_played=("played", "sum"))
        .reset_index()
    )

    # For each target season Y, we need cumulative sums over all seasons < Y
    rows: list[dict] = []
    for target_season in range(FIRST_SEASON, max_season + 1):
        prior = season_stats[season_stats["season"] < target_season]
        if prior.empty:
            continue
        cumulative = (
            prior.groupby("team")
            .agg(
                series_wins_cum=("series_wins", "sum"),
                series_played_cum=("series_played", "sum"),
                playoff_seasons=("season", "nunique"),
            )
            .reset_index()
        )
        # seasons_in_league = number of seasons the team has existed up to target_season - 1
        # approximate as number of distinct seasons the team appeared (in or out of playoffs)
        all_teams_up_to = set(
            season_stats.loc[season_stats["season"] < target_season, "team"].unique()
        )
        for _, row in cumulative.iterrows():
            rows.append(
                {
                    "season": target_season,
                    "team": row["team"],
                    "series_wins_cum": int(row["series_wins_cum"]),
                    "series_played_cum": int(row["series_played_cum"]),
                    "playoff_seasons": int(row["playoff_seasons"]),
                }
            )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["avg_series_wins"] = result["series_wins_cum"] / result["playoff_seasons"].clip(lower=1)
    result["avg_series_played"] = result["series_played_cum"] / result["playoff_seasons"].clip(lower=1)
    return result


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Attach historical playoff experience features to a series-level DataFrame.

    For each row in df (one row = one playoff series), looks up cumulative
    series wins/played for team_high and team_low in all prior seasons.
    Teams with no prior playoff history receive zeros.

    Args:
        df: Series-level DataFrame with columns: series_id, season,
            team_high, team_low.

    Returns:
        df with eight new columns added (four per team side).
    """
    if df.empty:
        logger.warning("playoff_experience.run received empty DataFrame — returning as-is.")
        return df

    max_season = int(df["season"].max())
    logger.info("Building playoff experience table up to season %d", max_season)
    exp = _build_team_experience_table(max_season)

    if exp.empty:
        logger.warning("No playoff experience data found; all features will be zero.")
        for side in ("high", "low"):
            for col in (
                f"playoff_series_wins_{side}",
                f"avg_playoff_series_wins_{side}",
                f"playoff_series_played_{side}",
                f"avg_playoff_series_played_{side}",
            ):
                df = df.copy()
                df[col] = 0.0
        return df

    exp_idx = exp.set_index(["season", "team"])

    def _lookup(season: int, team: str, col: str) -> float:
        try:
            return float(exp_idx.at[(season, team), col])
        except KeyError:
            return 0.0

    df = df.copy()
    for side, team_col in (("high", "team_high"), ("low", "team_low")):
        df[f"playoff_series_wins_{side}"] = [
            _lookup(row.season, getattr(row, team_col), "series_wins_cum")
            for row in df.itertuples()
        ]
        df[f"avg_playoff_series_wins_{side}"] = [
            _lookup(row.season, getattr(row, team_col), "avg_series_wins")
            for row in df.itertuples()
        ]
        df[f"playoff_series_played_{side}"] = [
            _lookup(row.season, getattr(row, team_col), "series_played_cum")
            for row in df.itertuples()
        ]
        df[f"avg_playoff_series_played_{side}"] = [
            _lookup(row.season, getattr(row, team_col), "avg_series_played")
            for row in df.itertuples()
        ]

    logger.info(
        "playoff_experience: added 8 features to %d series rows", len(df)
    )
    return df
