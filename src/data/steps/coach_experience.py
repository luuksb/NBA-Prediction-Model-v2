"""coach_experience.py — Step: cumulative playoff series win % per coach.

Data sources (two separate files cover the full 1980-2024 range):
  • data/raw/basketball_coaches.csv  — coachID, year, tmID (1946-2011)
  • data/raw/coaches_nba_api/{team_id}_{year}.json — season, team_id, coach_name (2012-2024)

Series outcomes come from data/raw/playoff_series/{year}_nba_api.csv.

Coach identity:
  • 1980-2011 rows use a coachID key  (Basketball-Reference slug, e.g. "popovgr01").
  • 2012-2024 rows use a normalised coach_name key (e.g. "gregg_popovich").
  • BRIDGE_COACHES maps coachID → normalised name for coaches who appeared in
    both periods so their career stats are accumulated correctly.

Features produced:
  coach_series_win_pct_high / _low  (float, NaN if coach has no prior playoff series)

Input df must have columns: series_id, season, team_high, team_low.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
COACHES_CSV = RAW_DIR / "basketball_coaches.csv"
COACHES_API_DIR = RAW_DIR / "coaches_nba_api"
PLAYOFF_SERIES_DIR = RAW_DIR / "playoff_series"

FIRST_SEASON = 1980

# nba_api team_id → BBRef-style abbreviation used in playoff_series CSVs
TEAM_ID_ABBREV: dict[int, str] = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612739: "CLE",
    1610612740: "NOP",
    1610612741: "CHI",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612751: "BRK",
    1610612752: "NYK",
    1610612753: "ORL",
    1610612754: "IND",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612760: "OKC",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612763: "MEM",
    1610612764: "WAS",
    1610612765: "DET",
    1610612766: "CHA",
}

# Maps BBRef coachID → normalised name for coaches who span the 2011/2012 data gap.
# Add entries here for any coach whose 1980-2011 record should be combined with
# their 2012-2024 record.  Format: coachID (lowercase) → _normalise_name() output.
BRIDGE_COACHES: dict[str, str] = {
    "popovgr01": "gregg_popovich",
    "riverdo01": "doc_rivers",
    "carliri01": "rick_carlisle",
    "mcmilna01": "nate_mcmillan",
    "brooksc01": "scott_brooks",
    "vogelde01": "frank_vogel",
    "blattda01": "david_blatt",
    "skilesd01": "scott_skiles",
    "sloanje01": "jerry_sloan",
    "riversdo01": "doc_rivers",  # alternate slug form
}


def _normalise_name(name: str) -> str:
    """Lowercase, replace spaces/punctuation with underscores for consistent keys."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_coach_team_year() -> pd.DataFrame:
    """Build a unified coach–team–season table for 1980-2024.

    Returns:
        DataFrame with columns: season (int), team_abbr (str), coach_key (str).
        coach_key is a normalised string that uniquely identifies a coach across
        both data sources (coachID mapped via BRIDGE_COACHES where applicable).
    """
    rows: list[dict] = []

    # ── 1980-2011: basketball_coaches.csv ────────────────────────────────────
    if COACHES_CSV.exists():
        csv = pd.read_csv(COACHES_CSV)
        csv = csv[
            (csv["lgID"] == "NBA")
            & (csv["year"].astype(int) >= FIRST_SEASON)
            & (csv["year"].astype(int) <= 2011)
        ].copy()
        for _, row in csv.iterrows():
            coach_id = str(row["coachID"]).strip().lower()
            key = BRIDGE_COACHES.get(coach_id, coach_id)
            rows.append(
                {
                    "season": int(row["year"]),
                    "team_abbr": str(row["tmID"]).strip().upper(),
                    "coach_key": key,
                }
            )
    else:
        logger.warning("basketball_coaches.csv not found; 1980-2011 coach data missing.")

    # ── 2012-2024: coaches_nba_api/*.json ────────────────────────────────────
    if COACHES_API_DIR.exists():
        for path in sorted(COACHES_API_DIR.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as fh:
                    d = json.load(fh)
                season = int(d["season"])
                team_id = int(d["team_id"])
                coach_name = str(d["coach_name"]).strip()
                abbrev = TEAM_ID_ABBREV.get(team_id)
                if abbrev is None:
                    logger.debug("Unknown team_id %d in %s — skipping.", team_id, path.name)
                    continue
                rows.append(
                    {
                        "season": season,
                        "team_abbr": abbrev,
                        "coach_key": _normalise_name(coach_name),
                    }
                )
            except Exception as exc:
                logger.warning("Could not parse %s: %s", path.name, exc)
    else:
        logger.warning("coaches_nba_api/ directory not found; 2012-2024 coach data missing.")

    if not rows:
        return pd.DataFrame(columns=["season", "team_abbr", "coach_key"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["season", "team_abbr"])
    logger.info("Loaded coach-team-year table: %d rows (seasons %d-%d)",
                len(df), df["season"].min(), df["season"].max())
    return df


def _load_all_series(up_to_season: int) -> pd.DataFrame:
    """Load playoff series data for all seasons < up_to_season."""
    frames: list[pd.DataFrame] = []
    for path in sorted(PLAYOFF_SERIES_DIR.glob("*_nba_api.csv")):
        year = int(path.stem.split("_")[0])
        if year < FIRST_SEASON or year >= up_to_season:
            continue
        df = pd.read_csv(
            path,
            usecols=["season", "team_high", "team_low", "higher_seed_wins"],
        )
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["season", "team_high", "team_low", "higher_seed_wins"])
    return pd.concat(frames, ignore_index=True)


def _build_coach_record_table(
    coach_team: pd.DataFrame, max_season: int
) -> pd.DataFrame:
    """Accumulate series wins/losses per coach_key across all prior seasons.

    Args:
        coach_team: DataFrame from _load_coach_team_year().
        max_season: Highest season to generate entries for (inclusive).

    Returns:
        DataFrame with columns: season, team_abbr, coach_key,
            coach_series_wins_cum, coach_series_losses_cum, coach_series_win_pct.
        One row per (season, team_abbr) representing the coach's record
        *before* that season.
    """
    all_series = _load_all_series(up_to_season=max_season + 1)
    if all_series.empty or coach_team.empty:
        return pd.DataFrame()

    # Expand series to per-team rows, annotate with coach_key
    high = all_series[["season", "team_high", "higher_seed_wins"]].copy()
    high.columns = ["season", "team_abbr", "higher_seed_wins"]
    high["won"] = (high["higher_seed_wins"] == 1).astype(int)

    low = all_series[["season", "team_low", "higher_seed_wins"]].copy()
    low.columns = ["season", "team_abbr", "higher_seed_wins"]
    low["won"] = (low["higher_seed_wins"] == 0).astype(int)

    team_series = pd.concat([high, low], ignore_index=True)
    team_series["played"] = 1

    # Attach coach key to each series
    coach_lookup = coach_team.set_index(["season", "team_abbr"])["coach_key"].to_dict()
    team_series["coach_key"] = [
        coach_lookup.get((row.season, row.team_abbr))
        for row in team_series.itertuples()
    ]
    team_series = team_series.dropna(subset=["coach_key"])

    # Aggregate to coach-season level
    coach_season = (
        team_series.groupby(["season", "coach_key"])
        .agg(series_wins=("won", "sum"), series_played=("played", "sum"))
        .reset_index()
    )

    # For each target season Y, build cumulative record per coach
    rows: list[dict] = []
    for target_season in range(FIRST_SEASON, max_season + 1):
        prior = coach_season[coach_season["season"] < target_season]
        if prior.empty:
            continue
        cumulative = (
            prior.groupby("coach_key")
            .agg(wins=("series_wins", "sum"), played=("series_played", "sum"))
            .reset_index()
        )
        cumulative["losses"] = cumulative["played"] - cumulative["wins"]
        cumulative["win_pct"] = cumulative["wins"] / cumulative["played"].clip(lower=1)

        # Map back to team for this target season
        season_coaches = coach_team[coach_team["season"] == target_season]
        cumulative_idx = cumulative.set_index("coach_key")
        for tc in season_coaches.itertuples():
            if tc.coach_key not in cumulative_idx.index:
                rows.append(
                    {
                        "season": target_season,
                        "team_abbr": tc.team_abbr,
                        "coach_key": tc.coach_key,
                        "coach_series_wins_cum": 0,
                        "coach_series_losses_cum": 0,
                        "coach_series_win_pct": np.nan,
                    }
                )
            else:
                r = cumulative_idx.loc[tc.coach_key]
                rows.append(
                    {
                        "season": target_season,
                        "team_abbr": tc.team_abbr,
                        "coach_key": tc.coach_key,
                        "coach_series_wins_cum": int(r["wins"]),
                        "coach_series_losses_cum": int(r["losses"]),
                        "coach_series_win_pct": float(r["win_pct"])
                        if int(r["played"]) > 0
                        else np.nan,
                    }
                )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Attach coach playoff series win-percentage features to a series DataFrame.

    For each row (one playoff series), looks up the cumulative series win%
    of the head coach for team_high and team_low using only data from prior seasons.
    NaN is assigned when the coach has no prior playoff series record.

    Args:
        df: Series-level DataFrame with columns: series_id, season,
            team_high, team_low.

    Returns:
        df with two new columns: coach_series_win_pct_high, coach_series_win_pct_low.
    """
    if df.empty:
        logger.warning("coach_experience.run received empty DataFrame — returning as-is.")
        return df

    max_season = int(df["season"].max())
    coach_team = _load_coach_team_year()

    if coach_team.empty:
        logger.warning("No coach-team data; coach_series_win_pct features will be NaN.")
        df = df.copy()
        df["coach_series_win_pct_high"] = np.nan
        df["coach_series_win_pct_low"] = np.nan
        return df

    logger.info("Building coach record table up to season %d", max_season)
    records = _build_coach_record_table(coach_team, max_season)

    if records.empty:
        logger.warning("Coach record table is empty; features will be NaN.")
        df = df.copy()
        df["coach_series_win_pct_high"] = np.nan
        df["coach_series_win_pct_low"] = np.nan
        return df

    rec_idx = records.set_index(["season", "team_abbr"])["coach_series_win_pct"].to_dict()

    def _lookup(season: int, team: str) -> float | None:
        return rec_idx.get((season, team), np.nan)

    df = df.copy()
    df["coach_series_win_pct_high"] = [
        _lookup(row.season, row.team_high) for row in df.itertuples()
    ]
    df["coach_series_win_pct_low"] = [
        _lookup(row.season, row.team_low) for row in df.itertuples()
    ]

    n_nan_high = df["coach_series_win_pct_high"].isna().sum()
    n_nan_low = df["coach_series_win_pct_low"].isna().sum()
    logger.info(
        "coach_experience: added 2 features; NaN count — high: %d, low: %d",
        n_nan_high,
        n_nan_low,
    )
    return df
