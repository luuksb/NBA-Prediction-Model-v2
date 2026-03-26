"""team_ratings.py — Step: regular-season team efficiency stats per series.

Loads all numeric columns from Team Stats Per 100 Poss.csv and Team Summaries.csv
and, for each playoff series, computes:

  delta_{stat} = stat_high − stat_low

for every available numeric stat. This produces exhaustive team-level delta features
without cherry-picking. Source files use playoffs=True to flag teams that made the
playoffs; those rows contain the full regular-season stats for playoff teams.

Data sources:
  data/raw/Team Stats Per 100 Poss.csv  — shooting / rebounding / etc. per 100 poss
  data/raw/Team Summaries.csv           — efficiency ratings, pace, advanced metrics

Anti-look-ahead: regular-season stats are fully finalised before the playoffs begin.

Input df must have columns: series_id, season, team_high, team_low.
Output df adds delta_{stat} for every numeric team stat in the two source files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PER100_CSV = RAW_DIR / "Team Stats Per 100 Poss.csv"
SUMMARIES_CSV = RAW_DIR / "Team Summaries.csv"

# Columns that are metadata / identifiers — excluded from delta features
_META_COLS = frozenset(
    ["season", "lg", "team", "abbreviation", "playoffs",
     "g", "mp",                         # games / minutes (per-100 file)
     "pw", "pl",                         # pythagorean wins/losses
     "arena", "attend", "attend_g"]      # attendance / arena
)

# Warn if any raw column exceeds this missing-value rate before delta computation
MISSINGNESS_STOP_THRESHOLD = 0.20


def _load_per100_stats(seasons: list[int]) -> pd.DataFrame:
    """Load Team Stats Per 100 Poss.csv and return playoff teams for given seasons.

    The playoffs=True flag marks teams that made the playoffs; those rows hold
    full regular-season per-100-possession stats.

    Args:
        seasons: Season end-years to include.

    Returns:
        DataFrame with columns: season, abbreviation + all numeric stat columns.
    """
    df = pd.read_csv(PER100_CSV)
    df = df[
        (df["lg"] == "NBA")
        & (df["playoffs"] == True)
        & (df["season"].isin(seasons))
    ].copy()

    stat_cols = [c for c in df.columns if c not in _META_COLS]
    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["season", "abbreviation"] + stat_cols].reset_index(drop=True)


def _load_summary_stats(seasons: list[int]) -> pd.DataFrame:
    """Load Team Summaries.csv and return playoff teams for given seasons.

    Args:
        seasons: Season end-years to include.

    Returns:
        DataFrame with columns: season, abbreviation + all numeric stat columns.
    """
    df = pd.read_csv(SUMMARIES_CSV)
    df = df[
        (df["lg"] == "NBA")
        & (df["playoffs"] == True)
        & (df["season"].isin(seasons))
    ].copy()

    stat_cols = [c for c in df.columns if c not in _META_COLS]
    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["season", "abbreviation"] + stat_cols].reset_index(drop=True)


def _check_missingness(df: pd.DataFrame, stat_cols: list[str], label: str) -> None:
    """Log warnings and raise if any column exceeds MISSINGNESS_STOP_THRESHOLD.

    Args:
        df: DataFrame of team stats (one row per team per season).
        stat_cols: Feature column names to check.
        label: Human-readable source label for log messages.

    Raises:
        ValueError: If any column exceeds the missingness stop threshold.
    """
    for col in stat_cols:
        rate = df[col].isna().mean()
        if rate > MISSINGNESS_STOP_THRESHOLD:
            raise ValueError(
                f"STOP CONDITION: column '{col}' in {label} has "
                f"{rate:.1%} missing values (threshold {MISSINGNESS_STOP_THRESHOLD:.0%}). "
                "Investigate before proceeding."
            )
        if rate > 0:
            logger.warning(
                "team_ratings: %s column '%s' has %.1f%% missing values",
                label, col, rate * 100,
            )


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Attach delta team stat features to a series-level DataFrame.

    For each playoff series in df:
      1. Looks up regular-season stats for team_high and team_low.
      2. Computes delta_{stat} = stat_high − stat_low for every numeric stat.

    Columns with >20% missingness in the source file raise ValueError (stop condition).

    Args:
        df: Series-level DataFrame with columns: series_id, season,
            team_high, team_low.

    Returns:
        df with delta_{stat} columns added for all numeric team stats.
    """
    if df.empty:
        logger.warning("team_ratings.run received empty DataFrame — returning as-is.")
        return df

    seasons = sorted(df["season"].unique().tolist())
    logger.info(
        "team_ratings: loading stats for %d seasons (%d–%d)",
        len(seasons), seasons[0], seasons[-1],
    )

    per100 = _load_per100_stats(seasons)
    summaries = _load_summary_stats(seasons)

    per100_feat_cols = [c for c in per100.columns if c not in ("season", "abbreviation")]
    summ_feat_cols = [c for c in summaries.columns if c not in ("season", "abbreviation")]

    _check_missingness(per100, per100_feat_cols, "Team Stats Per 100 Poss")
    _check_missingness(summaries, summ_feat_cols, "Team Summaries")

    # Merge the two sources on (season, abbreviation) — no column name overlap expected
    overlap = set(per100_feat_cols) & set(summ_feat_cols)
    if overlap:
        logger.warning(
            "team_ratings: overlapping stat columns between sources (%s); "
            "Team Summaries values will be suffixed '_summ'.",
            overlap,
        )
    team_stats = per100.merge(
        summaries,
        on=["season", "abbreviation"],
        how="outer",
        suffixes=("", "_summ"),
    )
    all_stat_cols = [c for c in team_stats.columns if c not in ("season", "abbreviation")]

    # Vectorised approach: left-join team stats for each side, then subtract.
    # team_stats has unique (season, abbreviation) rows so both joins are 1-to-1.
    keys = ["series_id", "season"]

    high_stats = (
        df[keys + ["team_high"]]
        .merge(
            team_stats,
            left_on=["season", "team_high"],
            right_on=["season", "abbreviation"],
            how="left",
        )
        .drop(columns=["team_high", "abbreviation"])
        .set_index("series_id")
    )

    low_stats = (
        df[keys + ["team_low"]]
        .merge(
            team_stats,
            left_on=["season", "team_low"],
            right_on=["season", "abbreviation"],
            how="left",
        )
        .drop(columns=["team_low", "abbreviation"])
        .set_index("series_id")
    )

    df = df.copy().set_index("series_id")
    n_missing_teams: dict[str, int] = {}

    for col in all_stat_cols:
        delta = high_stats[col] - low_stats[col]
        df[f"delta_{col}"] = delta.values
        n_miss = int(delta.isna().sum())
        if n_miss > 0:
            n_missing_teams[col] = n_miss

    df = df.reset_index()

    if n_missing_teams:
        logger.warning(
            "team_ratings: delta columns with NaN (team lookup misses): %s",
            n_missing_teams,
        )

    logger.info(
        "team_ratings: added %d delta columns to %d series rows",
        len(all_stat_cols), len(df),
    )
    return df
