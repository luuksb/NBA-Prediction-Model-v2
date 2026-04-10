"""playoff_experience.py — Step: roster-based playoff series wins/played per team.

Computes, for each series in the input DataFrame, the cumulative playoff
experience of the *current roster* of both teams — defined as the sum of
all roster players' individual historical series wins/played entering
the current season.

Using roster-based experience rather than franchise totals corrects for
roster turnover: a rebuilt franchise gets credit only for what its current
players have personally won before.

History source:
  1997+ : data/raw/playoff_player_stats/{year}_playoffs.csv (exact playoff rosters)
  pre-1997 (1980–1996): data/raw/Advanced.csv regular-season roster filtered to
      teams that appeared in that year's playoff_series CSVs.

Features produced (eight columns added to df):
  playoff_series_wins_high / _low        : sum of roster players' prior series wins
  avg_playoff_series_wins_high / _low    : series wins ÷ roster count
  playoff_series_played_high / _low      : sum of roster players' prior series played
  avg_playoff_series_played_high / _low  : series played ÷ roster count

Input df must have columns: series_id, season, team_high, team_low.
Output df has all original columns plus the eight features above.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.shared.text_utils import normalise_player_name as _normalise_name

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PLAYOFF_SERIES_DIR = RAW_DIR / "playoff_series"
PLAYOFF_PLAYER_STATS_DIR = RAW_DIR / "playoff_player_stats"
ADVANCED_CSV = RAW_DIR / "Advanced.csv"

FIRST_SEASON = 1980
AVAIL_START_SEASON = 1997  # first season with playoff_player_stats data
MIN_GAMES_ROSTER = 5  # min regular-season games to count as "on roster"


# ── Team-level series stats ────────────────────────────────────────────────────


def _load_team_series_stats(up_to_season: int) -> pd.DataFrame:
    """Build (season, team, series_wins, series_played) from playoff_series CSVs.

    Returns one row per (season, team) for all teams that appeared in the
    playoffs in seasons [FIRST_SEASON, up_to_season).

    Args:
        up_to_season: Exclude this season and later (anti-look-ahead).

    Returns:
        DataFrame with columns: season, team, series_wins, series_played.
    """
    frames: list[pd.DataFrame] = []
    for path in sorted(PLAYOFF_SERIES_DIR.glob("*_nba_api.csv")):
        year = int(path.stem.split("_")[0])
        if year < FIRST_SEASON or year >= up_to_season:
            continue
        df = pd.read_csv(path, usecols=["season", "team_high", "team_low", "higher_seed_wins"])
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["season", "team", "series_wins", "series_played"])

    all_series = pd.concat(frames, ignore_index=True)

    high = all_series[["season", "team_high", "higher_seed_wins"]].copy()
    high.columns = pd.Index(["season", "team", "won"])
    high["won"] = (high["won"] == 1).astype(int)

    low = all_series[["season", "team_low", "higher_seed_wins"]].copy()
    low.columns = pd.Index(["season", "team", "won"])
    low["won"] = (low["won"] == 0).astype(int)

    per_team = pd.concat([high, low], ignore_index=True)
    per_team["played"] = 1

    result = (
        per_team.groupby(["season", "team"])
        .agg(series_wins=("won", "sum"), series_played=("played", "sum"))
        .reset_index()
    )
    return result


# ── Player playoff participation ───────────────────────────────────────────────


def _build_player_participation(up_to_season: int) -> pd.DataFrame:
    """Return (player_norm, season, team) for all playoff participants.

    1997+: uses playoff_player_stats CSVs (exact playoff rosters from nba_api).
    pre-1997 (1980–1996): uses Advanced.csv regular-season assignments filtered
        to teams that appeared in playoff_series that year.

    Args:
        up_to_season: Exclude this season and later (anti-look-ahead).

    Returns:
        DataFrame with columns: player_norm, season, team.
        One row per (player, season, team) — player was on that team's
        playoff roster in that season.
    """
    parts: list[pd.DataFrame] = []

    # ── 1997+ from playoff_player_stats ──────────────────────────────────────
    post_start = max(FIRST_SEASON, AVAIL_START_SEASON)
    for path in sorted(PLAYOFF_PLAYER_STATS_DIR.glob("*_playoffs.csv")):
        year = int(path.stem.split("_")[0])
        if year < post_start or year >= up_to_season:
            continue
        df = pd.read_csv(path, usecols=["player_name_norm", "team_abbr", "season"])
        df = df.rename(columns={"player_name_norm": "player_norm", "team_abbr": "team"})
        parts.append(df[["player_norm", "season", "team"]])

    # ── pre-1997 from Advanced.csv ────────────────────────────────────────────
    pre_end = min(AVAIL_START_SEASON, up_to_season)
    if FIRST_SEASON < pre_end:
        adv = pd.read_csv(ADVANCED_CSV, usecols=["season", "lg", "player", "team", "g"])
        adv = adv[
            (adv["lg"] == "NBA")
            & (adv["season"] >= FIRST_SEASON)
            & (adv["season"] < pre_end)
            & (adv["team"] != "TOT")
            & (pd.to_numeric(adv["g"], errors="coerce").fillna(0) >= MIN_GAMES_ROSTER)
        ].copy()
        adv["player_norm"] = adv["player"].map(_normalise_name)

        # Filter to teams that actually made the playoffs
        team_stats = _load_team_series_stats(up_to_season=pre_end)
        playoff_team_seasons = set(zip(team_stats["season"].tolist(), team_stats["team"].tolist()))
        mask = [(row.season, row.team) in playoff_team_seasons for row in adv.itertuples()]
        pre_df = adv.loc[mask, ["player_norm", "season", "team"]].copy()
        parts.append(pre_df)

    if not parts:
        return pd.DataFrame(columns=["player_norm", "season", "team"])

    participation = pd.concat(parts, ignore_index=True)
    # A player traded mid-season may appear for two teams; keep all entries —
    # they earned credit for each playoff run they participated in.
    return participation.drop_duplicates(subset=["player_norm", "season", "team"])


# ── Player-season stats ────────────────────────────────────────────────────────


def _build_player_season_stats(max_season: int) -> pd.DataFrame:
    """Build (player_norm, season, series_wins, series_played) for all seasons.

    Joins player participation with the team's series results for that season
    so each player is credited with their team's wins/played.

    Args:
        max_season: Highest season to include (inclusive).

    Returns:
        DataFrame with columns: player_norm, season, series_wins, series_played.
    """
    team_stats = _load_team_series_stats(up_to_season=max_season + 1)
    participation = _build_player_participation(up_to_season=max_season + 1)

    if participation.empty or team_stats.empty:
        return pd.DataFrame(columns=["player_norm", "season", "series_wins", "series_played"])

    merged = participation.merge(team_stats, on=["season", "team"], how="inner")

    # If a player was on two playoff teams in one season, sum both contributions.
    result = (
        merged.groupby(["player_norm", "season"])
        .agg(series_wins=("series_wins", "sum"), series_played=("series_played", "sum"))
        .reset_index()
    )
    return result


# ── Current rosters ────────────────────────────────────────────────────────────


def _build_current_rosters(max_season: int) -> pd.DataFrame:
    """Build (season, team, player_norm) for each team-season.

    Uses Advanced.csv regular-season assignments. For players traded mid-season
    (multiple team rows), assigns them to the team where they played the most
    regular-season games.

    Args:
        max_season: Highest season to include (inclusive).

    Returns:
        DataFrame with columns: season, team, player_norm.
    """
    adv = pd.read_csv(ADVANCED_CSV, usecols=["season", "lg", "player", "team", "g"])
    adv = adv[
        (adv["lg"] == "NBA")
        & (adv["season"] >= FIRST_SEASON)
        & (adv["season"] <= max_season)
        & (adv["team"] != "TOT")
        & (pd.to_numeric(adv["g"], errors="coerce").fillna(0) >= MIN_GAMES_ROSTER)
    ].copy()

    adv["player_norm"] = adv["player"].map(_normalise_name)
    adv["g"] = pd.to_numeric(adv["g"], errors="coerce").fillna(0)

    # For multi-team players, keep the team with the most games played
    adv = (
        adv.sort_values("g", ascending=False)
        .drop_duplicates(subset=["season", "player_norm"])
        .reset_index(drop=True)
    )
    return adv[["season", "team", "player_norm"]].copy()


# ── Roster experience table ────────────────────────────────────────────────────


def _build_roster_experience_table(max_season: int) -> pd.DataFrame:
    """For each (team, target_season), aggregate cumulative experience of roster.

    For each player on the current roster, sums their playoff series wins and
    series played across all seasons before the target season. The result
    captures the collective playoff pedigree of the actual players on the team.

    Args:
        max_season: Highest target season to build (inclusive).

    Returns:
        DataFrame with columns: season, team, series_wins_cum, series_played_cum,
        roster_size, avg_series_wins, avg_series_played.
    """
    player_stats = _build_player_season_stats(max_season)
    rosters = _build_current_rosters(max_season)

    if player_stats.empty or rosters.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for target_season in range(FIRST_SEASON, max_season + 1):
        # Cumulative per-player experience across all prior seasons
        prior = player_stats[player_stats["season"] < target_season]
        if prior.empty:
            player_cum: pd.DataFrame = pd.DataFrame(
                columns=["player_norm", "wins_cum", "played_cum"]
            )
        else:
            player_cum = (
                prior.groupby("player_norm")
                .agg(wins_cum=("series_wins", "sum"), played_cum=("series_played", "sum"))
                .reset_index()
            )

        # Current rosters for this season
        season_roster = rosters[rosters["season"] == target_season]
        if season_roster.empty:
            continue

        player_cum_idx = (
            player_cum.set_index("player_norm")
            if not player_cum.empty
            else pd.DataFrame(columns=["wins_cum", "played_cum"])
        )

        for team, team_players in season_roster.groupby("team"):
            player_norms = team_players["player_norm"].tolist()
            roster_size = len(player_norms)

            wins_sum = 0
            played_sum = 0
            for pn in player_norms:
                if pn in player_cum_idx.index:
                    wins_sum += int(player_cum_idx.at[pn, "wins_cum"])
                    played_sum += int(player_cum_idx.at[pn, "played_cum"])

            rows.append(
                {
                    "season": target_season,
                    "team": team,
                    "series_wins_cum": wins_sum,
                    "series_played_cum": played_sum,
                    "roster_size": roster_size,
                }
            )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["avg_series_wins"] = result["series_wins_cum"] / result["roster_size"].clip(lower=1)
    result["avg_series_played"] = result["series_played_cum"] / result["roster_size"].clip(lower=1)
    return result


# ── Public entry point ─────────────────────────────────────────────────────────


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Attach roster-based playoff experience features to a series-level DataFrame.

    For each row in df (one row = one playoff series), looks up the cumulative
    playoff series wins/played of the current roster for team_high and team_low.
    Teams whose roster players have no prior playoff history receive zeros.

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
    logger.info("Building roster-based playoff experience table up to season %d", max_season)
    exp = _build_roster_experience_table(max_season)

    if exp.empty:
        logger.warning("No playoff experience data found; all features will be zero.")
        df = df.copy()
        for side in ("high", "low"):
            for col in (
                f"playoff_series_wins_{side}",
                f"avg_playoff_series_wins_{side}",
                f"playoff_series_played_{side}",
                f"avg_playoff_series_played_{side}",
            ):
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

    logger.info("playoff_experience: added 8 roster-based features to %d series rows", len(df))
    return df
