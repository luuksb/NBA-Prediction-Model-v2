"""player_availability.py — Step: regular-season games-played % for top-3 players.

Computes the pre-series availability signal: what fraction of regular-season
games did each team's top-3 players (ranked by BPM) actually play?

A higher value means those stars were healthy throughout the regular season
heading into the playoffs.

Feature produced:
  top3_gp_pct_delta : (mean GP% of top-3 for high seed) − (mean GP% of top-3
                       for low seed), where GP% = player_games / team_games.

Data sources:
  data/raw/Advanced.csv        — player BPM and regular-season games played (g)
  data/raw/Team Summaries.csv  — team total games (w + l) per season

Anti-look-ahead: only the regular-season stats for the current season are used
(they are finalised before the playoffs begin).

Input df must have columns: series_id, season, team_high, team_low.
Output df has one new column: top3_gp_pct_delta.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
ADVANCED_CSV = RAW_DIR / "Advanced.csv"
TEAM_SUMMARIES_CSV = RAW_DIR / "Team Summaries.csv"
FEATURES_YAML = Path("configs/features.yaml")

# Minimum games played to be eligible as a top-3 player.
# Filters out players with only a handful of appearances whose advanced stats
# are unreliable due to small samples.
MIN_GAMES = 10


def _load_top_player_config() -> int:
    """Read n_players from configs/features.yaml top_player_ranking section."""
    if FEATURES_YAML.exists():
        with open(FEATURES_YAML, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return int(cfg.get("top_player_ranking", {}).get("n_players", 3))
    return 3


def _load_player_gp(seasons: list[int]) -> pd.DataFrame:
    """Load player games-played and BPM from Advanced.csv for the given seasons.

    Args:
        seasons: List of season end-years to include.

    Returns:
        DataFrame with columns: season, team, player_id, bpm, g (games played).
        Rows with g < MIN_GAMES are excluded.
    """
    adv = pd.read_csv(
        ADVANCED_CSV,
        usecols=["season", "lg", "player_id", "team", "g", "bpm"],
    )
    adv = adv[
        (adv["lg"] == "NBA")
        & (adv["season"].isin(seasons))
        & (adv["team"] != "TOT")  # drop totals rows for players traded mid-season
        & (adv["g"] >= MIN_GAMES)
    ].copy()
    adv["bpm"] = pd.to_numeric(adv["bpm"], errors="coerce")
    return adv[["season", "team", "player_id", "bpm", "g"]]


def _load_team_games(seasons: list[int]) -> pd.DataFrame:
    """Load total regular-season games per team from Team Summaries.csv.

    Args:
        seasons: List of season end-years to include.

    Returns:
        DataFrame with columns: season, abbreviation, team_games.
    """
    ts = pd.read_csv(
        TEAM_SUMMARIES_CSV,
        usecols=["season", "lg", "abbreviation", "w", "l"],
    )
    ts = ts[
        (ts["lg"] == "NBA") & (ts["season"].isin(seasons)) & (ts["abbreviation"].notna())
    ].copy()
    ts["team_games"] = ts["w"] + ts["l"]
    return ts[["season", "abbreviation", "team_games"]]


def _compute_top3_gp_pct(
    player_gp: pd.DataFrame,
    team_games: pd.DataFrame,
    n_players: int,
) -> pd.DataFrame:
    """Compute mean GP% of the top-N players (by BPM) per team per season.

    Args:
        player_gp: From _load_player_gp().
        team_games: From _load_team_games().
        n_players: Number of top players to average over.

    Returns:
        DataFrame with columns: season, team_abbr, top3_gp_pct.
    """
    # Attach team_games to each player row via a left join
    player_gp = player_gp.merge(
        team_games.rename(columns={"abbreviation": "team"}),
        on=["season", "team"],
        how="left",
    )
    player_gp = player_gp.dropna(subset=["team_games"])
    player_gp["gp_pct"] = player_gp["g"] / player_gp["team_games"]

    # For each team-season, take top-N players by BPM and average their GP%
    def _top_n_mean_gp(grp: pd.DataFrame) -> float:
        top = grp.nlargest(n_players, "bpm")
        return float(top["gp_pct"].mean())

    result = (
        player_gp.groupby(["season", "team"]).apply(_top_n_mean_gp).reset_index(name="top3_gp_pct")
    )
    result.rename(columns={"team": "team_abbr"}, inplace=True)
    return result


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Attach top3_gp_pct_delta to a series-level DataFrame.

    GP% is computed from regular-season stats of the current season — data that
    is fully available before the playoffs begin (no look-ahead).

    Args:
        df: Series-level DataFrame with columns: series_id, season,
            team_high, team_low.

    Returns:
        df with one new column: top3_gp_pct_delta
            = top3_gp_pct[team_high] − top3_gp_pct[team_low].
    """
    if df.empty:
        logger.warning("player_availability.run received empty DataFrame — returning as-is.")
        return df

    seasons = sorted(df["season"].unique().tolist())
    n_players = _load_top_player_config()
    logger.info(
        "Computing top-%d GP%% for %d seasons (%d-%d)",
        n_players,
        len(seasons),
        seasons[0],
        seasons[-1],
    )

    player_gp = _load_player_gp(seasons)
    team_games = _load_team_games(seasons)
    gp_pct = _compute_top3_gp_pct(player_gp, team_games, n_players)

    gp_idx = gp_pct.set_index(["season", "team_abbr"])["top3_gp_pct"].to_dict()

    def _lookup(season: int, team: str) -> float:
        return gp_idx.get((season, team), np.nan)

    df = df.copy()
    df["top3_gp_pct_high"] = [_lookup(r.season, r.team_high) for r in df.itertuples()]
    df["top3_gp_pct_low"] = [_lookup(r.season, r.team_low) for r in df.itertuples()]
    df["top3_gp_pct_delta"] = df["top3_gp_pct_high"] - df["top3_gp_pct_low"]

    # Drop the intermediate columns — only the delta is a registered feature
    df = df.drop(columns=["top3_gp_pct_high", "top3_gp_pct_low"])

    n_nan = df["top3_gp_pct_delta"].isna().sum()
    logger.info(
        "player_availability: added top3_gp_pct_delta; NaN: %d/%d rows",
        n_nan,
        len(df),
    )
    return df
