"""player_ratings.py — Step: per-star player stats and availability-weighted sums.

Identifies the top-N players per team using a composite ranking (BPM, usage
rate, minutes per game — equally weighted, z-score normalised) for each
playoff season, computes their regular-season stats (BPM, PER, usage rate),
looks up their actual per-series availability from game logs, and produces
availability-weighted composite features.

Ranking weights are read from configs/features.yaml (top_player_ranking section)
and mirror the logic used in src/injury/identify_top_players.py.

Data sources:
  data/raw/Advanced.csv              — BPM, PER, usage, minutes per player per season
  data/raw/PlayerStatisticsMisc.csv  — game-level playoff appearances (1997+)
  data/raw/playoff_series/           — series metadata (series_id, games_played)

Features produced (side ∈ {high, low}, r ∈ {1, 2, 3}):

  Individual star features (each metric already multiplied by avail):
    star{r}_bpm_{side}   : BPM × avail  (avail=1.0 assumed for pre-1997)
    star{r}_per_{side}   : PER × avail
    star{r}_usg_{side}   : usage rate × avail
    star{r}_avail_{side} : raw availability fraction, 0–1 (NaN pre-1997)

  Availability-weighted sums (sum of the individual weighted columns above):
    bpm_avail_sum_{side}
    per_avail_sum_{side}
    usg_avail_sum_{side}

Anti-look-ahead: regular-season stats are finalised before the playoffs begin.
Series availability is the *actual* games-played fraction for historical series.

Input df must have columns: series_id, season, team_high, team_low.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
ADVANCED_CSV = RAW_DIR / "Advanced.csv"
PLAYER_STATS_MISC_CSV = RAW_DIR / "PlayerStatisticsMisc.csv"
PLAYOFF_SERIES_DIR = RAW_DIR / "playoff_series"
FEATURES_YAML = Path("configs/features.yaml")

MIN_GAMES = 10          # minimum regular-season games to be eligible as a star
AVAIL_START_SEASON = 1997  # first season with game-level playoff data
METRICS = ("bpm", "per", "usg")


def _load_ranking_weights() -> dict[str, float]:
    """Read top_player_ranking.weights from configs/features.yaml.

    Returns:
        Dict mapping metric name → weight. Falls back to BPM-only if config
        is missing.
    """
    if FEATURES_YAML.exists():
        with open(FEATURES_YAML, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cfg.get("top_player_ranking", {}).get("weights", {"bpm": 1.0})
    return {"bpm": 1.0}


def _compute_composite_rating(
    players_df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Compute a weighted composite rating for each player row.

    Each metric is z-score normalised across all rows before weighting so that
    metrics on different scales (BPM, USG%, MPG) contribute equally when their
    weights are equal. Mirrors compute_composite_rating() in
    src/injury/identify_top_players.py.

    Args:
        players_df: DataFrame with columns for each metric in weights.
        weights: Metric name → weight mapping (need not sum to 1).

    Returns:
        Series of composite ratings aligned to players_df index.
    """
    total_weight = sum(weights.values())
    rating = pd.Series(0.0, index=players_df.index)
    for metric, weight in weights.items():
        if metric not in players_df.columns:
            logger.warning("Ranking metric %r not in players DataFrame — skipping.", metric)
            continue
        col = pd.to_numeric(players_df[metric], errors="coerce").fillna(0.0)
        std = col.std()
        z = (col - col.mean()) / std if std > 0 else pd.Series(0.0, index=players_df.index)
        rating += (weight / total_weight) * z
    return rating


def _normalise_name(name: str) -> str:
    """Lowercase, strip diacritics, replace spaces/punctuation with underscores.

    Diacritic stripping ensures names like 'Kukoč' (Advanced.csv) and 'Kukoc'
    (nba_api) normalise to the same key.
    """
    ascii_name = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def _load_n_stars() -> int:
    """Read n_players from configs/features.yaml top_player_ranking section."""
    if FEATURES_YAML.exists():
        with open(FEATURES_YAML, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return int(cfg.get("top_player_ranking", {}).get("n_players", 3))
    return 3


def _load_player_stats(seasons: list[int]) -> pd.DataFrame:
    """Load BPM, PER, usage rate, and minutes from Advanced.csv for the given seasons.

    For players traded mid-season (multiple team rows), keeps the row with
    the most games played to maximise sample reliability.

    Args:
        seasons: Season end-years to include.

    Returns:
        DataFrame with columns: season, team, player_norm, bpm, per, usg,
        usg_percent, mpg.  ``usg`` and ``usg_percent`` hold the same value;
        ``usg`` is used downstream for metric columns, ``usg_percent`` matches
        the key name in the ranking-weights config.  ``mpg`` = mp / g.
        Rows with g < MIN_GAMES are excluded.
    """
    adv = pd.read_csv(
        ADVANCED_CSV,
        usecols=["season", "lg", "player", "team", "g", "mp", "bpm", "per", "usg_percent"],
    )
    adv = adv[
        (adv["lg"] == "NBA")
        & (adv["season"].isin(seasons))
        & (adv["team"] != "TOT")
        & (adv["g"] >= MIN_GAMES)
    ].copy()

    for col in ("bpm", "per", "usg_percent", "mp"):
        adv[col] = pd.to_numeric(adv[col], errors="coerce")

    adv["usg"] = adv["usg_percent"]
    adv["mpg"] = adv["mp"] / adv["g"].replace(0, np.nan)
    adv["player_norm"] = adv["player"].map(_normalise_name)

    # For multi-team players keep the row with the most games
    adv = (
        adv.sort_values("g", ascending=False)
        .drop_duplicates(subset=["season", "player_norm"])
        .reset_index(drop=True)
    )

    return adv[["season", "team", "player_norm", "bpm", "per", "usg", "usg_percent", "mpg"]]


def _identify_top_n(player_stats: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rank the top-N players per (season, team) by composite rating.

    The composite rating is a weighted average of z-score-normalised BPM,
    usage rate, and MPG, with weights read from configs/features.yaml
    (top_player_ranking.weights).  Z-scores are computed per season so that
    each year's player pool is normalised independently, matching the
    behaviour of src/injury/identify_top_players.py.

    Args:
        player_stats: From _load_player_stats().
        n: Number of top players to retain.

    Returns:
        DataFrame with columns: season, team, star_rank, player_norm, bpm, per, usg.
        star_rank runs from 1 (highest composite rating) to N.
    """
    weights = _load_ranking_weights()

    # Z-score per season to match injury-module behaviour
    rated_seasons: list[pd.DataFrame] = []
    for _, season_df in player_stats.groupby("season"):
        season_df = season_df.copy()
        season_df["composite_rating"] = _compute_composite_rating(season_df, weights)
        rated_seasons.append(season_df)

    if not rated_seasons:
        return pd.DataFrame(
            columns=["season", "team", "star_rank", "player_norm", "bpm", "per", "usg"]
        )

    all_rated = pd.concat(rated_seasons, ignore_index=True)

    def _rank_group(grp: pd.DataFrame) -> pd.DataFrame:
        top = grp.nlargest(n, "composite_rating").copy().reset_index(drop=True)
        top["star_rank"] = range(1, len(top) + 1)
        return top

    result = (
        all_rated.groupby(["season", "team"], group_keys=False)
        .apply(_rank_group)
        .reset_index(drop=True)
    )
    return result[["season", "team", "star_rank", "player_norm", "bpm", "per", "usg"]]


def _load_series_game_counts(seasons: list[int]) -> dict[str, int]:
    """Return {series_id: games_played} for the given seasons.

    Args:
        seasons: Season end-years whose series files should be loaded.

    Returns:
        Dict mapping series_id → total games played in that series.
    """
    frames = []
    for path in sorted(PLAYOFF_SERIES_DIR.glob("*_nba_api.csv")):
        year = int(path.stem.split("_")[0])
        if year in seasons:
            frames.append(pd.read_csv(path, usecols=["series_id", "games_played"]))
    if not frames:
        return {}
    return (
        pd.concat(frames, ignore_index=True)
        .set_index("series_id")["games_played"]
        .to_dict()
    )


def _build_series_availability(
    series_df: pd.DataFrame,
) -> dict[tuple[str, str], float]:
    """Compute the fraction of series games played per player per series.

    Reads PlayerStatisticsMisc.csv (game-level; 1997+), matches each game to
    a series via (season, {team_a, team_b}), and counts distinct games per
    player.

    Args:
        series_df: Series DataFrame with series_id, season, team_high, team_low.

    Returns:
        Dict keyed by (series_id, player_norm) → avail in [0, 1].
        Only seasons >= AVAIL_START_SEASON are covered; earlier seasons are
        absent from the dict (callers should treat missing keys as NaN).
    """
    avail_seasons = [s for s in series_df["season"].unique() if s >= AVAIL_START_SEASON]
    if not avail_seasons:
        return {}

    misc = pd.read_csv(
        PLAYER_STATS_MISC_CSV,
        usecols=["gameId", "playerName", "teamAbbreviation", "gameType", "gameDate"],
    )
    playoffs = misc[misc["gameType"] == "Playoffs"].copy()
    playoffs["player_norm"] = playoffs["playerName"].map(_normalise_name)
    playoffs["season"] = pd.to_datetime(playoffs["gameDate"]).dt.year
    playoffs = playoffs[playoffs["season"].isin(avail_seasons)]

    if playoffs.empty:
        logger.warning("player_ratings: no playoff game rows found for seasons %s", avail_seasons)
        return {}

    # (season, frozenset{team_high, team_low}) → series_id
    series_key: dict[tuple, str] = {}
    for row in series_df[series_df["season"].isin(avail_seasons)].itertuples():
        key = (row.season, frozenset([row.team_high, row.team_low]))
        series_key[key] = row.series_id

    # Identify both teams per game
    game_teams = (
        playoffs.groupby(["gameId", "season"])["teamAbbreviation"]
        .apply(lambda x: frozenset(x.unique()))
        .reset_index()
    )
    game_teams.columns = ["gameId", "season", "teams"]
    game_teams["series_id"] = [
        series_key.get((r.season, r.teams)) for r in game_teams.itertuples()
    ]
    game_teams = game_teams.dropna(subset=["series_id"])

    if game_teams.empty:
        logger.warning("player_ratings: no games could be matched to series IDs.")
        return {}

    playoffs = playoffs.merge(game_teams[["gameId", "series_id"]], on="gameId", how="inner")

    # Count distinct games per (series_id, player_norm)
    appearances = (
        playoffs.groupby(["series_id", "player_norm"])["gameId"]
        .nunique()
        .reset_index()
        .rename(columns={"gameId": "games_played"})
    )

    series_games = _load_series_game_counts(avail_seasons)
    appearances["series_games"] = appearances["series_id"].map(series_games)
    appearances = appearances.dropna(subset=["series_games"])
    appearances["avail"] = (
        appearances["games_played"] / appearances["series_games"]
    ).clip(0.0, 1.0)

    return {
        (row.series_id, row.player_norm): row.avail
        for row in appearances.itertuples()
    }


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-star player ratings and availability-weighted composite features.

    For each series row:
      1. Identifies top-N players per team by composite rating (BPM, USG%, MPG).
      2. Records their BPM, PER, and usage rate as individual columns.
      3. Looks up their actual series-level availability from game logs
         (NaN for pre-1997 series).
      4. Computes availability-weighted sums: Σ(metric_r × avail_r) for r=1..N.
         When avail is NaN, full availability (1.0) is assumed for the sum.

    Args:
        df: Series-level DataFrame with columns: series_id, season,
            team_high, team_low.

    Returns:
        df with new columns:
          star{r}_avail_{side}             raw availability (NaN pre-1997)
          star{r}_{bpm,per,usg}_{side}     metric × avail (avail=1.0 when unknown)
          {bpm,per,usg}_avail_sum_{side}   sum of the three weighted star columns
    """
    if df.empty:
        logger.warning("player_ratings.run received empty DataFrame — returning as-is.")
        return df

    seasons = sorted(df["season"].unique().tolist())
    n = _load_n_stars()
    logger.info(
        "player_ratings: top-%d stars for %d seasons (%d–%d)",
        n, len(seasons), seasons[0], seasons[-1],
    )

    player_stats = _load_player_stats(seasons)
    top_n = _identify_top_n(player_stats, n)
    top_n_idx = top_n.set_index(["season", "team", "star_rank"])

    avail_lookup = _build_series_availability(df)
    logger.info(
        "player_ratings: availability entries loaded: %d (series, player) pairs",
        len(avail_lookup),
    )

    df = df.copy()

    for side, team_col in (("high", "team_high"), ("low", "team_low")):
        # Accumulate values for all ranks in one pass over df rows
        rank_cols: dict[int, dict[str, list]] = {
            r: {"bpm": [], "per": [], "usg": [], "avail": []}
            for r in range(1, n + 1)
        }

        for row in df.itertuples():
            team = getattr(row, team_col)
            season = row.season
            series_id = row.series_id

            for r in range(1, n + 1):
                try:
                    star = top_n_idx.loc[(season, team, r)]
                    bpm = float(star["bpm"]) if not pd.isna(star["bpm"]) else np.nan
                    per = float(star["per"]) if not pd.isna(star["per"]) else np.nan
                    usg = float(star["usg"]) if not pd.isna(star["usg"]) else np.nan
                    player_norm = str(star["player_norm"])
                except KeyError:
                    bpm = per = usg = np.nan
                    player_norm = ""

                avail = avail_lookup.get((series_id, player_norm), np.nan)

                rank_cols[r]["bpm"].append(bpm)
                rank_cols[r]["per"].append(per)
                rank_cols[r]["usg"].append(usg)
                rank_cols[r]["avail"].append(avail)

        for r in range(1, n + 1):
            # Raw availability (0–1 or NaN for pre-1997); stored for reference
            df[f"star{r}_avail_{side}"] = rank_cols[r]["avail"]

            # Individual metric columns are availability-weighted:
            # metric × avail, with avail=1.0 assumed when unknown (pre-1997)
            avail_s = pd.Series(rank_cols[r]["avail"], index=df.index).fillna(1.0)
            for metric, vals in (
                ("bpm", rank_cols[r]["bpm"]),
                ("per", rank_cols[r]["per"]),
                ("usg", rank_cols[r]["usg"]),
            ):
                df[f"star{r}_{metric}_{side}"] = (
                    pd.Series(vals, index=df.index) * avail_s
                )

        # Weighted sums are now just the sum of the already-weighted individual cols
        for metric in METRICS:
            df[f"{metric}_avail_sum_{side}"] = sum(
                df[f"star{r}_{metric}_{side}"] for r in range(1, n + 1)
            )

    for side in ("high", "low"):
        n_nan = df[f"star1_avail_{side}"].isna().sum()
        logger.info(
            "player_ratings: %s side — star1 avail NaN %d/%d "
            "(pre-%d series have no game-level data)",
            side, n_nan, len(df), AVAIL_START_SEASON,
        )

    return df
