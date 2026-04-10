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
EPM_PARQUET = RAW_DIR / "epm.parquet"
FEATURES_YAML = Path("configs/features.yaml")

MIN_GAMES = 10  # minimum regular-season games to be eligible as a star
AVAIL_START_SEASON = 1997  # first season with game-level playoff data
EPM_START_SEASON = 2002  # first season with EPM data
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
        # z-score normalisation: centres and scales each metric before weighting
        z = (col - col.mean()) / std if std > 0 else pd.Series(0.0, index=players_df.index)
        rating += (weight / total_weight) * z
    return rating


def _normalise_name(name: str) -> str:
    """Lowercase, strip diacritics, replace spaces/punctuation with underscores.

    Diacritic stripping ensures names like 'Kukoč' (Advanced.csv) and 'Kukoc'
    (nba_api) normalise to the same key.

    Generational suffixes (Jr., Sr., II, III, IV, V) are stripped so that
    PlayerStatisticsMisc entries like 'Jimmy Butler III' match the Advanced.csv
    entry 'Jimmy Butler'.
    """
    ascii_name = (
        unicodedata.normalize("NFKD", str(name)).encode("ascii", errors="ignore").decode("ascii")
    )
    normalised = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    return re.sub(r"_(?:jr|sr|ii|iii|iv|v)$", "", normalised)


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


def _load_epm_set(seasons: list[int]) -> set[tuple[int, str]]:
    """Return the set of (season, player_norm) pairs present in the EPM top-5 data.

    The scraped EPM file contains only the top-5 visible players per season
    (paywall limits the rest). Membership in this set is used as a binary
    superstar indicator: a team whose best player appears here has a genuine
    top-5 EPM player.

    EPM data is available from 2002 onward. Seasons outside that range yield
    no entries, so the indicator will be 0 for all pre-2002 series.

    Args:
        seasons: Season end-years to include.

    Returns:
        Set of (season, player_norm) tuples for players present in EPM data.
    """
    if not EPM_PARQUET.exists():
        logger.warning("EPM parquet not found at %s — star_epm_avail will be 0.", EPM_PARQUET)
        return set()

    epm_seasons = [s for s in seasons if s >= EPM_START_SEASON]
    if not epm_seasons:
        return set()

    epm = pd.read_parquet(EPM_PARQUET, columns=["season", "player_name"])
    epm = epm[(epm["season"].isin(epm_seasons)) & (epm["player_name"] != "Locked Player")].copy()
    epm["player_norm"] = epm["player_name"].map(_normalise_name)

    return {(int(row.season), row.player_norm) for row in epm.itertuples()}


def _load_bpm_top5_set(
    player_stats: pd.DataFrame,
    seasons: list[int],
    n_top: int = 5,
) -> set[tuple[int, str]]:
    """Return the set of (season, player_norm) for the top-N players by BPM.

    Used to backfill the superstar indicator for seasons before EPM_START_SEASON.
    Mirrors the EPM top-5 concept: a player in this set is considered a genuine
    league-best talent for that year.

    Args:
        player_stats: From _load_player_stats() — must contain season, player_norm, bpm.
        seasons: Seasons to include (typically those before EPM_START_SEASON).
        n_top: Number of top players per season to flag (default 5).

    Returns:
        Set of (season, player_norm) tuples for BPM top-N players.
    """
    subset = player_stats[player_stats["season"].isin(seasons)].copy()
    subset["bpm"] = pd.to_numeric(subset["bpm"], errors="coerce")

    result: set[tuple[int, str]] = set()
    for season, grp in subset.groupby("season"):
        top = grp.nlargest(n_top, "bpm")
        for row in top.itertuples():
            result.add((int(season), row.player_norm))
    return result


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
    return pd.concat(frames, ignore_index=True).set_index("series_id")["games_played"].to_dict()


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
    game_teams["series_id"] = [series_key.get((r.season, r.teams)) for r in game_teams.itertuples()]
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
    appearances["avail"] = (appearances["games_played"] / appearances["series_games"]).clip(
        0.0, 1.0
    )

    return {(row.series_id, row.player_norm): row.avail for row in appearances.itertuples()}


def _accumulate_rank_cols(
    df: pd.DataFrame,
    top_n_idx: pd.DataFrame,
    avail_lookup: dict,
    series_with_avail: set,
    team_col: str,
    n: int,
) -> dict[int, dict[str, list]]:
    """Accumulate per-star stats and availability for every row in df.

    For each row and each star rank 1..n, looks up the star's metrics from
    top_n_idx and their series availability from avail_lookup.

    Args:
        df: Series-level DataFrame (must have series_id, season, team_col).
        top_n_idx: DataFrame indexed by (season, team, star_rank) with columns
            bpm, per, usg, player_norm.
        avail_lookup: Dict keyed by (series_id, player_norm) → avail in [0, 1].
        series_with_avail: Set of series_ids that have any avail data; used to
            distinguish "0 games played" (0.0) from "pre-1997 unknown" (NaN).
        team_col: Column name for the team side ('team_high' or 'team_low').
        n: Number of star ranks to accumulate.

    Returns:
        Dict mapping star_rank → {'bpm', 'per', 'usg', 'avail', 'player_norm'}
        where each value is a list aligned to df.index.
    """
    rank_cols: dict[int, dict[str, list]] = {
        r: {"bpm": [], "per": [], "usg": [], "avail": [], "player_norm": []}
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
            # For series with avail data, a missing entry means 0 games played
            # (injured/DNP). For pre-1997 series (no game logs), NaN signals
            # "unknown" and is later treated as full availability.
            if series_id in series_with_avail:
                avail = avail_lookup.get((series_id, player_norm), 0.0)
            else:
                avail = avail_lookup.get((series_id, player_norm), np.nan)
            rank_cols[r]["bpm"].append(bpm)
            rank_cols[r]["per"].append(per)
            rank_cols[r]["usg"].append(usg)
            rank_cols[r]["avail"].append(avail)
            rank_cols[r]["player_norm"].append(player_norm)
    return rank_cols


def _compute_star_flags(
    df: pd.DataFrame,
    rank_cols: dict[int, dict[str, list]],
    n: int,
    superstar_set: set,
) -> list[float]:
    """Compute star_flag values for each row in df.

    star_flag = avail if any top-N player is in superstar_set, else 0.0.
    Scans stars by rank order and uses the first match's availability.
    For pre-2002 seasons (no EPM data) the value will always be 0.0.

    Args:
        df: Series-level DataFrame (must have a 'season' column).
        rank_cols: Output of _accumulate_rank_cols.
        n: Number of star ranks.
        superstar_set: Set of (season, player_norm) tuples for EPM/BPM top-5.

    Returns:
        List of float values aligned to df.index.
    """
    flag_vals: list[float] = []
    for i in range(len(df)):
        val = 0.0
        season_i = df["season"].iat[i]
        for r in range(1, n + 1):
            pnorm = rank_cols[r]["player_norm"][i]
            if (season_i, pnorm) in superstar_set:
                avail_raw = rank_cols[r]["avail"][i]
                val = 1.0 if pd.isna(avail_raw) else float(avail_raw)
                break
        flag_vals.append(val)
    return flag_vals


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
        n,
        len(seasons),
        seasons[0],
        seasons[-1],
    )

    player_stats = _load_player_stats(seasons)
    top_n = _identify_top_n(player_stats, n)
    top_n_idx = top_n.set_index(["season", "team", "star_rank"])
    avail_lookup = _build_series_availability(df)

    # Superstar set: EPM top-5 for 2002+, BPM top-5 for earlier seasons.
    pre_epm_seasons = [s for s in seasons if s < EPM_START_SEASON]
    epm_set = _load_epm_set(seasons)
    bpm_set = _load_bpm_top5_set(player_stats, pre_epm_seasons)
    superstar_set = epm_set | bpm_set
    logger.info(
        "player_ratings: superstar set — %d EPM entries (2002+), %d BPM entries (pre-2002), "
        "%d total",
        len(epm_set),
        len(bpm_set),
        len(superstar_set),
    )
    logger.info(
        "player_ratings: availability entries loaded: %d (series, player) pairs",
        len(avail_lookup),
    )

    # Series that have at least one avail entry — any star absent from these
    # played 0 games and should get avail=0.0, not NaN (which would be
    # misread as "unknown / pre-1997" and filled to 1.0 downstream).
    series_with_avail: set[str] = {sid for sid, _ in avail_lookup}

    df = df.copy()

    for side, team_col in (("high", "team_high"), ("low", "team_low")):
        rank_cols = _accumulate_rank_cols(
            df, top_n_idx, avail_lookup, series_with_avail, team_col, n
        )

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
                df[f"star{r}_{metric}_{side}"] = pd.Series(vals, index=df.index) * avail_s

        # Weighted sums are now just the sum of the already-weighted individual cols
        for metric in METRICS:
            df[f"{metric}_avail_sum_{side}"] = sum(
                df[f"star{r}_{metric}_{side}"] for r in range(1, n + 1)
            )

        # BPM-weighted availability: fraction of raw BPM quality that is available.
        # = bpm_avail_sum / sum(raw_bpm_r)  →  1.0 when all stars play fully.
        # Pre-1997: avail assumed 1.0, so this equals 1.0 for all teams.
        raw_bpm_sum = sum(
            pd.Series(rank_cols[r]["bpm"], index=df.index).fillna(0.0) for r in range(1, n + 1)
        )
        df[f"bpm_weighted_avail_{side}"] = (
            df[f"bpm_avail_sum_{side}"] / raw_bpm_sum.where(raw_bpm_sum > 0)
        ).fillna(0.0)

        # Top star's BPM × availability: isolates the best player's contribution.
        df[f"star_bpm_avail_{side}"] = df[f"star1_bpm_{side}"]

        df[f"star_flag_{side}"] = _compute_star_flags(df, rank_cols, n, superstar_set)

    for side in ("high", "low"):
        n_nan = df[f"star1_avail_{side}"].isna().sum()
        logger.info(
            "player_ratings: %s side — star1 avail NaN %d/%d "
            "(pre-%d series have no game-level data)",
            side,
            n_nan,
            len(df),
            AVAIL_START_SEASON,
        )

    return df
