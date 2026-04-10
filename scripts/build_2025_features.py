#!/usr/bin/env python3
"""build_2025_features.py — Build team_season_features rows for the 2025 playoff bracket.

2025 is the validation year: no playoff series CSV exists yet (the pipeline
only ingests historical series up through 2024).  This script reconstructs
the 2025 per-team features from available raw data, using the pre-drawn
injury simulation results in results/injury_sims/injury_meta_2025.json as
the availability weights for bpm_avail_sum (the same role that actual
game-log availability plays for historical years).

Outputs: appends 16 rows (one per 2025 playoff team) to
         data/final/team_season_features.parquet.

Usage:
    python scripts/build_2025_features.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

BRACKET_SEEDS_CONFIG = Path("configs/bracket_seeds.yaml")
INJURY_META = Path("results/injury_sims/injury_meta_2025.json")
TEAM_FEATURES_PATH = Path("data/final/team_season_features.parquet")
PLAYOFF_SERIES_DIR = Path("data/raw/playoff_series")
ADVANCED_CSV = Path("data/raw/Advanced.csv")
EPM_PARQUET = Path("data/raw/epm.parquet")
FEATURES_YAML = Path("configs/features.yaml")

YEAR = 2025
EPM_START = 2002


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_2025_teams() -> list[str]:
    with open(BRACKET_SEEDS_CONFIG) as f:
        cfg = yaml.safe_load(f)
    seeds = cfg["bracket_seeds"][YEAR]
    return seeds["east"] + seeds["west"]


def _load_injury_meta() -> tuple[list[str], list[list[float]], list[list[float]]]:
    """Return (teams, player_bpm, mean_rates) from injury_meta_2025.json."""
    with open(INJURY_META) as f:
        meta = json.load(f)
    return meta["teams"], meta["player_bpm"], meta["mean_rates"]


def _compute_cumulative_series_wins_through_2024() -> dict[str, int]:
    """Count each team's total playoff series wins across all seasons 1980–2024."""
    wins: dict[str, int] = {}
    for csv_path in sorted(PLAYOFF_SERIES_DIR.glob("*_nba_api.csv")):
        year = int(csv_path.stem.split("_")[0])
        if year > 2024:
            continue
        df = pd.read_csv(csv_path, usecols=["team_high", "team_low", "higher_seed_wins"])
        for _, row in df.iterrows():
            winner = row["team_high"] if int(row["higher_seed_wins"]) == 1 else row["team_low"]
            wins[winner] = wins.get(winner, 0) + 1
    return wins


def _compute_coach_cum_wins_for_2025(teams: list[str]) -> dict[str, float]:
    """Approximate coach cumulative series wins entering 2025.

    Strategy:
      - For teams in the 2024 playoffs: pre-2024 cum value + 2024 series wins.
      - For teams absent from 2024: use their pre-last-appearance cum value +
        wins from that season (the coach's record at that point).
    """
    ce = pd.read_parquet("data/intermediate/coach_experience.parquet")

    # Build (season, series_id) → (team, side, cum_wins) lookup
    df_2024 = pd.read_csv(PLAYOFF_SERIES_DIR / "2024_nba_api.csv",
                          usecols=["series_id", "team_high", "team_low", "higher_seed_wins"])

    result: dict[str, float] = {}

    for team in teams:
        # Find all appearances in coach_experience
        as_high = ce[ce["series_id"].str.contains(f"_{team}_|_{team}$", regex=True)]
        # More robust: match via series_id prefix pattern or explicit team cols aren't stored
        # Instead reconstruct from series_id names
        team_rows_h = ce[ce["series_id"].apply(
            lambda s: s.split("_")[1] == team or
            (len(s.split("_")) > 2 and s.split("_")[2] == team)
        )]
        # simpler: check if team appears as high or low from original raw CSVs
        # For robustness just scan all series
        all_raw = []
        for csv_path in sorted(PLAYOFF_SERIES_DIR.glob("*_nba_api.csv")):
            yr = int(csv_path.stem.split("_")[0])
            if yr > 2024:
                continue
            tmp = pd.read_csv(csv_path, usecols=["season", "series_id", "team_high",
                                                   "team_low", "higher_seed_wins"])
            team_rows = tmp[(tmp["team_high"] == team) | (tmp["team_low"] == team)]
            all_raw.append(team_rows)

        if not all_raw:
            result[team] = 0.0
            continue

        team_all = pd.concat(all_raw, ignore_index=True).sort_values("season")

        if team_all.empty:
            result[team] = 0.0
            continue

        # Get the most recent season this team appeared in
        last_season = int(team_all["season"].max())
        last_season_rows = team_all[team_all["season"] == last_season]

        # Get pre-last-season cum from coach_experience for first series of that season
        first_series_id = last_season_rows.iloc[0]["series_id"]
        ce_row = ce[ce["series_id"] == first_series_id]

        if ce_row.empty:
            pre_cum = 0.0
        else:
            ce_row = ce_row.iloc[0]
            if last_season_rows.iloc[0]["team_high"] == team:
                pre_cum = float(ce_row.get("coach_series_wins_cum_high", 0.0))
            else:
                pre_cum = float(ce_row.get("coach_series_wins_cum_low", 0.0))

        # Count series wins in last_season for this team
        season_wins = 0
        for _, row in last_season_rows.iterrows():
            winner = row["team_high"] if int(row["higher_seed_wins"]) == 1 else row["team_low"]
            if winner == team:
                season_wins += 1

        result[team] = pre_cum + season_wins

    return result


def _compute_player_features(
    teams: list[str],
    inj_teams: list[str],
    player_bpm: list[list[float]],
    mean_rates: list[list[float]],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return (bpm_avail_sum, per_avail_sum, star_flag) per team.

    bpm_avail_sum: computed directly from injury_meta (bpm × mean_rate for top-3).
    per_avail_sum: computed from Advanced.csv top-3 players with mean_rate weights.
    star_flag:     1 × mean_rate if top player is in EPM/BPM top-5, else 0.
    """
    # Build injury-meta lookup by team
    meta_by_team: dict[str, tuple[list[float], list[float]]] = {}
    for i, t in enumerate(inj_teams):
        meta_by_team[t] = (player_bpm[i], mean_rates[i])

    # bpm_avail_sum directly from meta
    bpm_avail: dict[str, float] = {}
    for team in teams:
        if team in meta_by_team:
            bpms, rates = meta_by_team[team]
            bpm_avail[team] = sum(b * r for b, r in zip(bpms, rates))
        else:
            logger.warning("Team %s not found in injury meta — bpm_avail_sum=0", team)
            bpm_avail[team] = 0.0

    # Load Advanced.csv for 2025 player PER data
    adv = pd.read_csv(
        ADVANCED_CSV,
        usecols=["season", "lg", "player", "team", "g", "per", "bpm", "usg_percent", "mp"],
    )
    adv = adv[
        (adv["lg"] == "NBA") & (adv["season"] == YEAR) &
        (adv["team"] != "TOT") & (adv["g"] >= 10)
    ].copy()
    adv["per"] = pd.to_numeric(adv["per"], errors="coerce")
    adv["bpm"] = pd.to_numeric(adv["bpm"], errors="coerce")
    adv["usg"] = pd.to_numeric(adv["usg_percent"], errors="coerce")
    adv["mpg"] = pd.to_numeric(adv["mp"], errors="coerce") / adv["g"].replace(0, float("nan"))

    # For multi-team players keep the row with the most games
    adv = (adv.sort_values("g", ascending=False)
           .drop_duplicates(subset=["player"])
           .reset_index(drop=True))

    # Rank top-3 per team by composite score (z-norm BPM + usg + mpg equally weighted)
    import numpy as np

    def _z(col: pd.Series) -> pd.Series:
        std = col.std()
        return (col - col.mean()) / std if std > 0 else pd.Series(0.0, index=col.index)

    adv_all = adv[adv["team"].isin(teams)].copy()
    adv_all["composite"] = (
        _z(adv_all["bpm"].fillna(0)) +
        _z(adv_all["usg"].fillna(0)) +
        _z(adv_all["mpg"].fillna(0))
    ) / 3.0

    per_avail: dict[str, float] = {}
    star_flag: dict[str, float] = {}

    # EPM / BPM top-5 set for 2025
    superstar_norm: set[str] = set()
    from src.shared.text_utils import normalise_player_name as _norm

    if EPM_PARQUET.exists():
        epm = pd.read_parquet(EPM_PARQUET, columns=["season", "player_name"])
        epm25 = epm[(epm["season"] == YEAR) & (epm["player_name"] != "Locked Player")]
        superstar_norm = {_norm(n) for n in epm25["player_name"]}
        logger.info("EPM top-5 for 2025: %d entries", len(superstar_norm))

    if not superstar_norm:
        # Fall back to BPM top-5 across all 2025 NBA players
        bpm_top5 = (adv.nlargest(5, "bpm")["player"]
                    .map(_norm).tolist())
        superstar_norm = set(bpm_top5)
        logger.info("BPM top-5 fallback for 2025: %s", bpm_top5)

    for team in teams:
        team_players = (adv_all[adv_all["team"] == team]
                        .nlargest(3, "composite")
                        .reset_index(drop=True))

        rates = meta_by_team.get(team, ([0.85, 0.85, 0.85], [0.85, 0.85, 0.85]))[1]

        # per_avail_sum
        per_sum = 0.0
        for rank, row in enumerate(team_players.itertuples()):
            rate = rates[rank] if rank < len(rates) else 0.85
            per_val = float(row.per) if not pd.isna(row.per) else 0.0
            per_sum += per_val * rate
        per_avail[team] = per_sum

        # star_flag
        flag = 0.0
        for rank, row in enumerate(team_players.itertuples()):
            if _norm(row.player) in superstar_norm:
                rate = rates[rank] if rank < len(rates) else 0.85
                flag = rate
                break
        star_flag[team] = flag

    return bpm_avail, per_avail, star_flag


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== build_2025_features START ===")

    teams = _load_2025_teams()
    logger.info("2025 playoff teams (%d): %s", len(teams), teams)

    # ── 1. Team ratings (from Kaggle CSVs) ────────────────────────────────────
    from src.data.steps.team_ratings import build_team_stats
    team_stats = build_team_stats([YEAR])
    team_stats = team_stats[team_stats["team"].isin(teams)].copy()
    team_stats = team_stats.rename(columns={"season": "year"})
    logger.info("Team ratings: %d rows", len(team_stats))

    # ── 2. Injury-meta based player features ──────────────────────────────────
    inj_teams, player_bpm, mean_rates = _load_injury_meta()
    bpm_avail, per_avail, star_flag = _compute_player_features(
        teams, inj_teams, player_bpm, mean_rates
    )

    team_stats["bpm_avail_sum"] = team_stats["team"].map(bpm_avail)
    team_stats["per_avail_sum"] = team_stats["team"].map(per_avail)
    team_stats["star_flag"] = team_stats["team"].map(star_flag)
    logger.info("Player features computed.")

    # ── 3. Playoff series wins through 2024 ───────────────────────────────────
    cum_wins = _compute_cumulative_series_wins_through_2024()
    team_stats["playoff_series_wins"] = team_stats["team"].map(cum_wins).fillna(0.0)
    logger.info("Playoff series wins computed.")

    # ── 4. Coach cumulative wins entering 2025 ────────────────────────────────
    coach_wins = _compute_coach_cum_wins_for_2025(teams)
    team_stats["coach_series_wins_cum"] = team_stats["team"].map(coach_wins).fillna(0.0)
    logger.info("Coach wins computed.")

    # ── 5. Merge with existing team_season_features ────────────────────────────
    existing = pd.read_parquet(TEAM_FEATURES_PATH)

    # Drop any stale 2025 rows (idempotent re-runs)
    existing = existing[existing["year"] != YEAR]

    # Align columns: keep only columns present in existing
    col_order = existing.columns.tolist()
    for col in col_order:
        if col not in team_stats.columns:
            team_stats[col] = float("nan")
    team_stats = team_stats[col_order]

    combined = pd.concat([existing, team_stats], ignore_index=True)
    combined.to_parquet(TEAM_FEATURES_PATH, index=False)

    logger.info(
        "Saved %d total rows to %s  (added %d rows for year %d)",
        len(combined), TEAM_FEATURES_PATH, len(team_stats), YEAR,
    )

    # Print summary
    print("\n2025 feature values:")
    print(combined[combined["year"] == YEAR][
        ["team", "bpm_avail_sum", "playoff_series_wins", "ts_percent",
         "per_avail_sum", "star_flag", "coach_series_wins_cum"]
    ].sort_values("bpm_avail_sum", ascending=False).to_string(index=False))

    logger.info("=== build_2025_features DONE ===")


if __name__ == "__main__":
    main()
