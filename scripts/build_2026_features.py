#!/usr/bin/env python3
"""build_2026_features.py — Build team_season_features rows for the 2026 playoff bracket.

2026 is the prediction year: no playoff series CSV exists yet.  This script
reconstructs the 2026 per-team features from available raw data, using the
pre-drawn injury simulation results in results/injury_sims/injury_meta_2026.json
as the availability weights for bpm_avail_sum (the same role that actual
game-log availability plays for historical years).

Outputs: appends 16 rows (one per 2026 playoff team) to
         data/final/team_season_features.parquet.

Usage:
    python scripts/build_2026_features.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

BRACKET_SEEDS_CONFIG = Path("configs/bracket_seeds.yaml")
INJURY_META = Path("results/injury_sims/injury_meta_2026.json")
TEAM_FEATURES_PATH = Path("data/final/team_season_features.parquet")
PLAYOFF_SERIES_DIR = Path("data/raw/playoff_series")
ADVANCED_CSV = Path("data/raw/Advanced.csv")
EPM_PARQUET = Path("data/raw/epm.parquet")

YEAR = 2026
EPM_START = 2002


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_2026_teams() -> list[str]:
    with open(BRACKET_SEEDS_CONFIG) as f:
        cfg = yaml.safe_load(f)
    seeds = cfg["bracket_seeds"][YEAR]
    return seeds["east"] + seeds["west"]


def _load_injury_meta() -> tuple[list[str], list[list[float]], list[list[float]]]:
    """Return (teams, player_bpm, mean_rates) from injury_meta_2026.json."""
    with open(INJURY_META) as f:
        meta = json.load(f)
    return meta["teams"], meta["player_bpm"], meta["mean_rates"]


def _compute_roster_series_wins(teams: list[str]) -> dict[str, float]:
    """Compute roster-based cumulative playoff series wins entering 2026.

    Uses playoff_experience._build_roster_experience_table to sum each current
    roster player's personal prior series wins — the same logic used for
    historical years in the main data pipeline.
    """
    from src.data.steps.playoff_experience import _build_roster_experience_table

    exp = _build_roster_experience_table(YEAR)
    exp_year = exp[exp["season"] == YEAR].set_index("team")
    return {
        team: float(exp_year.at[team, "series_wins_cum"]) if team in exp_year.index else 0.0
        for team in teams
    }


def _compute_coach_cum_wins_for_2026(teams: list[str]) -> dict[str, float]:
    """Approximate coach cumulative series wins entering 2026.

    Strategy:
      - Uses coach_experience.parquet (built from coaches_nba_api/ through 2025)
        plus any 2025 playoff series results to derive the entering-2026 total.
      - For teams absent from known series data: cumulative = 0.
    """
    ce = pd.read_parquet("data/intermediate/coach_experience.parquet")

    result: dict[str, float] = {}

    for team in teams:
        all_raw = []
        for csv_path in sorted(PLAYOFF_SERIES_DIR.glob("*_nba_api.csv")):
            yr = int(csv_path.stem.split("_")[0])
            if yr > 2025:
                continue
            tmp = pd.read_csv(
                csv_path,
                usecols=["season", "series_id", "team_high", "team_low", "higher_seed_wins"],
            )
            team_rows = tmp[(tmp["team_high"] == team) | (tmp["team_low"] == team)]
            all_raw.append(team_rows)

        if not all_raw:
            result[team] = 0.0
            continue

        team_all = pd.concat(all_raw, ignore_index=True).sort_values("season")

        if team_all.empty:
            result[team] = 0.0
            continue

        last_season = int(team_all["season"].max())
        last_season_rows = team_all[team_all["season"] == last_season]
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
    star_flag:     1 × mean_rate if top player is in EPM top-5, else BPM top-5 fallback.
    """
    meta_by_team: dict[str, tuple[list[float], list[float]]] = {
        t: (player_bpm[i], mean_rates[i]) for i, t in enumerate(inj_teams)
    }

    # bpm_avail_sum directly from meta
    bpm_avail: dict[str, float] = {}
    for team in teams:
        if team in meta_by_team:
            bpms, rates = meta_by_team[team]
            bpm_avail[team] = sum(b * r for b, r in zip(bpms, rates))
        else:
            logger.warning("Team %s not found in injury meta — bpm_avail_sum=0", team)
            bpm_avail[team] = 0.0

    # Load Advanced.csv for 2026 player PER data
    adv = pd.read_csv(
        ADVANCED_CSV,
        usecols=["season", "lg", "player", "team", "g", "per", "bpm", "usg_percent", "mp"],
    )
    adv = adv[
        (adv["lg"] == "NBA") & (adv["season"] == YEAR) &
        (~adv["team"].str.match(r"^\dTM$")) & (adv["g"] >= 10)
    ].copy()
    adv["per"] = pd.to_numeric(adv["per"], errors="coerce")
    adv["bpm"] = pd.to_numeric(adv["bpm"], errors="coerce")
    adv["usg"] = pd.to_numeric(adv["usg_percent"], errors="coerce")
    adv["mpg"] = pd.to_numeric(adv["mp"], errors="coerce") / adv["g"].replace(0, float("nan"))

    # For multi-team players keep the row with the most games
    adv = (
        adv.sort_values("g", ascending=False)
        .drop_duplicates(subset=["player"])
        .reset_index(drop=True)
    )

    def _z(col: pd.Series) -> pd.Series:
        std = col.std()
        return (col - col.mean()) / std if std > 0 else pd.Series(0.0, index=col.index)

    adv_all = adv[adv["team"].isin(teams)].copy()
    adv_all["composite"] = (
        _z(adv_all["bpm"].fillna(0)) +
        _z(adv_all["usg"].fillna(0)) +
        _z(adv_all["mpg"].fillna(0))
    ) / 3.0

    # EPM top-5 for 2026 (now available after scrape)
    from src.shared.text_utils import normalise_player_name as _norm

    superstar_norm: set[str] = set()
    if EPM_PARQUET.exists():
        epm = pd.read_parquet(EPM_PARQUET, columns=["season", "player_name"])
        epm26 = epm[(epm["season"] == YEAR) & (epm["player_name"] != "Locked Player")]
        if not epm26.empty:
            superstar_norm = {_norm(n) for n in epm26["player_name"]}
            logger.info("EPM entries for 2026: %d", len(superstar_norm))

    if not superstar_norm:
        bpm_top5 = adv.nlargest(5, "bpm")["player"].map(_norm).tolist()
        superstar_norm = set(bpm_top5)
        logger.warning("EPM 2026 unavailable — using BPM top-5 fallback: %s", bpm_top5)

    per_avail: dict[str, float] = {}
    star_flag: dict[str, float] = {}

    for team in teams:
        team_players = (
            adv_all[adv_all["team"] == team]
            .nlargest(3, "composite")
            .reset_index(drop=True)
        )

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


def _load_team_stats_for_year(teams: list[str], year: int) -> pd.DataFrame:
    """Load team ratings for a given year by team abbreviation list.

    The standard build_team_stats() relies on the playoffs==True flag in the
    raw CSVs.  For 2026 that flag was only set for teams that clinched early, so
    we bypass it and filter by the known 16-team playoff bracket instead.

    Args:
        teams: List of team abbreviations (e.g. ['DET', 'BOS', ...]).
        year: Season end-year (e.g. 2026).

    Returns:
        DataFrame with columns: year, team, <all numeric stat columns>.
    """
    per100_path = Path("data/raw/Team Stats Per 100 Poss.csv")
    summaries_path = Path("data/raw/Team Summaries.csv")

    _meta = frozenset(["season", "lg", "team", "abbreviation", "playoffs", "g", "mp",
                        "pw", "pl", "arena", "attend", "attend_g"])

    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df[(df["lg"] == "NBA") & (df["season"] == year) &
                (df["abbreviation"].isin(teams))].copy()
        stat_cols = [c for c in df.columns if c not in _meta]
        for col in stat_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["season", "abbreviation"] + stat_cols].reset_index(drop=True)

    per100 = _load(per100_path)
    summaries = _load(summaries_path)
    merged = per100.merge(summaries, on=["season", "abbreviation"],
                          how="outer", suffixes=("", "_summ"))
    return merged.rename(columns={"season": "year", "abbreviation": "team"})


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== build_2026_features START ===")

    teams = _load_2026_teams()
    logger.info("2026 playoff teams (%d): %s", len(teams), teams)

    # ── 1. Team ratings (from Kaggle CSVs) ────────────────────────────────────
    # Note: build_team_stats filters playoffs==True, which only captured the 3
    # teams that clinched early in the 2026 season CSV.  For 2026 we load by
    # team abbreviation directly so all 16 playoff teams are included.
    team_stats = _load_team_stats_for_year(teams, YEAR)
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

    # ── 3. Roster-based playoff series wins entering 2026 ─────────────────────
    cum_wins = _compute_roster_series_wins(teams)
    team_stats["playoff_series_wins"] = team_stats["team"].map(cum_wins).fillna(0.0)
    logger.info("Playoff series wins computed.")

    # ── 4. Coach cumulative wins entering 2026 ────────────────────────────────
    coach_wins = _compute_coach_cum_wins_for_2026(teams)
    team_stats["coach_series_wins_cum"] = team_stats["team"].map(coach_wins).fillna(0.0)
    logger.info("Coach wins computed.")

    # ── 5. Merge with existing team_season_features ────────────────────────────
    existing = pd.read_parquet(TEAM_FEATURES_PATH)

    # Drop any stale 2026 rows (idempotent re-runs)
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
    print(f"\n{YEAR} feature values:")
    print(
        combined[combined["year"] == YEAR][
            ["team", "bpm_avail_sum", "playoff_series_wins", "ts_percent",
             "per_avail_sum", "star_flag", "coach_series_wins_cum"]
        ]
        .sort_values("bpm_avail_sum", ascending=False)
        .to_string(index=False)
    )

    logger.info("=== build_2026_features DONE ===")


if __name__ == "__main__":
    main()
