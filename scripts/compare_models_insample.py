#!/usr/bin/env python3
"""compare_models_insample.py — In-sample comparison: Locked Model (LM) vs LM + Star Flag (LM+SF).

For each of the three training windows, fits both model variants on the full window
dataset, saves the LM+SF artifact, then runs Monte Carlo bracket simulations for every
season in the window's year range.  Prints three side-by-side comparison tables.

Usage:
    python scripts/compare_models_insample.py [--n-sims N]

Forbidden: 2025 data is never loaded or touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.fit import fit_logit
from src.simulation.bracket import advance_bracket, build_bracket
from src.simulation.simulate_series import predict_win_prob

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "model_selection"
FINAL_DIR = ROOT / "data" / "final"
RAW_DIR = ROOT / "data" / "raw"

LM_FEATURES = ["delta_bpm_avail_sum", "delta_playoff_series_wins", "delta_ts_percent"]
LMSF_FEATURES = LM_FEATURES + ["delta_star_flag"]

VALIDATION_YEAR = 2025  # Sacred holdout — never touched


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_windows() -> list[dict]:
    with open(ROOT / "configs" / "training_windows.yaml") as f:
        return yaml.safe_load(f)["windows"]


def _load_series_dataset() -> pd.DataFrame:
    return pd.read_parquet(FINAL_DIR / "series_dataset.parquet")


def _load_all_team_features() -> pd.DataFrame:
    return pd.read_parquet(FINAL_DIR / "team_season_features.parquet")


def _infer_conference_split(
    csv_df: pd.DataFrame,
    year: int,
) -> tuple[set[str], set[str]]:
    """For years where conference='unknown', infer East/West by tracing the bracket.

    Starts from the two conf_finals matchups and walks backward through R2/R1
    to collect all 8 teams per conference.  Conference identity is resolved via
    a static set of historically-East franchises.

    Args:
        csv_df: Full playoff CSV for the year (all rounds).
        year: Season year (used for error messages only).

    Returns:
        (east_teams, west_teams) as sets of team abbreviations.

    Raises:
        ValueError: If exactly two conf_finals matchups cannot be found.
    """
    # Build round lookup using CSV columns (not series_id, which is alphabetical)
    rounds: dict[int, list[dict]] = {}
    for _, row in csv_df.iterrows():
        rn = int(row["round_num"])
        t_high = str(row["team_high"])
        t_low = str(row["team_low"])
        winner = t_high if int(row["higher_seed_wins"]) == 1 else t_low
        entry = {"high": t_high, "low": t_low, "winner": winner}
        rounds.setdefault(rn, []).append(entry)

    cf = rounds.get(3, [])
    if len(cf) != 2:
        raise ValueError(
            f"Year {year}: expected 2 conf_finals matchups, got {len(cf)}."
        )

    def _collect_half(cf_series: dict) -> set[str]:
        """Trace backwards from one conf_finals to gather all 8 teams."""
        teams: set[str] = {cf_series["high"], cf_series["low"]}
        if 2 in rounds:
            for s2 in rounds[2]:
                if s2["winner"] in teams:
                    teams |= {s2["high"], s2["low"]}
                    if 1 in rounds:
                        for s1 in rounds[1]:
                            if s1["winner"] in {s2["high"], s2["low"]}:
                                teams |= {s1["high"], s1["low"]}
        elif 1 in rounds:
            for s1 in rounds[1]:
                if s1["winner"] in teams:
                    teams |= {s1["high"], s1["low"]}
        return teams

    half1 = _collect_half(cf[0])
    half2 = _collect_half(cf[1])

    # Resolve which half is East using historically East-aligned franchises
    _hist_east = {
        "BOS", "NYK", "PHI", "DET", "CHI", "ATL", "MIL", "CLE", "IND", "MIA",
        "ORL", "NJN", "WSB", "WAS", "CHA", "TOR", "BKN", "NJN", "BUF",
    }
    east1 = len(half1 & _hist_east)
    east2 = len(half2 & _hist_east)

    if east1 >= east2:
        return half1, half2
    return half2, half1


def _load_bracket_seeds(year: int) -> tuple[list[str], list[str]]:
    """Return (east_seeds, west_seeds) ordered 1–8 for the given year.

    Strategy by year range:
    * 1980-1983 : only 4 R1 games — incompatible bracket format, raises ValueError.
    * 1984-2002 : 8 R1 games, conference='unknown' — inferred via bracket tracing
                  then ordered by regular-season wins.
    * 2003+     : 8 R1 games with explicit east/west labels.
    """
    # Priority: bracket_seeds.yaml (manual overrides for any year)
    seeds_yaml = ROOT / "configs" / "bracket_seeds.yaml"
    if seeds_yaml.exists():
        with open(seeds_yaml) as f:
            cfg = yaml.safe_load(f)
        year_seeds = (cfg.get("bracket_seeds") or {}).get(year)
        if year_seeds:
            return year_seeds["east"], year_seeds["west"]

    csv_path = RAW_DIR / "playoff_series" / f"{year}_nba_api.csv"
    df = pd.read_csv(csv_path)
    r1 = df[df["round_num"] == 1]

    if len(r1) < 8:
        raise ValueError(
            f"Year {year}: only {len(r1)} R1 matchups — "
            "pre-1984 format is incompatible with the 16-team bracket model."
        )

    # Build wins lookup (used for seed ordering when explicit seed cols absent)
    summaries = pd.read_csv(RAW_DIR / "Team Summaries.csv")
    summaries = summaries[(summaries["lg"] == "NBA") & (summaries["season"] == year)]
    year_wins: dict[str, int] = {
        str(row["abbreviation"]): int(row["w"]) if pd.notna(row["w"]) else 0
        for _, row in summaries.iterrows()
    }

    def _order_by_wins(teams: set[str]) -> list[str]:
        return sorted(teams, key=lambda t: (-year_wins.get(t, 0), t))

    conf_known = r1["conference"].str.lower().isin(["east", "west"]).all()

    if conf_known and r1["seed_high"].notna().all() and r1["seed_low"].notna().all():
        # Best case: explicit seeds + conference labels (2003+ typically)
        def _from_seed_cols(conf_df: pd.DataFrame) -> list[str]:
            seed_map: dict[int, str] = {}
            for _, row in conf_df.iterrows():
                seed_map[int(row["seed_high"])] = str(row["team_high"])
                seed_map[int(row["seed_low"])] = str(row["team_low"])
            return [seed_map[i] for i in range(1, 9)]

        east = _from_seed_cols(r1[r1["conference"].str.lower() == "east"])
        west = _from_seed_cols(r1[r1["conference"].str.lower() == "west"])

    elif conf_known:
        # Explicit conference labels but no seed cols — order by wins
        def _from_wins(conf_df: pd.DataFrame) -> list[str]:
            teams = set(
                [str(r["team_high"]) for _, r in conf_df.iterrows()]
                + [str(r["team_low"]) for _, r in conf_df.iterrows()]
            )
            return _order_by_wins(teams)

        east = _from_wins(r1[r1["conference"].str.lower() == "east"])
        west = _from_wins(r1[r1["conference"].str.lower() == "west"])

    else:
        # 1984-2002: infer east/west from bracket structure, order by wins
        east_set, west_set = _infer_conference_split(df, year)
        east = _order_by_wins(east_set)
        west = _order_by_wins(west_set)

    if len(east) != 8 or len(west) != 8:
        raise ValueError(
            f"Year {year}: inferred {len(east)} East / {len(west)} West teams "
            "from bracket structure — likely malformed source data."
        )
    return east, west


def _actual_champion(year: int) -> str:
    """Get actual champion for a season from the playoff CSV (team_high/low are authoritative)."""
    csv_path = RAW_DIR / "playoff_series" / f"{year}_nba_api.csv"
    if not csv_path.exists():
        return "Unknown"
    df = pd.read_csv(csv_path)
    finals = df[df["round_num"] == 4]
    if finals.empty:
        return "Unknown"
    row = finals.iloc[0]
    return str(row["team_high"]) if row["higher_seed_wins"] == 1 else str(row["team_low"])


# ── Simulation ─────────────────────────────────────────────────────────────────

def _run_bracket_sim(
    year: int,
    east_seeds: list[str],
    west_seeds: list[str],
    team_features: pd.DataFrame,
    spec: dict,
    n_sims: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Run n_sims bracket iterations and return championship probability per team.

    Optimised: feature dicts and pairwise win probabilities are pre-computed once
    per season, so predict_win_prob is never called inside the hot loop.

    Args:
        year: Season year (used to build the bracket).
        east_seeds: 8 Eastern Conference teams ordered 1–8.
        west_seeds: 8 Western Conference teams ordered 1–8.
        team_features: DataFrame indexed by team with raw feature columns.
        spec: Model spec dict (features, intercept, coefficients).
        n_sims: Monte Carlo iterations.
        rng: Numpy random generator.

    Returns:
        Dict mapping team_id → championship probability.
    """
    all_teams = east_seeds + west_seeds

    # Pre-compute feature dicts to avoid repeated .loc + .to_dict() in the loop
    feat_cache: dict[str, dict] = {
        t: (team_features.loc[t].to_dict() if t in team_features.index else {})
        for t in all_teams
    }

    # Pre-compute win probabilities for every possible matchup (deterministic)
    prob_cache: dict[tuple[str, str], float] = {}

    def _p(high: str, low: str) -> float:
        key = (high, low)
        if key not in prob_cache:
            prob_cache[key] = predict_win_prob(feat_cache[high], feat_cache[low], spec)
        return prob_cache[key]

    champion_counts: Counter = Counter()
    random_draws = rng.random(n_sims * 15)  # pre-draw enough random numbers
    draw_idx = 0

    for _ in range(n_sims):
        bracket = build_bracket(year, list(east_seeds), list(west_seeds))

        while True:
            for series in bracket.rounds[-1]:
                p = _p(series.high_seed, series.low_seed)
                series.winner = (
                    series.high_seed if random_draws[draw_idx] < p else series.low_seed
                )
                draw_idx += 1

            result = advance_bracket(bracket)
            if result is None:
                champion = bracket.rounds[-1][0].winner
                break

        champion_counts[champion] += 1

    total = sum(champion_counts.values())
    return {team: count / total for team, count in champion_counts.items()}


# ── Artifact persistence ───────────────────────────────────────────────────────

def _save_lmsf_spec(spec: dict, window: str) -> Path:
    """Serialize LM+SF model spec to results/model_selection/chosen_model_star_flag_{window}.json."""
    path = RESULTS_DIR / f"chosen_model_star_flag_{window}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
    return path


# ── Table rendering ────────────────────────────────────────────────────────────

def _print_table(
    window_name: str,
    start_year: int,
    end_year: int,
    rows: list[dict],
) -> None:
    """Print one side-by-side comparison table for a training window."""
    W = 138
    print(f"\n{'=' * W}")
    print(f"  WINDOW: {window_name.upper()}  ({start_year}-{end_year})   "
          f"[{len(rows)} seasons, n_sims per season shown in header]")
    print(f"{'=' * W}")

    hdr = (
        f"{'Season':>6}  {'Actual Champion':<16}  "
        f"{'LM: Predicted Winner':<20}  {'LM: Pred. Prob':>14}  {'LM: Actual Champ Prob':>21}  |  "
        f"{'LM+SF: Predicted Winner':<23}  {'LM+SF: Pred. Prob':>17}  {'LM+SF: Actual Champ Prob':>24}"
    )
    print(hdr)
    print("-" * W)

    for r in rows:
        disagree_flag = "  *" if r["lm_winner"] != r["lmsf_winner"] else "   "
        line = (
            f"{r['season']:>6}  {r['actual_champion']:<16}  "
            f"{r['lm_winner']:<20}  {r['lm_pred_prob']:>14.1%}  {r['lm_champ_prob']:>21.1%}  |  "
            f"{r['lmsf_winner']:<23}  {r['lmsf_pred_prob']:>17.1%}  {r['lmsf_champ_prob']:>24.1%}"
            f"{disagree_flag}"
        )
        print(line)

    print(f"\n  * = LM and LM+SF disagree on predicted winner")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(n_sims: int = 5_000) -> None:
    """Fit both model variants on all three windows, run bracket sims, print tables."""
    print(f"\nLoading series dataset and team features…")
    series_df = _load_series_dataset()
    all_team_features = _load_all_team_features()
    windows = _load_windows()

    # Confirm no 2025 data is present in the loaded dataset
    assert VALIDATION_YEAR not in series_df["year"].values, (
        f"STOP: validation year {VALIDATION_YEAR} found in series_dataset — aborting."
    )

    all_window_rows: dict[str, list[dict]] = {}

    for window in windows:
        wname = window["name"]
        start_year: int = window["start_year"]
        end_year: int = window["end_year"]

        print(f"\n{'-' * 60}")
        print(f"Window: {wname.upper()}  ({start_year}-{end_year})")
        print(f"{'-' * 60}")

        # ── Fit both models on the full window ──────────────────────────────
        print(f"  [1/3] Fitting LM   on {wname} window…")
        lm_spec = dict(fit_logit(series_df, LM_FEATURES, wname, start_year, end_year))
        print(f"        n_obs={lm_spec['n_obs']}  intercept={lm_spec['intercept']:.4f}")
        for feat, coef in lm_spec["coefficients"].items():
            print(f"          {feat}: {coef:.6f}")

        print(f"  [1/3] Fitting LM+SF on {wname} window…")
        lmsf_spec = dict(fit_logit(series_df, LMSF_FEATURES, wname, start_year, end_year))
        print(f"        n_obs={lmsf_spec['n_obs']}  intercept={lmsf_spec['intercept']:.4f}")
        for feat, coef in lmsf_spec["coefficients"].items():
            print(f"          {feat}: {coef:.6f}")

        # ── Checkpoint 1: save LM+SF artifact ──────────────────────────────
        artifact_path = _save_lmsf_spec(lmsf_spec, wname)
        print(f"\n  [2/3] Saved LM+SF artifact -> {artifact_path.relative_to(ROOT)}")

        # ── Simulate every season in this window ───────────────────────────
        season_years = sorted(
            y for y in range(start_year, end_year + 1) if y != VALIDATION_YEAR
        )
        print(f"\n  [3/3] Running bracket simulations for {len(season_years)} seasons "
              f"(n_sims={n_sims:,} each)…")

        rng_lm = np.random.default_rng(42)
        rng_lmsf = np.random.default_rng(42)

        table_rows: list[dict] = []

        for idx, year in enumerate(season_years):
            try:
                east_seeds, west_seeds = _load_bracket_seeds(year)
            except Exception as exc:
                print(f"    WARNING year {year}: could not load seeds — {exc}")
                continue

            team_feats = (
                all_team_features[all_team_features["year"] == year]
                .set_index("team")
                .drop(columns=["year"], errors="ignore")
            )
            if team_feats.empty:
                print(f"    WARNING year {year}: no team features found — skipping.")
                continue

            actual_champ = _actual_champion(year)

            lm_probs = _run_bracket_sim(
                year, east_seeds, west_seeds, team_feats, lm_spec, n_sims, rng_lm
            )
            lmsf_probs = _run_bracket_sim(
                year, east_seeds, west_seeds, team_feats, lmsf_spec, n_sims, rng_lmsf
            )

            # Validate: no NaN or Inf
            for label, probs in (("LM", lm_probs), ("LM+SF", lmsf_probs)):
                for team, p in probs.items():
                    if not np.isfinite(p):
                        raise ValueError(
                            f"STOP: {label} produced non-finite probability {p} "
                            f"for {team} in {year} ({wname} window)."
                        )

            lm_winner = max(lm_probs, key=lm_probs.get)
            lmsf_winner = max(lmsf_probs, key=lmsf_probs.get)

            table_rows.append({
                "season": year,
                "actual_champion": actual_champ,
                "lm_winner": lm_winner,
                "lm_pred_prob": lm_probs[lm_winner],
                "lm_champ_prob": lm_probs.get(actual_champ, 0.0),
                "lmsf_winner": lmsf_winner,
                "lmsf_pred_prob": lmsf_probs[lmsf_winner],
                "lmsf_champ_prob": lmsf_probs.get(actual_champ, 0.0),
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(season_years):
                print(f"    {idx + 1}/{len(season_years)} seasons done…")

        all_window_rows[wname] = table_rows

        # ── Checkpoint 3: print table for this window ───────────────────────
        _print_table(wname, start_year, end_year, table_rows)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    for wname, rows in all_window_rows.items():
        disagreements = [r for r in rows if r["lm_winner"] != r["lmsf_winner"]]
        neither_correct = [
            r for r in rows
            if r["lm_winner"] != r["actual_champion"]
            and r["lmsf_winner"] != r["actual_champion"]
        ]

        print(f"\nWindow: {wname.upper()}")
        print(f"  Seasons where LM != LM+SF predicted winner  ({len(disagreements)}):")
        if disagreements:
            for r in disagreements:
                print(
                    f"    {r['season']}  LM={r['lm_winner']} ({r['lm_pred_prob']:.1%})  "
                    f"LM+SF={r['lmsf_winner']} ({r['lmsf_pred_prob']:.1%})  "
                    f"Actual={r['actual_champion']}"
                )
        else:
            print("    (none)")

        print(f"\n  Seasons where NEITHER model predicted the actual champion  ({len(neither_correct)}):")
        if neither_correct:
            for r in neither_correct:
                print(
                    f"    {r['season']}  Actual={r['actual_champion']}  "
                    f"LM={r['lm_winner']}  LM+SF={r['lmsf_winner']}"
                )
        else:
            print("    (none)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="In-sample comparison: Locked Model vs LM + Star Flag."
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=5_000,
        help="Monte Carlo iterations per season (default: 5000).",
    )
    args = parser.parse_args()
    main(n_sims=args.n_sims)
