#!/usr/bin/env python3
"""backtest_bracket.py — Batch backtesting of bracket simulation across historical years.

For each year in a training window, runs the bracket simulation and reports:
  - Predicted champion (highest probability)
  - Actual champion
  - Probability assigned to actual champion

Usage:
    python scripts/backtest_bracket.py --window full   [--n-sims 10000] [--seed 42]
    python scripts/backtest_bracket.py --window modern [--n-sims 10000] [--seed 42]
    python scripts/backtest_bracket.py --window recent [--n-sims 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation import aggregate, run_bracket

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
FINAL_DIR = Path("data/final")
WINDOWS_CONFIG = Path("configs/training_windows.yaml")

_win_totals: pd.DataFrame | None = None


def _get_win_totals() -> pd.DataFrame:
    """Load win totals from Team Summaries CSV (cached)."""
    global _win_totals
    if _win_totals is None:
        df = pd.read_csv(RAW_DIR / "Team Summaries.csv")
        _win_totals = df[df["lg"] == "NBA"][["season", "abbreviation", "w"]].copy()
    return _win_totals


def get_actual_champion(year: int) -> str | None:
    """Return the actual NBA champion for the given year."""
    csv_path = RAW_DIR / "playoff_series" / f"{year}_nba_api.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    finals = df[df["round_num"] == 4]
    if finals.empty:
        return None
    row = finals.iloc[0]
    return str(row["team_high"]) if row["wins_high"] > row["wins_low"] else str(row["team_low"])


def load_bracket_seeds(year: int) -> tuple[list[str], list[str]]:
    """Load east/west seeds ordered 1–8, inferred from wins when seed columns are absent.

    For years where conference labels are 'unknown' (pre-2003), groups teams into
    two conferences by tracing shared pre-Finals series, then ranks by wins.
    The East/West label may be swapped for those years, but champion accuracy
    is unaffected since each group simulates independently.
    """
    csv_path = RAW_DIR / "playoff_series" / f"{year}_nba_api.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No playoff series CSV for {year}: {csv_path}")

    df = pd.read_csv(csv_path)
    r1 = df[df["round_num"] == 1]

    wins = _get_win_totals()
    wdf = wins[wins["season"] == year]
    year_wins: dict[str, int] = {
        str(row["abbreviation"]): int(row["w"]) if pd.notna(row["w"]) else 0
        for _, row in wdf.iterrows()
    }

    def _sort_by_wins(teams: list[str]) -> list[str]:
        """Sort teams by wins descending (seed 1 = most wins); ties broken alphabetically."""
        return sorted(teams, key=lambda t: (-year_wins.get(t, 0), t))

    # If conference labels are available, use them directly
    if set(r1["conference"].unique()) <= {"east", "west"}:

        def _conf_teams(conf: str) -> list[str]:
            rows = r1[r1["conference"] == conf]
            teams = [str(r["team_high"]) for _, r in rows.iterrows()] + [
                str(r["team_low"]) for _, r in rows.iterrows()
            ]
            if len(teams) != 8:
                raise ValueError(f"Expected 8 teams for '{conf}', got {len(teams)}: {teams}")
            return _sort_by_wins(teams)

        return _conf_teams("east"), _conf_teams("west")

    # Unknown conference labels — infer two groups via transitive closure on pre-Finals series
    pre_finals = df[df["round_num"] < 4]
    adjacency: dict[str, set[str]] = {}
    for _, row in pre_finals.iterrows():
        a, b = str(row["team_high"]), str(row["team_low"])
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    all_teams = list(adjacency.keys())
    visited: set[str] = set()
    groups: list[set[str]] = []
    for start in all_teams:
        if start in visited:
            continue
        group: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            group.add(node)
            stack.extend(adjacency.get(node, set()) - visited)
        groups.append(group)

    if len(groups) != 2:
        raise ValueError(f"Expected 2 conference groups for {year}, found {len(groups)}: {groups}")

    group_a = _sort_by_wins(list(groups[0]))
    group_b = _sort_by_wins(list(groups[1]))

    if len(group_a) != 8 or len(group_b) != 8:
        raise ValueError(
            f"Expected 8 teams per conference for {year}, "
            f"got {len(group_a)} and {len(group_b)} — non-standard bracket format."
        )

    return group_a, group_b


def load_team_features(year: int) -> pd.DataFrame:
    """Load team features for the given year."""
    path = FINAL_DIR / "team_season_features.parquet"
    df = pd.read_parquet(path)
    year_df = df[df["year"] == year]
    if year_df.empty:
        raise ValueError(f"No team features for year {year}")
    return year_df.set_index("team").drop(columns=["year"], errors="ignore")


def get_window_years(window: str) -> list[int]:
    """Return the list of years (end-years) covered by the given training window."""
    with open(WINDOWS_CONFIG) as f:
        config = yaml.safe_load(f)
    for w in config["windows"]:
        if w["name"] == window:
            return list(range(w["start_year"], w["end_year"] + 1))
    raise ValueError(f"Unknown window '{window}'")


def run_year(
    year: int,
    window: str,
    n_sims: int,
    seed: int | None,
) -> dict | None:
    """Simulate one year and return summary dict, or None if data is missing."""
    try:
        team_features = load_team_features(year)
    except ValueError:
        return None

    try:
        east_seeds, west_seeds = load_bracket_seeds(year)
    except (FileNotFoundError, ValueError) as e:
        logger.warning("Seed loading failed for %d: %s — skipping", year, e)
        return None

    actual_champion = get_actual_champion(year)

    try:
        outcomes = run_bracket.run_simulations(
            year=year,
            east_seeds=east_seeds,
            west_seeds=west_seeds,
            team_features=team_features,
            window=window,
            n_sims=n_sims,
            seed=seed,
        )
    except Exception as e:
        logger.warning("Simulation failed for %d: %s — skipping", year, e)
        return None

    all_teams = east_seeds + west_seeds
    aggregated = aggregate.aggregate_outcomes(outcomes, all_teams)
    champ_probs: dict[str, float] = aggregated["championship_prob"]

    predicted_champion = max(champ_probs, key=champ_probs.get)
    predicted_prob = round(champ_probs[predicted_champion] * 100, 1)
    actual_prob = round(champ_probs.get(actual_champion, 0.0) * 100, 1) if actual_champion else None
    correct = predicted_champion == actual_champion

    return {
        "year": year,
        "predicted_champion": predicted_champion,
        "predicted_prob_pct": predicted_prob,
        "actual_champion": actual_champion,
        "actual_champion_prob_pct": actual_prob,
        "correct": correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest bracket simulation over historical years."
    )
    parser.add_argument("--window", required=True, help="Training window: full, modern, or recent")
    parser.add_argument(
        "--n-sims", type=int, default=10_000, help="Monte Carlo iterations per year"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    years = get_window_years(args.window)
    print(
        f"\nBacktesting window='{args.window}' ({years[0]}–{years[-1]}), n_sims={args.n_sims:,}\n"
    )

    rows = []
    for year in years:
        result = run_year(year, args.window, args.n_sims, args.seed)
        if result is None:
            continue
        rows.append(result)
        marker = "OK" if result["correct"] else "--"
        print(
            f"  {year}  predicted={result['predicted_champion']} ({result['predicted_prob_pct']}%)  "
            f"actual={result['actual_champion']} ({result['actual_champion_prob_pct']}%)  {marker}"
        )

    if not rows:
        print("No results — check data availability.")
        return

    df = pd.DataFrame(rows)
    accuracy = df["correct"].mean() * 100
    avg_actual_prob = df["actual_champion_prob_pct"].mean()

    print(f"\n{'='*70}")
    print(
        f"Window: {args.window}  |  Years: {len(df)}  |  "
        f"Accuracy: {accuracy:.1f}%  |  Avg P(actual winner): {avg_actual_prob:.1f}%"
    )
    print(f"{'='*70}\n")

    out_dir = Path("results/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_{args.window}.csv"
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
