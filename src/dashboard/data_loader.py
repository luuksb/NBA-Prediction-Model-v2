"""data_loader.py — File-based data access layer for the NBA Playoff dashboard.

All functions read from disk and return plain Python structures or DataFrames.
No cross-module imports from src.model, src.simulation, src.data, or src.injury.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_CONFIG_PATH = Path("configs/dashboard.yaml")
_cfg_cache: dict[str, Any] = {}


def load_dashboard_config() -> dict[str, Any]:
    """Load and cache configs/dashboard.yaml.

    Returns:
        Parsed YAML dict with keys: paths, logos, ui.
    """
    global _cfg_cache
    if not _cfg_cache:
        with open(_CONFIG_PATH) as f:
            _cfg_cache = yaml.safe_load(f)
    return _cfg_cache


def list_available_runs() -> list[str]:
    """Return sorted list of available simulation run identifiers.

    Scans results/simulations/ for subdirectories that contain summary.json.
    Each identifier is a string like '2025_full' or '2024_modern'.

    Returns:
        List of run identifiers sorted ascending by year then window name.
        Returns empty list if the results directory does not exist.
    """
    cfg = load_dashboard_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    if not results_dir.exists():
        return []

    runs = [p.parent.name for p in results_dir.glob("*/summary.json")]

    def _sort_key(run_id: str) -> tuple[int, str]:
        parts = run_id.split("_", 1)
        year = int(parts[0]) if parts[0].isdigit() else 0
        window = parts[1] if len(parts) > 1 else ""
        return (year, window)

    return sorted(runs, key=_sort_key)


def load_simulation_results(run_id: str) -> dict[str, Any]:
    """Load all output files for one simulation run.

    Args:
        run_id: Identifier string, e.g. '2025_full'. Must correspond to a
            subdirectory of results/simulations/ containing summary.json,
            round_advancement.parquet, and championship_probs.parquet.

    Returns:
        Dict with keys:
            summary (dict): Parsed summary.json.
            round_advancement (pd.DataFrame): columns [team, round, advancement_prob].
            championship_probs (pd.DataFrame): columns [team, championship_prob].

    Raises:
        FileNotFoundError: If run_id directory or summary.json is missing.
        ValueError: If parquet files are absent.
    """
    cfg = load_dashboard_config()
    run_dir = Path(cfg["paths"]["results_dir"]) / run_id

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found for run '{run_id}'")

    with open(summary_path) as f:
        summary = json.load(f)

    adv_path = run_dir / "round_advancement.parquet"
    if not adv_path.exists():
        raise ValueError(f"round_advancement.parquet missing for run '{run_id}'")
    adv_df = pd.read_parquet(adv_path)
    adv_df["round"] = adv_df["round"].astype(int)

    champ_path = run_dir / "championship_probs.parquet"
    if not champ_path.exists():
        raise ValueError(f"championship_probs.parquet missing for run '{run_id}'")
    champ_df = pd.read_parquet(champ_path)

    # Direct matchup win rates: (team_a, team_b, round) → P(team_a wins)
    # team_a < team_b alphabetically; absent when file predates this feature.
    matchup_wins: dict[tuple[str, str, int], float | None] = {}
    mw_path = run_dir / "matchup_wins.parquet"
    if mw_path.exists():
        mw_df = pd.read_parquet(mw_path)
        for _, row in mw_df.iterrows():
            total = int(row["total"])
            key: tuple[str, str, int] = (str(row["team_a"]), str(row["team_b"]), int(row["round"]))
            matchup_wins[key] = int(row["wins_a"]) / total if total > 0 else None

    return {
        "summary": summary,
        "round_advancement": adv_df,
        "championship_probs": champ_df,
        "matchup_wins": matchup_wins,
    }


def load_model_spec(window: str) -> dict[str, Any]:
    """Load the locked model specification for a training window.

    Args:
        window: One of 'full', 'modern', 'recent'.

    Returns:
        Dict with keys: features, window, intercept, coefficients, n_obs.
        Note: Pseudo R² and AUC are NOT present in this file.

    Raises:
        FileNotFoundError: If chosen_model_{window}.json does not exist.
    """
    cfg = load_dashboard_config()
    path = Path(cfg["paths"]["model_selection_dir"]) / f"chosen_model_{window}.json"
    if not path.exists():
        raise FileNotFoundError(f"Model spec not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_team_features(year: int) -> pd.DataFrame:
    """Load per-team raw feature values for a given year.

    Reads data/final/team_season_features.parquet, filters to the target year,
    and returns a DataFrame indexed by team abbreviation.

    Args:
        year: Season year (e.g. 2025).

    Returns:
        DataFrame indexed by team abbreviation with all raw feature columns.
        Returns an empty DataFrame if the file is missing or the year has no rows.
    """
    cfg = load_dashboard_config()
    path = Path(cfg["paths"]["team_features_path"])
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    year_df = df[df["year"] == year]
    if year_df.empty:
        return pd.DataFrame()
    return year_df.set_index("team").drop(columns=["year"], errors="ignore")


def load_bracket_seeds(year: int) -> dict[str, list[str]]:
    """Return the bracket seedings for a given year.

    Reads configs/bracket_seeds.yaml. Returns a dict with keys 'east' and
    'west', each a list of 8 team abbreviations ordered 1st to 8th seed.

    Args:
        year: Season year (e.g. 2025).

    Returns:
        Dict {'east': [...], 'west': [...]} or empty dict if year not found.
    """
    cfg = load_dashboard_config()
    seeds_path = Path(cfg["paths"]["bracket_seeds_config"])
    if not seeds_path.exists():
        return {}
    with open(seeds_path) as f:
        data = yaml.safe_load(f)
    bracket_seeds = data.get("bracket_seeds", {})
    return bracket_seeds.get(year, {})


_ESPN_ABBREV_MAP: dict[str, str] = {
    # Kaggle abbrev → ESPN CDN abbrev
    "GSW": "gs",
    "NYK": "ny",
    "SAS": "sa",
    "NOP": "no",
    "NOH": "no",
    "UTA": "utah",
    "PHO": "phx",
    "WAS": "wsh",
    "BRK": "bkn",
    "CHO": "cha",
    "CHH": "cha",
    # Historical franchises (best-effort; may not exist on ESPN CDN)
    "NJN": "nj",
    "SEA": "sea",
    "WSB": "wsh",
}


def logo_url(abbrev: str) -> str:
    """Return the ESPN CDN logo URL for a team abbreviation.

    Args:
        abbrev: Team abbreviation, e.g. 'BOS'. Case-insensitive.

    Returns:
        Full URL string using the ESPN CDN template from dashboard.yaml.
    """
    cfg = load_dashboard_config()
    template = cfg["logos"]["espn_cdn_template"]
    espn_abbrev = _ESPN_ABBREV_MAP.get(abbrev.upper(), abbrev).lower()
    return template.format(abbrev=espn_abbrev)
