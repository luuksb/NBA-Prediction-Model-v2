"""Export Monte Carlo simulation results to a static JSON file for the web app.

Usage:
    python scripts/export_for_web.py
    python scripts/export_for_web.py --run_id 2025_modern --output nba_results.json

Reads simulation outputs from results/simulations/<run_id>/, model spec from
results/model_selection/chosen_model_<window>.json, and bracket seeds from
configs/bracket_seeds.yaml. Writes a self-contained JSON file consumable by
the NBA_prediction_web_app frontend.

Output schema
-------------
metadata          season, n_simulations, training_window, features, model metrics
teams             per-team display info (name, colors, seed, conference)
bracket           West/East/Finals matchups with win probabilities per round
championship_probs  per-team championship probability from Monte Carlo
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.dashboard.bracket_builder import build_bracket_structure
from src.dashboard.data_loader import (
    load_bracket_seeds,
    load_model_spec,
    load_simulation_results,
    load_team_features,
)
from src.dashboard.html_renderer import TEAM_COLORS

# ---------------------------------------------------------------------------
# Team full names (display only)
# ---------------------------------------------------------------------------

_TEAM_NAMES: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


# ---------------------------------------------------------------------------
# Model metrics lookup
# ---------------------------------------------------------------------------


def load_model_metrics(features: list[str], window: str) -> dict[str, Optional[float]]:
    """Search all_models*.parquet files for metrics matching feature set and window.

    Args:
        features: Feature list from the chosen model spec.
        window: Training window identifier ('full', 'modern', 'recent').

    Returns:
        Dict with keys 'mcfadden_r2', 'brier_score', 'auc_roc'.
        Values are None if no matching row is found.
    """
    feature_set = set(features)
    pattern = str(Path("results/model_selection") / "all_models*.parquet")
    for fpath in sorted(glob.glob(pattern)):
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue
        if "window" not in df.columns or "features" not in df.columns:
            continue
        for _, row in df[df["window"] == window].iterrows():
            if set(row["features"]) == feature_set:
                return {
                    "mcfadden_r2": float(row["mcfadden_r2"]) if "mcfadden_r2" in row.index else None,
                    "brier_score": float(row["brier_score"]) if "brier_score" in row.index else None,
                    "auc_roc": float(row["auc_roc"]) if "auc_roc" in row.index else None,
                }
    return {"mcfadden_r2": None, "brier_score": None, "auc_roc": None}


# ---------------------------------------------------------------------------
# Bracket conversion helpers
# ---------------------------------------------------------------------------


def matchup_to_dict(matchup: dict[str, Any], matchup_id: str) -> dict[str, Any]:
    """Convert a MatchupNode dict to the JSON export schema.

    Args:
        matchup: Dict with 'high' and 'low' TeamNode dicts.
        matchup_id: Unique identifier string for this matchup.

    Returns:
        Dict with keys: matchup_id, top_team, bottom_team,
        top_seed, bottom_seed, top_win_prob.
    """
    hi = matchup["high"]
    lo = matchup["low"]
    return {
        "matchup_id": matchup_id,
        "top_team": hi["abbrev"],
        "bottom_team": lo["abbrev"],
        "top_seed": hi["seed"],
        "bottom_seed": lo["seed"],
        "top_win_prob": round(float(hi["cond_win_prob"]), 4),
    }


def build_bracket_json(bracket: dict[str, Any]) -> dict[str, Any]:
    """Convert a BracketStructure dict to the JSON export schema.

    Args:
        bracket: Output of build_bracket_structure() with keys
            'west', 'east', 'finals', 'champion'.

    Returns:
        Dict with 'West', 'East', 'Finals' keys matching the web schema.
        West/East each have 'R1' (4 matchups), 'R2' (2 matchups), 'CF' (1 matchup).
        Finals is a single matchup dict (not a list).
    """
    west_r1 = [matchup_to_dict(m, f"W_R1_{i+1}") for i, m in enumerate(bracket["west"][1])]
    west_r2 = [matchup_to_dict(m, f"W_R2_{i+1}") for i, m in enumerate(bracket["west"][2])]
    west_cf = [matchup_to_dict(bracket["west"][3][0], "W_CF")]

    east_r1 = [matchup_to_dict(m, f"E_R1_{i+1}") for i, m in enumerate(bracket["east"][1])]
    east_r2 = [matchup_to_dict(m, f"E_R2_{i+1}") for i, m in enumerate(bracket["east"][2])]
    east_cf = [matchup_to_dict(bracket["east"][3][0], "E_CF")]

    finals = matchup_to_dict(bracket["finals"][4][0], "Finals")

    return {
        "West": {"R1": west_r1, "R2": west_r2, "CF": west_cf},
        "East": {"R1": east_r1, "R2": east_r2, "CF": east_cf},
        "Finals": finals,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_export(output: dict[str, Any]) -> None:
    """Assert invariants on the exported dict.

    Args:
        output: The full export dict produced by export_run().

    Raises:
        AssertionError: If championship_probs don't sum to ~1.0, any bracket
            team abbreviation is absent from the teams dict, or any win_prob
            is outside [0, 1].
    """
    champ_probs: dict[str, float] = output["championship_probs"]
    prob_sum = sum(champ_probs.values())
    assert abs(prob_sum - 1.0) < 1e-2, (
        f"championship_probs sum to {prob_sum:.4f}, expected ~1.0"
    )

    teams: dict[str, Any] = output["teams"]
    bracket: dict[str, Any] = output["bracket"]
    bracket_abbrevs: set[str] = set()
    for conf in ("West", "East"):
        for rnd in ("R1", "R2", "CF"):
            for m in bracket[conf][rnd]:
                bracket_abbrevs.add(m["top_team"])
                bracket_abbrevs.add(m["bottom_team"])
    bracket_abbrevs.add(bracket["Finals"]["top_team"])
    bracket_abbrevs.add(bracket["Finals"]["bottom_team"])
    missing = bracket_abbrevs - set(teams.keys())
    assert not missing, f"Bracket teams missing from teams dict: {missing}"

    all_probs: list[float] = []
    for conf in ("West", "East"):
        for rnd in ("R1", "R2", "CF"):
            for m in bracket[conf][rnd]:
                all_probs.append(m["top_win_prob"])
    all_probs.append(bracket["Finals"]["top_win_prob"])
    out_of_range = [p for p in all_probs if not 0.0 <= p <= 1.0]
    assert not out_of_range, f"Win probs outside [0, 1]: {out_of_range}"


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export_run(run_id: str, output_path: str) -> dict[str, Any]:
    """Export one simulation run to the web JSON format.

    Args:
        run_id: Simulation run identifier, e.g. '2025_modern'.
        output_path: File path where the JSON will be written.

    Returns:
        The exported dict (also written to disk at output_path).

    Raises:
        AssertionError: If validation fails (see validate_export).
        FileNotFoundError: If the run_id directory or required files are missing.
        ValueError: If run_id format is invalid (expected '<year>_<window>').
    """
    parts = run_id.split("_", 1)
    if len(parts) != 2 or not parts[0].isdigit():
        raise ValueError(f"run_id must be '<year>_<window>', got: '{run_id}'")
    year = int(parts[0])
    window = parts[1]

    results = load_simulation_results(run_id)
    summary = results["summary"]
    adv_df = results["round_advancement"]
    champ_df = results["championship_probs"]

    seeds = load_bracket_seeds(year)
    spec = load_model_spec(window)
    team_features = load_team_features(year)
    metrics = load_model_metrics(spec["features"], window)

    bracket = build_bracket_structure(
        east_seeds=seeds["east"],
        west_seeds=seeds["west"],
        adv_df=adv_df,
        logo_url_fn=lambda _: "",
        predicted_champion=summary.get("predicted_champion"),
        team_features=team_features if not team_features.empty else None,
        spec=spec,
    )

    teams: dict[str, Any] = {}
    for conf_key, conf_label in [("east", "East"), ("west", "West")]:
        for i, abbrev in enumerate(seeds[conf_key]):
            colors = TEAM_COLORS.get(abbrev, ("#1a3a5c", "#0a1929", "#0a1929"))
            teams[abbrev] = {
                "name": _TEAM_NAMES.get(abbrev, abbrev),
                "abbreviation": abbrev,
                "conference": conf_label,
                "seed": i + 1,
                "color_primary": colors[0],
                "color_secondary": colors[1],
            }

    champ_probs = {
        str(row["team"]): float(row["championship_prob"])
        for _, row in champ_df.iterrows()
    }

    _window_labels = {"modern": "2000–2024", "full": "1980–2024", "recent": "2014–2024"}
    window_label = f"{window} ({_window_labels.get(window, window)})"

    output: dict[str, Any] = {
        "metadata": {
            "season": year,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_simulations": int(summary["n_sims"]),
            "training_window": window_label,
            "features": spec["features"],
            "pseudo_r2": round(metrics["mcfadden_r2"], 4) if metrics["mcfadden_r2"] is not None else None,
            "auc": round(metrics["auc_roc"], 4) if metrics["auc_roc"] is not None else None,
            "brier_score": round(metrics["brier_score"], 4) if metrics["brier_score"] is not None else None,
        },
        "teams": teams,
        "bracket": build_bracket_json(bracket),
        "championship_probs": champ_probs,
    }

    validate_export(output)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported {run_id} -> {output_path}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export NBA simulation results to JSON for the web app."
    )
    parser.add_argument(
        "--run_id",
        default="2025_modern",
        help="Simulation run ID, e.g. '2025_modern' (default: 2025_modern)",
    )
    parser.add_argument(
        "--output",
        default="nba_results.json",
        help="Output JSON file path (default: nba_results.json)",
    )
    args = parser.parse_args()
    export_run(args.run_id, args.output)


if __name__ == "__main__":
    main()
