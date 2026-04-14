"""Export Monte Carlo simulation results to static JSON files for the web app.

Usage:
    # Single run:
    python scripts/export_for_web.py --run_id 2025_modern --output nba_results.json

    # All runs (writes one JSON per run + index.json):
    python scripts/export_for_web.py --all --output_dir ../NBA_prediction_web_app/public/data/seasons

Reads simulation outputs from results/simulations/<run_id>/, model spec from
results/model_selection/chosen_model_<window>.json, and bracket seeds from
configs/bracket_seeds.yaml. Writes self-contained JSON files consumable by
the NBA_prediction_web_app frontend.

Output schema (per run)
-----------------------
metadata            season, n_simulations, training_window, features,
                    coefficients, intercept, model metrics
teams               per-team display info (name, colors, seed, conference)
bracket             West/East/Finals matchups with win probabilities per round
championship_probs  per-team championship probability from Monte Carlo
actual_champion     actual champion string (null for future years)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from src.dashboard.bracket_builder import build_bracket_structure
from src.dashboard.data_loader import (
    load_bracket_seeds,
    load_model_spec,
    load_simulation_results,
    load_team_features,
)
from src.dashboard.html_renderer import TEAM_COLORS

# ---------------------------------------------------------------------------
# Team full names (display only) — includes historical/defunct franchises
# ---------------------------------------------------------------------------

_TEAM_NAMES: dict[str, str] = {
    # Current franchises
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
    # Historical/defunct franchises
    "WSB": "Washington Bullets",
    "NJN": "New Jersey Nets",
    "SEA": "Seattle SuperSonics",
    "VAN": "Vancouver Grizzlies",
    "CHH": "Charlotte Hornets",
    "NOH": "New Orleans Hornets",
    "NOK": "New Orleans/Oklahoma City Hornets",
    "PHO": "Phoenix Suns",
    "NJA": "New Jersey Americans",
    "SDC": "San Diego Clippers",
    "KCK": "Kansas City Kings",
    "GSW": "Golden State Warriors",
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
# In-sample fit metrics
# ---------------------------------------------------------------------------


def compute_insample_fit(window: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Compute in-sample series and champion prediction accuracy for a training window.

    Uses the stored model coefficients (no refit) to compute logistic predictions
    on the training data. Champion accuracy is read from per-year simulation
    summary.json files.

    Args:
        window: Training window name ('full', 'modern', 'recent').
        spec: Model spec dict with 'features', 'coefficients', 'intercept'.

    Returns:
        Dict with keys: correct_series, total_series, correct_champs, total_champs.
        Returns zeros if series_dataset.parquet or training_windows.yaml are missing.
    """
    dataset_path = Path("data/final/series_dataset.parquet")
    windows_config = Path("configs/training_windows.yaml")
    if not dataset_path.exists() or not windows_config.exists():
        return {"correct_series": None, "total_series": None, "correct_champs": None, "total_champs": None}

    with open(windows_config) as f:
        tw_cfg = yaml.safe_load(f)
    window_row = next((w for w in tw_cfg["windows"] if w["name"] == window), None)
    if window_row is None:
        return {"correct_series": None, "total_series": None, "correct_champs": None, "total_champs": None}
    start_year, end_year = window_row["start_year"], window_row["end_year"]

    features = spec["features"]
    coefs = spec["coefficients"]
    intercept = spec["intercept"]

    df = pd.read_parquet(dataset_path)
    sub = df[(df["year"] >= start_year) & (df["year"] <= end_year)].dropna(
        subset=features + ["higher_seed_wins"]
    )

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    logit_vals = sub.apply(
        lambda row: intercept + sum(row[f] * coefs[f] for f in features), axis=1
    )
    probs = logit_vals.apply(_sigmoid)
    y = sub["higher_seed_wins"].values
    preds = (probs >= 0.5).astype(int).values
    correct_series = int((preds == y).sum())
    total_series = int(len(y))

    # Load actual champions from raw playoff CSVs (same source as the dashboard)
    _champion_overrides: dict[int, str] = {1980: "LAL", 1981: "BOS", 1982: "LAL", 1983: "PHI", 2025: "OKC"}
    actual_champions: dict[int, str] = dict(_champion_overrides)
    playoff_series_dir = Path("data/raw/playoff_series")
    if playoff_series_dir.exists():
        for csv_file in sorted(playoff_series_dir.glob("*_nba_api.csv")):
            try:
                csv_df = pd.read_csv(csv_file)
                yr = int(csv_df["season"].iloc[0])
                if yr in _champion_overrides:
                    continue
                finals_rows = csv_df[csv_df["round"] == "finals"]
                if len(finals_rows) != 1:
                    continue
                row = finals_rows.iloc[0]
                actual_champions[yr] = row["team_high"] if int(row["higher_seed_wins"]) == 1 else row["team_low"]
            except Exception:
                continue

    # Champion accuracy: compare simulation predicted_champion vs actual
    sim_dir = Path("results/simulations")
    correct_champs = 0
    total_champs = 0
    for yr in range(start_year, end_year + 1):
        summary_path = sim_dir / f"{yr}_{window}" / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            sim_summary = json.load(f)
        predicted = sim_summary.get("predicted_champion")
        actual = actual_champions.get(yr)
        if not predicted or actual is None:
            continue
        total_champs += 1
        if predicted == actual:
            correct_champs += 1

    return {
        "correct_series": correct_series,
        "total_series": total_series,
        "correct_champs": correct_champs,
        "total_champs": total_champs,
    }


# ---------------------------------------------------------------------------
# Inference statistics (z-values and p-values)
# ---------------------------------------------------------------------------


def compute_inference_stats(window: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Refit the chosen model to extract z-values and p-values per feature.

    Refits using the same data slice and features as the locked model spec.
    Uses statsmodels Logit so we get Wald z-statistics and two-sided p-values.

    Args:
        window: Training window name ('full', 'modern', 'recent').
        spec: Model spec dict with 'features', 'coefficients', 'intercept'.

    Returns:
        Dict with keys 'z_values' and 'p_values', each mapping feature name
        to float. Also includes 'intercept_z' and 'intercept_p' for the
        constant term. Returns empty dicts if data is unavailable.
    """
    dataset_path = Path("data/final/series_dataset.parquet")
    windows_config = Path("configs/training_windows.yaml")
    if not dataset_path.exists() or not windows_config.exists():
        return {"z_values": {}, "p_values": {}}

    with open(windows_config) as f:
        tw_cfg = yaml.safe_load(f)
    window_row = next((w for w in tw_cfg["windows"] if w["name"] == window), None)
    if window_row is None:
        return {"z_values": {}, "p_values": {}}

    start_year, end_year = window_row["start_year"], window_row["end_year"]
    features = spec["features"]

    df = pd.read_parquet(dataset_path)
    sub = df[(df["year"] >= start_year) & (df["year"] <= end_year)].dropna(
        subset=features + ["higher_seed_wins"]
    )

    X = sm.add_constant(sub[features].astype(float))
    y = sub["higher_seed_wins"].astype(int)

    try:
        result = sm.Logit(y, X).fit(disp=False)
    except Exception:
        return {"z_values": {}, "p_values": {}}

    z_vals = result.tvalues
    p_vals = result.pvalues

    return {
        "z_values": {feat: round(float(z_vals[feat]), 4) for feat in features},
        "p_values": {feat: round(float(p_vals[feat]), 4) for feat in features},
        "intercept_z": round(float(z_vals["const"]), 4),
        "intercept_p": round(float(p_vals["const"]), 4),
    }


# ---------------------------------------------------------------------------
# Injury impact
# ---------------------------------------------------------------------------


def load_injury_impact(run_id: str) -> Optional[dict[str, Any]]:
    """Compute injury impact statistics from a run's iterations.parquet.

    Replicates the injury impact section shown in the dashboard's left panel.
    Only meaningful for out-of-sample years where injury draws were used.

    Args:
        run_id: Simulation run identifier, e.g. '2025_modern'.

    Returns:
        Dict with keys:
            pct_finals_with_injury: fraction of Finals with at least one injured star.
            healthy_finalist_win_rate: win rate of the healthy finalist in one-sided
                injury matchups (one finalist healthy, other injured).
            pct_champ_with_injury: fraction of Finals where the champion had an injury.
        Returns None if iterations.parquet doesn't exist or has no injury data.
    """
    iter_path = Path("results/simulations") / run_id / "iterations.parquet"
    if not iter_path.exists():
        return None

    iter_df = pd.read_parquet(iter_path)
    inj_df = iter_df.dropna(
        subset=["finalist_east_injuries", "finalist_west_injuries"]
    ).copy()
    if inj_df.empty:
        return None

    # Check if injury data is actually present (non-trivial)
    if (inj_df["finalist_east_injuries"] == 0).all() and (inj_df["finalist_west_injuries"] == 0).all():
        return None

    inj_df["_east_inj"] = inj_df["finalist_east_injuries"].astype(int)
    inj_df["_west_inj"] = inj_df["finalist_west_injuries"].astype(int)
    inj_df["_total_inj"] = inj_df["_east_inj"] + inj_df["_west_inj"]
    inj_df["_champ_is_east"] = inj_df["champion"] == inj_df["finalist_east"]
    inj_df["_champ_inj"] = inj_df["_east_inj"].where(inj_df["_champ_is_east"], inj_df["_west_inj"])
    inj_df["_loser_inj"] = inj_df["_west_inj"].where(inj_df["_champ_is_east"], inj_df["_east_inj"])

    pct_any = round(float((inj_df["_total_inj"] > 0).mean()), 4)
    one_sided = inj_df[(inj_df["_champ_inj"] == 0) != (inj_df["_loser_inj"] == 0)]
    healthy_won = int(((one_sided["_champ_inj"] == 0) & (one_sided["_loser_inj"] > 0)).sum())
    healthy_win_rate = round(healthy_won / len(one_sided), 4) if len(one_sided) > 0 else 0.0
    pct_inj_champ = round(float((inj_df["_champ_inj"] > 0).mean()), 4)

    return {
        "pct_finals_with_injury": pct_any,
        "healthy_finalist_win_rate": healthy_win_rate,
        "pct_champ_with_injury": pct_inj_champ,
    }


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
    insample_fit = compute_insample_fit(window, spec)
    injury_impact = load_injury_impact(run_id)

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
            "n_obs": spec.get("n_obs"),
            "features": spec["features"],
            "coefficients": spec.get("coefficients", {}),
            "intercept": spec.get("intercept"),
            "pseudo_r2": round(metrics["mcfadden_r2"], 4) if metrics["mcfadden_r2"] is not None else None,
            "auc": round(metrics["auc_roc"], 4) if metrics["auc_roc"] is not None else None,
            "brier_score": round(metrics["brier_score"], 4) if metrics["brier_score"] is not None else None,
            "insample_fit": insample_fit,
        },
        "actual_champion": summary.get("actual_champion"),
        "teams": teams,
        "bracket": build_bracket_json(bracket),
        "championship_probs": champ_probs,
        "injury_impact": injury_impact,
    }

    validate_export(output)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported {run_id} -> {output_path}")
    return output


# ---------------------------------------------------------------------------
# Bulk export
# ---------------------------------------------------------------------------


def discover_run_ids() -> list[str]:
    """Return all available run_ids sorted by year then window.

    Returns:
        Sorted list of run_id strings (e.g. ['1986_full', '1986_modern', ...]).
    """
    sim_dir = Path("results/simulations")
    run_ids = []
    window_order = {"full": 0, "modern": 1, "recent": 2}
    for d in sim_dir.iterdir():
        if not d.is_dir():
            continue
        if not (d / "summary.json").exists():
            continue
        parts = d.name.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            run_ids.append(d.name)
    run_ids.sort(key=lambda r: (int(r.split("_")[0]), window_order.get(r.split("_", 1)[1], 99)))
    return run_ids


def export_all(output_dir: str) -> None:
    """Export all discovered simulation runs and write an index.json manifest.

    Individual run files are written to output_dir/{run_id}.json.
    An index file is written to {parent_of_output_dir}/index.json.

    Args:
        output_dir: Directory path for per-run JSON files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    run_ids = discover_run_ids()
    print(f"Found {len(run_ids)} simulation runs.")

    index_entries: list[dict[str, Any]] = []
    errors: list[str] = []

    for run_id in run_ids:
        file_path = out_path / f"{run_id}.json"
        try:
            exported = export_run(run_id, str(file_path))
            parts = run_id.split("_", 1)
            year = int(parts[0])
            window = parts[1]
            summary_path = Path("results/simulations") / run_id / "summary.json"
            with open(summary_path) as f:
                summary = json.load(f)
            index_entries.append({
                "run_id": run_id,
                "year": year,
                "window": window,
                "year_type": summary.get("year_type", "historical"),
                "predicted_champion": summary.get("predicted_champion"),
                "actual_champion": summary.get("actual_champion"),
                "file": f"seasons/{run_id}.json",
            })
        except Exception as exc:
            print(f"  ERROR exporting {run_id}: {exc}")
            errors.append(f"{run_id}: {exc}")

    index_path = out_path.parent / "index.json"
    with open(index_path, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "runs": index_entries,
            },
            f,
            indent=2,
        )
    print(f"Wrote index -> {index_path}  ({len(index_entries)} runs)")

    if errors:
        print(f"\n{len(errors)} run(s) failed:")
        for e in errors:
            print(f"  {e}")


# ---------------------------------------------------------------------------
# Model overview export (all windows in one file)
# ---------------------------------------------------------------------------


def export_model_overview(output_path: str) -> dict[str, Any]:
    """Export a model overview JSON with metrics for all three training windows.

    Produces a self-contained file with n_obs, fit metrics, and in-sample
    accuracy for each window ('full', 'modern', 'recent'). Intended as the
    nba_results.json consumed by the web app's model-info panel.

    Args:
        output_path: File path where the JSON will be written.

    Returns:
        The exported dict (also written to disk).
    """
    _window_labels = {"full": "1980–2024", "modern": "2000–2024", "recent": "2014–2024"}
    windows_out: dict[str, Any] = {}

    for window in ("full", "modern", "recent"):
        spec = load_model_spec(window)
        metrics = load_model_metrics(spec["features"], window)
        insample_fit = compute_insample_fit(window, spec)
        inference = compute_inference_stats(window, spec)
        windows_out[window] = {
            "window_span": _window_labels.get(window, window),
            "n_obs": spec.get("n_obs"),
            "features": spec["features"],
            "coefficients": spec.get("coefficients", {}),
            "intercept": spec.get("intercept"),
            "z_values": inference.get("z_values", {}),
            "p_values": inference.get("p_values", {}),
            "intercept_z": inference.get("intercept_z"),
            "intercept_p": inference.get("intercept_p"),
            "pseudo_r2": round(metrics["mcfadden_r2"], 4) if metrics["mcfadden_r2"] is not None else None,
            "auc": round(metrics["auc_roc"], 4) if metrics["auc_roc"] is not None else None,
            "brier_score": round(metrics["brier_score"], 4) if metrics["brier_score"] is not None else None,
            "insample_fit": insample_fit,
        }

    output: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_windows": windows_out,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported model overview -> {output_path}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export NBA simulation results to JSON for the web app."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run_id",
        default=None,
        help="Single simulation run ID, e.g. '2025_modern'",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Export all discovered simulation runs",
    )
    mode.add_argument(
        "--model-overview",
        action="store_true",
        help="Export model overview JSON with metrics for all three training windows",
    )
    parser.add_argument(
        "--output",
        default="nba_results.json",
        help="Output JSON file path for single-run or model-overview mode (default: nba_results.json)",
    )
    parser.add_argument(
        "--output_dir",
        default="../NBA_prediction_web_app/public/data/seasons",
        help="Output directory for --all mode (default: ../NBA_prediction_web_app/public/data/seasons)",
    )
    args = parser.parse_args()

    if args.all:
        export_all(args.output_dir)
    elif args.model_overview:
        export_model_overview(args.output)
    else:
        run_id = args.run_id or "2025_modern"
        export_run(run_id, args.output)


if __name__ == "__main__":
    main()
