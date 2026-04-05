"""report.py — Produce and save standardised simulation output reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/simulations")
WINDOWS_CONFIG = Path("configs/training_windows.yaml")


def _window_name(year: int) -> str:
    """Return 'validation' or 'prediction' label for special years, else 'historical'."""
    with open(WINDOWS_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if year == cfg.get("validation_year"):
        return "validation"
    if year == cfg.get("prediction_year"):
        return "prediction"
    return "historical"


def save_simulation_report(
    aggregated: dict,
    year: int,
    window: str,
    actual_champion: str | None = None,
    outcomes: list[dict] | None = None,
) -> Path:
    """Write simulation results to results/simulations/{year}_{window}/.

    Args:
        aggregated: Output of aggregate.aggregate_outcomes().
        year: Season year.
        window: Training window name (e.g. 'modern').
        actual_champion: Actual champion for historical/validation years.
        outcomes: Optional raw per-iteration outcome list from run_simulations().
            When provided, saved as iterations.parquet with per-sim finalist and
            injury columns for post-hoc analysis.

    Returns:
        Path to the output directory.
    """
    out_dir = RESULTS_DIR / f"{year}_{window}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    summary = {
        "year": year,
        "window": window,
        "year_type": _window_name(year),
        "n_sims": aggregated["n_sims"],
        "predicted_champion": aggregated["most_common_champion"],
        "actual_champion": actual_champion,
        "most_common_finals": list(aggregated["most_common_finals"]) if aggregated["most_common_finals"] else None,
        "championship_prob": aggregated["championship_prob"],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Round-by-round advancement parquet
    rows = []
    for team, rounds in aggregated["round_advancement"].items():
        for rnd, prob in rounds.items():
            rows.append({"team": team, "round": rnd, "advancement_prob": prob})
    adv_df = pd.DataFrame(rows)
    adv_df.to_parquet(out_dir / "round_advancement.parquet", index=False)

    # Championship probabilities parquet
    champ_df = pd.DataFrame(
        [{"team": t, "championship_prob": p} for t, p in aggregated["championship_prob"].items()]
    ).sort_values("championship_prob", ascending=False)
    champ_df.to_parquet(out_dir / "championship_probs.parquet", index=False)

    # Per-iteration parquet (finalist + injury tracking)
    if outcomes is not None:
        iter_rows = [
            {
                "iteration": o["iteration"],
                "champion": o["champion"],
                "finalist_east": o.get("finalist_east"),
                "finalist_west": o.get("finalist_west"),
                "finalist_east_injuries": o.get("finalist_east_injuries"),
                "finalist_west_injuries": o.get("finalist_west_injuries"),
            }
            for o in outcomes
        ]
        iter_df = pd.DataFrame(iter_rows)
        iter_df.to_parquet(out_dir / "iterations.parquet", index=False)
        logger.info("Per-iteration data saved to %s", out_dir / "iterations.parquet")

    logger.info("Simulation report saved to %s", out_dir)
    return out_dir
