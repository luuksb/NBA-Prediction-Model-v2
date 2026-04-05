"""export.py — Export injury simulation results to results/injury_sims/.

Produces a summary (mean, median, percentiles) per team-series combination
that can be merged into the series dataset as the availability_pct feature.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INJURY_SIM_DIR = Path("results/injury_sims")


def summarise_draws(draws: np.ndarray) -> dict[str, float]:
    """Compute summary statistics from a Monte Carlo draw array.

    Args:
        draws: 1-D array of simulated availability percentages.

    Returns:
        Dict with keys: mean, median, p10, p25, p75, p90, std.
    """
    return {
        "mean": float(np.mean(draws)),
        "median": float(np.median(draws)),
        "p10": float(np.percentile(draws, 10)),
        "p25": float(np.percentile(draws, 25)),
        "p75": float(np.percentile(draws, 75)),
        "p90": float(np.percentile(draws, 90)),
        "std": float(np.std(draws)),
    }


def export_injury_sims(
    sim_records: list[dict],
    year: int,
    output_dir: Path = INJURY_SIM_DIR,
) -> Path:
    """Save injury simulation summaries to parquet.

    Args:
        sim_records: List of dicts with keys: team_id, series_id, and all
            summary statistic keys from summarise_draws().
        year: Season year (used in the output filename).
        output_dir: Parent directory for outputs.

    Returns:
        Path to the written parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"injury_sims_{year}.parquet"
    df = pd.DataFrame(sim_records)
    df.to_parquet(out_path, index=False)
    logger.info("Injury sim results for %d saved to %s  (%d rows)", year, out_path, len(df))
    return out_path


def export_injury_draws(
    draws: np.ndarray,
    meta: dict,
    year: int,
    output_dir: Path = INJURY_SIM_DIR,
) -> tuple[Path, Path]:
    """Save pre-drawn binary injury array and metadata for bracket simulation.

    Persists a uniform random array of shape (n_teams, n_stars, n_rounds, n_sims)
    to a .npy file, and a companion JSON with team ordering, per-player raw BPM
    values, and mean availability rates.

    Args:
        draws: Uniform random array of shape (n_teams, n_stars, n_rounds, n_sims).
        meta: Dict with keys 'teams' (list of team IDs), 'player_bpm'
            (n_teams × n_stars raw BPM values), 'mean_rates'
            (n_teams × n_stars historical availability rates).
        year: Season year (used in output filenames).
        output_dir: Parent directory for outputs.

    Returns:
        Tuple (npy_path, json_path) of the written files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"injury_draws_{year}.npy"
    json_path = output_dir / f"injury_meta_{year}.json"
    np.save(npy_path, draws)
    with open(json_path, "w") as f:
        json.dump(meta, f)
    logger.info(
        "Injury draws for %d saved to %s  shape=%s",
        year, npy_path, draws.shape,
    )
    return npy_path, json_path
