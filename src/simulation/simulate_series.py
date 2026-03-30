"""simulate_series.py — Simulate a single playoff series outcome.

Loads the chosen model spec from results/model_selection/chosen_model_{window}.json.
For 2025/2026, optionally applies team availability draws from the injury simulation.
No imports from src.model — model spec is loaded from JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path("results/model_selection")

_spec_cache: dict[str, dict] = {}


def load_spec(window: str) -> dict:
    """Load and cache the chosen model spec for the given training window.

    Args:
        window: One of 'full', 'modern', 'recent'.

    Returns:
        Model spec dict with keys: features, intercept, coefficients, window, n_obs.
    """
    if window not in _spec_cache:
        path = MODEL_DIR / f"chosen_model_{window}.json"
        with open(path) as f:
            _spec_cache[window] = json.load(f)
    return _spec_cache[window]


def predict_win_prob(
    features_high: dict[str, float],
    features_low: dict[str, float],
    spec: dict,
    series_deltas: dict[str, float] | None = None,
) -> float:
    """Compute P(high seed wins the series) using the chosen model spec.

    Each feature in spec['features'] is named 'delta_X'. The delta is computed
    as features_high['X'] - features_low['X'] using the raw per-team values,
    unless a pre-computed value is provided in series_deltas (used for
    series-level features like delta_home_court_advantage).

    Args:
        features_high: Raw feature values for the higher-seeded team.
        features_low: Raw feature values for the lower-seeded team.
        spec: Model spec dict (from load_spec()).
        series_deltas: Optional pre-computed delta values keyed by full
            feature name (e.g. 'delta_home_court_advantage'). Takes
            precedence over per-team feature lookup.

    Returns:
        Probability in [0, 1] that the higher seed wins.
    """
    logit = spec["intercept"]
    for feat in spec["features"]:
        if series_deltas and feat in series_deltas:
            delta = series_deltas[feat]
        else:
            raw = feat.removeprefix("delta_")
            delta = features_high.get(raw, 0.0) - features_low.get(raw, 0.0)
        logit += spec["coefficients"][feat] * delta
    return float(1.0 / (1.0 + np.exp(-logit)))


def simulate_series(
    high_seed: str,
    low_seed: str,
    team_features: pd.DataFrame,
    rng: np.random.Generator,
    spec: dict,
    injury_draws: dict[str, np.ndarray] | None = None,
    draw_index: int = 0,
    series_deltas: dict[str, float] | None = None,
) -> str:
    """Simulate a single series and return the winner.

    Args:
        high_seed: Team ID for the higher seed.
        low_seed: Team ID for the lower seed.
        team_features: DataFrame indexed by team ID with raw feature columns.
        rng: Numpy random generator for the win/loss draw.
        spec: Model spec dict (from load_spec()).
        injury_draws: Optional dict mapping team_id → array of simulated
            availability percentages (one per bracket iteration).
        draw_index: Index into injury_draws arrays for this iteration.
        series_deltas: Optional pre-computed delta values for series-level
            features (e.g. {'delta_home_court_advantage': 1.0}). Passed
            through to predict_win_prob.

    Returns:
        Team ID of the winner.
    """
    row_high = team_features.loc[high_seed].to_dict() if high_seed in team_features.index else {}
    row_low = team_features.loc[low_seed].to_dict() if low_seed in team_features.index else {}

    # Apply injury availability: scale availability-weighted features by the draw
    if injury_draws:
        for team_id, row in ((high_seed, row_high), (low_seed, row_low)):
            if team_id in injury_draws:
                avail = float(injury_draws[team_id][draw_index])
                row = dict(row)  # avoid mutating the original
                for feat in ("bpm_avail_sum", "per_avail_sum", "usg_avail_sum"):
                    if feat in row and row[feat] is not None:
                        row[feat] = row[feat] * avail
                if team_id == high_seed:
                    row_high = row
                else:
                    row_low = row

    p_high_wins = predict_win_prob(row_high, row_low, spec=spec, series_deltas=series_deltas)
    return high_seed if rng.random() < p_high_wins else low_seed
