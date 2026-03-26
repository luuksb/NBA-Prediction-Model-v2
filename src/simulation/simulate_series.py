"""simulate_series.py — Simulate a single playoff series outcome.

Loads the chosen model spec from results/model_selection/chosen_model.json.
For 2025/2026, optionally draws team availability from injury sim distributions.
No imports from src.model — model spec is loaded from JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CHOSEN_MODEL_PATH = Path("results/model_selection/chosen_model.json")

_cached_spec: dict | None = None


def _load_spec() -> dict:
    global _cached_spec
    if _cached_spec is None:
        with open(CHOSEN_MODEL_PATH) as f:
            _cached_spec = json.load(f)
    return _cached_spec


def predict_win_prob(
    features_high: dict[str, float],
    features_low: dict[str, float],
    spec: dict | None = None,
) -> float:
    """Compute P(high seed wins the series) using the chosen model spec.

    Args:
        features_high: Feature values for the higher-seeded team.
        features_low: Feature values for the lower-seeded team.
        spec: Model spec dict. Defaults to loading chosen_model.json.

    Returns:
        Probability in [0, 1] that the higher seed wins.
    """
    if spec is None:
        spec = _load_spec()

    logit = spec["intercept"]
    for feat in spec["features"]:
        # Feature names in the dataset are suffixed _high / _low
        val_high = features_high.get(feat.replace("_low", "_high").replace("_high", "_high"), 0.0)
        val_low = features_low.get(feat.replace("_high", "_low").replace("_low", "_low"), 0.0)
        # Use the raw feature value — the spec was trained on paired (high, low) columns
        coef = spec["coefficients"].get(feat, 0.0)
        val = features_high.get(feat, features_low.get(feat, 0.0))
        logit += coef * val

    return float(1.0 / (1.0 + np.exp(-logit)))


def simulate_series(
    high_seed: str,
    low_seed: str,
    team_features: pd.DataFrame,
    rng: np.random.Generator,
    spec: dict | None = None,
    injury_draws: dict[str, np.ndarray] | None = None,
    draw_index: int = 0,
) -> str:
    """Simulate a single series and return the winner.

    Args:
        high_seed: Team ID for the higher seed.
        low_seed: Team ID for the lower seed.
        team_features: DataFrame indexed by team ID with feature columns.
        rng: Numpy random generator for the win/loss draw.
        spec: Model spec. Defaults to chosen_model.json.
        injury_draws: Optional dict mapping team_id → array of simulated
            availability percentages (one per bracket iteration).
        draw_index: Index into injury_draws arrays for this iteration.

    Returns:
        Team ID of the winner.
    """
    if spec is None:
        spec = _load_spec()

    row_high = team_features.loc[high_seed].to_dict() if high_seed in team_features.index else {}
    row_low = team_features.loc[low_seed].to_dict() if low_seed in team_features.index else {}

    # Inject injury availability if provided
    if injury_draws:
        if high_seed in injury_draws:
            row_high["availability_pct_high"] = injury_draws[high_seed][draw_index]
        if low_seed in injury_draws:
            row_low["availability_pct_low"] = injury_draws[low_seed][draw_index]

    p_high_wins = predict_win_prob(row_high, row_low, spec=spec)
    return high_seed if rng.random() < p_high_wins else low_seed
