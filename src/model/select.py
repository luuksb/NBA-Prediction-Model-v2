"""select.py — Lock in and serialize the chosen model specification.

After reviewing the leaderboard, this module serialises the chosen model to
results/model_selection/chosen_model.json so the simulation module can load
it without importing anything from src.model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.model.fit import ModelSpec

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/model_selection")
CHOSEN_MODEL_PATH = RESULTS_DIR / "chosen_model.json"


def save_chosen_model(spec: ModelSpec) -> Path:
    """Serialize the chosen model spec to JSON.

    Args:
        spec: Fitted model specification from fit.fit_logit().

    Returns:
        Path to the written JSON file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHOSEN_MODEL_PATH, "w") as f:
        json.dump(spec, f, indent=2)
    logger.info("Chosen model saved to %s", CHOSEN_MODEL_PATH)
    return CHOSEN_MODEL_PATH


def load_chosen_model() -> ModelSpec:
    """Load the chosen model spec from disk.

    Returns:
        ModelSpec dict loaded from results/model_selection/chosen_model.json.

    Raises:
        FileNotFoundError: If chosen_model.json does not exist.
    """
    if not CHOSEN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{CHOSEN_MODEL_PATH} not found. Run model selection first."
        )
    with open(CHOSEN_MODEL_PATH) as f:
        return json.load(f)


def select_top_model(leaderboard: pd.DataFrame) -> ModelSpec:
    """Return the top model from the leaderboard as a ModelSpec.

    Note: this function returns the spec for inspection only. Call
    save_chosen_model() explicitly to lock it in.

    Args:
        leaderboard: Ranked leaderboard from evaluate.build_leaderboard().

    Returns:
        ModelSpec for the highest-ranked model.
    """
    row = leaderboard.iloc[0]
    return ModelSpec(
        features=row["features"],
        window=row["window"],
        intercept=row.get("intercept", 0.0),
        coefficients=row.get("coefficients", {}),
        n_obs=int(row.get("n_obs", 0)),
    )
