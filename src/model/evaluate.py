"""evaluate.py — Compute evaluation metrics for fitted logit models.

Metrics: McFadden pseudo-R², Brier score, AUC-ROC, log-loss, calibration error.
Ranks models and returns top-N per metric and a composite-weighted top-N.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.model.fit import ModelSpec, predict_proba

logger = logging.getLogger(__name__)

MODEL_SELECTION_CONFIG = Path("configs/model_selection.yaml")
TARGET_COL = "high_seed_wins"


def _load_ms_config() -> dict:
    with open(MODEL_SELECTION_CONFIG) as f:
        return yaml.safe_load(f)


def mcfadden_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute McFadden pseudo-R² for a logit model.

    Args:
        y_true: Binary ground-truth labels.
        y_pred: Predicted probabilities in (0, 1).

    Returns:
        McFadden pseudo-R² in [0, 1].
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    ll_model = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    base_rate = np.mean(y_true)
    ll_null = len(y_true) * (base_rate * np.log(base_rate) + (1 - base_rate) * np.log(1 - base_rate))
    return float(1.0 - ll_model / ll_null)


def max_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Compute maximum absolute calibration error across probability bins.

    Args:
        y_true: Binary ground-truth labels.
        y_pred: Predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        Maximum absolute calibration error.
    """
    fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")
    return float(np.max(np.abs(fraction_pos - mean_pred)))


def evaluate_model(
    spec: ModelSpec,
    df: pd.DataFrame,
) -> dict:
    """Compute all evaluation metrics for a single model spec.

    Args:
        spec: Fitted model specification.
        df: Series dataset filtered to the evaluation window.

    Returns:
        Dict with keys: features, window, n_obs, mcfadden_r2, brier_score,
        auc_roc, log_loss, max_cal_error.
    """
    eval_df = df.dropna(subset=spec["features"] + [TARGET_COL])
    y_true = eval_df[TARGET_COL].astype(int).values
    y_pred = predict_proba(spec, eval_df)

    return {
        "features": spec["features"],
        "window": spec["window"],
        "n_obs": len(eval_df),
        "mcfadden_r2": mcfadden_r2(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred),
        "max_cal_error": max_calibration_error(y_true, y_pred),
    }


def build_leaderboard(
    specs: list[ModelSpec],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate all model specs and build a ranked leaderboard.

    Args:
        specs: List of fitted ModelSpec dicts.
        df: Full series dataset (evaluation uses each spec's own window).

    Returns:
        DataFrame sorted by composite score descending, with a composite_score column.
    """
    ms_cfg = _load_ms_config()
    weights = ms_cfg["composite_weights"]
    top_n = ms_cfg["top_n_models"]

    rows = []
    for spec in specs:
        try:
            metrics = evaluate_model(spec, df)
            rows.append(metrics)
        except Exception as exc:
            logger.warning("Could not evaluate spec features=%s window=%s: %s",
                           spec["features"], spec["window"], exc)

    leaderboard = pd.DataFrame(rows)

    # Composite score: higher is better. Negate lower-is-better metrics.
    leaderboard["composite_score"] = (
        weights["mcfadden_r2"] * leaderboard["mcfadden_r2"]
        - weights["brier_score"] * leaderboard["brier_score"]
        + weights["auc_roc"] * leaderboard["auc_roc"]
        - weights["log_loss"] * leaderboard["log_loss"]
    )

    return leaderboard.sort_values("composite_score", ascending=False).reset_index(drop=True)
