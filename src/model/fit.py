"""fit.py — Fit logistic regression models for each feature set × training window.

Uses statsmodels for coefficient access and sklearn for metrics compatibility.
The chosen model is serialized as a plain JSON spec (no pickle).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

logger = logging.getLogger(__name__)

WINDOWS_CONFIG = Path("configs/training_windows.yaml")
TARGET_COL = "higher_seed_wins"  # 1 if higher seed won the series, 0 otherwise


class ModelSpec(TypedDict):
    """Serialisable specification of a fitted logit model."""

    features: list[str]
    window: str
    intercept: float
    coefficients: dict[str, float]
    n_obs: int


def _load_windows() -> list[dict]:
    with open(WINDOWS_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return cfg["windows"]


def fit_logit(
    df: pd.DataFrame,
    features: list[str],
    window_name: str,
    start_year: int,
    end_year: int,
) -> ModelSpec:
    """Fit a logistic regression for a single feature set and training window.

    Args:
        df: Full series-level dataset.
        features: Feature column names to use as predictors.
        window_name: Label for the training window (e.g. 'modern').
        start_year: First year (inclusive) of the training window.
        end_year: Last year (inclusive) of the training window.

    Returns:
        ModelSpec with coefficients and metadata.

    Raises:
        ValueError: If the window contains fewer than 10 complete rows.
    """
    mask = (df["year"] >= start_year) & (df["year"] <= end_year)
    window_df = df[mask].dropna(subset=features + [TARGET_COL])

    if len(window_df) < 10:
        raise ValueError(
            f"Window {window_name!r} has only {len(window_df)} complete rows — "
            "too few to fit a reliable model."
        )

    X = sm.add_constant(window_df[features].astype(float))
    y = window_df[TARGET_COL].astype(int)

    result = sm.Logit(y, X).fit(disp=False)

    coef = result.params
    intercept = float(coef["const"])
    coefficients = {feat: float(coef[feat]) for feat in features}

    logger.info(
        "Fitted logit | window=%s | features=%s | n=%d | pseudo-R²=%.4f",
        window_name,
        features,
        len(window_df),
        result.prsquared,
    )

    return ModelSpec(
        features=features,
        window=window_name,
        intercept=intercept,
        coefficients=coefficients,
        n_obs=len(window_df),
    )


def fit_all(
    df: pd.DataFrame,
    candidate_feature_sets: list[list[str]],
) -> list[tuple[ModelSpec, object]]:
    """Fit logit models for every feature set × training window combination.

    Args:
        df: Full series-level dataset.
        candidate_feature_sets: Output of feature_sets.get_candidate_feature_sets().

    Returns:
        List of (ModelSpec, statsmodels_result) tuples, one per successful fit.
    """
    windows = _load_windows()
    results = []
    total = len(candidate_feature_sets) * len(windows)
    logger.info("Fitting %d models (%d feature sets × %d windows)…",
                total, len(candidate_feature_sets), len(windows))

    for features in candidate_feature_sets:
        for w in windows:
            try:
                spec = fit_logit(df, features, w["name"], w["start_year"], w["end_year"])
                results.append(spec)
            except Exception as exc:
                logger.warning("Skipped features=%s window=%s: %s", features, w["name"], exc)

    logger.info("Successfully fitted %d / %d models.", len(results), total)
    return results


def predict_proba(spec: ModelSpec, X: pd.DataFrame) -> np.ndarray:
    """Compute P(high seed wins) for each row using a serialised ModelSpec.

    Args:
        spec: Fitted model specification (from fit_logit or loaded from JSON).
        X: DataFrame with columns matching spec['features'].

    Returns:
        1-D numpy array of win probabilities in [0, 1].
    """
    logit = spec["intercept"] + sum(
        spec["coefficients"][feat] * X[feat].values for feat in spec["features"]
    )
    return 1.0 / (1.0 + np.exp(-logit))
