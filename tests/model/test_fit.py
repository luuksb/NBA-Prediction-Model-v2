"""tests/model/test_fit.py — Unit tests for src/model/fit.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model.fit import ModelSpec, fit_logit, predict_proba


def _make_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = 0.5 + 1.2 * x1 - 0.8 * x2
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({
        "year": [2000 + i % 24 for i in range(n)],
        "feat1": x1,
        "feat2": x2,
        "high_seed_wins": y,
    })


class TestFitLogit:
    def test_returns_model_spec(self):
        df = _make_df()
        spec = fit_logit(df, ["feat1", "feat2"], "test_window", 2000, 2023)
        assert isinstance(spec, dict)
        assert "intercept" in spec
        assert "coefficients" in spec
        assert set(spec["features"]) == {"feat1", "feat2"}

    def test_raises_on_too_few_rows(self):
        df = _make_df(5)
        with pytest.raises(ValueError, match="too few"):
            fit_logit(df, ["feat1", "feat2"], "tiny", 2000, 2023)

    def test_n_obs_matches_window(self):
        df = _make_df(50)
        spec = fit_logit(df, ["feat1"], "recent", 2010, 2023)
        in_window = ((df["year"] >= 2010) & (df["year"] <= 2023)).sum()
        assert spec["n_obs"] <= in_window


class TestPredictProba:
    def test_output_in_unit_interval(self):
        df = _make_df()
        spec = fit_logit(df, ["feat1", "feat2"], "w", 2000, 2023)
        probs = predict_proba(spec, df)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_output_length_matches_input(self):
        df = _make_df(30)
        spec = fit_logit(df, ["feat1"], "w", 2000, 2023)
        probs = predict_proba(spec, df)
        assert len(probs) == len(df)
