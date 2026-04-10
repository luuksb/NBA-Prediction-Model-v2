"""Tests for src/simulation/simulate_series.py."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.simulation.simulate_series import load_spec, predict_win_prob, simulate_series


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_SPEC = {
    "features": ["delta_bpm_avail_sum", "delta_playoff_series_wins", "delta_ts_percent"],
    "window": "full",
    "intercept": 1.0,
    "coefficients": {
        "delta_bpm_avail_sum": 0.1,
        "delta_playoff_series_wins": 0.02,
        "delta_ts_percent": 10.0,
    },
    "n_obs": 100,
}

FEATURES_HIGH = {"bpm_avail_sum": 5.0, "playoff_series_wins": 20.0, "ts_percent": 0.58}
FEATURES_LOW = {"bpm_avail_sum": 2.0, "playoff_series_wins": 10.0, "ts_percent": 0.55}


# ── predict_win_prob ───────────────────────────────────────────────────────────

class TestPredictWinProb:
    def test_output_in_unit_interval(self):
        prob = predict_win_prob(FEATURES_HIGH, FEATURES_LOW, spec=SAMPLE_SPEC)
        assert 0.0 < prob < 1.0

    def test_known_value(self):
        # delta_bpm = 3.0, delta_wins = 10.0, delta_ts = 0.03
        # logit = 1.0 + 0.1*3 + 0.02*10 + 10*0.03 = 1.0 + 0.3 + 0.2 + 0.3 = 1.8
        # sigmoid(1.8) ≈ 0.8581
        prob = predict_win_prob(FEATURES_HIGH, FEATURES_LOW, spec=SAMPLE_SPEC)
        expected = 1.0 / (1.0 + math.exp(-1.8))
        assert abs(prob - expected) < 1e-9

    def test_equal_teams_above_50pct(self):
        # Positive intercept → high seed favoured even when features are equal
        features = {"bpm_avail_sum": 5.0, "playoff_series_wins": 20.0, "ts_percent": 0.58}
        prob = predict_win_prob(features, features, spec=SAMPLE_SPEC)
        assert prob > 0.5

    def test_zero_intercept_equal_features_exactly_50pct(self):
        spec = {**SAMPLE_SPEC, "intercept": 0.0}
        features = {"bpm_avail_sum": 5.0, "playoff_series_wins": 20.0, "ts_percent": 0.58}
        prob = predict_win_prob(features, features, spec=spec)
        assert abs(prob - 0.5) < 1e-9

    def test_missing_feature_defaults_to_zero(self):
        # If a feature is missing from both dicts, delta = 0 → no contribution
        prob_full = predict_win_prob(FEATURES_HIGH, FEATURES_LOW, spec=SAMPLE_SPEC)
        spec_extra = {
            **SAMPLE_SPEC,
            "features": SAMPLE_SPEC["features"] + ["delta_unknown_feat"],
            "coefficients": {**SAMPLE_SPEC["coefficients"], "delta_unknown_feat": 99.0},
        }
        prob_with_missing = predict_win_prob(FEATURES_HIGH, FEATURES_LOW, spec=spec_extra)
        assert abs(prob_full - prob_with_missing) < 1e-9


# ── simulate_series ────────────────────────────────────────────────────────────

class TestSimulateSeries:
    @pytest.fixture
    def team_features(self):
        return pd.DataFrame(
            [FEATURES_HIGH, FEATURES_LOW],
            index=["BOS", "MIA"],
        )

    def test_returns_one_of_two_teams(self, team_features):
        rng = np.random.default_rng(0)
        winner = simulate_series("BOS", "MIA", team_features, rng, spec=SAMPLE_SPEC)
        assert winner in ("BOS", "MIA")

    def test_deterministic_with_fixed_seed(self, team_features):
        rng1 = np.random.default_rng(42)
        w1 = simulate_series("BOS", "MIA", team_features, rng1, spec=SAMPLE_SPEC)
        rng2 = np.random.default_rng(42)
        w2 = simulate_series("BOS", "MIA", team_features, rng2, spec=SAMPLE_SPEC)
        assert w1 == w2

    def test_strong_favourite_wins_most(self, team_features):
        # High seed has much better features — should win >> 50% with 1000 trials
        rng = np.random.default_rng(0)
        wins = sum(
            simulate_series("BOS", "MIA", team_features, rng, spec=SAMPLE_SPEC) == "BOS"
            for _ in range(1000)
        )
        assert wins > 700

    def test_injury_draws_applied(self, team_features):
        # With mean_rates=0.0 for all BOS stars, every draw > 0 → no healthy star
        # → bpm_avail_sum=0 for BOS, weakening them significantly.
        n_sims = 500
        # draws_arr shape: (n_teams=1, n_stars=3, n_rounds=4, n_sims)
        draws_arr = np.full((1, 3, 4, n_sims), 0.5)
        injury_draws = {
            "draws": draws_arr,
            "team_index": {"BOS": 0},
            "player_bpm": [[5.0, 5.0, 5.0]],   # BOS stars have high BPM
            "mean_rates": [[0.0, 0.0, 0.0]],    # availability=0 → always injured
        }
        wins_no_injury = sum(
            simulate_series("BOS", "MIA", team_features, np.random.default_rng(i), spec=SAMPLE_SPEC)
            == "BOS"
            for i in range(n_sims)
        )
        wins_injured = sum(
            simulate_series(
                "BOS", "MIA", team_features, np.random.default_rng(i),
                spec=SAMPLE_SPEC, injury_draws=injury_draws, draw_index=i,
            ) == "BOS"
            for i in range(n_sims)
        )
        assert wins_injured < wins_no_injury


# ── load_spec ──────────────────────────────────────────────────────────────────

class TestLoadSpec:
    def test_loads_correct_window(self, tmp_path):
        spec_path = tmp_path / "chosen_model_modern.json"
        spec_path.write_text(json.dumps(SAMPLE_SPEC))
        with patch("src.simulation.simulate_series.MODEL_DIR", tmp_path):
            import src.simulation.simulate_series as ss
            ss._spec_cache.clear()
            spec = ss.load_spec("modern")
        assert spec["window"] == "full"  # value from our fixture
        assert "intercept" in spec

    def test_caches_spec(self, tmp_path):
        spec_path = tmp_path / "chosen_model_full.json"
        spec_path.write_text(json.dumps(SAMPLE_SPEC))
        with patch("src.simulation.simulate_series.MODEL_DIR", tmp_path):
            import src.simulation.simulate_series as ss
            ss._spec_cache.clear()
            s1 = ss.load_spec("full")
            s2 = ss.load_spec("full")
        assert s1 is s2
