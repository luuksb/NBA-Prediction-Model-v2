"""tests/data/test_quality.py — Unit tests for src/data/quality.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.quality import _iqr_outlier_rate, run_feature_checks


class TestIqrOutlierRate:
    def test_no_outliers(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        assert _iqr_outlier_rate(s) == 0.0

    def test_detects_extreme_outlier(self):
        # Use a spread series so IQR > 0, then add a value far beyond 3*IQR
        s = pd.Series(list(range(1, 101)) + [10000])
        rate = _iqr_outlier_rate(s)
        assert rate > 0.0

    def test_zero_iqr_returns_zero(self):
        s = pd.Series([5.0] * 50)
        assert _iqr_outlier_rate(s) == 0.0


class TestRunFeatureChecks:
    @pytest.fixture()
    def sample_df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "year": list(range(2010, 2020)) * 5,
            "feat_a": rng.normal(0, 1, 50),
            "feat_b": rng.normal(5, 2, 50),
        })

    def test_returns_one_row_per_feature(self, sample_df):
        result = run_feature_checks(sample_df, ["feat_a", "feat_b"])
        assert len(result) == 2

    def test_missing_column_flagged(self, sample_df):
        result = run_feature_checks(sample_df, ["feat_a", "nonexistent"])
        row = result[result["feature"] == "nonexistent"].iloc[0]
        assert not row["pass_flag"]

    def test_pass_flag_true_for_clean_data(self, sample_df):
        result = run_feature_checks(sample_df, ["feat_a"])
        assert result.iloc[0]["pass_flag"]

    def test_fail_flag_for_high_missingness(self, sample_df):
        df = sample_df.copy()
        df.loc[:40, "feat_a"] = np.nan  # >80% missing
        result = run_feature_checks(df, ["feat_a"], thresholds={"max_missingness_rate": 0.05})
        assert not result.iloc[0]["pass_flag"]
