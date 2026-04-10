"""tests/data/test_assemble.py — Unit tests for src/data/assemble.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.assemble import _join_intermediates, compute_deltas, load_active_features, save_final_dataset


class TestLoadActiveFeatures:
    def test_returns_list_of_strings(self):
        features = load_active_features()
        assert isinstance(features, list)
        assert all(isinstance(f, str) for f in features)

    def test_non_empty(self):
        assert len(load_active_features()) > 0


class TestJoinIntermediates:
    def test_raises_when_no_intermediates(self, tmp_path):
        base = pd.DataFrame({"season": [2020], "series_id": ["a"]})
        with pytest.raises(FileNotFoundError):
            _join_intermediates(base, intermediate_dir=tmp_path)

    def test_joins_on_season_series_id(self, tmp_path):
        base = pd.DataFrame({"season": [2020, 2021], "series_id": ["a", "b"]})
        step1 = pd.DataFrame({"season": [2020, 2021], "series_id": ["a", "b"], "feat1": [1.0, 2.0]})
        step2 = pd.DataFrame({"season": [2020, 2021], "series_id": ["a", "b"], "feat2": [3.0, 4.0]})
        step1.to_parquet(tmp_path / "step1.parquet", index=False)
        step2.to_parquet(tmp_path / "step2.parquet", index=False)

        result = _join_intermediates(base, intermediate_dir=tmp_path)
        assert "feat1" in result.columns
        assert "feat2" in result.columns
        assert len(result) == 2


class TestComputeDeltas:
    def test_paired_columns_become_delta(self):
        df = pd.DataFrame({"off_rtg_high": [110.0], "off_rtg_low": [105.0]})
        result = compute_deltas(df)
        assert "delta_off_rtg" in result.columns
        assert "off_rtg_high" not in result.columns
        assert "off_rtg_low" not in result.columns
        assert result["delta_off_rtg"].iloc[0] == pytest.approx(5.0)

    def test_delta_is_high_minus_low(self):
        df = pd.DataFrame({"bpm_top3_mean_high": [3.0], "bpm_top3_mean_low": [5.0]})
        result = compute_deltas(df)
        assert result["delta_bpm_top3_mean"].iloc[0] == pytest.approx(-2.0)

    def test_unmatched_columns_untouched(self):
        df = pd.DataFrame({"off_rtg_high": [110.0], "some_other_col": [1.0]})
        result = compute_deltas(df)
        assert "off_rtg_high" in result.columns
        assert "some_other_col" in result.columns

    def test_no_high_low_columns_returns_unchanged(self):
        df = pd.DataFrame({"year": [2020], "series_id": ["a"], "feat_delta": [1.0]})
        result = compute_deltas(df)
        assert list(result.columns) == list(df.columns)

    def test_multiple_pairs(self):
        df = pd.DataFrame({
            "off_rtg_high": [110.0], "off_rtg_low": [105.0],
            "bpm_top3_mean_high": [3.0], "bpm_top3_mean_low": [1.0],
        })
        result = compute_deltas(df)
        assert set(result.columns) == {"delta_off_rtg", "delta_bpm_top3_mean"}


class TestSaveFinalDataset:
    def test_writes_parquet(self, tmp_path):
        df = pd.DataFrame({"year": [2020], "series_id": ["x"], "feat": [1.0]})
        out = save_final_dataset(df, output_dir=tmp_path)
        assert out.exists()
        loaded = pd.read_parquet(out)
        assert len(loaded) == 1
