"""tests/data/test_assemble.py — Unit tests for src/data/assemble.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.assemble import assemble_dataset, load_active_features, save_final_dataset


class TestLoadActiveFeatures:
    def test_returns_list_of_strings(self):
        features = load_active_features()
        assert isinstance(features, list)
        assert all(isinstance(f, str) for f in features)

    def test_non_empty(self):
        assert len(load_active_features()) > 0


class TestAssembleDataset:
    def test_raises_when_no_intermediates(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            assemble_dataset(intermediate_dir=tmp_path)

    def test_joins_on_year_series_id(self, tmp_path):
        base = pd.DataFrame({"year": [2020, 2021], "series_id": ["a", "b"], "feat1": [1.0, 2.0]})
        extra = pd.DataFrame({"year": [2020, 2021], "series_id": ["a", "b"], "feat2": [3.0, 4.0]})
        base.to_parquet(tmp_path / "step1.parquet", index=False)
        extra.to_parquet(tmp_path / "step2.parquet", index=False)

        result = assemble_dataset(intermediate_dir=tmp_path)
        assert "feat1" in result.columns
        assert "feat2" in result.columns
        assert len(result) == 2


class TestSaveFinalDataset:
    def test_writes_parquet(self, tmp_path):
        df = pd.DataFrame({"year": [2020], "series_id": ["x"], "feat": [1.0]})
        out = save_final_dataset(df, output_dir=tmp_path)
        assert out.exists()
        loaded = pd.read_parquet(out)
        assert len(loaded) == 1
