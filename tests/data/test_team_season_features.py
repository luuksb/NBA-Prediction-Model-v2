"""Tests for build_team_season_features() in src/data/assemble.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.assemble import build_team_season_features


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_base() -> pd.DataFrame:
    """Minimal base series DataFrame with Round 1 entries for two teams per conf."""
    return pd.DataFrame([
        {"series_id": "2020_BOS_MIA", "season": 2020, "round": 1,
         "conference": "East", "team_high": "BOS", "team_low": "MIA",
         "seed_high": 1, "seed_low": 8, "higher_seed_wins": 1},
        {"series_id": "2020_MIL_ORL", "season": 2020, "round": 1,
         "conference": "East", "team_high": "MIL", "team_low": "ORL",
         "seed_high": 2, "seed_low": 7, "higher_seed_wins": 1},
        {"series_id": "2020_LAL_POR", "season": 2020, "round": 1,
         "conference": "West", "team_high": "LAL", "team_low": "POR",
         "seed_high": 1, "seed_low": 8, "higher_seed_wins": 1},
        # Round 2 series — should NOT appear in team_season_features
        {"series_id": "2020_BOS_MIL", "season": 2020, "round": 2,
         "conference": "East", "team_high": "BOS", "team_low": "MIL",
         "seed_high": 1, "seed_low": 2, "higher_seed_wins": 0},
    ])


def _make_player_ratings_intermediate() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2020, "series_id": "2020_BOS_MIA",
         "bpm_avail_sum_high": 15.0, "bpm_avail_sum_low": 8.0,
         "per_avail_sum_high": 25.0, "per_avail_sum_low": 20.0},
        {"season": 2020, "series_id": "2020_MIL_ORL",
         "bpm_avail_sum_high": 12.0, "bpm_avail_sum_low": 5.0,
         "per_avail_sum_high": 22.0, "per_avail_sum_low": 18.0},
        {"season": 2020, "series_id": "2020_LAL_POR",
         "bpm_avail_sum_high": 18.0, "bpm_avail_sum_low": 7.0,
         "per_avail_sum_high": 28.0, "per_avail_sum_low": 19.0},
    ])


def _make_playoff_experience_intermediate() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2020, "series_id": "2020_BOS_MIA",
         "playoff_series_wins_high": 50.0, "playoff_series_wins_low": 10.0},
        {"season": 2020, "series_id": "2020_MIL_ORL",
         "playoff_series_wins_high": 30.0, "playoff_series_wins_low": 5.0},
        {"season": 2020, "series_id": "2020_LAL_POR",
         "playoff_series_wins_high": 80.0, "playoff_series_wins_low": 15.0},
    ])


def _make_team_stats() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2020, "team": "BOS", "ts_percent": 0.58, "o_rtg": 112.0},
        {"season": 2020, "team": "MIA", "ts_percent": 0.55, "o_rtg": 108.0},
        {"season": 2020, "team": "MIL", "ts_percent": 0.57, "o_rtg": 115.0},
        {"season": 2020, "team": "ORL", "ts_percent": 0.54, "o_rtg": 106.0},
        {"season": 2020, "team": "LAL", "ts_percent": 0.59, "o_rtg": 116.0},
        {"season": 2020, "team": "POR", "ts_percent": 0.56, "o_rtg": 110.0},
    ])


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildTeamSeasonFeatures:
    @pytest.fixture
    def setup(self, tmp_path):
        """Write intermediates to tmp_path and mock team_ratings.build_team_stats."""
        intermediate_dir = tmp_path / "intermediate"
        intermediate_dir.mkdir()
        output_dir = tmp_path / "final"

        _make_player_ratings_intermediate().to_parquet(
            intermediate_dir / "player_ratings.parquet", index=False
        )
        _make_playoff_experience_intermediate().to_parquet(
            intermediate_dir / "playoff_experience.parquet", index=False
        )

        return tmp_path, intermediate_dir, output_dir

    def test_one_row_per_team_per_year(self, setup):
        _, intermediate_dir, output_dir = setup
        base = _make_base()
        with patch("src.data.steps.team_ratings.build_team_stats", return_value=_make_team_stats()):
            result = build_team_season_features(base, intermediate_dir, output_dir)
        assert result.duplicated(subset=["year", "team"]).sum() == 0

    def test_correct_number_of_rows(self, setup):
        _, intermediate_dir, output_dir = setup
        base = _make_base()
        with patch("src.data.steps.team_ratings.build_team_stats", return_value=_make_team_stats()):
            result = build_team_season_features(base, intermediate_dir, output_dir)
        # 5 unique teams in Round 1: BOS, MIA, MIL, ORL, LAL, POR — but only 6 in our base
        assert len(result) == 6

    def test_year_column_not_season(self, setup):
        _, intermediate_dir, output_dir = setup
        base = _make_base()
        with patch("src.data.steps.team_ratings.build_team_stats", return_value=_make_team_stats()):
            result = build_team_season_features(base, intermediate_dir, output_dir)
        assert "year" in result.columns
        assert "season" not in result.columns

    def test_bpm_avail_sum_correct_for_high_seed(self, setup):
        _, intermediate_dir, output_dir = setup
        base = _make_base()
        with patch("src.data.steps.team_ratings.build_team_stats", return_value=_make_team_stats()):
            result = build_team_season_features(base, intermediate_dir, output_dir)
        bos_row = result[result["team"] == "BOS"].iloc[0]
        assert bos_row["bpm_avail_sum"] == pytest.approx(15.0)

    def test_bpm_avail_sum_correct_for_low_seed(self, setup):
        _, intermediate_dir, output_dir = setup
        base = _make_base()
        with patch("src.data.steps.team_ratings.build_team_stats", return_value=_make_team_stats()):
            result = build_team_season_features(base, intermediate_dir, output_dir)
        mia_row = result[result["team"] == "MIA"].iloc[0]
        assert mia_row["bpm_avail_sum"] == pytest.approx(8.0)

    def test_team_ratings_features_present(self, setup):
        _, intermediate_dir, output_dir = setup
        base = _make_base()
        with patch("src.data.steps.team_ratings.build_team_stats", return_value=_make_team_stats()):
            result = build_team_season_features(base, intermediate_dir, output_dir)
        assert "ts_percent" in result.columns

    def test_parquet_file_saved(self, setup):
        _, intermediate_dir, output_dir = setup
        base = _make_base()
        with patch("src.data.steps.team_ratings.build_team_stats", return_value=_make_team_stats()):
            build_team_season_features(base, intermediate_dir, output_dir)
        assert (output_dir / "team_season_features.parquet").exists()

    def test_round2_team_not_duplicated(self, setup):
        """BOS appears in Round 1 and Round 2 — result must have only one BOS row."""
        _, intermediate_dir, output_dir = setup
        base = _make_base()
        with patch("src.data.steps.team_ratings.build_team_stats", return_value=_make_team_stats()):
            result = build_team_season_features(base, intermediate_dir, output_dir)
        assert len(result[result["team"] == "BOS"]) == 1

    def test_missing_intermediate_skipped_gracefully(self, setup):
        """If an intermediate is absent, the function logs a warning and continues."""
        _, intermediate_dir, output_dir = setup
        (intermediate_dir / "player_ratings.parquet").unlink()
        base = _make_base()
        with patch("src.data.steps.team_ratings.build_team_stats", return_value=_make_team_stats()):
            result = build_team_season_features(base, intermediate_dir, output_dir)
        assert "year" in result.columns
        assert len(result) > 0
