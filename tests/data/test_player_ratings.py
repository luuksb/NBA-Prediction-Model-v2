"""tests/data/test_player_ratings.py — Unit tests for src/data/steps/player_ratings.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.data.steps.player_ratings import run


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_series_df() -> pd.DataFrame:
    """Two series: one post-1997 (game-log data available) and one pre-1997."""
    return pd.DataFrame([
        {
            "series_id": "2015_CLE_GSW",
            "season": 2015,
            "team_high": "GSW",   # higher seed despite alphabetical CLE < GSW
            "team_low": "CLE",
        },
        {
            "series_id": "1993_CHI_PHX",
            "season": 1993,
            "team_high": "CHI",
            "team_low": "PHX",
        },
    ])


def _make_player_stats() -> pd.DataFrame:
    """One star per team — minimal set sufficient for _identify_top_n with n=1."""
    return pd.DataFrame([
        {"season": 2015, "team": "GSW", "player_norm": "stephen_curry",
         "bpm": 7.0, "per": 25.0, "usg": 30.0, "usg_percent": 30.0, "mpg": 35.0},
        {"season": 2015, "team": "CLE", "player_norm": "kevin_love",
         "bpm": 1.0, "per": 18.0, "usg": 22.0, "usg_percent": 22.0, "mpg": 28.0},
        {"season": 1993, "team": "CHI", "player_norm": "michael_jordan",
         "bpm": 8.0, "per": 30.0, "usg": 33.0, "usg_percent": 33.0, "mpg": 38.0},
        {"season": 1993, "team": "PHX", "player_norm": "charles_barkley",
         "bpm": 6.0, "per": 26.0, "usg": 28.0, "usg_percent": 28.0, "mpg": 35.0},
    ])


# ── Test: high-seed label is correctly routed to _high columns ─────────────────

class TestHighSeedLabelRouting:
    """Verify that team_high's player stats land in _high columns, not swapped."""

    def test_high_seed_stats_in_high_columns(self):
        """GSW is team_high; its star's BPM (7.0) should appear in bpm_avail_sum_high."""
        avail_lookup = {
            ("2015_CLE_GSW", "stephen_curry"): 1.0,
            ("2015_CLE_GSW", "kevin_love"): 1.0,
        }
        with (
            patch("src.data.steps.player_ratings._load_n_stars", return_value=1),
            patch("src.data.steps.player_ratings._load_ranking_weights", return_value={"bpm": 1.0}),
            patch("src.data.steps.player_ratings._load_player_stats", return_value=_make_player_stats()),
            patch("src.data.steps.player_ratings._build_series_availability", return_value=avail_lookup),
        ):
            result = run(_make_series_df())

        row_2015 = result[result["series_id"] == "2015_CLE_GSW"].iloc[0]
        # GSW star (curry, BPM=7.0) → _high; CLE star (love, BPM=1.0) → _low
        assert row_2015["star1_bpm_high"] == pytest.approx(7.0 * 1.0)
        assert row_2015["star1_bpm_low"] == pytest.approx(1.0 * 1.0)

    def test_swapped_assignment_would_fail(self):
        """Confirm the values for high and low are distinct so the test above is meaningful."""
        avail_lookup = {
            ("2015_CLE_GSW", "stephen_curry"): 1.0,
            ("2015_CLE_GSW", "kevin_love"): 1.0,
        }
        with (
            patch("src.data.steps.player_ratings._load_n_stars", return_value=1),
            patch("src.data.steps.player_ratings._load_ranking_weights", return_value={"bpm": 1.0}),
            patch("src.data.steps.player_ratings._load_player_stats", return_value=_make_player_stats()),
            patch("src.data.steps.player_ratings._build_series_availability", return_value=avail_lookup),
        ):
            result = run(_make_series_df())

        row_2015 = result[result["series_id"] == "2015_CLE_GSW"].iloc[0]
        # If assignment were swapped, high would show 1.0 and low 7.0 — assert that's NOT the case
        assert row_2015["star1_bpm_high"] != pytest.approx(1.0)
        assert row_2015["star1_bpm_low"] != pytest.approx(7.0)


# ── Test: post-1997 availability is 0.0 for absent players ────────────────────

class TestPostSeasonAvailability:
    """Verify the bug fix: absent players in post-1997 series get avail=0.0, not NaN→1.0."""

    def _run_with_partial_avail(self) -> pd.DataFrame:
        """Run player_ratings with kevin_love absent from avail_lookup (2015 Finals injury)."""
        avail_lookup = {
            # stephen_curry played all games — present in lookup
            ("2015_CLE_GSW", "stephen_curry"): 1.0,
            # kevin_love is intentionally absent — 0 games played (injured)
        }
        with (
            patch("src.data.steps.player_ratings._load_n_stars", return_value=1),
            patch("src.data.steps.player_ratings._load_ranking_weights", return_value={"bpm": 1.0}),
            patch("src.data.steps.player_ratings._load_player_stats", return_value=_make_player_stats()),
            patch("src.data.steps.player_ratings._build_series_availability", return_value=avail_lookup),
        ):
            return run(_make_series_df())

    def test_absent_post1997_player_gets_zero_availability(self):
        """Kevin Love (absent from game logs in 2015) should get avail=0.0, not 1.0."""
        result = self._run_with_partial_avail()
        row = result[result["series_id"] == "2015_CLE_GSW"].iloc[0]
        # CLE is team_low; kevin_love played 0 games → avail must be exactly 0.0
        assert row["star1_avail_low"] == pytest.approx(0.0), (
            f"Expected 0.0 availability for absent post-1997 player, got {row['star1_avail_low']}"
        )

    def test_absent_post1997_player_contributes_zero_to_bpm_sum(self):
        """A player with avail=0.0 should contribute 0.0 to bpm_avail_sum, not their full BPM."""
        result = self._run_with_partial_avail()
        row = result[result["series_id"] == "2015_CLE_GSW"].iloc[0]
        # kevin_love BPM=1.0, avail=0.0 → contribution = 0.0
        assert row["bpm_avail_sum_low"] == pytest.approx(0.0), (
            f"Injured player should contribute 0.0 to BPM sum, got {row['bpm_avail_sum_low']}"
        )

    def test_pre1997_absent_player_gets_nan_availability(self):
        """Pre-1997 series have no game logs; absent players should get NaN (unknown), not 0.0."""
        result = self._run_with_partial_avail()
        row = result[result["series_id"] == "1993_CHI_PHX"].iloc[0]
        # 1993 series has no avail data at all — both sides should be NaN
        assert pd.isna(row["star1_avail_high"]), (
            f"Expected NaN for pre-1997 player, got {row['star1_avail_high']}"
        )
        assert pd.isna(row["star1_avail_low"]), (
            f"Expected NaN for pre-1997 player, got {row['star1_avail_low']}"
        )

    def test_pre1997_player_assumes_full_availability_in_bpm_sum(self):
        """NaN availability (pre-1997) should be treated as 1.0 when computing BPM sums."""
        result = self._run_with_partial_avail()
        row = result[result["series_id"] == "1993_CHI_PHX"].iloc[0]
        # michael_jordan BPM=8.0, avail=NaN→1.0 → bpm_avail_sum_high = 8.0
        assert row["bpm_avail_sum_high"] == pytest.approx(8.0), (
            f"Expected BPM sum of 8.0 for pre-1997 player (full avail assumed), "
            f"got {row['bpm_avail_sum_high']}"
        )
