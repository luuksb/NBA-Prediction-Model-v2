"""tests/injury/test_availability_history.py — Unit tests for src/injury/availability_history.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.injury.availability_history import compute_availability_rates


def _make_games_df() -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": ["p1", "p1", "p1", "p2", "p2"],
        "year":       [2020, 2021, 2022, 2021, 2022],
        "games_played":  [6, 4, 5, 7, 3],
        "games_possible": [7, 7, 7, 7, 7],
    })


class TestComputeAvailabilityRates:
    def test_excludes_target_year(self):
        df = _make_games_df()
        rates = compute_availability_rates(df, target_year=2022)
        # Only 2020 and 2021 data should be used
        p1_row = rates[rates["player_id"] == "p1"].iloc[0]
        # (6+4)/(7+7) = 10/14 ≈ 0.714
        assert abs(p1_row["availability_rate"] - 10 / 14) < 1e-6

    def test_returns_one_row_per_player(self):
        df = _make_games_df()
        rates = compute_availability_rates(df, target_year=2023)
        assert rates["player_id"].nunique() == len(rates)

    def test_rate_clipped_to_unit_interval(self):
        df = pd.DataFrame({
            "player_id": ["p1"],
            "year": [2019],
            "games_played": [10],
            "games_possible": [7],  # > 1 — should clip to 1.0
        })
        rates = compute_availability_rates(df, target_year=2025)
        assert rates.iloc[0]["availability_rate"] <= 1.0

    def test_n_series_counted(self):
        df = _make_games_df()
        rates = compute_availability_rates(df, target_year=2023)
        p1_row = rates[rates["player_id"] == "p1"].iloc[0]
        assert p1_row["n_series"] == 3
