"""tests/data/test_home_court.py — Unit tests for src/data/steps/home_court.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.steps.home_court import run, NEUTRAL_SITE_SEASONS


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _make_base(seasons: list[int]) -> pd.DataFrame:
    """Build a minimal base DataFrame with one series row per season."""
    return pd.DataFrame(
        {
            "season": seasons,
            "series_id": [f"{s}_TMA_TMB" for s in seasons],
            "team_high": "TMA",
            "team_low": "TMB",
            "higher_seed_wins": 1,
        }
    )


# ── Column presence ─────────────────────────────────────────────────────────────

def test_run_adds_column() -> None:
    base = _make_base([2019, 2020, 2021])
    result = run(base)
    assert "delta_home_court_advantage" in result.columns


def test_run_preserves_all_original_columns() -> None:
    base = _make_base([2019, 2020])
    result = run(base)
    for col in base.columns:
        assert col in result.columns


def test_run_does_not_mutate_input() -> None:
    base = _make_base([2019, 2020])
    original_cols = list(base.columns)
    run(base)
    assert list(base.columns) == original_cols


# ── Values ──────────────────────────────────────────────────────────────────────

def test_normal_season_is_one() -> None:
    base = _make_base([2019])
    result = run(base)
    assert result.loc[0, "delta_home_court_advantage"] == 1


def test_bubble_season_2020_is_zero() -> None:
    base = _make_base([2020])
    result = run(base)
    assert result.loc[0, "delta_home_court_advantage"] == 0


def test_mixed_seasons() -> None:
    seasons = [2018, 2019, 2020, 2021, 2022]
    base = _make_base(seasons)
    result = run(base)
    expected = [1, 1, 0, 1, 1]
    assert list(result["delta_home_court_advantage"]) == expected


def test_no_neutral_site_seasons_all_ones() -> None:
    seasons = list(range(1980, 2020)) + list(range(2021, 2025))
    base = _make_base(seasons)
    result = run(base)
    assert (result["delta_home_court_advantage"] == 1).all()


# ── Data type ───────────────────────────────────────────────────────────────────

def test_column_is_integer_dtype() -> None:
    base = _make_base([2019, 2020])
    result = run(base)
    assert result["delta_home_court_advantage"].dtype in (int, "int32", "int64")


# ── NEUTRAL_SITE_SEASONS constant ───────────────────────────────────────────────

def test_neutral_site_seasons_contains_2020() -> None:
    assert 2020 in NEUTRAL_SITE_SEASONS


def test_neutral_site_seasons_does_not_contain_2019() -> None:
    assert 2019 not in NEUTRAL_SITE_SEASONS
