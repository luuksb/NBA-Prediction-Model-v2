"""tests/data/test_playoff_experience.py — Unit tests for src/data/steps/playoff_experience.py."""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import patch

from src.data.steps.playoff_experience import (
    _normalise_name,
    _load_team_series_stats,
    _build_player_participation,
    _build_player_season_stats,
    _build_current_rosters,
    _build_roster_experience_table,
    run,
)


# ── Name normalisation ─────────────────────────────────────────────────────────

def test_normalise_name_basic() -> None:
    assert _normalise_name("Michael Jordan") == "michael_jordan"


def test_normalise_name_diacritics() -> None:
    assert _normalise_name("Toni Kukoč") == "toni_kukoc"


def test_normalise_name_suffix_stripped() -> None:
    assert _normalise_name("Jimmy Butler III") == "jimmy_butler"


# ── _load_team_series_stats ────────────────────────────────────────────────────

def _make_series_csv_df(
    season: int,
    matchups: list[tuple[str, str, int]],
) -> pd.DataFrame:
    """Helper: build a small playoff_series CSV DataFrame."""
    return pd.DataFrame(
        [
            {"season": season, "team_high": h, "team_low": l, "higher_seed_wins": w}
            for h, l, w in matchups
        ]
    )


def test_load_team_series_stats_basic() -> None:
    """Each team's wins and played counts are tallied correctly."""
    fake_csv = _make_series_csv_df(
        2010,
        [
            ("LAL", "OKC", 1),  # LAL wins
            ("LAL", "SAS", 1),  # LAL wins (second series)
            ("BOS", "MIA", 0),  # MIA wins
        ],
    )
    with patch("src.data.steps.playoff_experience.PLAYOFF_SERIES_DIR") as mock_dir:
        from pathlib import Path
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "2010_nba_api.csv"
            fake_csv.to_csv(path, index=False)

            mock_dir.glob.return_value = [path]

            result = _load_team_series_stats(up_to_season=2011)

    lal = result[result["team"] == "LAL"].iloc[0]
    assert lal["series_wins"] == 2
    assert lal["series_played"] == 2

    okc = result[result["team"] == "OKC"].iloc[0]
    assert okc["series_wins"] == 0
    assert okc["series_played"] == 1

    mia = result[result["team"] == "MIA"].iloc[0]
    assert mia["series_wins"] == 1
    assert mia["series_played"] == 1


def test_load_team_series_stats_anti_look_ahead() -> None:
    """up_to_season is exclusive — the matching year is excluded."""
    fake_csv = _make_series_csv_df(2015, [("GSW", "CLE", 1)])
    with patch("src.data.steps.playoff_experience.PLAYOFF_SERIES_DIR") as mock_dir:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "2015_nba_api.csv"
            fake_csv.to_csv(path, index=False)
            mock_dir.glob.return_value = [path]

            result = _load_team_series_stats(up_to_season=2015)  # excludes 2015

    assert result.empty


# ── _build_player_season_stats ─────────────────────────────────────────────────

def test_build_player_season_stats_credits_player() -> None:
    """A player on a winning team accumulates series wins."""
    team_stats = pd.DataFrame([
        {"season": 2010, "team": "LAL", "series_wins": 4, "series_played": 4},
    ])
    participation = pd.DataFrame([
        {"player_norm": "kobe_bryant", "season": 2010, "team": "LAL"},
        {"player_norm": "pau_gasol",   "season": 2010, "team": "LAL"},
    ])

    with (
        patch("src.data.steps.playoff_experience._load_team_series_stats", return_value=team_stats),
        patch("src.data.steps.playoff_experience._build_player_participation", return_value=participation),
    ):
        result = _build_player_season_stats(max_season=2010)

    kobe = result[result["player_norm"] == "kobe_bryant"].iloc[0]
    assert kobe["series_wins"] == 4
    assert kobe["series_played"] == 4


def test_build_player_season_stats_traded_player_sums_both_teams() -> None:
    """A player traded between two playoff teams in one season gets credit from both."""
    team_stats = pd.DataFrame([
        {"season": 2012, "team": "MIA", "series_wins": 4, "series_played": 4},
        {"season": 2012, "team": "BOS", "series_wins": 2, "series_played": 3},
    ])
    participation = pd.DataFrame([
        {"player_norm": "ray_allen", "season": 2012, "team": "MIA"},
        {"player_norm": "ray_allen", "season": 2012, "team": "BOS"},
    ])

    with (
        patch("src.data.steps.playoff_experience._load_team_series_stats", return_value=team_stats),
        patch("src.data.steps.playoff_experience._build_player_participation", return_value=participation),
    ):
        result = _build_player_season_stats(max_season=2012)

    ray = result[result["player_norm"] == "ray_allen"].iloc[0]
    assert ray["series_wins"] == 6   # 4 + 2
    assert ray["series_played"] == 7  # 4 + 3


# ── _build_roster_experience_table ────────────────────────────────────────────

def test_roster_experience_aggregates_correctly() -> None:
    """Roster-level wins = sum of individual players' prior wins."""
    player_stats = pd.DataFrame([
        {"player_norm": "lebron_james", "season": 2012, "series_wins": 4, "series_played": 4},
        {"player_norm": "dwyane_wade",  "season": 2012, "series_wins": 4, "series_played": 4},
        {"player_norm": "chris_bosh",   "season": 2012, "series_wins": 4, "series_played": 4},
    ])
    rosters = pd.DataFrame([
        {"season": 2013, "team": "MIA", "player_norm": "lebron_james"},
        {"season": 2013, "team": "MIA", "player_norm": "dwyane_wade"},
        {"season": 2013, "team": "MIA", "player_norm": "chris_bosh"},
        {"season": 2013, "team": "IND", "player_norm": "paul_george"},  # no prior wins
    ])

    with (
        patch("src.data.steps.playoff_experience._build_player_season_stats", return_value=player_stats),
        patch("src.data.steps.playoff_experience._build_current_rosters", return_value=rosters),
    ):
        result = _build_roster_experience_table(max_season=2013)

    mia = result[(result["season"] == 2013) & (result["team"] == "MIA")].iloc[0]
    assert mia["series_wins_cum"] == 12   # 3 players × 4 wins
    assert mia["series_played_cum"] == 12
    assert mia["roster_size"] == 3
    assert abs(mia["avg_series_wins"] - 4.0) < 1e-9

    ind = result[(result["season"] == 2013) & (result["team"] == "IND")].iloc[0]
    assert ind["series_wins_cum"] == 0
    assert ind["avg_series_wins"] == 0.0


def test_roster_experience_only_uses_prior_seasons() -> None:
    """Experience from the target season itself must not be counted (anti-look-ahead)."""
    # Player has experience in 2015; target is also 2015 — should not count
    player_stats = pd.DataFrame([
        {"player_norm": "steph_curry", "season": 2015, "series_wins": 4, "series_played": 4},
    ])
    rosters = pd.DataFrame([
        {"season": 2015, "team": "GSW", "player_norm": "steph_curry"},
    ])

    with (
        patch("src.data.steps.playoff_experience._build_player_season_stats", return_value=player_stats),
        patch("src.data.steps.playoff_experience._build_current_rosters", return_value=rosters),
    ):
        result = _build_roster_experience_table(max_season=2015)

    gsw_2015 = result[(result["season"] == 2015) & (result["team"] == "GSW")].iloc[0]
    assert gsw_2015["series_wins_cum"] == 0  # 2015 experience not counted for target 2015


# ── run ────────────────────────────────────────────────────────────────────────

def _make_series_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"series_id": "2016_GSW_CLE", "season": 2016, "team_high": "GSW", "team_low": "CLE"},
        {"series_id": "2016_SAS_MEM", "season": 2016, "team_high": "SAS", "team_low": "MEM"},
    ])


def test_run_adds_eight_columns() -> None:
    """run() must add exactly the eight expected feature columns."""
    exp_table = pd.DataFrame([
        {"season": 2016, "team": "GSW",
         "series_wins_cum": 10, "series_played_cum": 12, "roster_size": 5,
         "avg_series_wins": 2.0, "avg_series_played": 2.4},
        {"season": 2016, "team": "CLE",
         "series_wins_cum": 5, "series_played_cum": 8, "roster_size": 5,
         "avg_series_wins": 1.0, "avg_series_played": 1.6},
        {"season": 2016, "team": "SAS",
         "series_wins_cum": 20, "series_played_cum": 22, "roster_size": 6,
         "avg_series_wins": 3.33, "avg_series_played": 3.67},
        {"season": 2016, "team": "MEM",
         "series_wins_cum": 3, "series_played_cum": 6, "roster_size": 4,
         "avg_series_wins": 0.75, "avg_series_played": 1.5},
    ])

    with patch(
        "src.data.steps.playoff_experience._build_roster_experience_table",
        return_value=exp_table,
    ):
        result = run(_make_series_df())

    expected_cols = {
        "playoff_series_wins_high", "playoff_series_wins_low",
        "avg_playoff_series_wins_high", "avg_playoff_series_wins_low",
        "playoff_series_played_high", "playoff_series_played_low",
        "avg_playoff_series_played_high", "avg_playoff_series_played_low",
    }
    assert expected_cols.issubset(set(result.columns))


def test_run_correct_values() -> None:
    """run() correctly looks up values for high and low seeds."""
    exp_table = pd.DataFrame([
        {"season": 2016, "team": "GSW",
         "series_wins_cum": 10, "series_played_cum": 12, "roster_size": 5,
         "avg_series_wins": 2.0, "avg_series_played": 2.4},
        {"season": 2016, "team": "CLE",
         "series_wins_cum": 5, "series_played_cum": 8, "roster_size": 5,
         "avg_series_wins": 1.0, "avg_series_played": 1.6},
    ])
    series = pd.DataFrame([
        {"series_id": "2016_GSW_CLE", "season": 2016, "team_high": "GSW", "team_low": "CLE"},
    ])

    with patch(
        "src.data.steps.playoff_experience._build_roster_experience_table",
        return_value=exp_table,
    ):
        result = run(series)

    row = result.iloc[0]
    assert row["playoff_series_wins_high"] == 10
    assert row["playoff_series_wins_low"] == 5
    assert row["playoff_series_played_high"] == 12
    assert row["playoff_series_played_low"] == 8


def test_run_unknown_team_returns_zero() -> None:
    """Teams not in the experience table receive zero (new franchise / expansion)."""
    exp_table = pd.DataFrame(columns=[
        "season", "team", "series_wins_cum", "series_played_cum",
        "roster_size", "avg_series_wins", "avg_series_played",
    ])
    series = pd.DataFrame([
        {"series_id": "2021_NEW_XYZ", "season": 2021, "team_high": "NEW", "team_low": "XYZ"},
    ])

    with patch(
        "src.data.steps.playoff_experience._build_roster_experience_table",
        return_value=exp_table,
    ):
        result = run(series)

    assert result.iloc[0]["playoff_series_wins_high"] == 0.0
    assert result.iloc[0]["playoff_series_wins_low"] == 0.0


def test_run_empty_input() -> None:
    """run() on an empty DataFrame returns it unchanged."""
    empty = pd.DataFrame(columns=["series_id", "season", "team_high", "team_low"])
    result = run(empty)
    assert result.empty
    assert list(result.columns) == list(empty.columns)
