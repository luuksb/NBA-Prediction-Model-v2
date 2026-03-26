"""tests/simulation/test_bracket.py — Unit tests for src/simulation/bracket.py."""

from __future__ import annotations

import pytest

from src.simulation.bracket import advance_bracket, build_bracket, build_first_round


EAST = [f"E{i}" for i in range(1, 9)]
WEST = [f"W{i}" for i in range(1, 9)]


class TestBuildFirstRound:
    def test_produces_four_series(self):
        series = build_first_round(EAST, "East")
        assert len(series) == 4

    def test_matchup_seeds(self):
        series = build_first_round(EAST, "East")
        # 1v8, 2v7, 3v6, 4v5
        assert series[0].high_seed == "E1" and series[0].low_seed == "E8"
        assert series[1].high_seed == "E2" and series[1].low_seed == "E7"

    def test_raises_wrong_seed_count(self):
        with pytest.raises(ValueError):
            build_first_round(EAST[:7], "East")


class TestBuildBracket:
    def test_round1_has_eight_series(self):
        bracket = build_bracket(2025, EAST, WEST)
        assert len(bracket.rounds[0]) == 8

    def test_year_stored(self):
        bracket = build_bracket(2024, EAST, WEST)
        assert bracket.year == 2024


class TestAdvanceBracket:
    def _fill_round(self, series_list, pick_high=True):
        for s in series_list:
            s.winner = s.high_seed if pick_high else s.low_seed

    def test_round2_has_four_series(self):
        bracket = build_bracket(2025, EAST, WEST)
        self._fill_round(bracket.rounds[0])
        result = advance_bracket(bracket)
        assert result is not None
        assert len(bracket.rounds[1]) == 4

    def test_finals_has_one_series(self):
        bracket = build_bracket(2025, EAST, WEST)
        for _ in range(3):
            self._fill_round(bracket.rounds[-1])
            advance_bracket(bracket)
        assert len(bracket.rounds[-1]) == 1
        assert bracket.rounds[-1][0].round_num == 4

    def test_returns_none_after_finals(self):
        bracket = build_bracket(2025, EAST, WEST)
        for _ in range(4):
            self._fill_round(bracket.rounds[-1])
            result = advance_bracket(bracket)
        assert result is None
