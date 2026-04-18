"""Tests for src/simulation/aggregate.py."""

from __future__ import annotations

import pytest

from src.simulation.aggregate import aggregate_outcomes


ALL_TEAMS = ["BOS", "MIA", "NYK", "MIL", "OKC", "LAL", "DEN", "PHX",
             "CLE", "ATL", "TOR", "CHI", "GSW", "SAC", "MEM", "HOU"]


def _make_outcome(champion: str, finalist_east: str, finalist_west: str,
                  round_exits: dict, series_results: list | None = None) -> dict:
    return {
        "iteration": 0,
        "champion": champion,
        "finalist_east": finalist_east,
        "finalist_west": finalist_west,
        "round_exits": round_exits,
        "series_results": series_results or [],
    }


class TestAggregateOutcomes:
    def test_championship_probs_sum_to_one(self):
        outcomes = [
            _make_outcome("BOS", "BOS", "OKC",
                          {t: 1 for t in ALL_TEAMS if t not in ("BOS", "OKC")}),
            _make_outcome("OKC", "BOS", "OKC",
                          {t: 1 for t in ALL_TEAMS if t not in ("BOS", "OKC")}),
        ]
        result = aggregate_outcomes(outcomes, ALL_TEAMS)
        total = sum(result["championship_prob"].values())
        assert abs(total - 1.0) < 1e-9

    def test_champion_prob_correct(self):
        # BOS wins 3 out of 4
        outcomes = []
        for i, champ in enumerate(["BOS", "BOS", "BOS", "OKC"]):
            exits = {t: 1 for t in ALL_TEAMS if t not in ("BOS", "OKC")}
            outcomes.append(_make_outcome(champ, "BOS", "OKC", exits))
        result = aggregate_outcomes(outcomes, ALL_TEAMS)
        assert abs(result["championship_prob"]["BOS"] - 0.75) < 1e-9
        assert abs(result["championship_prob"]["OKC"] - 0.25) < 1e-9

    def test_most_common_champion(self):
        exits = {t: 1 for t in ALL_TEAMS if t not in ("BOS", "OKC")}
        outcomes = [_make_outcome("BOS", "BOS", "OKC", exits)] * 3
        outcomes += [_make_outcome("OKC", "BOS", "OKC", exits)]
        result = aggregate_outcomes(outcomes, ALL_TEAMS)
        assert result["most_common_champion"] == "BOS"

    def test_round_advancement_champion_reaches_all_rounds(self):
        exits = {t: 1 for t in ALL_TEAMS if t not in ("BOS", "OKC")}
        outcomes = [_make_outcome("BOS", "BOS", "OKC", exits)]
        result = aggregate_outcomes(outcomes, ALL_TEAMS)
        for r in range(1, 5):
            assert result["round_advancement"]["BOS"][r] == 1.0

    def test_round_advancement_r1_exit_reaches_only_r1(self):
        exits = {t: 1 for t in ALL_TEAMS if t not in ("BOS", "OKC")}
        outcomes = [_make_outcome("BOS", "BOS", "OKC", exits)]
        result = aggregate_outcomes(outcomes, ALL_TEAMS)
        # A team that exited in round 1 reaches only round 1
        r1_exit_team = next(t for t, r in exits.items() if r == 1)
        assert result["round_advancement"][r1_exit_team][1] == 1.0
        assert result["round_advancement"][r1_exit_team][2] == 0.0

    def test_n_sims_matches_input(self):
        exits = {t: 1 for t in ALL_TEAMS if t not in ("BOS", "OKC")}
        outcomes = [_make_outcome("BOS", "BOS", "OKC", exits)] * 7
        result = aggregate_outcomes(outcomes, ALL_TEAMS)
        assert result["n_sims"] == 7

    def test_all_teams_present_in_output(self):
        exits = {t: 1 for t in ALL_TEAMS if t not in ("BOS", "OKC")}
        outcomes = [_make_outcome("BOS", "BOS", "OKC", exits)]
        result = aggregate_outcomes(outcomes, ALL_TEAMS)
        for team in ALL_TEAMS:
            assert team in result["championship_prob"]
            assert team in result["round_advancement"]

    def test_matchup_wins_counts_correctly(self):
        # BOS beats MIA in R1 twice, MIA beats BOS once
        series = [{"team_a": "BOS", "team_b": "MIA", "round": 1, "winner": "BOS"}]
        exits = {t: 1 for t in ALL_TEAMS if t not in ("BOS", "OKC")}
        outcomes = [
            _make_outcome("BOS", "BOS", "OKC", exits, series_results=series),
            _make_outcome("BOS", "BOS", "OKC", exits, series_results=series),
            _make_outcome("BOS", "BOS", "OKC", exits, series_results=[
                {"team_a": "BOS", "team_b": "MIA", "round": 1, "winner": "MIA"}
            ]),
        ]
        result = aggregate_outcomes(outcomes, ALL_TEAMS)
        mw = result["matchup_wins"]
        key = ("BOS", "MIA", 1)
        assert key in mw
        assert mw[key]["wins_a"] == 2   # BOS=team_a (alphabetically first)
        assert mw[key]["wins_b"] == 1   # MIA=team_b
        assert mw[key]["total"] == 3

    def test_matchup_wins_empty_when_no_series_results(self):
        exits = {t: 1 for t in ALL_TEAMS if t not in ("BOS", "OKC")}
        outcomes = [_make_outcome("BOS", "BOS", "OKC", exits)]
        result = aggregate_outcomes(outcomes, ALL_TEAMS)
        assert result["matchup_wins"] == {}
