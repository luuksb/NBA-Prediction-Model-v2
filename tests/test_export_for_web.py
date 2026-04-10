"""tests/test_export_for_web.py — Unit and integration tests for scripts/export_for_web.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.export_for_web import (
    build_bracket_json,
    export_run,
    load_model_metrics,
    matchup_to_dict,
    validate_export,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TEAM_NODE_HIGH = {
    "abbrev": "OKC",
    "seed": 1,
    "conference": "west",
    "adv_prob": 0.90,
    "cond_win_prob": 0.78,
    "logo_url": "",
}

_TEAM_NODE_LOW = {
    "abbrev": "MEM",
    "seed": 8,
    "conference": "west",
    "adv_prob": 0.10,
    "cond_win_prob": 0.22,
    "logo_url": "",
}


def _make_matchup(hi: dict, lo: dict) -> dict[str, Any]:
    return {"high": hi, "low": lo}


def _make_team_node(abbrev: str, seed: int, conf: str, cond_win_prob: float) -> dict:
    return {
        "abbrev": abbrev,
        "seed": seed,
        "conference": conf,
        "adv_prob": 0.5,
        "cond_win_prob": cond_win_prob,
        "logo_url": "",
    }


def _minimal_bracket() -> dict[str, Any]:
    """Build a minimal but structurally valid bracket dict for testing."""
    # West
    w_r1 = [
        _make_matchup(_make_team_node("OKC", 1, "west", 0.78), _make_team_node("MEM", 8, "west", 0.22)),
        _make_matchup(_make_team_node("HOU", 2, "west", 0.65), _make_team_node("GSW", 7, "west", 0.35)),
        _make_matchup(_make_team_node("LAL", 3, "west", 0.55), _make_team_node("MIN", 6, "west", 0.45)),
        _make_matchup(_make_team_node("DEN", 4, "west", 0.52), _make_team_node("LAC", 5, "west", 0.48)),
    ]
    w_r2 = [
        _make_matchup(_make_team_node("OKC", 1, "west", 0.72), _make_team_node("DEN", 4, "west", 0.28)),
        _make_matchup(_make_team_node("HOU", 2, "west", 0.60), _make_team_node("LAL", 3, "west", 0.40)),
    ]
    w_cf = _make_matchup(_make_team_node("OKC", 1, "west", 0.65), _make_team_node("HOU", 2, "west", 0.35))

    # East
    e_r1 = [
        _make_matchup(_make_team_node("CLE", 1, "east", 0.80), _make_team_node("MIA", 8, "east", 0.20)),
        _make_matchup(_make_team_node("BOS", 2, "east", 0.70), _make_team_node("ORL", 7, "east", 0.30)),
        _make_matchup(_make_team_node("NYK", 3, "east", 0.58), _make_team_node("MIL", 5, "east", 0.42)),
        _make_matchup(_make_team_node("IND", 4, "east", 0.55), _make_team_node("DET", 6, "east", 0.45)),
    ]
    e_r2 = [
        _make_matchup(_make_team_node("CLE", 1, "east", 0.68), _make_team_node("IND", 4, "east", 0.32)),
        _make_matchup(_make_team_node("BOS", 2, "east", 0.62), _make_team_node("NYK", 3, "east", 0.38)),
    ]
    e_cf = _make_matchup(_make_team_node("CLE", 1, "east", 0.55), _make_team_node("BOS", 2, "east", 0.45))

    finals = _make_matchup(_make_team_node("OKC", 1, "west", 0.58), _make_team_node("CLE", 1, "east", 0.42))

    return {
        "west": {1: w_r1, 2: w_r2, 3: [w_cf]},
        "east": {1: e_r1, 2: e_r2, 3: [e_cf]},
        "finals": {4: [finals]},
        "champion": _make_team_node("OKC", 1, "west", 0.58),
    }


def _minimal_output(bracket_json: dict) -> dict[str, Any]:
    """Build a minimal valid export dict for validation tests."""
    teams = {
        "OKC": {"name": "Oklahoma City Thunder", "abbreviation": "OKC", "conference": "West", "seed": 1, "color_primary": "#007AC1", "color_secondary": "#EF3B24"},
        "MEM": {"name": "Memphis Grizzlies", "abbreviation": "MEM", "conference": "West", "seed": 8, "color_primary": "#5D76A9", "color_secondary": "#12173F"},
        "HOU": {"name": "Houston Rockets", "abbreviation": "HOU", "conference": "West", "seed": 2, "color_primary": "#CE1141", "color_secondary": "#000000"},
        "GSW": {"name": "Golden State Warriors", "abbreviation": "GSW", "conference": "West", "seed": 7, "color_primary": "#1D428A", "color_secondary": "#FFC72C"},
        "LAL": {"name": "Los Angeles Lakers", "abbreviation": "LAL", "conference": "West", "seed": 3, "color_primary": "#552583", "color_secondary": "#FDB927"},
        "MIN": {"name": "Minnesota Timberwolves", "abbreviation": "MIN", "conference": "West", "seed": 6, "color_primary": "#0C2340", "color_secondary": "#236192"},
        "DEN": {"name": "Denver Nuggets", "abbreviation": "DEN", "conference": "West", "seed": 4, "color_primary": "#0E2240", "color_secondary": "#FEC524"},
        "LAC": {"name": "Los Angeles Clippers", "abbreviation": "LAC", "conference": "West", "seed": 5, "color_primary": "#C8102E", "color_secondary": "#1D428A"},
        "CLE": {"name": "Cleveland Cavaliers", "abbreviation": "CLE", "conference": "East", "seed": 1, "color_primary": "#860038", "color_secondary": "#FDBB30"},
        "MIA": {"name": "Miami Heat", "abbreviation": "MIA", "conference": "East", "seed": 8, "color_primary": "#98002E", "color_secondary": "#F9A01B"},
        "BOS": {"name": "Boston Celtics", "abbreviation": "BOS", "conference": "East", "seed": 2, "color_primary": "#007A33", "color_secondary": "#BA9653"},
        "ORL": {"name": "Orlando Magic", "abbreviation": "ORL", "conference": "East", "seed": 7, "color_primary": "#0077C0", "color_secondary": "#C4CED4"},
        "NYK": {"name": "New York Knicks", "abbreviation": "NYK", "conference": "East", "seed": 3, "color_primary": "#006BB6", "color_secondary": "#F58426"},
        "MIL": {"name": "Milwaukee Bucks", "abbreviation": "MIL", "conference": "East", "seed": 5, "color_primary": "#00471B", "color_secondary": "#EEE1C6"},
        "IND": {"name": "Indiana Pacers", "abbreviation": "IND", "conference": "East", "seed": 4, "color_primary": "#002D62", "color_secondary": "#FDBB30"},
        "DET": {"name": "Detroit Pistons", "abbreviation": "DET", "conference": "East", "seed": 6, "color_primary": "#C8102E", "color_secondary": "#1D428A"},
    }
    # Sum championship probs to exactly 1.0
    champ_probs = {t: 1.0 / 16 for t in teams}

    return {
        "metadata": {
            "season": 2025,
            "generated_at": "2025-04-01T00:00:00+00:00",
            "n_simulations": 50000,
            "training_window": "modern (2000–2024)",
            "features": ["delta_bpm_avail_sum", "delta_playoff_series_wins"],
            "pseudo_r2": 0.10,
            "auc": 0.72,
            "brier_score": 0.15,
        },
        "teams": teams,
        "bracket": bracket_json,
        "championship_probs": champ_probs,
    }


# ---------------------------------------------------------------------------
# Tests: matchup_to_dict
# ---------------------------------------------------------------------------


class TestMatchupToDict:
    def test_returns_required_keys(self):
        m = _make_matchup(_TEAM_NODE_HIGH, _TEAM_NODE_LOW)
        result = matchup_to_dict(m, "W_R1_1")
        for key in ("matchup_id", "top_team", "bottom_team", "top_seed", "bottom_seed", "top_win_prob"):
            assert key in result

    def test_top_team_is_high_seed(self):
        m = _make_matchup(_TEAM_NODE_HIGH, _TEAM_NODE_LOW)
        result = matchup_to_dict(m, "W_R1_1")
        assert result["top_team"] == "OKC"
        assert result["bottom_team"] == "MEM"

    def test_matchup_id_set_correctly(self):
        m = _make_matchup(_TEAM_NODE_HIGH, _TEAM_NODE_LOW)
        result = matchup_to_dict(m, "Finals")
        assert result["matchup_id"] == "Finals"

    def test_top_win_prob_rounded_to_4dp(self):
        hi = {**_TEAM_NODE_HIGH, "cond_win_prob": 0.7812345}
        m = _make_matchup(hi, _TEAM_NODE_LOW)
        result = matchup_to_dict(m, "W_R1_1")
        assert result["top_win_prob"] == pytest.approx(0.7812, abs=1e-4)

    def test_seeds_are_correct(self):
        m = _make_matchup(_TEAM_NODE_HIGH, _TEAM_NODE_LOW)
        result = matchup_to_dict(m, "W_R1_1")
        assert result["top_seed"] == 1
        assert result["bottom_seed"] == 8


# ---------------------------------------------------------------------------
# Tests: build_bracket_json
# ---------------------------------------------------------------------------


class TestBuildBracketJson:
    def setup_method(self):
        self.bracket = _minimal_bracket()
        self.result = build_bracket_json(self.bracket)

    def test_has_west_east_finals_keys(self):
        assert "West" in self.result
        assert "East" in self.result
        assert "Finals" in self.result

    def test_west_has_r1_r2_cf(self):
        assert "R1" in self.result["West"]
        assert "R2" in self.result["West"]
        assert "CF" in self.result["West"]

    def test_r1_has_four_matchups(self):
        assert len(self.result["West"]["R1"]) == 4
        assert len(self.result["East"]["R1"]) == 4

    def test_r2_has_two_matchups(self):
        assert len(self.result["West"]["R2"]) == 2
        assert len(self.result["East"]["R2"]) == 2

    def test_cf_has_one_matchup(self):
        assert len(self.result["West"]["CF"]) == 1
        assert len(self.result["East"]["CF"]) == 1

    def test_finals_is_single_dict(self):
        assert isinstance(self.result["Finals"], dict)
        assert self.result["Finals"]["matchup_id"] == "Finals"

    def test_matchup_ids_are_unique(self):
        ids = []
        for conf in ("West", "East"):
            for rnd in ("R1", "R2", "CF"):
                for m in self.result[conf][rnd]:
                    ids.append(m["matchup_id"])
        ids.append(self.result["Finals"]["matchup_id"])
        assert len(ids) == len(set(ids))

    def test_west_r1_ids_prefixed_w(self):
        for m in self.result["West"]["R1"]:
            assert m["matchup_id"].startswith("W_R1_")

    def test_east_r1_ids_prefixed_e(self):
        for m in self.result["East"]["R1"]:
            assert m["matchup_id"].startswith("E_R1_")


# ---------------------------------------------------------------------------
# Tests: validate_export
# ---------------------------------------------------------------------------


class TestValidateExport:
    def setup_method(self):
        bracket = _minimal_bracket()
        self.bracket_json = build_bracket_json(bracket)
        self.output = _minimal_output(self.bracket_json)

    def test_passes_for_valid_output(self):
        # Should not raise
        validate_export(self.output)

    def test_fails_when_probs_dont_sum_to_one(self):
        bad = dict(self.output)
        bad["championship_probs"] = {"OKC": 0.9, "CLE": 0.5}
        with pytest.raises(AssertionError, match="championship_probs"):
            validate_export(bad)

    def test_fails_when_bracket_team_missing_from_teams(self):
        bad_teams = {k: v for k, v in self.output["teams"].items() if k != "OKC"}
        bad = {**self.output, "teams": bad_teams}
        with pytest.raises(AssertionError, match="missing from teams dict"):
            validate_export(bad)

    def test_fails_when_win_prob_out_of_range(self):
        bracket_copy = json.loads(json.dumps(self.bracket_json))
        bracket_copy["West"]["R1"][0]["top_win_prob"] = 1.5
        bad = {**self.output, "bracket": bracket_copy}
        with pytest.raises(AssertionError, match="Win probs outside"):
            validate_export(bad)


# ---------------------------------------------------------------------------
# Tests: load_model_metrics
# ---------------------------------------------------------------------------


class TestLoadModelMetrics:
    def test_returns_none_for_unknown_features(self):
        result = load_model_metrics(["nonexistent_feature_xyz"], "modern")
        assert result["mcfadden_r2"] is None
        assert result["brier_score"] is None
        assert result["auc_roc"] is None

    def test_returns_floats_for_known_features(self):
        known = ["delta_bpm_avail_sum", "delta_playoff_series_wins", "delta_ts_percent"]
        result = load_model_metrics(known, "modern")
        # Will find match in all_models_22feat_size2-4.parquet
        assert result["mcfadden_r2"] is not None
        assert isinstance(result["mcfadden_r2"], float)
        assert 0.0 < result["mcfadden_r2"] < 1.0

    def test_returns_dict_with_three_keys(self):
        result = load_model_metrics(["x"], "modern")
        assert set(result.keys()) == {"mcfadden_r2", "brier_score", "auc_roc"}


# ---------------------------------------------------------------------------
# Integration test: export_run with real 2025_modern data
# ---------------------------------------------------------------------------


class TestExportRunIntegration:
    def test_export_produces_valid_json(self, tmp_path):
        out = tmp_path / "nba_results.json"
        result = export_run("2025_modern", str(out))

        assert out.exists()
        with open(out) as f:
            parsed = json.load(f)

        assert parsed["metadata"]["season"] == 2025
        assert parsed["metadata"]["n_simulations"] == 50000
        assert len(parsed["teams"]) == 16
        assert len(parsed["bracket"]["West"]["R1"]) == 4
        assert len(parsed["bracket"]["East"]["R1"]) == 4
        assert "Finals" in parsed["bracket"]

    def test_championship_probs_sum_to_one(self, tmp_path):
        out = tmp_path / "nba_results.json"
        result = export_run("2025_modern", str(out))
        total = sum(result["championship_probs"].values())
        assert abs(total - 1.0) < 1e-2

    def test_all_win_probs_in_range(self, tmp_path):
        out = tmp_path / "nba_results.json"
        result = export_run("2025_modern", str(out))
        bracket = result["bracket"]
        for conf in ("West", "East"):
            for rnd in ("R1", "R2", "CF"):
                for m in bracket[conf][rnd]:
                    assert 0.0 <= m["top_win_prob"] <= 1.0, f"Bad prob in {m['matchup_id']}"
        assert 0.0 <= bracket["Finals"]["top_win_prob"] <= 1.0

    def test_raises_on_invalid_run_id(self, tmp_path):
        with pytest.raises(ValueError, match="run_id must be"):
            export_run("invalid", str(tmp_path / "out.json"))
