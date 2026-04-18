"""tests/injury/test_injury_overrides.py — Unit tests for src/injury/injury_overrides.py.

Tests cover:
- _normalize_name: diacritics, suffixes, apostrophes
- _parse_return_date: specific dates, status keywords, empty inputs
- compute_round_availability: all four boundary cases + pro-rata
- _apply_scalar_override: zero / partial / full availability, isolation of slice
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.injury.injury_overrides import (
    ROUND_SCHEDULE,
    _apply_scalar_override,
    _normalize_name,
    _parse_return_date,
    compute_round_availability,
)


# ── _normalize_name ──────────────────────────────────────────────────────────

class TestNormalizeName:
    def test_strips_diacritics(self) -> None:
        assert _normalize_name("Nikola Jokić") == "nikola_jokic"

    def test_lowercase_ascii(self) -> None:
        assert _normalize_name("LeBron James") == "lebron_james"

    def test_strips_roman_suffix(self) -> None:
        assert _normalize_name("Jimmy Butler III") == "jimmy_butler"

    def test_strips_jr_suffix(self) -> None:
        assert _normalize_name("Gary Payton Jr.") == "gary_payton"

    def test_apostrophe_becomes_underscore(self) -> None:
        assert _normalize_name("De'Aaron Fox") == "de_aaron_fox"

    def test_cedilla_stripped(self) -> None:
        assert _normalize_name("Luka Dončić") == "luka_doncic"

    def test_spaces_become_underscores(self) -> None:
        assert _normalize_name("Kevin Durant") == "kevin_durant"


# ── _parse_return_date ───────────────────────────────────────────────────────

class TestParseReturnDate:
    def test_specific_date_no_year(self) -> None:
        d = _parse_return_date("Apr 20", reference_year=2026)
        assert d == date(2026, 4, 20)

    def test_specific_date_with_year(self) -> None:
        d = _parse_return_date("April 20, 2026")
        assert d == date(2026, 4, 20)

    def test_month_abbreviation_dot(self) -> None:
        d = _parse_return_date("May. 5", reference_year=2026)
        assert d == date(2026, 5, 5)

    def test_out_returns_none(self) -> None:
        assert _parse_return_date("Out") is None

    def test_injured_reserve_returns_none(self) -> None:
        assert _parse_return_date("Injured Reserve") is None

    def test_indefinite_returns_none(self) -> None:
        assert _parse_return_date("Out Indefinitely") is None

    def test_doubtful_returns_none(self) -> None:
        assert _parse_return_date("Doubtful") is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_return_date("") is None

    def test_dash_returns_none(self) -> None:
        assert _parse_return_date("—") is None
        assert _parse_return_date("-") is None

    def test_day_to_day_returns_today(self) -> None:
        d = _parse_return_date("Day-To-Day")
        assert d == date.today()

    def test_questionable_returns_today(self) -> None:
        d = _parse_return_date("Questionable")
        assert d == date.today()

    def test_probable_returns_today(self) -> None:
        d = _parse_return_date("Probable")
        assert d == date.today()

    def test_unrecognised_returns_none(self) -> None:
        # Unknown strings fall back to None (conservative)
        assert _parse_return_date("TBD") is None


# ── compute_round_availability ───────────────────────────────────────────────

class TestComputeRoundAvailability:
    """All four availability cases plus range invariant."""

    # Round 1: 2026-04-19 to 2026-05-04 (15 days)

    def test_indefinitely_out_all_zeros(self) -> None:
        avails = compute_round_availability(None)
        assert avails == [0.0, 0.0, 0.0, 0.0]

    def test_returns_after_finals_all_zeros(self) -> None:
        avails = compute_round_availability(date(2026, 7, 1))
        assert avails == [0.0, 0.0, 0.0, 0.0]

    def test_available_before_round1_start(self) -> None:
        avails = compute_round_availability(date(2026, 4, 18))
        assert avails[0] == 1.0

    def test_returns_on_round1_start_date(self) -> None:
        # Exact start date is still fully available (≤ round_start branch).
        avails = compute_round_availability(date(2026, 4, 19))
        assert avails[0] == 1.0

    def test_returns_mid_round1(self) -> None:
        # Return Apr 28: (May 4 - Apr 28).days = 6 / (May 4 - Apr 19).days = 15
        r_start, r_end = ROUND_SCHEDULE[0]
        return_date = date(2026, 4, 28)
        expected = (r_end - return_date).days / (r_end - r_start).days
        avails = compute_round_availability(return_date)
        assert abs(avails[0] - expected) < 1e-9
        # Available for round 2 onward (return before round 2 start 2026-05-05)
        assert avails[1] == 1.0
        assert avails[2] == 1.0
        assert avails[3] == 1.0

    def test_returns_mid_round2(self) -> None:
        # Return May 10: misses R1, partial R2, full R3/R4
        r_start, r_end = ROUND_SCHEDULE[1]
        return_date = date(2026, 5, 10)
        expected_r2 = (r_end - return_date).days / (r_end - r_start).days
        avails = compute_round_availability(return_date)
        assert avails[0] == 0.0  # misses round 1
        assert abs(avails[1] - expected_r2) < 1e-9
        assert avails[2] == 1.0
        assert avails[3] == 1.0

    def test_returns_after_round1_end(self) -> None:
        # May 5 is round 2 start; return on May 4 → round 1 availability = 0, round 2 full.
        avails = compute_round_availability(date(2026, 5, 4))
        assert avails[0] == 0.0
        assert avails[1] == 1.0

    def test_all_values_in_unit_interval(self) -> None:
        for return_date in [None, date(2026, 4, 17), date(2026, 4, 28), date(2026, 7, 1)]:
            for a in compute_round_availability(return_date):
                assert 0.0 <= a <= 1.0


# ── _apply_scalar_override ───────────────────────────────────────────────────

class TestApplyScalarOverride:
    N_SIMS = 10_000
    N_TEAMS = 3
    N_STARS = 3
    N_ROUNDS = 4

    def _make_draws(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.uniform(size=(self.N_TEAMS, self.N_STARS, self.N_ROUNDS, self.N_SIMS))

    def test_zero_availability_sets_all_draws_to_one(self) -> None:
        draws = self._make_draws()
        _apply_scalar_override(draws, 0, 1, 0, 0.0, rng=None)
        assert np.all(draws[0, 1, 0, :] == 1.0)

    def test_full_availability_sets_all_draws_to_zero(self) -> None:
        # availability=1.0 is normally skipped by the caller, but the primitive
        # itself should set all draws to 0.0 (all healthy).
        draws = self._make_draws()
        _apply_scalar_override(draws, 0, 0, 0, 1.0, rng=None)
        assert np.all(draws[0, 0, 0, :] == 0.0)

    def test_partial_availability_correct_fraction(self) -> None:
        draws = self._make_draws()
        target = 0.6
        _apply_scalar_override(draws, 0, 1, 2, target, rng=None)
        n_healthy = int(np.sum(draws[0, 1, 2, :] == 0.0))
        assert abs(n_healthy / self.N_SIMS - target) < 1e-9  # deterministic split

    def test_only_target_slice_modified(self) -> None:
        draws = self._make_draws()
        original = draws.copy()
        _apply_scalar_override(draws, 1, 2, 3, 0.0, rng=None)
        # Every element except the overridden slice must be unchanged.
        mask = np.ones(draws.shape, dtype=bool)
        mask[1, 2, 3, :] = False
        np.testing.assert_array_equal(draws[mask], original[mask])

    def test_shuffle_preserves_fraction(self) -> None:
        draws = self._make_draws()
        rng = np.random.default_rng(0)
        target = 0.4
        _apply_scalar_override(draws, 0, 0, 0, target, rng=rng)
        n_healthy = int(np.sum(draws[0, 0, 0, :] == 0.0))
        assert abs(n_healthy / self.N_SIMS - target) < 1e-9

    def test_shuffle_changes_ordering_vs_no_shuffle(self) -> None:
        # Without shuffle the first n_healthy positions are 0.0.
        draws_noshuffle = self._make_draws()
        _apply_scalar_override(draws_noshuffle, 0, 0, 0, 0.5, rng=None)
        draws_shuffle = self._make_draws()
        _apply_scalar_override(draws_shuffle, 0, 0, 0, 0.5, rng=np.random.default_rng(99))
        # Both have the same count of zeros but (very likely) different positions.
        assert int(np.sum(draws_noshuffle[0, 0, 0, :] == 0.0)) == int(
            np.sum(draws_shuffle[0, 0, 0, :] == 0.0)
        )
        assert not np.array_equal(draws_noshuffle[0, 0, 0, :], draws_shuffle[0, 0, 0, :])

    def test_team_index_zero_isolation(self) -> None:
        # Overriding team 0 should not affect team 1.
        draws = self._make_draws()
        original_team1 = draws[1, :, :, :].copy()
        _apply_scalar_override(draws, 0, 0, 0, 0.0, rng=None)
        np.testing.assert_array_equal(draws[1, :, :, :], original_team1)
