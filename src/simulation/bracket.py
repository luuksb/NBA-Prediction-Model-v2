"""bracket.py — NBA playoff bracket structure and seeding rules.

No play-in tournament. Bracket starts from 16 teams for all years including 2021+.
Follows standard NBA seeding: 1v8, 2v7, 3v6, 4v5 in each conference first round.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Series:
    """A single playoff series between two teams."""

    high_seed: str  # Team identifier for the higher seed
    low_seed: str  # Team identifier for the lower seed
    round_num: int  # 1=first round, 2=conf semis, 3=conf finals, 4=finals
    conference: str  # 'East', 'West', or 'Finals'
    winner: str | None = None


@dataclass
class Bracket:
    """Full 16-team NBA playoff bracket."""

    year: int
    # East seeds 1–8 and West seeds 1–8 (team identifiers, ordered 1st to 8th)
    east_seeds: list[str] = field(default_factory=list)
    west_seeds: list[str] = field(default_factory=list)
    rounds: list[list[Series]] = field(default_factory=list)
    # Regular-season win totals used to break cross-conference ties (e.g. Finals)
    team_wins: dict[str, int] = field(default_factory=dict)


def build_first_round(seeds: list[str], conference: str) -> list[Series]:
    """Build the four first-round series for one conference.

    Seeding: 1v8, 2v7, 3v6, 4v5.

    Args:
        seeds: 8 team identifiers ordered 1st (best) to 8th (worst).
        conference: 'East' or 'West'.

    Returns:
        List of four Series objects.

    Raises:
        ValueError: If seeds does not contain exactly 8 entries.
    """
    if len(seeds) != 8:
        raise ValueError(f"Expected 8 seeds for {conference}, got {len(seeds)}.")
    matchups = [(0, 7), (1, 6), (2, 5), (3, 4)]  # 0-indexed
    return [
        Series(
            high_seed=seeds[hi],
            low_seed=seeds[lo],
            round_num=1,
            conference=conference,
        )
        for hi, lo in matchups
    ]


def build_bracket(
    year: int,
    east_seeds: list[str],
    west_seeds: list[str],
    team_wins: dict[str, int] | None = None,
) -> Bracket:
    """Construct an empty bracket ready for simulation.

    Args:
        year: Season year.
        east_seeds: 8 Eastern Conference team IDs, ordered 1st to 8th.
        west_seeds: 8 Western Conference team IDs, ordered 1st to 8th.
        team_wins: Optional dict mapping team ID → regular-season win total.
            Used to award home-court advantage when two teams have the same
            within-conference seed rank (i.e. the Finals, where both
            conference champions are seed #1).

    Returns:
        Bracket with Round 1 series populated; subsequent rounds are empty
        until winners are filled in by the simulation.
    """
    round1 = build_first_round(east_seeds, "East") + build_first_round(west_seeds, "West")
    return Bracket(
        year=year,
        east_seeds=east_seeds,
        west_seeds=west_seeds,
        rounds=[round1],
        team_wins=team_wins or {},
    )


def _seed_rank(team: str, bracket: Bracket) -> int:
    """Return the original 1-based seed rank of a team (1 = best seed)."""
    if team in bracket.east_seeds:
        return bracket.east_seeds.index(team) + 1
    return bracket.west_seeds.index(team) + 1


def _make_series(
    team_a: str, team_b: str, round_num: int, conference: str, bracket: Bracket
) -> Series:
    """Create a Series, assigning high/low seed based on original seedings.

    When both teams share the same within-conference seed rank (only possible
    in the Finals, where both conference champions are seed #1), regular-season
    win totals from bracket.team_wins break the tie. If win totals are
    unavailable or equal, team_a is treated as the higher seed.
    """
    rank_a = _seed_rank(team_a, bracket)
    rank_b = _seed_rank(team_b, bracket)
    if rank_a != rank_b:
        if rank_a < rank_b:
            return Series(team_a, team_b, round_num, conference)
        return Series(team_b, team_a, round_num, conference)
    # Tied seed ranks — break by regular-season wins (more wins = higher seed)
    wins_a = bracket.team_wins.get(team_a, 0)
    wins_b = bracket.team_wins.get(team_b, 0)
    if wins_a >= wins_b:
        return Series(team_a, team_b, round_num, conference)
    return Series(team_b, team_a, round_num, conference)


def advance_bracket(bracket: Bracket) -> Bracket | None:
    """Build the next round's matchups from the completed previous round.

    Standard NBA bracket (non-reseeded):
      R2: (1/8 winner) vs (4/5 winner), (2/7 winner) vs (3/6 winner)
      R3: top-half R2 winner vs bottom-half R2 winner

    Args:
        bracket: Bracket whose latest round has all winners filled in.

    Returns:
        Bracket with the next round appended, or None if the bracket is complete
        (the Finals winner has been determined).
    """
    current_round = bracket.rounds[-1]
    round_num = current_round[0].round_num

    if round_num == 4:
        return None  # Finals done

    winners = [s.winner for s in current_round]
    if any(w is None for w in winners):
        raise ValueError("Cannot advance bracket: not all series in current round have winners.")

    if round_num == 1:
        # R1 order: [1v8, 2v7, 3v6, 4v5] × 2 conferences
        # R2 pairings: (1/8 winner) vs (4/5 winner), (2/7 winner) vs (3/6 winner)
        east_winners = winners[:4]  # indices: 0=1/8w, 1=2/7w, 2=3/6w, 3=4/5w
        west_winners = winners[4:]
        next_series = [
            _make_series(east_winners[0], east_winners[3], 2, "East", bracket),
            _make_series(east_winners[1], east_winners[2], 2, "East", bracket),
            _make_series(west_winners[0], west_winners[3], 2, "West", bracket),
            _make_series(west_winners[1], west_winners[2], 2, "West", bracket),
        ]
    elif round_num == 2:
        east_conf_winners = winners[:2]
        west_conf_winners = winners[2:]
        next_series = [
            _make_series(east_conf_winners[0], east_conf_winners[1], 3, "East", bracket),
            _make_series(west_conf_winners[0], west_conf_winners[1], 3, "West", bracket),
        ]
    elif round_num == 3:
        next_series = [_make_series(winners[0], winners[1], 4, "Finals", bracket)]
    else:
        return None

    bracket.rounds.append(next_series)
    return bracket
