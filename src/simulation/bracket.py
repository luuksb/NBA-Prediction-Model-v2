"""bracket.py — NBA playoff bracket structure and seeding rules.

No play-in tournament. Bracket starts from 16 teams for all years including 2021+.
Follows standard NBA seeding: 1v8, 2v7, 3v6, 4v5 in each conference first round.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Series:
    """A single playoff series between two teams."""

    high_seed: str   # Team identifier for the higher seed
    low_seed: str    # Team identifier for the lower seed
    round_num: int   # 1=first round, 2=conf semis, 3=conf finals, 4=finals
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


def build_bracket(year: int, east_seeds: list[str], west_seeds: list[str]) -> Bracket:
    """Construct an empty bracket ready for simulation.

    Args:
        year: Season year.
        east_seeds: 8 Eastern Conference team IDs, ordered 1st to 8th.
        west_seeds: 8 Western Conference team IDs, ordered 1st to 8th.

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
    )


def advance_bracket(bracket: Bracket) -> Bracket | None:
    """Build the next round's matchups from the completed previous round.

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
        # Winners 0,1 → East semi; winners 2,3 → East semi; 4,5 → West; 6,7 → West
        east_winners = winners[:4]
        west_winners = winners[4:]
        next_series = [
            Series(east_winners[0], east_winners[1], 2, "East"),
            Series(east_winners[2], east_winners[3], 2, "East"),
            Series(west_winners[0], west_winners[1], 2, "West"),
            Series(west_winners[2], west_winners[3], 2, "West"),
        ]
    elif round_num == 2:
        east_conf_winners = winners[:2]
        west_conf_winners = winners[2:]
        next_series = [
            Series(east_conf_winners[0], east_conf_winners[1], 3, "East"),
            Series(west_conf_winners[0], west_conf_winners[1], 3, "West"),
        ]
    elif round_num == 3:
        next_series = [Series(winners[0], winners[1], 4, "Finals")]
    else:
        return None

    bracket.rounds.append(next_series)
    return bracket
