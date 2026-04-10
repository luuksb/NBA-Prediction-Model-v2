"""aggregate.py — Aggregate 50,000 bracket simulation outcomes into summary statistics."""

from __future__ import annotations

from collections import Counter

import pandas as pd


def aggregate_outcomes(outcomes: list[dict], all_teams: list[str]) -> dict:
    """Compute championship and round-advancement probabilities from simulation outcomes.

    Args:
        outcomes: List of per-iteration dicts from run_bracket.run_simulations().
        all_teams: All 16 participating team IDs.

    Returns:
        Dict with keys:
        - championship_prob: {team: probability}
        - round_advancement: {team: {round_num: probability}}
        - most_common_champion: team ID
        - most_common_finals: (team_a, team_b) tuple
        - n_sims: number of iterations
    """
    n = len(outcomes)
    champ_counts: Counter = Counter()
    finals_counts: Counter = Counter()
    # round_reached[team][round] = count of iterations where team reached that round
    round_reached: dict[str, Counter] = {t: Counter() for t in all_teams}

    for outcome in outcomes:
        champ = outcome["champion"]
        champ_counts[champ] += 1

        finalist_pair = tuple(sorted([outcome["finalist_east"], outcome["finalist_west"]]))
        finals_counts[finalist_pair] += 1

        # A team "reached" round R if it either won round R-1 (i.e., exited in R or later)
        # or won the championship (exited after all rounds)
        for team in all_teams:
            exit_round = outcome["round_exits"].get(team)
            if exit_round is None:
                # Champion — reached all 4 rounds
                for r in range(1, 5):
                    round_reached[team][r] += 1
            else:
                # Reached up to and including their exit round
                for r in range(1, exit_round + 1):
                    round_reached[team][r] += 1

    championship_prob = {t: champ_counts[t] / n for t in all_teams}
    round_advancement = {t: {r: round_reached[t][r] / n for r in range(1, 5)} for t in all_teams}
    most_common_champion = champ_counts.most_common(1)[0][0] if champ_counts else None
    most_common_finals = finals_counts.most_common(1)[0][0] if finals_counts else None

    return {
        "championship_prob": championship_prob,
        "round_advancement": round_advancement,
        "most_common_champion": most_common_champion,
        "most_common_finals": most_common_finals,
        "n_sims": n,
    }
