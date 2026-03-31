"""bracket_builder.py — Build a display-ready bracket structure from simulation outputs.

Consumes seeds from load_bracket_seeds() and round_advancement DataFrame from
load_simulation_results(). Produces a nested dict for HTML rendering.

No cross-module imports from src.model, src.simulation, src.data, or src.injury.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Logit win probability
# ---------------------------------------------------------------------------

def compute_win_prob(
    high_abbrev: str,
    low_abbrev: str,
    team_features: pd.DataFrame,
    spec: dict[str, Any],
) -> float:
    """Compute P(high_abbrev wins the series) using the locked logit model spec.

    Args:
        high_abbrev: Abbreviation of the higher-seeded team.
        low_abbrev: Abbreviation of the lower-seeded team.
        team_features: DataFrame indexed by team abbreviation with raw feature columns.
        spec: Model spec dict with keys: features, intercept, coefficients.

    Returns:
        Probability in [0, 1] that the higher seed wins. Returns 0.5 if either
        team is missing from team_features.
    """
    if team_features.empty:
        return 0.5
    if high_abbrev not in team_features.index or low_abbrev not in team_features.index:
        return 0.5
    row_high = team_features.loc[high_abbrev].to_dict()
    row_low = team_features.loc[low_abbrev].to_dict()
    logit = spec["intercept"]
    for feat in spec["features"]:
        raw = feat.removeprefix("delta_")
        delta = float(row_high.get(raw, 0.0)) - float(row_low.get(raw, 0.0))
        logit += spec["coefficients"][feat] * delta
    return 1.0 / (1.0 + math.exp(-logit))


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _get_adv_prob(abbrev: str, round_num: int, adv_df: pd.DataFrame) -> float:
    """Look up advancement probability for a team in a given round.

    Args:
        abbrev: Team abbreviation.
        round_num: Round number (1–4).
        adv_df: round_advancement DataFrame with columns [team, round, advancement_prob].

    Returns:
        Advancement probability, or 0.0 if team/round not found.
    """
    mask = (adv_df["team"] == abbrev) & (adv_df["round"] == round_num)
    rows = adv_df.loc[mask, "advancement_prob"]
    return float(rows.iloc[0]) if not rows.empty else 0.0


def _cond_win_prob(abbrev: str, round_num: int, adv_df: pd.DataFrame) -> float:
    """Compute P(winning round_num | reached round_num).

    Formula: adv_prob(round_num + 1) / adv_prob(round_num).
    Guards against division by zero when a team never reached this round.

    Args:
        abbrev: Team abbreviation.
        round_num: Round number (1–3). For round 4 (Finals), uses
            adv_prob(r4) / adv_prob(r3) where r4 == championship_prob.
        adv_df: round_advancement DataFrame.

    Returns:
        Float in [0, 1]. Returns 0.0 if adv_prob(round_num) == 0.
    """
    denom = _get_adv_prob(abbrev, round_num, adv_df)
    if denom == 0.0:
        return 0.0
    numer = _get_adv_prob(abbrev, round_num + 1, adv_df)
    return min(numer / denom, 1.0)


# ---------------------------------------------------------------------------
# TeamNode builder
# ---------------------------------------------------------------------------

def _build_team_node(
    abbrev: str,
    seed: int,
    conference: str,
    round_num: int,
    adv_df: pd.DataFrame,
    logo_url_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Build a single TeamNode dict for use in the bracket structure.

    Args:
        abbrev: Team abbreviation, e.g. 'BOS'.
        seed: Original conference seed (1–8).
        conference: 'east' or 'west'.
        round_num: The round this node participates in (1–4).
        adv_df: round_advancement DataFrame.
        logo_url_fn: Callable mapping abbrev -> URL string.

    Returns:
        TeamNode dict with keys: abbrev, seed, conference, adv_prob,
        cond_win_prob, logo_url.
    """
    adv = _get_adv_prob(abbrev, round_num, adv_df)
    cond = _cond_win_prob(abbrev, round_num, adv_df)
    return {
        "abbrev": abbrev,
        "seed": seed,
        "conference": conference,
        "adv_prob": adv,
        "cond_win_prob": cond,
        "logo_url": logo_url_fn(abbrev),
    }


# ---------------------------------------------------------------------------
# Round builders
# ---------------------------------------------------------------------------

def _build_r1_matchups(
    seeds: list[str],
    conference: str,
    adv_df: pd.DataFrame,
    logo_url_fn: Callable[[str], str],
    team_features: pd.DataFrame | None = None,
    spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the four Round 1 matchup dicts for one conference.

    Seeding pairs: 1v8, 2v7, 3v6, 4v5 (0-indexed: (0,7),(1,6),(2,5),(3,4)).

    Args:
        seeds: 8 team abbreviations, ordered 1st (index 0) to 8th (index 7).
        conference: 'east' or 'west'.
        adv_df: round_advancement DataFrame.
        logo_url_fn: Callable mapping abbrev -> URL string.
        team_features: Optional DataFrame of raw per-team features (indexed by team).
        spec: Optional model spec dict for direct logit win probability.

    Returns:
        List of 4 matchup dicts in order 1v8, 2v7, 3v6, 4v5.
        Each matchup: {'high': TeamNode, 'low': TeamNode}.
        When team_features and spec are provided, cond_win_prob is the direct
        logit model prediction; otherwise it falls back to adv_prob ratios.
    """
    pairs = [(0, 7), (1, 6), (2, 5), (3, 4)]
    matchups = []
    for hi_idx, lo_idx in pairs:
        hi = _build_team_node(seeds[hi_idx], hi_idx + 1, conference, 1, adv_df, logo_url_fn)
        lo = _build_team_node(seeds[lo_idx], lo_idx + 1, conference, 1, adv_df, logo_url_fn)
        if team_features is not None and spec is not None:
            p = compute_win_prob(seeds[hi_idx], seeds[lo_idx], team_features, spec)
            hi["cond_win_prob"] = p
            lo["cond_win_prob"] = 1.0 - p
        matchups.append({"high": hi, "low": lo})
    return matchups


def _pick_rep(
    candidates: list[str],
    seeds_map: dict[str, int],
    conference: str,
    round_num: int,
    adv_df: pd.DataFrame,
    logo_url_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Pick the most-likely representative team from a set of candidates.

    The representative is the candidate with the highest advancement_prob
    for round_num (i.e. the team most likely to appear in this round).

    Args:
        candidates: Team abbreviations to choose from.
        seeds_map: Maps abbrev -> original seed number.
        conference: 'east', 'west', or 'finals'.
        round_num: The round this node participates in.
        adv_df: round_advancement DataFrame.
        logo_url_fn: Callable mapping abbrev -> URL string.

    Returns:
        A TeamNode dict for the chosen representative.
    """
    best = max(candidates, key=lambda t: _get_adv_prob(t, round_num, adv_df))
    return _build_team_node(best, seeds_map[best], conference, round_num, adv_df, logo_url_fn)


def _build_r2_matchups(
    r1_matchups: list[dict[str, Any]],
    conference: str,
    adv_df: pd.DataFrame,
    logo_url_fn: Callable[[str], str],
    team_features: pd.DataFrame | None = None,
    spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the two Round 2 matchup dicts for one conference.

    NBA bracket pairing: (1/8 winner) vs (4/5 winner), (2/7 winner) vs (3/6 winner).
    Representative team per slot = candidate with highest advancement_prob for R2.

    Args:
        r1_matchups: Output of _build_r1_matchups (4 matchups: 1v8, 2v7, 3v6, 4v5).
        conference: 'east' or 'west'.
        adv_df: round_advancement DataFrame.
        logo_url_fn: Callable mapping abbrev -> URL string.
        team_features: Optional DataFrame of raw per-team features (indexed by team).
        spec: Optional model spec dict for direct logit win probability.

    Returns:
        List of 2 matchup dicts:
            [0]: upper bracket (1/8 vs 4/5)
            [1]: lower bracket (2/7 vs 3/6)
    """
    # Slot groups: r1[0]=1v8, r1[3]=4v5 ; r1[1]=2v7, r1[2]=3v6
    slot_groups = [
        (r1_matchups[0], r1_matchups[3]),
        (r1_matchups[1], r1_matchups[2]),
    ]

    matchups = []
    for g_a, g_b in slot_groups:
        teams_a = [g_a["high"]["abbrev"], g_a["low"]["abbrev"]]
        seeds_a = {g_a["high"]["abbrev"]: g_a["high"]["seed"],
                   g_a["low"]["abbrev"]: g_a["low"]["seed"]}
        teams_b = [g_b["high"]["abbrev"], g_b["low"]["abbrev"]]
        seeds_b = {g_b["high"]["abbrev"]: g_b["high"]["seed"],
                   g_b["low"]["abbrev"]: g_b["low"]["seed"]}

        rep_a = _pick_rep(teams_a, seeds_a, conference, 2, adv_df, logo_url_fn)
        rep_b = _pick_rep(teams_b, seeds_b, conference, 2, adv_df, logo_url_fn)

        # Assign high/low by original seed number (lower number = higher seed)
        if rep_a["seed"] <= rep_b["seed"]:
            hi, lo = rep_a, rep_b
        else:
            hi, lo = rep_b, rep_a

        if team_features is not None and spec is not None:
            p = compute_win_prob(hi["abbrev"], lo["abbrev"], team_features, spec)
            hi["cond_win_prob"] = p
            lo["cond_win_prob"] = 1.0 - p

        matchups.append({"high": hi, "low": lo})

    return matchups


def _build_r3_matchup(
    r2_matchups: list[dict[str, Any]],
    conference: str,
    adv_df: pd.DataFrame,
    logo_url_fn: Callable[[str], str],
    team_features: pd.DataFrame | None = None,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the conference finals (Round 3) matchup dict.

    Representative teams are the R2 participants with highest advancement_prob for R3.

    Args:
        r2_matchups: Output of _build_r2_matchups (2 matchups).
        conference: 'east' or 'west'.
        adv_df: round_advancement DataFrame.
        logo_url_fn: Callable mapping abbrev -> URL string.
        team_features: Optional DataFrame of raw per-team features (indexed by team).
        spec: Optional model spec dict for direct logit win probability.

    Returns:
        Single matchup dict {'high': TeamNode, 'low': TeamNode}.
    """
    seeds_map: dict[str, int] = {}
    for m in r2_matchups:
        for side in ("high", "low"):
            t = m[side]
            seeds_map[t["abbrev"]] = t["seed"]

    rep_a_candidates = [r2_matchups[0]["high"]["abbrev"], r2_matchups[0]["low"]["abbrev"]]
    rep_b_candidates = [r2_matchups[1]["high"]["abbrev"], r2_matchups[1]["low"]["abbrev"]]
    seeds_a = {t: seeds_map[t] for t in rep_a_candidates}
    seeds_b = {t: seeds_map[t] for t in rep_b_candidates}

    rep_a = _pick_rep(rep_a_candidates, seeds_a, conference, 3, adv_df, logo_url_fn)
    rep_b = _pick_rep(rep_b_candidates, seeds_b, conference, 3, adv_df, logo_url_fn)

    if rep_a["seed"] <= rep_b["seed"]:
        hi, lo = rep_a, rep_b
    else:
        hi, lo = rep_b, rep_a

    if team_features is not None and spec is not None:
        p = compute_win_prob(hi["abbrev"], lo["abbrev"], team_features, spec)
        hi["cond_win_prob"] = p
        lo["cond_win_prob"] = 1.0 - p

    return {"high": hi, "low": lo}


def _build_finals_matchup(
    east_r3: dict[str, Any],
    west_r3: dict[str, Any],
    adv_df: pd.DataFrame,
    logo_url_fn: Callable[[str], str],
    team_features: pd.DataFrame | None = None,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the NBA Finals (Round 4) matchup dict.

    Representative teams are the R3 participants with highest advancement_prob for R4.

    Args:
        east_r3: Output of _build_r3_matchup for East.
        west_r3: Output of _build_r3_matchup for West.
        adv_df: round_advancement DataFrame.
        logo_url_fn: Callable mapping abbrev -> URL string.
        team_features: Optional DataFrame of raw per-team features (indexed by team).
        spec: Optional model spec dict for direct logit win probability.

    Returns:
        Single matchup dict {'high': TeamNode, 'low': TeamNode}.
        Nodes carry conference='east'/'west' from their source.
    """
    east_candidates = [east_r3["high"]["abbrev"], east_r3["low"]["abbrev"]]
    east_seeds = {east_r3["high"]["abbrev"]: east_r3["high"]["seed"],
                  east_r3["low"]["abbrev"]: east_r3["low"]["seed"]}
    west_candidates = [west_r3["high"]["abbrev"], west_r3["low"]["abbrev"]]
    west_seeds = {west_r3["high"]["abbrev"]: west_r3["high"]["seed"],
                  west_r3["low"]["abbrev"]: west_r3["low"]["seed"]}

    east_rep = _pick_rep(east_candidates, east_seeds, "east", 4, adv_df, logo_url_fn)
    west_rep = _pick_rep(west_candidates, west_seeds, "west", 4, adv_df, logo_url_fn)

    # For Finals, assign high by higher adv_prob at R4
    if east_rep["adv_prob"] >= west_rep["adv_prob"]:
        hi, lo = east_rep, west_rep
    else:
        hi, lo = west_rep, east_rep

    if team_features is not None and spec is not None:
        p = compute_win_prob(hi["abbrev"], lo["abbrev"], team_features, spec)
        hi["cond_win_prob"] = p
        lo["cond_win_prob"] = 1.0 - p

    return {"high": hi, "low": lo}


# ---------------------------------------------------------------------------
# Top-level public functions
# ---------------------------------------------------------------------------

def build_bracket_structure(
    east_seeds: list[str],
    west_seeds: list[str],
    adv_df: pd.DataFrame,
    logo_url_fn: Callable[[str], str],
    predicted_champion: str | None = None,
    team_features: pd.DataFrame | None = None,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete display-ready bracket structure.

    Args:
        east_seeds: 8 Eastern Conference team abbreviations, seed 1–8 order.
        west_seeds: 8 Western Conference team abbreviations, seed 1–8 order.
        adv_df: round_advancement DataFrame from load_simulation_results().
        logo_url_fn: Callable mapping abbrev -> URL string.
        predicted_champion: Abbreviation of predicted champion (from summary).
        team_features: Optional DataFrame of raw per-team features (indexed by team).
            When provided together with spec, win probabilities are computed as
            direct logit model predictions rather than simulation advancement ratios.
        spec: Optional model spec dict (features, intercept, coefficients).

    Returns:
        Dict with keys:
            'west': {1: [4 matchups], 2: [2 matchups], 3: [1 matchup]}
            'east': {1: [4 matchups], 2: [2 matchups], 3: [1 matchup]}
            'finals': {4: [1 matchup]}
            'champion': TeamNode for the predicted champion (or None)
    """
    # Build each conference
    west_r1 = _build_r1_matchups(west_seeds, "west", adv_df, logo_url_fn, team_features, spec)
    west_r2 = _build_r2_matchups(west_r1, "west", adv_df, logo_url_fn, team_features, spec)
    west_r3 = _build_r3_matchup(west_r2, "west", adv_df, logo_url_fn, team_features, spec)

    east_r1 = _build_r1_matchups(east_seeds, "east", adv_df, logo_url_fn, team_features, spec)
    east_r2 = _build_r2_matchups(east_r1, "east", adv_df, logo_url_fn, team_features, spec)
    east_r3 = _build_r3_matchup(east_r2, "east", adv_df, logo_url_fn, team_features, spec)

    finals = _build_finals_matchup(east_r3, west_r3, adv_df, logo_url_fn, team_features, spec)

    # Champion node
    champion: dict[str, Any] | None = None
    if predicted_champion:
        # Find the predicted champion in the Finals matchup
        for side in ("high", "low"):
            if finals[side]["abbrev"] == predicted_champion:
                champion = finals[side]
                break
        # Fallback: build node directly if not in Finals representative
        if champion is None:
            all_seeds = {t: (i + 1) for i, t in enumerate(east_seeds)}
            all_seeds.update({t: (i + 1) for i, t in enumerate(west_seeds)})
            conf = "east" if predicted_champion in east_seeds else "west"
            champion = _build_team_node(
                predicted_champion,
                all_seeds.get(predicted_champion, 0),
                conf,
                4,
                adv_df,
                logo_url_fn,
            )

    return {
        "west": {1: west_r1, 2: west_r2, 3: [west_r3]},
        "east": {1: east_r1, 2: east_r2, 3: [east_r3]},
        "finals": {4: [finals]},
        "champion": champion,
    }


def get_upsets(
    east_seeds: list[str],
    west_seeds: list[str],
    adv_df: pd.DataFrame,
    upset_threshold: float = 0.50,
    team_features: pd.DataFrame | None = None,
    spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Identify notable upsets from simulation output.

    Round 1: all lower seeds listed (seeds 5–8).
    Rounds 2–4: only lower seeds with cond_win_prob > upset_threshold.

    Args:
        east_seeds: 8 Eastern Conference team abbreviations, seed 1–8 order.
        west_seeds: 8 Western Conference team abbreviations, seed 1–8 order.
        adv_df: round_advancement DataFrame.
        upset_threshold: Min conditional win prob for rounds 2–4 upsets.
        team_features: Optional DataFrame of raw per-team features (indexed by team).
            When provided together with spec, win probabilities are direct logit
            model predictions rather than simulation advancement ratios.
        spec: Optional model spec dict (features, intercept, coefficients).

    Returns:
        List of upset dicts sorted ascending by cond_win_prob (most surprising
        first). Each dict has: matchup, underdog, underdog_seed, favourite,
        favourite_seed, cond_win_prob, round.
    """
    upsets: list[dict[str, Any]] = []

    # Round 1 — deterministic matchups: 1v8, 2v7, 3v6, 4v5
    r1_pairs = [(0, 7), (1, 6), (2, 5), (3, 4)]
    for seeds_list, conf in [(east_seeds, "East"), (west_seeds, "West")]:
        for hi_idx, lo_idx in r1_pairs:
            hi = seeds_list[hi_idx]
            lo = seeds_list[lo_idx]
            if team_features is not None and spec is not None:
                lo_prob = 1.0 - compute_win_prob(hi, lo, team_features, spec)
            else:
                lo_prob = _cond_win_prob(lo, 1, adv_df)
            upsets.append({
                "matchup": f"{conf}: #{hi_idx + 1} {hi} vs #{lo_idx + 1} {lo}",
                "underdog": lo,
                "underdog_seed": lo_idx + 1,
                "favourite": hi,
                "favourite_seed": hi_idx + 1,
                "cond_win_prob": lo_prob,
                "round": 1,
            })

    # Rounds 2–4: derive matchups from advancement probs
    # Build representative bracket to find round 2–4 matchups
    _dummy_logo = lambda a: ""  # noqa: E731
    west_r1 = _build_r1_matchups(west_seeds, "west", adv_df, _dummy_logo, team_features, spec)
    west_r2 = _build_r2_matchups(west_r1, "west", adv_df, _dummy_logo, team_features, spec)
    west_r3 = _build_r3_matchup(west_r2, "west", adv_df, _dummy_logo, team_features, spec)
    east_r1 = _build_r1_matchups(east_seeds, "east", adv_df, _dummy_logo, team_features, spec)
    east_r2 = _build_r2_matchups(east_r1, "east", adv_df, _dummy_logo, team_features, spec)
    east_r3 = _build_r3_matchup(east_r2, "east", adv_df, _dummy_logo, team_features, spec)
    finals = _build_finals_matchup(east_r3, west_r3, adv_df, _dummy_logo, team_features, spec)

    later_matchups: list[tuple[dict[str, Any], int, str]] = []
    for m in west_r2 + east_r2:
        later_matchups.append((m, 2, "Conf Semis"))
    for m in [west_r3, east_r3]:
        later_matchups.append((m, 3, "Conf Finals"))
    later_matchups.append((finals, 4, "Finals"))

    for matchup, rnd, label in later_matchups:
        hi = matchup["high"]
        lo = matchup["low"]
        lo_prob = lo["cond_win_prob"]
        if lo_prob > upset_threshold:
            upsets.append({
                "matchup": f"{label}: #{hi['seed']} {hi['abbrev']} vs #{lo['seed']} {lo['abbrev']}",
                "underdog": lo["abbrev"],
                "underdog_seed": lo["seed"],
                "favourite": hi["abbrev"],
                "favourite_seed": hi["seed"],
                "cond_win_prob": lo_prob,
                "round": rnd,
            })

    return sorted(upsets, key=lambda u: u["cond_win_prob"])
