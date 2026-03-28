"""feature_sets.py — Generate candidate feature combinations for model selection.

Reads configs/features.yaml and configs/model_selection.yaml to produce the
list of feature sets to evaluate. Supports two strategies:
- all_subsets: all subsets of active features up to max_feature_set_size
- curated_groups: fixed groups defined in model_selection.yaml
- both: union of the two
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import yaml

FEATURES_CONFIG = Path("configs/features.yaml")
MODEL_SELECTION_CONFIG = Path("configs/model_selection.yaml")


def _load_configs() -> tuple[dict, dict]:
    with open(FEATURES_CONFIG) as f:
        feat_cfg = yaml.safe_load(f)
    with open(MODEL_SELECTION_CONFIG) as f:
        ms_cfg = yaml.safe_load(f)
    return feat_cfg, ms_cfg


def get_active_features() -> list[str]:
    """Return the names of all active features from configs/features.yaml.

    Returns:
        List of active feature name strings.
    """
    feat_cfg, _ = _load_configs()
    return [f["name"] for f in feat_cfg["features"] if f.get("active", True)]


def generate_all_subsets(
    features: list[str],
    max_size: int,
    min_size: int = 1,
) -> list[list[str]]:
    """Generate all subsets of features with sizes in [min_size, max_size].

    Args:
        features: Pool of available feature names.
        max_size: Maximum number of features per subset (inclusive).
        min_size: Minimum number of features per subset (inclusive).

    Returns:
        List of feature subsets (each subset is a sorted list of names).
    """
    result: list[list[str]] = []
    for size in range(min_size, min(max_size, len(features)) + 1):
        for combo in combinations(features, size):
            result.append(sorted(combo))
    return result


def get_curated_groups() -> list[list[str]]:
    """Return curated feature groups from configs/model_selection.yaml.

    Returns:
        List of feature lists, one per curated group.
    """
    _, ms_cfg = _load_configs()
    return [group["features"] for group in ms_cfg.get("curated_groups", [])]


def get_forbidden_pairs() -> list[tuple[str, str]]:
    """Return forbidden feature pairs from configs/model_selection.yaml.

    Returns:
        List of (feature_a, feature_b) tuples that must not appear together
        in the same model. Only pairs where both features are active matter.
    """
    _, ms_cfg = _load_configs()
    raw = ms_cfg.get("forbidden_pairs", [])
    return [(pair[0], pair[1]) for pair in raw]


def filter_forbidden_pairs(
    combos: list[list[str]],
    forbidden_pairs: list[tuple[str, str]],
) -> list[list[str]]:
    """Remove combinations that contain any forbidden feature pair.

    Args:
        combos: List of feature combinations (each a sorted list of names).
        forbidden_pairs: Pairs of features that must not appear together.

    Returns:
        Filtered list with all invalid combinations removed.
    """
    if not forbidden_pairs:
        return combos
    result = []
    for combo in combos:
        combo_set = set(combo)
        if not any(a in combo_set and b in combo_set for a, b in forbidden_pairs):
            result.append(combo)
    return result


def get_candidate_feature_sets() -> list[list[str]]:
    """Return all candidate feature sets according to the configured strategy.

    Reads feature_set_strategy from configs/model_selection.yaml.

    Returns:
        Deduplicated list of feature-name lists to evaluate.
    """
    feat_cfg, ms_cfg = _load_configs()
    active = [f["name"] for f in feat_cfg["features"] if f.get("active", True)]
    max_size = ms_cfg.get("max_feature_set_size", 6)
    strategy = ms_cfg.get("feature_set_strategy", "both")

    subsets: list[list[str]] = []
    if strategy in ("all_subsets", "both"):
        subsets.extend(generate_all_subsets(active, max_size))
    if strategy in ("curated_groups", "both"):
        subsets.extend(get_curated_groups())

    # Deduplicate while preserving order
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for fs in subsets:
        key = tuple(sorted(fs))
        if key not in seen:
            seen.add(key)
            unique.append(fs)
    return unique
