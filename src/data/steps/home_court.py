"""home_court.py — Step: home-court advantage indicator.

In NBA playoffs the higher seed hosts games 1, 2, 5, and 7; the lower seed
never has a home-court edge. The sole exception is the 2020 COVID bubble
(Walt Disney World), where all games were played at a neutral site.

This step adds one series-level binary column:
    delta_home_court_advantage — 1 for all normal playoff years, 0 for 2020.

The value is already a delta (high-seed perspective) so no _high/_low
pair is needed; the column name starts with delta_ and is preserved as-is
by assemble.compute_deltas().
"""

from __future__ import annotations

import pandas as pd

# Seasons played at a neutral site — home-court advantage did not apply.
NEUTRAL_SITE_SEASONS: frozenset[int] = frozenset({2020})


def run(base: pd.DataFrame) -> pd.DataFrame:
    """Add delta_home_court_advantage to the base series DataFrame.

    Args:
        base: Base series DataFrame with at least a 'season' column.

    Returns:
        DataFrame with delta_home_court_advantage added.
        1 for all normal seasons; 0 for neutral-site seasons (2020 bubble).
    """
    df = base.copy()
    df["delta_home_court_advantage"] = (
        ~df["season"].isin(NEUTRAL_SITE_SEASONS)
    ).astype(int)
    return df
