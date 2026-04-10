"""text_utils.py — Shared text normalisation utilities.

These helpers are used by multiple steps within src/data/steps/ to produce
consistent player-name keys across data sources that use different spellings,
diacritics, or generational suffixes (Jr., Sr., II, III, …).
"""

from __future__ import annotations

import re
import unicodedata


def normalise_player_name(name: str) -> str:
    """Lowercase, strip diacritics, and normalise a player name to a plain key.

    Steps applied in order:
      1. NFKD decomposition + ASCII encoding to strip diacritics
         (e.g. 'Kukoč' → 'Kukoc').
      2. Lowercase and replace any non-alphanumeric run with a single underscore.
      3. Strip leading/trailing underscores.
      4. Remove a trailing generational suffix: _jr, _sr, _ii, _iii, _iv, _v
         (e.g. 'jimmy_butler_iii' → 'jimmy_butler').

    This allows names from Advanced.csv, PlayerStatisticsMisc.csv, and the
    nba_api to resolve to the same key despite minor formatting differences.

    Args:
        name: Raw player name string.

    Returns:
        Normalised lowercase key with underscores and no diacritics or suffixes.
    """
    ascii_name = (
        unicodedata.normalize("NFKD", str(name)).encode("ascii", errors="ignore").decode("ascii")
    )
    normalised = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    return re.sub(r"_(?:jr|sr|ii|iii|iv|v)$", "", normalised)
