"""fetch.py — Raw data acquisition for the NBA playoff model.

Priority order:
1. Static Kaggle CSVs placed in data/raw/
2. nba_api for supplemental data
3. Basketball Reference — one-shot fallback only. If blocked, log and skip.

All outputs land in data/raw/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def load_kaggle_csv(filename: str) -> pd.DataFrame:
    """Load a static Kaggle CSV from data/raw/.

    Args:
        filename: Name of the CSV file (e.g. 'nba_games.csv').

    Returns:
        DataFrame with the raw CSV contents.

    Raises:
        FileNotFoundError: If the file does not exist in data/raw/.
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the Kaggle CSV in data/raw/ before running."
        )
    logger.info("Loading Kaggle CSV: %s", path)
    return pd.read_csv(path)


def fetch_nba_api(endpoint: str, params: dict) -> pd.DataFrame:
    """Fetch a dataset from nba_api.

    Args:
        endpoint: nba_api endpoint class name (e.g. 'leaguegamefinder').
        params: Keyword arguments forwarded to the endpoint constructor.

    Returns:
        First result set as a DataFrame.
    """
    # Import deferred so the rest of the module works without nba_api installed.
    from nba_api.stats.endpoints import leaguegamefinder  # noqa: F401 — example import

    raise NotImplementedError(
        f"Implement fetch_nba_api for endpoint={endpoint!r}. "
        "Use nba_api.<endpoint>(**params).get_data_frames()[0]."
    )


def fetch_bref_once(url: str) -> pd.DataFrame | None:
    """Attempt a one-shot Basketball Reference scrape.

    If the request fails (403, rate-limit, CAPTCHA, or any HTTP error) this
    function logs the failure and returns None. **Never call this more than
    once per session** — if it fails, switch to an alternative source.

    Args:
        url: Full Basketball Reference URL to scrape.

    Returns:
        DataFrame of the first HTML table found, or None on failure.
    """
    import requests

    logger.info("Attempting one-shot Basketball Reference scrape: %s", url)
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as exc:
        logger.warning(
            "Basketball Reference fetch FAILED (%s). Will not retry. "
            "Log: %s — switching to alternative source.",
            type(exc).__name__,
            exc,
        )
        return None

    tables = pd.read_html(resp.text)
    if not tables:
        logger.warning("No tables found at %s. Returning None.", url)
        return None
    return tables[0]
