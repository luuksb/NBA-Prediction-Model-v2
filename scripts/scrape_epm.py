"""One-time scraper for EPM data from dunksandthrees.com.

Fetches end-of-season EPM for seasons 2002-2025 (one request per season).
Saves combined data to data/raw/epm.parquet.
"""

import re
import time

import pandas as pd
import pyjson5
import requests

BASE_URL = "https://dunksandthrees.com/epm"
SEASONS = range(2002, 2026)
OUTPUT_PATH = "data/raw/epm.parquet"
REQUEST_DELAY = 2.0  # seconds between requests


def extract_bracket(text: str, start: int) -> str:
    """Extract a balanced [...] or {...} block starting at `start`.

    Args:
        text: Full string to search in.
        start: Index of the opening bracket character.

    Returns:
        The full balanced block as a string.
    """
    open_char = text[start]
    close_char = "]" if open_char == "[" else "}"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def extract_stats(html: str, season: int) -> list[dict]:
    """Extract player stats array from SvelteKit page HTML.

    The data is embedded as a JS object literal (unquoted keys) in the form:
    {type:"data",data:{date:"...",stats:[{season:...,player_name:...}]}}

    Args:
        html: Raw HTML of the EPM page.
        season: Season year (used for logging).

    Returns:
        List of player stat dicts, empty list if extraction fails.
    """
    # Locate the start of the stats array
    match = re.search(r'\bstats:\[', html)
    if not match:
        print(f"  [WARN] No stats array found for season {season}")
        return []

    array_str = extract_bracket(html, match.end() - 1)
    if not array_str:
        print(f"  [WARN] Could not extract balanced array for season {season}")
        return []

    try:
        return pyjson5.loads(array_str)
    except Exception as e:
        print(f"  [WARN] Parse error for season {season}: {e}")
        return []


def scrape_season(session: requests.Session, season: int) -> pd.DataFrame:
    """Fetch and parse EPM data for a single season.

    Args:
        session: Requests session to reuse.
        season: Season end-year (e.g. 2015 = 2014-15 season).

    Returns:
        DataFrame with player EPM data for that season, or empty DataFrame.
    """
    url = f"{BASE_URL}?season={season}"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Request failed for season {season}: {e}")
        return pd.DataFrame()

    records = extract_stats(resp.text, season)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"  season={season}: {len(df)} players, cols={list(df.columns[:8])}")
    return df


def main() -> None:
    """Scrape all seasons and save to parquet."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research/personal use)"})

    all_frames = []
    for season in SEASONS:
        print(f"Fetching season {season}...")
        df = scrape_season(session, season)
        if not df.empty:
            all_frames.append(df)
        time.sleep(REQUEST_DELAY)

    if not all_frames:
        print("No data collected — aborting.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal rows: {len(combined)}")
    print(f"Seasons covered: {sorted(combined['season'].unique())}")
    print(f"Columns: {list(combined.columns)}")

    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
