"""injury_overrides.py — Apply real-world injury overrides to pre-drawn injury arrays.

Scrapes current NBA injury data from CBS Sports and overwrites the corresponding
players' simulated availability values in the injury draws matrix.  Players absent
from the injury report are left completely untouched.

Injury draw schema (confirmed from export.py / run_bracket_sim.py):
    draws : np.ndarray, shape (n_teams, n_stars, n_rounds, n_sims), dtype float64
        Uniform [0, 1) values.  Healthy-check in simulate_series.py:
            ``draw <= mean_rate  →  player healthy``
        Override strategy:
            - availability = 0.0  →  set all draws to 1.0 (1.0 > any mean_rate ≤ 0.99)
            - 0 < availability < 1  →  set round(availability × n_sims) draws to 0.0,
              remainder to 1.0; shuffle with rng for unbiased iteration assignment.
            - availability = 1.0  →  skip (no override needed).

Usage
-----
>>> from src.injury.injury_overrides import apply_known_injuries
>>> injury_draws = load_injury_draws(year=2026, ...)   # dict from run_bracket_sim.py
>>> team_rosters = identify_top_players(...)           # DataFrame
>>> injury_draws = apply_known_injuries(injury_draws, team_rosters)
"""

from __future__ import annotations

import copy
import logging
import re
import unicodedata
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CBS_INJURIES_URL = "https://www.cbssports.com/nba/injuries/"

# ── 2026 playoff round schedule ─────────────────────────────────────────────
# Round end = day before next round starts (conservative upper bound).
# Finals end = start + max 7-game series duration (≈ 21 days).
ROUND_SCHEDULE: list[tuple[date, date]] = [
    (date(2026, 4, 19), date(2026, 5, 4)),   # Round 1
    (date(2026, 5, 5),  date(2026, 5, 19)),  # Round 2
    (date(2026, 5, 20), date(2026, 6, 3)),   # Conference Finals
    (date(2026, 6, 4),  date(2026, 6, 25)),  # NBA Finals
]

_MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s*(\d{4})?\b",
    re.IGNORECASE,
)
_SUFFIX_PAT = re.compile(r"_(jr|sr|ii|iii|iv|v)$")

# CBS Sports full team name (lower-case) → NBA 3-letter abbreviation
_CBS_TEAM_TO_ABBREV: dict[str, str] = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "golden st.": "GSW",
    "golden st": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "los angeles clippers": "LAC",
    "la clippers": "LAC",
    "l.a. clippers": "LAC",
    "los angeles lakers": "LAL",
    "la lakers": "LAL",
    "l.a. lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}


# ── Name normalisation ───────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Normalise a player display name to lowercase ASCII underscore format.

    Strips diacritics via NFKD decomposition (ć→c, č→c, etc.) and removes
    generational suffixes (Jr., III, …) so that scraped names match the keys
    produced by the injury simulation pipeline.

    Args:
        name: Display name, e.g. "Nikola Jokić" or "Jimmy Butler III".

    Returns:
        Normalised string, e.g. "nikola_jokic" or "jimmy_butler".
    """
    ascii_name = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )
    norm = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    return _SUFFIX_PAT.sub("", norm)


# ── Return-date parsing ──────────────────────────────────────────────────────

def _parse_return_date(text: str, reference_year: int = 2026) -> Optional[date]:
    """Parse a return-date or status string from CBS Sports into a date object.

    Mapping rules applied in order:
    - Empty / dash → ``None`` (indefinitely out).
    - "Probable" / "Questionable" / "Day-To-Day" / "Active" / "Game Time" →
      ``date.today()`` (treat as available; no override applied by caller).
    - "Out" / "Injured Reserve" / "Season" / "Indefinite" / "Doubtful" →
      ``None`` (no known return date → availability 0.0 for all rounds).
    - Specific date string ("Apr 20", "April 20, 2026") → parsed date.
    - Unrecognised string → ``None`` with a warning log (conservative).

    Args:
        text: Raw return-date or status text from CBS Sports.
        reference_year: Year to assume when the date string omits the year.

    Returns:
        Parsed ``date``, ``date.today()`` for near-term statuses, or ``None``
        when no return date is known.
    """
    text = text.strip()
    if not text or text in ("—", "-", "N/A", "n/a", "–"):
        return None

    # Try a specific date first — CBS often embeds it in descriptive text like
    # "Expected to be out until at least Apr 18". Parsing the date takes
    # precedence over keyword matching so that "out" substring matches don't
    # swallow strings that contain a concrete return date.
    m = _DATE_RE.search(text)
    if m:
        month_str = m.group(1)[:3].lower()
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else reference_year
        month = _MONTH_MAP.get(month_str)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    upper = text.upper()
    if any(kw in upper for kw in ("PROBABLE", "QUESTIONABLE", "DAY-TO-DAY", "ACTIVE", "GAME TIME")):
        return date.today()
    if any(kw in upper for kw in ("OUT", "INJURED RESERVE", "SEASON", "INDEFINITE", "DOUBTFUL")):
        return None

    logger.warning("Could not parse return date %r — treating as indefinitely out.", text)
    return None


# ── Availability scalar logic ────────────────────────────────────────────────

def compute_round_availability(return_date: Optional[date]) -> list[float]:
    """Compute per-round availability scalars for a player given their return date.

    Uses the module-level ``ROUND_SCHEDULE`` for the 2026 NBA playoffs.

    Per-round logic:
    - ``return_date`` is ``None`` → 0.0 for all rounds (indefinitely out).
    - ``return_date > round_end`` → 0.0 (misses the entire round).
    - ``return_date <= round_start`` → 1.0 (fully available; caller should skip
      override for this round).
    - ``return_date`` within the round →
      ``(round_end − return_date).days / (round_end − round_start).days``
      (pro-rata fraction of the round remaining after the player returns).

    Args:
        return_date: Expected return date, or ``None`` for indefinitely out.

    Returns:
        List of 4 floats in [0.0, 1.0], one per playoff round (R1 → Finals).
    """
    scalars: list[float] = []
    for round_start, round_end in ROUND_SCHEDULE:
        if return_date is None:
            scalars.append(0.0)
        elif return_date > round_end:
            scalars.append(0.0)
        elif return_date <= round_start:
            scalars.append(1.0)
        else:
            total_days = (round_end - round_start).days
            remaining_days = (round_end - return_date).days
            scalars.append(remaining_days / total_days)
    return scalars


# ── Draw-level override primitive ────────────────────────────────────────────

def _apply_scalar_override(
    draws: np.ndarray,
    team_idx: int,
    star_idx: int,
    round_idx: int,
    availability: float,
    rng: Optional[np.random.Generator],
) -> None:
    """Overwrite one player/round draw slice to achieve target availability.

    Sets ``round(availability × n_sims)`` draws to ``0.0`` (player healthy,
    because ``0.0 ≤`` any ``mean_rate > 0``) and the remaining draws to
    ``1.0`` (player injured, because ``1.0 >`` any ``mean_rate ≤ 0.99``).
    If ``rng`` is provided the resulting array is shuffled so that the
    healthy/injured assignment is random across iterations rather than
    front-loaded.

    Mutates ``draws`` in-place on the specified slice only.

    Args:
        draws: Full draws array shape (n_teams, n_stars, n_rounds, n_sims).
        team_idx: Index into axis 0 (team).
        star_idx: Index into axis 1 (star player, 0 = top-rated).
        round_idx: Index into axis 2 (0-indexed playoff round).
        availability: Target healthy fraction in [0.0, 1.0].
        rng: Optional generator for random shuffle ordering.
    """
    n_sims = draws.shape[3]
    n_healthy = int(round(availability * n_sims))
    new_vals = np.ones(n_sims, dtype=draws.dtype)
    if n_healthy > 0:
        new_vals[:n_healthy] = 0.0
    if rng is not None:
        rng.shuffle(new_vals)
    draws[team_idx, star_idx, round_idx, :] = new_vals


# ── CBS Sports scraper ───────────────────────────────────────────────────────

def _fetch_rendered_html(url: str, timeout_ms: int = 30_000) -> str:
    """Fetch fully JavaScript-rendered HTML using a headless Chromium browser.

    Args:
        url: Page URL to load.
        timeout_ms: Navigation timeout in milliseconds.

    Returns:
        Full rendered HTML string.

    Raises:
        RuntimeError: If Playwright or Chromium is not available.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run: "
            "pip install playwright && python -m playwright install chromium"
        ) from exc

    logger.info("Launching headless Chromium to fetch %s", url)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=timeout_ms)
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
        # Injury rows are populated by a secondary XHR after the initial load.
        # Wait until at least one <tbody> row appears in a TableBase table.
        try:
            page.wait_for_selector(
                ".TableBase-table tbody tr", timeout=timeout_ms
            )
        except Exception:
            logger.warning(
                "Timed out waiting for injury table rows — "
                "page structure may have changed."
            )
        html = page.content()
        browser.close()
    return html


def _scrape_cbs_injuries(url: str = CBS_INJURIES_URL) -> list[dict]:
    """Scrape NBA injury report from CBS Sports using a headless browser.

    CBS Sports renders injury data via JavaScript, so a plain ``requests`` GET
    returns a skeleton page.  This function uses Playwright + headless Chromium
    to wait for the page to fully render before parsing.

    Attempts two strategies in order:

    1. **TableBase strategy** — CBS Sports wraps each team's injuries in a
       ``<div class="TableBase">`` block containing a heading and a ``<table>``.
       Column indices (player, game-status, projected return) are inferred from
       the ``<thead>`` row so the parser is resilient to column reordering.

    2. **Generic table fallback** — if strategy 1 yields nothing (e.g. the CSS
       class was renamed), fall back to scanning every ``<table>`` and inferring
       the team name from surrounding text.

    Args:
        url: CBS Sports NBA injuries page URL.

    Returns:
        List of dicts with keys: ``player_name`` (str), ``player_name_norm``
        (str), ``team_abbrev`` (str | None), ``status`` (str),
        ``return_date`` (date | None).

    Raises:
        RuntimeError: If Playwright is not installed or Chromium launch fails.
    """
    html = _fetch_rendered_html(url)
    soup = BeautifulSoup(html, "html.parser")

    injuries: list[dict] = []

    # ── Strategy 1: CBS TableBase design-system blocks ────────────────────────
    # Each team section is wrapped in a `div.TableBase` (exact class).  That
    # div contains the team name as leading text, then a nested
    # `div.TableBase-overflow` which directly holds the `<table>`.  Using the
    # exact class avoids matching `TableBase-overflow` / `TableBase-shadows`
    # wrappers, which would produce duplicate entries (4× per player).
    table_blocks = soup.find_all("div", class_="TableBase")
    for block in table_blocks:
        team_abbrev = _extract_team_abbrev(block)
        table = block.find("table")
        if table is None:
            continue
        injuries.extend(_parse_injury_table(table, team_abbrev))

    if injuries:
        logger.info("Strategy 1 (TableBase): %d injured players found.", len(injuries))
        return injuries

    # ── Strategy 2: generic <table> fallback ─────────────────────────────────
    for table in soup.find_all("table"):
        team_abbrev = _extract_team_abbrev_from_context(table)
        parsed = _parse_injury_table(table, team_abbrev)
        injuries.extend(parsed)

    if injuries:
        logger.info("Strategy 2 (generic tables): %d injured players found.", len(injuries))
    else:
        logger.warning(
            "No injury rows found in HTML from %s. "
            "The page is likely JavaScript-rendered — see apply_known_injuries docstring.",
            url,
        )
    return injuries


def _extract_team_abbrev(block: "BeautifulSoup") -> Optional[str]:
    """Extract team abbreviation from a CBS TableBase block.

    The team name appears as leading text in the ``div.TableBase`` wrapper,
    before the nested table.  Strategy: look for a heading element first, then
    fall back to the first non-empty direct text string in the block.
    """
    # 1. Named heading element with a team/title class
    heading = block.find(
        ["h4", "h3", "h2", "span"],
        class_=re.compile(r"title|team|name", re.I),
    )
    if heading is None:
        heading = block.find(["h4", "h3", "h2"])
    if heading is not None:
        abbrev = _team_name_to_abbrev(heading.get_text(strip=True))
        if abbrev:
            return abbrev

    # 2. First non-empty direct string in the block (CBS puts city/team name
    #    as a plain text node before the inner overflow div).
    for string in block.strings:
        text = string.strip()
        if text and len(text) > 2:
            abbrev = _team_name_to_abbrev(text)
            if abbrev:
                return abbrev
    return None


def _extract_team_abbrev_from_context(table: "BeautifulSoup") -> Optional[str]:
    """Extract team abbreviation from the nearest surrounding text of a table."""
    parent = table.parent
    for node in ([parent] + (list(parent.find_all_previous(limit=5)) if parent else [])):
        text = node.get_text(strip=True) if hasattr(node, "get_text") else ""
        abbrev = _team_name_to_abbrev(text)
        if abbrev:
            return abbrev
    return None


def _team_name_to_abbrev(text: str) -> Optional[str]:
    """Map a free-form team-name string to a 3-letter abbreviation."""
    lower = text.lower().strip()
    # Exact match first
    if lower in _CBS_TEAM_TO_ABBREV:
        return _CBS_TEAM_TO_ABBREV[lower]
    # Full name is a substring of the text (e.g. "Atlanta Hawks Injuries")
    for full_name, abbrev in _CBS_TEAM_TO_ABBREV.items():
        if full_name in lower:
            return abbrev
    # Text is a substring of a full name (e.g. "Atlanta" matches "atlanta hawks").
    # Only apply when text is at least 5 chars to avoid spurious city matches.
    if len(lower) >= 4:
        for full_name, abbrev in _CBS_TEAM_TO_ABBREV.items():
            if lower in full_name:
                return abbrev
    return None


def _parse_injury_table(table: "BeautifulSoup", team_abbrev: Optional[str]) -> list[dict]:
    """Parse one CBS Sports injury ``<table>`` into a list of player dicts.

    CBS Sports table columns (as of 2026): Player | Position | Updated |
    Injury | Injury Status.  "Injury Status" is the column with descriptive
    return-date text (e.g. "Expected to be out until at least Apr 18").
    """
    # Defaults match the live CBS Sports 5-column layout.
    col: dict[str, int] = {"player": 0, "status": 3, "return": 4}
    thead = table.find("thead")
    if thead:
        ths = [th.get_text(strip=True).lower() for th in thead.find_all("th")]
        for i, h in enumerate(ths):
            if "player" in h:
                col["player"] = i
            elif "status" in h:
                # "Injury Status" column contains descriptive return-date text.
                col["return"] = i
            elif "injury" in h or "game" in h:
                # "Injury" column contains injury type (Rest, Ankle, …).
                col["status"] = i
            elif "return" in h or "projected" in h or "expected" in h:
                col["return"] = i

    tbody = table.find("tbody")
    if tbody is None:
        return []

    rows: list[dict] = []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        min_cols = max(col.values()) + 1
        if len(cells) < min_cols:
            continue

        # CBS renders two name variants inside the cell: short ("C. McCollum")
        # and long ("CJ McCollum").  Prefer the long-name span when present.
        player_cell = cells[col["player"]]
        long_span = player_cell.find("span", class_=re.compile(r"long", re.I))
        player_name = (long_span or player_cell).get_text(strip=True)
        if not player_name or len(player_name) < 3:
            continue

        status_text = cells[col["status"]].get_text(strip=True) if len(cells) > col["status"] else ""
        return_text = cells[col["return"]].get_text(strip=True) if len(cells) > col["return"] else ""

        # Prefer the projected-return cell; fall back to game-status text.
        date_text = return_text if return_text else status_text

        # Skip players whose only status is "Probable" or "Active" — they need
        # no override (compute_round_availability would return 1.0 anyway).
        upper = date_text.upper()
        if any(kw in upper for kw in ("PROBABLE", "ACTIVE")):
            continue

        return_date = _parse_return_date(date_text)

        rows.append({
            "player_name": player_name,
            "player_name_norm": _normalize_name(player_name),
            "team_abbrev": team_abbrev,
            "status": status_text,
            "return_date": return_date,
        })
    return rows


# ── Public API ───────────────────────────────────────────────────────────────

def apply_known_injuries(
    injury_draws: dict,
    team_rosters: pd.DataFrame,
    url: str = CBS_INJURIES_URL,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Apply real-world injury overrides to pre-drawn injury arrays.

    Scrapes the current NBA injury report from CBS Sports, matches each injured
    player against the top-N roster, computes per-round availability scalars,
    and overwrites the corresponding draws.  Players absent from the report are
    left entirely untouched.

    The draws array has shape ``(n_teams, n_stars, n_rounds, n_sims)``.  The
    simulation's healthy-check is ``draw <= mean_rate``.  Overrides work by
    replacing draws with values that achieve the target healthy fraction:

    - ``availability = 0.0`` → all draws set to ``1.0`` (always injured).
    - ``0 < availability < 1`` → ``round(availability × n_sims)`` draws set to
      ``0.0`` (healthy), remainder set to ``1.0`` (injured).
    - ``availability = 1.0`` → skipped; original random draws retained.

    Args:
        injury_draws: Dict produced by ``load_injury_draws()`` with keys:
            ``draws`` (ndarray n_teams × n_stars × n_rounds × n_sims),
            ``team_index`` (dict team_id → int), ``player_bpm``,
            ``mean_rates``, ``teams``.  **Never mutated** — a deep copy is
            returned.
        team_rosters: DataFrame from ``identify_top_players()`` containing at
            minimum the columns ``team``, ``player_name_norm``, and
            ``composite_rating``.  Players are sorted by ``composite_rating``
            descending within each team to assign star indices
            (0 = top-rated).
        url: CBS Sports NBA injuries URL to scrape.
        rng: Optional numpy random generator for shuffling the healthy/injured
            draw assignment (ensures unbiased per-iteration outcomes).

    Returns:
        Deep copy of ``injury_draws`` with overridden draws and an added
        ``_override_log`` key — a list of dicts (one per overridden player)
        with fields: ``player_name``, ``player_name_norm``, ``team``,
        ``return_date``, ``star_idx``, ``round_1`` … ``round_4``,
        ``rounds_overridden``.

    Raises:
        RuntimeError: If Playwright is not installed, Chromium launch fails,
            or the page yields zero parseable player entries after rendering.
    """
    scraped = _scrape_cbs_injuries(url)
    if not scraped:
        raise RuntimeError(
            "CBS Sports injury report yielded no parseable player/return-date pairs "
            "even after JavaScript rendering. The page structure may have changed — "
            "inspect the HTML returned by _fetch_rendered_html() and update the "
            "CSS selectors in _scrape_cbs_injuries() accordingly."
        )

    n_stars = injury_draws["draws"].shape[1]

    # Build roster lookup: team_abbrev → [(player_name_norm, star_idx), ...]
    # Star index 0 = highest composite_rating within the team.
    roster_lookup: dict[str, list[tuple[str, int]]] = {}
    for team, group in team_rosters.groupby("team"):
        top_n = (
            group.sort_values("composite_rating", ascending=False)
            .head(n_stars)
            .reset_index(drop=True)
        )
        roster_lookup[str(team)] = [
            (str(row["player_name_norm"]), i) for i, (_, row) in enumerate(top_n.iterrows())
        ]

    result = copy.deepcopy(injury_draws)
    draws: np.ndarray = result["draws"]
    team_index: dict[str, int] = result["team_index"]
    override_log: list[dict] = []

    for entry in scraped:
        team_abbrev: Optional[str] = entry["team_abbrev"]
        player_norm: str = entry["player_name_norm"]
        return_date: Optional[date] = entry["return_date"]

        if not team_abbrev or team_abbrev not in team_index:
            logger.debug(
                "Team %r not a playoff team — skipping %s.", team_abbrev, entry["player_name"]
            )
            continue

        team_idx = team_index[team_abbrev]
        star_idx: Optional[int] = None
        for pname, sidx in roster_lookup.get(team_abbrev, []):
            if pname == player_norm:
                star_idx = sidx
                break

        if star_idx is None:
            logger.debug(
                "Player %r (%s) not in top-%d roster — skipping.",
                player_norm,
                team_abbrev,
                n_stars,
            )
            continue

        round_avails = compute_round_availability(return_date)
        rounds_overridden: list[int] = []
        for round_idx, avail in enumerate(round_avails):
            if avail >= 1.0:
                continue  # fully available — original draws retained
            _apply_scalar_override(draws, team_idx, star_idx, round_idx, avail, rng)
            rounds_overridden.append(round_idx + 1)

        if rounds_overridden:
            log_entry = {
                "player_name": entry["player_name"],
                "player_name_norm": player_norm,
                "team": team_abbrev,
                "return_date": str(return_date) if return_date else "indefinite",
                "star_idx": star_idx,
                "round_1": round_avails[0],
                "round_2": round_avails[1],
                "round_3": round_avails[2],
                "round_4": round_avails[3],
                "rounds_overridden": rounds_overridden,
            }
            override_log.append(log_entry)
            logger.info(
                "Override: %-26s (%s star=%d)  return=%-12s  "
                "R1=%.2f  R2=%.2f  R3=%.2f  R4=%.2f",
                entry["player_name"],
                team_abbrev,
                star_idx,
                log_entry["return_date"],
                *round_avails,
            )

    result["_override_log"] = override_log
    logger.info(
        "Injury overrides applied: %d player(s) overridden out of %d scraped.",
        len(override_log),
        len(scraped),
    )
    return result
