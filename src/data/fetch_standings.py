"""Fetch historical NBA playoff standings (seeds 1-8 per conference) via nba_api.

Writes results to configs/bracket_seeds.yaml. Run directly:

    python src/data/fetch_standings.py
"""
from __future__ import annotations

import time
from pathlib import Path

from nba_api.stats.endpoints import leaguestandings

# ---------------------------------------------------------------------------
# TeamID → abbreviation, year-sensitive for franchises that changed names
# ---------------------------------------------------------------------------

# Base map: current NBA team IDs → current abbreviations (stable IDs)
_BASE: dict[int, str] = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612751: "BRK",  # overridden below for pre-2013
    1610612766: "CHO",  # overridden below for pre-2015
    1610612741: "CHI",
    1610612739: "CLE",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612765: "DET",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612754: "IND",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612763: "MEM",  # overridden below for pre-2002
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612740: "NOP",  # overridden below for pre-2013
    1610612752: "NYK",
    1610612760: "OKC",  # overridden below for pre-2009
    1610612753: "ORL",
    1610612755: "PHI",
    1610612756: "PHO",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612764: "WAS",  # overridden below for pre-1998
}

# (team_id, first_season_year, last_season_year_inclusive) → abbreviation
# season_year = the calendar year the season ENDS (e.g. 2016 = 2015-16)
#
# Franchise notes:
#   1610612766 = original Charlotte Hornets (1989-2002 as CHH), then Bobcats
#                (2005-2014 as CHA), then Hornets again (2015+ as CHO per base)
#   1610612740 = New Orleans franchise: Hornets (2003-2012 as NOH), Pelicans (2013+ as NOP per base)
#   1610612751 = New Jersey Nets (NJN) → Brooklyn Nets (2013+ as BRK per base)
#   1610612764 = Washington Bullets (WSB) → Wizards (1998+ as WAS per base)
#   1610612760 = Seattle SuperSonics (SEA) → OKC Thunder (2009+ as OKC per base)
#   1610612763 = Vancouver Grizzlies (VAN) → Memphis (2002+ as MEM per base)
_OVERRIDES: list[tuple[int, int, int, str]] = [
    (1610612766, 1989, 2002, "CHH"),  # original Charlotte Hornets
    (1610612766, 2005, 2014, "CHA"),  # Charlotte Bobcats
    (1610612740, 2003, 2012, "NOH"),  # New Orleans Hornets
    (1610612751, 1968, 2012, "NJN"),  # New Jersey Nets
    (1610612764, 1963, 1997, "WSB"),  # Washington Bullets
    (1610612760, 1968, 2008, "SEA"),  # Seattle SuperSonics
    (1610612763, 1996, 2001, "VAN"),  # Vancouver Grizzlies
]


def team_abbr(team_id: int, season_year: int) -> str:
    """Return the abbreviation used in playoff_series CSVs for a given season."""
    for tid, first, last, abbr in _OVERRIDES:
        if tid == team_id and first <= season_year <= last:
            return abbr
    return _BASE.get(team_id, f"ID{team_id}")


# ---------------------------------------------------------------------------
# Season string helpers
# ---------------------------------------------------------------------------

def season_str(year: int) -> str:
    """Convert season end-year to NBA API format, e.g. 2016 → '2015-16'."""
    return f"{year - 1}-{str(year)[-2:]}"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

SLEEP_SECONDS = 1.0  # polite delay between API calls


def fetch_seeds_for_year(year: int) -> dict[str, list[str]]:
    """Return {east: [s1..s8], west: [s1..s8]} for the given season end-year."""
    r = leaguestandings.LeagueStandings(
        season=season_str(year),
        timeout=30,
    )
    df = r.get_data_frames()[0]

    result: dict[str, list[str]] = {}
    for conf in ("East", "West"):
        conf_df = (
            df[df["Conference"] == conf]
            .sort_values("PlayoffRank")
            .head(8)
        )
        result[conf.lower()] = [
            team_abbr(int(row["TeamID"]), year)
            for _, row in conf_df.iterrows()
        ]
    return result


# ---------------------------------------------------------------------------
# YAML writer
# ---------------------------------------------------------------------------

YAML_HEADER = """\
# Bracket seeding configuration
# For each year listed here, seeds take precedence over the playoff_series CSV.
# Historical years (1980-2024) are loaded automatically from data/raw/playoff_series/.
# Add out-of-sample years here manually once seedings are known.
#
# Format:
#   bracket_seeds:
#     <year>:
#       east: [seed1, seed2, seed3, seed4, seed5, seed6, seed7, seed8]
#       west: [seed1, seed2, seed3, seed4, seed5, seed6, seed7, seed8]
#
# Team abbreviations must match those in data/raw/*.csv (e.g. BOS, NYK, MIL).

bracket_seeds:
"""

YAML_2025 = """\
  2025:
    # East: CLE 1, BOS 2, NYK 3, IND 4, MIL 5, DET 6, ORL 7, MIA 8
    east: [CLE, BOS, NYK, IND, MIL, DET, ORL, MIA]
    # West: OKC 1, HOU 2, LAL 3, DEN 4, LAC 5, MIN 6, GSW 7, MEM 8
    west: [OKC, HOU, LAL, DEN, LAC, MIN, GSW, MEM]

  # 2026 seedings -- fill in once playoff bracket is set
  # 2026:
  #   east: [BOS, CLE, NYK, MIL, ORL, IND, MIA, CHI]
  #   west: [OKC, HOU, MIN, LAL, GSW, DEN, SAC, MEM]
"""


def fmt_seeds(seeds: list[str]) -> str:
    return "[" + ", ".join(seeds) + "]"


def write_yaml(all_seeds: dict[int, dict[str, list[str]]], path: Path) -> None:
    lines = [YAML_HEADER]
    for year in sorted(all_seeds):
        east_seeds = all_seeds[year]["east"]
        west_seeds = all_seeds[year]["west"]
        e_comment = ", ".join(f"{t} {i+1}" for i, t in enumerate(east_seeds))
        w_comment = ", ".join(f"{t} {i+1}" for i, t in enumerate(west_seeds))
        lines.append(f"  {year}:")
        lines.append(f"    # East: {e_comment}")
        lines.append(f"    east: {fmt_seeds(east_seeds)}")
        lines.append(f"    # West: {w_comment}")
        lines.append(f"    west: {fmt_seeds(west_seeds)}")
        lines.append("")
    lines.append(YAML_2025)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_path = repo_root / "configs" / "bracket_seeds.yaml"

    all_seeds: dict[int, dict[str, list[str]]] = {}

    for year in range(1986, 2025):
        print(f"Fetching {season_str(year)} ...", end=" ", flush=True)
        try:
            seeds = fetch_seeds_for_year(year)
            all_seeds[year] = seeds
            print(f"E={seeds['east']}  W={seeds['west']}")
        except Exception as exc:
            print(f"ERROR: {exc}")
        time.sleep(SLEEP_SECONDS)

    write_yaml(all_seeds, out_path)


if __name__ == "__main__":
    main()
