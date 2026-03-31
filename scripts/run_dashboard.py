#!/usr/bin/env python3
"""run_dashboard.py — Module 5: NBA Playoff Prediction Dashboard.

Streamlit application. Reads only from results/ and configs/.
No cross-module imports from src.model, src.simulation, src.data, or src.injury.

Usage:
    streamlit run scripts/run_dashboard.py
    python scripts/run_dashboard.py   (re-launches via streamlit internally)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root so src/dashboard imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.dashboard.bracket_builder import build_bracket_structure, get_upsets
from src.dashboard.data_loader import (
    list_available_runs,
    load_bracket_seeds,
    load_dashboard_config,
    load_model_spec,
    load_simulation_results,
    load_team_features,
    logo_url,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NBA Playoff Prediction Model",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

cfg = load_dashboard_config()
COLORS = cfg["ui"]["colors"]

# ---------------------------------------------------------------------------
# NBA team colors  (primary, secondary, tertiary)
# Source: official NBA brand guidelines; equivalent to nbacolors.com values.
# ---------------------------------------------------------------------------
_TEAM_COLORS: dict[str, tuple[str, str, str]] = {
    "ATL": ("#E03A3E", "#C1D32F", "#26282A"),  # Hawks
    "BOS": ("#007A33", "#BA9653", "#000000"),  # Celtics
    "BKN": ("#000000", "#FFFFFF", "#AAAAAA"),  # Nets
    "CHA": ("#1D1160", "#00788C", "#A1A1A4"),  # Hornets
    "CHI": ("#CE1141", "#000000", "#FFFFFF"),  # Bulls
    "CLE": ("#860038", "#FDBB30", "#002D62"),  # Cavaliers
    "DAL": ("#00538C", "#002B5E", "#B8C4CA"),  # Mavericks
    "DEN": ("#0E2240", "#FEC524", "#8B2131"),  # Nuggets
    "DET": ("#C8102E", "#1D428A", "#BEC0C2"),  # Pistons
    "GSW": ("#1D428A", "#FFC72C", "#FFFFFF"),  # Warriors
    "HOU": ("#CE1141", "#000000", "#C4CED4"),  # Rockets
    "IND": ("#002D62", "#FDBB30", "#BEC0C2"),  # Pacers
    "LAC": ("#C8102E", "#1D428A", "#BEC0C2"),  # Clippers
    "LAL": ("#552583", "#FDB927", "#000000"),  # Lakers
    "MEM": ("#5D76A9", "#12173F", "#F5B112"),  # Grizzlies
    "MIA": ("#98002E", "#F9A01B", "#000000"),  # Heat
    "MIL": ("#00471B", "#EEE1C6", "#000000"),  # Bucks
    "MIN": ("#0C2340", "#236192", "#78BE20"),  # Timberwolves
    "NOP": ("#0C2340", "#C8102E", "#85714D"),  # Pelicans
    "NYK": ("#006BB6", "#F58426", "#BEC0C2"),  # Knicks
    "OKC": ("#007AC1", "#EF3B24", "#002D62"),  # Thunder
    "ORL": ("#0077C0", "#C4CED4", "#000000"),  # Magic
    "PHI": ("#006BB6", "#ED174C", "#002B5C"),  # 76ers
    "PHX": ("#1D1160", "#E56020", "#F9AD1B"),  # Suns
    "POR": ("#E03A3E", "#000000", "#FFFFFF"),  # Trail Blazers
    "SAC": ("#5A2D81", "#63727A", "#000000"),  # Kings
    "SAS": ("#C4CED4", "#000000", "#FFFFFF"),  # Spurs
    "TOR": ("#CE1141", "#000000", "#A1A1A4"),  # Raptors
    "UTA": ("#002B5C", "#00471B", "#F9A01B"),  # Jazz
    "WAS": ("#002B5C", "#E31837", "#C4CED4"),  # Wizards
}

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_GLOBAL_CSS = f"""
<style>
  /* ── Page background ─────────────────────────────────────────────────── */
  .stApp {{ background-color: {COLORS['background']}; }}
  section[data-testid="stSidebar"] {{ background-color: #0d1f33; }}

  /* ── Left panel column ───────────────────────────────────────────────── */
  div[data-testid="stColumn"]:first-child {{
    position: relative !important;
    background-color: #0d2438 !important;
    border-radius: 8px !important;
    padding: 8px !important;
  }}

  /* ── Bracket root container ──────────────────────────────────────────── */
  .bracket-root {{
    display: flex;
    flex-direction: row;
    align-items: stretch;
    justify-content: center;
    width: 100%;
    height: 620px;
    gap: 0;
    padding: 12px 8px 16px 8px;
    overflow-x: auto;
    box-sizing: border-box;
    background: {COLORS['background']};
    border-radius: 8px;
  }}

  /* ── Conference blocks ───────────────────────────────────────────────── */
  .conf-block {{
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 0;
  }}
  .conf-label {{
    font-size: 11px;
    font-weight: 700;
    color: {COLORS['accent']};
    text-align: center;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
    padding-top: 4px;
  }}
  .conf-rounds {{
    display: flex;
    flex-direction: row;
    flex: 1;
    min-height: 0;
    align-items: stretch;
  }}

  /* ── Round columns ───────────────────────────────────────────────────── */
  .round-col {{
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
    min-width: 120px;
    max-width: 155px;
    padding: 0 3px;
    box-sizing: border-box;
  }}
  /* West: R1 (1st) outward; R2 (2nd) toward center; CF (3rd) away from center */
  .conf-block.west .conf-rounds > .round-col:nth-child(1) {{ padding-right: 18px; }}
  .conf-block.west .conf-rounds > .round-col:nth-child(2) {{ padding-left: 14px; }}
  .conf-block.west .conf-rounds > .round-col:nth-child(3) {{ padding-right: 14px; }}
  /* East: CF (1st) away from center; R2 (2nd) toward center; R1 (3rd) outward */
  .conf-block.east .conf-rounds > .round-col:nth-child(1) {{ padding-left: 14px; }}
  .conf-block.east .conf-rounds > .round-col:nth-child(2) {{ padding-right: 14px; }}
  .conf-block.east .conf-rounds > .round-col:nth-child(3) {{ padding-left: 18px; }}
  .round-col-header {{
    flex-shrink: 0;
  }}
  .round-col-body {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    align-items: stretch;
    min-height: 0;
    position: relative;
  }}
  /* Matchup pair wrapper: distributes the two matchups inside it */
  .matchup-pair {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    position: relative;
    min-height: 0;
  }}
  .round-label {{
    font-size: 9px;
    font-weight: 600;
    color: #607d8b;
    text-align: center;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }}

  /* ── Finals column ───────────────────────────────────────────────────── */
  .finals-col {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    min-width: 148px;
    max-width: 168px;
    padding: 0 6px;
    box-sizing: border-box;
  }}
  .finals-col-header {{
    flex-shrink: 0;
    text-align: center;
    padding-top: 4px;
  }}
  .finals-col-body {{
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: 0;
  }}
  .finals-label {{
    font-size: 9px;
    font-weight: 700;
    color: {COLORS['accent']};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 4px;
  }}
  .champion-label {{
    font-size: 10px;
    font-weight: 700;
    color: {COLORS['champion_border']};
    text-align: center;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
  }}

  /* ── Matchup wrapper ─────────────────────────────────────────────────── */
  .matchup {{
    display: flex;
    flex-direction: column;
    gap: 3px;
    width: 100%;
    margin: 3px 0;
    position: relative;
  }}

  /* ── Team node card (2K26 pill style) ───────────────────────────────── */
  .team-node {{
    display: flex;
    flex-direction: row;
    align-items: stretch;
    background: #0d1520;
    border: 1px solid #1e2d3d;
    border-radius: 4px;
    min-height: 36px;
    position: relative;
    box-sizing: border-box;
    width: 100%;
    overflow: hidden;
  }}
  .team-node:hover {{ border-color: #4a7aaa; }}
  .team-node.champion {{
    border-color: {COLORS['champion_border']};
    box-shadow: 0 0 8px 2px rgba(255,215,0,0.28);
  }}

  /* ── Colored pill (left section, primary team color) ─────────────────── */
  .team-pill {{
    display: flex;
    flex-direction: row;
    align-items: center;
    flex: 1;
    min-width: 0;
  }}

  /* ── Seed badge ──────────────────────────────────────────────────────── */
  .team-seed-badge {{
    width: 18px;
    min-width: 18px;
    align-self: stretch;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    font-weight: 900;
    color: rgba(255,255,255,0.85);
    background: rgba(0,0,0,0.28);
    flex-shrink: 0;
  }}

  /* ── Team abbreviation ───────────────────────────────────────────────── */
  .team-abbrev {{
    font-size: 10px;
    font-weight: 800;
    color: #ffffff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0 4px 0 5px;
  }}

  /* ── Logo area (right of pill) ───────────────────────────────────────── */
  .team-logo-area {{
    width: 30px;
    min-width: 30px;
    align-self: stretch;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }}
  .team-logo {{
    width: 22px;
    height: 22px;
    object-fit: contain;
  }}
  .team-logo-fallback {{
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 7px;
    font-weight: 700;
    color: #8a9ab0;
  }}

  /* ── Win probability ─────────────────────────────────────────────────── */
  .team-prob {{
    font-size: 10px;
    font-weight: 700;
    color: #8aadcc;
    white-space: nowrap;
    flex-shrink: 0;
    align-self: center;
    padding: 0 5px;
    min-width: 28px;
    text-align: right;
  }}
  .team-node.champion .team-prob {{ color: {COLORS['champion_border']}; }}

  /* ── Bracket connector lines (90-degree arms) ───────────────────────── */
  /* Step 1 — internal matchup arm: vertical bar spanning the two team-card centers
     within every matchup (24%–76% of matchup height ≈ top-card mid to bottom-card mid).
     Positioned 7px outside the content edge, in the column gap. */
  .conf-block.west .round-col-body .matchup::after {{
    content: '';
    position: absolute;
    right: -7px;
    top: 24%;
    height: 52%;
    width: 0;
    border-right: 1.5px solid rgba(255,255,255,0.85);
  }}
  .conf-block.east .round-col-body .matchup::after {{
    content: '';
    position: absolute;
    left: -7px;
    top: 24%;
    height: 52%;
    width: 0;
    border-left: 1.5px solid rgba(255,255,255,0.85);
  }}
  /* Step 2 — pair arm: vertical bar on .matchup-pair spanning 25%–75% of its height,
     connecting the midpoints of the two matchups inside it (the "gather" bar). */
  .conf-block.west .matchup-pair::after {{
    content: '';
    position: absolute;
    right: -7px;
    top: 25%;
    height: 50%;
    width: 0;
    border-right: 1.5px solid rgba(255,255,255,0.85);
  }}
  .conf-block.east .matchup-pair::after {{
    content: '';
    position: absolute;
    left: -7px;
    top: 25%;
    height: 50%;
    width: 0;
    border-left: 1.5px solid rgba(255,255,255,0.85);
  }}
  /* Step 3 — bridge: horizontal line from the pair arm's midpoint (top: 50%) to the
     next round's matchup. Width covers the column gap; card backgrounds hide any overlap.
     West R1→R2: R1 right-pad 18px + R2 left-pad 14px − 7px arm = 25px → right:-32px w:25px.
     West R2→CF: R2 right-pad 3px + CF left-pad 3px − 7px arm ≈ −1px (arm already overlaps CF).
     West CF→Finals: CF right-pad 14px + Finals left-pad 6px − 7px arm = 13px → right:-20px w:13px.
     CF has no matchup-pair wrapper, so the bridge uses matchup::before instead. */
  .conf-block.west .r1 .matchup-pair::before {{
    content: '';
    position: absolute;
    right: -32px;
    top: 50%;
    transform: translateY(-50%);
    width: 25px;
    height: 1.5px;
    background: rgba(255,255,255,0.85);
  }}
  .conf-block.west .r3 .matchup::before {{
    content: '';
    position: absolute;
    right: -20px;
    top: 50%;
    transform: translateY(-50%);
    width: 13px;
    height: 1.5px;
    background: rgba(255,255,255,0.85);
  }}
  .conf-block.east .r1 .matchup-pair::before {{
    content: '';
    position: absolute;
    left: -32px;
    top: 50%;
    transform: translateY(-50%);
    width: 25px;
    height: 1.5px;
    background: rgba(255,255,255,0.85);
  }}
  .conf-block.east .r3 .matchup::before {{
    content: '';
    position: absolute;
    left: -20px;
    top: 50%;
    transform: translateY(-50%);
    width: 13px;
    height: 1.5px;
    background: rgba(255,255,255,0.85);
  }}

  /* ── Upsets panel ────────────────────────────────────────────────────── */
  .upsets-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 8px;
  }}
  .upset-card {{
    background: {COLORS['card_background']};
    border: 1.5px solid {COLORS['default_border']};
    border-radius: 6px;
    padding: 10px 14px;
    min-width: 200px;
    max-width: 270px;
  }}
  .upset-round  {{ font-size: 10px; color: #607d8b; margin-bottom: 2px; }}
  .upset-matchup {{ font-size: 11px; color: #90a4ae; margin-bottom: 4px; }}
  .upset-underdog {{ font-size: 14px; font-weight: 700; color: #ffffff; }}
  .upset-prob   {{ font-size: 13px; font-weight: 700; color: #ff9f40; margin-top: 2px; }}

  /* ── Text visibility on dark background ──────────────────────────────── */
  .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: #ffffff !important; }}
  .stApp p {{ color: #e0e0e0 !important; }}
  [data-testid="stMetricValue"] {{ color: #ffffff !important; }}
  [data-testid="stMetricLabel"] > div {{ color: #90a4ae !important; }}
  [data-testid="stMarkdownContainer"] p {{ color: #e0e0e0 !important; }}
  .stCaption {{ color: #90a4ae !important; }}

  /* ── Sidebar: always visible, no collapse arrow ───────────────────────── */
  [data-testid="stSidebarCollapseButton"],
  [data-testid="collapsedControl"],
  button[data-testid="stBaseButton-headerNoPadding"],
  button[title="Close sidebar"],
  button[aria-label="Close sidebar"],
  button[title="Open sidebar"],
  button[aria-label="Open sidebar"] {{
    display: none !important;
  }}
</style>
"""

st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _team_node_html(node: dict, is_champion: bool = False) -> str:
    """Render a single team card as HTML (2K26 pill style)."""
    abbrev = node["abbrev"]
    seed = node["seed"]
    url = node["logo_url"]
    prob_pct = f"{node['cond_win_prob']:.0%}"
    champ_cls = " champion" if is_champion else ""

    # Primary team color → gradient fading right so dark logo area blends in
    primary = _TEAM_COLORS.get(abbrev, ("#1a3a5c", "#0a1929", "#0a1929"))[0]
    pill_bg = (
        f"linear-gradient(90deg, {primary} 0%, {primary}cc 55%, {primary}55 85%, transparent 100%)"
    )
    # Darker solid version of primary for logo background: 40% opacity over dark base
    logo_bg = f"{primary}66"

    logo = (
        f'<img class="team-logo" src="{url}" '
        f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\';" '
        f'alt="{abbrev}">'
        f'<span class="team-logo-fallback" style="display:none">{abbrev}</span>'
    )
    return (
        f'<div class="team-node{champ_cls}">'
        f'<div class="team-pill" style="background:{pill_bg}">'
        f'<div class="team-seed-badge">{seed}</div>'
        f'<span class="team-abbrev">{abbrev}</span>'
        f'</div>'
        f'<div class="team-logo-area" style="background:{logo_bg}">{logo}</div>'
        f'<span class="team-prob">{prob_pct}</span>'
        f'</div>'
    )


def _matchup_html(
    matchup: dict, champion_abbrev: str | None = None, is_finals: bool = False
) -> str:
    """Render a matchup (two team cards) as HTML."""
    hi = matchup["high"]
    lo = matchup["low"]
    hi_champ = is_finals and hi["abbrev"] == champion_abbrev
    lo_champ = is_finals and lo["abbrev"] == champion_abbrev
    return (
        f'<div class="matchup">'
        f'{_team_node_html(hi, is_champion=hi_champ)}'
        f'{_team_node_html(lo, is_champion=lo_champ)}'
        f'</div>'
    )


def _round_col_html(
    matchups: list[dict],
    label: str,
    champion_abbrev: str | None = None,
    is_finals: bool = False,
    css_class: str = "",
) -> str:
    """Render one round column (label + all matchups)."""
    rendered = [
        _matchup_html(m, champion_abbrev=champion_abbrev, is_finals=is_finals)
        for m in matchups
    ]
    # Wrap consecutive pairs in .matchup-pair for bracket-arm connectors.
    inner = ""
    for i in range(0, len(rendered), 2):
        pair = rendered[i:i + 2]
        if len(pair) == 2:
            inner += f'<div class="matchup-pair">{"".join(pair)}</div>'
        else:
            inner += pair[0]
    extra_cls = f" {css_class}" if css_class else ""
    return (
        f'<div class="round-col{extra_cls}">'
        f'<div class="round-col-body">{inner}</div>'
        f'</div>'
    )


def _render_bracket_html(bracket: dict) -> str:
    """Build the full bracket HTML string.

    Column order left → right:
        W_R1 | W_R2 | W_R3 | Finals | E_R3 | E_R2 | E_R1
    """
    champ = bracket.get("champion")
    champ_abbrev = champ["abbrev"] if champ else None

    # West columns (R1 outermost left, R3 innermost).
    # R1 display order: [1v8, 4v5, 2v7, 3v6] (indices 0,3,1,2) so that each
    # adjacent pair of R1 matchups feeds the same R2 slot, aligning their midpoints.
    west_r1_raw = bracket["west"][1]
    west_r1_ordered = [west_r1_raw[0], west_r1_raw[3], west_r1_raw[1], west_r1_raw[2]]
    w_r1 = _round_col_html(west_r1_ordered, "R1", css_class="r1")
    w_r2 = _round_col_html(bracket["west"][2], "R2", css_class="r2")
    w_r3 = _round_col_html(bracket["west"][3], "Conf Finals", css_class="r3")

    # East columns (R3 innermost, R1 outermost right).
    # East R1 uses same vertical order for left-right symmetry.
    east_r1_raw = bracket["east"][1]
    east_r1_ordered = [east_r1_raw[0], east_r1_raw[3], east_r1_raw[1], east_r1_raw[2]]
    e_r3 = _round_col_html(bracket["east"][3], "Conf Finals", css_class="r3")
    e_r2 = _round_col_html(bracket["east"][2], "R2", css_class="r2")
    e_r1 = _round_col_html(east_r1_ordered, "R1", css_class="r1")

    # Finals centre column: labels in a separate header so the matchup card
    # is vertically centered in the remaining body space (aligns with R3 cards).
    finals_matchup = bracket["finals"][4][0]
    finals_col = (
        f'<div class="finals-col">'
        f'<div class="finals-col-body">'
        f'{_matchup_html(finals_matchup, champion_abbrev=champ_abbrev, is_finals=True)}'
        f'</div>'
        f'</div>'
    )

    west_block = (
        f'<div class="conf-block west">'
        f'<div class="conf-rounds">{w_r1}{w_r2}{w_r3}</div>'
        f'</div>'
    )
    east_block = (
        f'<div class="conf-block east">'
        f'<div class="conf-rounds">{e_r3}{e_r2}{e_r1}</div>'
        f'</div>'
    )

    return (
        f'<div class="bracket-root">'
        f'{west_block}'
        f'{finals_col}'
        f'{east_block}'
        f'</div>'
    )


def _render_upsets_panel(upsets: list[dict]) -> None:
    """Render the upsets panel as styled HTML cards."""
    if not upsets:
        st.info("No notable upsets predicted.")
        return

    round_labels = {1: "Round 1", 2: "Conf Semis", 3: "Conf Finals", 4: "Finals"}
    cards = ""
    for u in upsets:
        rl = round_labels.get(u["round"], f"Round {u['round']}")
        cards += (
            f'<div class="upset-card">'
            f'<div class="upset-round">{rl}</div>'
            f'<div class="upset-matchup">{u["matchup"]}</div>'
            f'<div class="upset-underdog">{u["underdog"]} (#{u["underdog_seed"]})</div>'
            f'<div class="upset-prob">{u["cond_win_prob"]:.0%} win probability</div>'
            f'</div>'
        )
    st.markdown(f'<div class="upsets-grid">{cards}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load available runs (needed before any widget)
# ---------------------------------------------------------------------------
available_runs = list_available_runs()
if not available_runs:
    st.warning("No simulation results found. Run `python scripts/run_bracket_sim.py` first.")
    st.stop()

_years = sorted({r.split("_", 1)[0] for r in available_runs}, reverse=True)
_windows = sorted({r.split("_", 1)[1] for r in available_runs if "_" in r})

# ---------------------------------------------------------------------------
# Page layout: left panel | main content
# ---------------------------------------------------------------------------
panel_col, _gap_col, main_col = st.columns([1.2, 0.2, 5])

with panel_col:
    st.markdown('<div style="height:0.7rem"></div>', unsafe_allow_html=True)
    st.markdown('<h1 style="font-size:1.6rem;margin:0 0 1rem 0">Configuration</h1>', unsafe_allow_html=True)
    selected_year = st.selectbox("Year", _years, index=0)
    selected_window = st.selectbox("Window", _windows, index=0)
    selected_run = f"{selected_year}_{selected_window}"
    if selected_run not in available_runs:
        st.error(f"No results for {selected_year} / {selected_window}.")
        st.stop()

# Load data
sim = load_simulation_results(selected_run)
summary = sim["summary"]
adv_df = sim["round_advancement"]
champ_df = sim["championship_probs"]
year: int = summary["year"]
window: str = summary["window"]

team_features = load_team_features(year)
try:
    spec = load_model_spec(window)
except FileNotFoundError:
    spec = None

_item_style = (
    "display:block;background:#1e1e1e;border:1px solid #333;"
    "border-radius:3px;padding:2px 6px;font-family:monospace;"
    "font-size:12px;color:#ffffff;margin:3px 0;"
)

with panel_col:
    st.markdown("---")
    st.markdown('<h1 style="font-size:1.6rem;margin:0 0 1rem 0">Model Specification</h1>', unsafe_allow_html=True)
    if spec is not None:
        st.markdown(f"**Training window:** {spec['window']}")
        st.markdown(f"**Observations (N):** {spec['n_obs']}")
        st.markdown("**Features:**")
        for feat in spec["features"]:
            coeff = spec["coefficients"].get(feat, 0.0)
            st.markdown(
                f'<div style="{_item_style}">{feat} ({coeff:+.4f})</div>',
                unsafe_allow_html=True,
            )
        intercept = spec["intercept"]
        st.markdown(
            f'<div style="{_item_style}">intercept ({intercept:+.4f})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"Model spec not found for window '{window}'.")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
with main_col:
    st.markdown('<h1 style="margin-top:0rem;margin-bottom:0.8rem">NBA Playoff Prediction Model</h1>', unsafe_allow_html=True)
    st.caption("Monte Carlo bracket simulation — 50,000 iterations")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Year", str(year))
    c2.metric("Window", window.capitalize())
    c3.metric("Predicted Champion", summary.get("predicted_champion") or "—")
    c4.metric("Actual Champion", summary.get("actual_champion") or "TBD")
    n_sims = summary.get("n_sims", 0)
    c5.metric("Simulations", f"{n_sims // 1000}k" if n_sims >= 1000 else str(n_sims))

    st.divider()

    seeds = load_bracket_seeds(year)
    if not seeds:
        st.warning(
            f"No bracket seeds found for {year} in `configs/bracket_seeds.yaml`. "
            "Showing championship probabilities instead."
        )
        st.bar_chart(champ_df.set_index("team")["championship_prob"])
    else:
        bracket = build_bracket_structure(
            east_seeds=seeds["east"],
            west_seeds=seeds["west"],
            adv_df=adv_df,
            logo_url_fn=logo_url,
            predicted_champion=summary.get("predicted_champion"),
            team_features=team_features,
            spec=spec,
        )
        st.markdown(_render_bracket_html(bracket), unsafe_allow_html=True)

        st.subheader("Notable Upsets")
        upsets = get_upsets(
            seeds["east"],
            seeds["west"],
            adv_df,
            upset_threshold=cfg["ui"]["upset_threshold"],
            team_features=team_features,
            spec=spec,
        )
        _render_upsets_panel(upsets)
