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

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import streamlit.components.v1 as components
import yaml

from src.dashboard.bracket_builder import build_bracket_structure
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

with open(cfg["paths"]["training_windows_config"]) as _f:
    _tw_cfg = yaml.safe_load(_f)
_WINDOW_SPANS: dict[str, str] = {
    w["name"]: f"{w['start_year']}–{w['end_year']}"
    for w in _tw_cfg["windows"]
}

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

def _team_node_html(
    node: dict,
    is_champion: bool = False,
    prob_mode: str = "Matchup Win %",
    champ_prob: float | None = None,
    is_finals: bool = False,
) -> str:
    """Render a single team card as HTML (2K26 pill style)."""
    abbrev = node["abbrev"]
    seed = node["seed"]
    url = node["logo_url"]
    if prob_mode == "Championship %" and champ_prob is not None:
        prob_pct = f"🏆 {champ_prob:.0%}" if is_champion else f"{champ_prob:.0%}"
    else:
        prob_pct = f"{node['cond_win_prob']:.0%}"
    finalist_cls = " finalist" if (is_champion and prob_mode == "Championship %") else ""
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
        f'<div class="team-node{champ_cls}{finalist_cls}">'
        f'<div class="team-pill" style="background:{pill_bg}">'
        f'<div class="team-seed-badge">{seed}</div>'
        f'<span class="team-abbrev">{abbrev}</span>'
        f'</div>'
        f'<div class="team-logo-area" style="background:{logo_bg}">{logo}</div>'
        f'<span class="team-prob">{prob_pct}</span>'
        f'</div>'
    )


def _matchup_html(
    matchup: dict,
    champion_abbrev: str | None = None,
    is_finals: bool = False,
    prob_mode: str = "Matchup Win %",
    champ_probs: dict | None = None,
) -> str:
    """Render a matchup (two team cards) as HTML."""
    hi = matchup["high"]
    lo = matchup["low"]
    hi_champ = is_finals and hi["abbrev"] == champion_abbrev
    lo_champ = is_finals and lo["abbrev"] == champion_abbrev
    hi_cp = champ_probs.get(hi["abbrev"]) if champ_probs else None
    lo_cp = champ_probs.get(lo["abbrev"]) if champ_probs else None
    return (
        f'<div class="matchup">'
        f'{_team_node_html(hi, is_champion=hi_champ, prob_mode=prob_mode, champ_prob=hi_cp, is_finals=is_finals)}'
        f'{_team_node_html(lo, is_champion=lo_champ, prob_mode=prob_mode, champ_prob=lo_cp, is_finals=is_finals)}'
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


def _render_bracket_html_canvas(
    bracket: dict,
    prob_mode: str = "Matchup Win %",
    champ_probs: dict | None = None,
) -> str:
    """Build the full bracket HTML string using an absolutely-positioned canvas.

    All 15 matchup boxes (W R1×4, W R2×2, W CF×1, Finals×1, E CF×1, E R2×2, E R1×4)
    are placed at explicit (x, y) pixel coordinates inside a single positioned
    container. No connector lines (added in Prompt 2).
    """
    # ── Layout constants ──────────────────────────────────────────────────────
    CANVAS_WIDTH  = 900   # px, full bracket width
    CANVAS_HEIGHT = 550    # px
    BOX_W = 130            # px, matchup box width
    BOX_H = 105            # px, matchup box height (two 36px team rows)

    # ── X positions per round column ─────────────────────────────────────────
    west_r1_x = 0
    west_r2_x = int(CANVAS_WIDTH * 0.16)          # 211
    west_cf_x = int(CANVAS_WIDTH * 0.22)           # 382
    finals_x  = int(CANVAS_WIDTH * 0.50 - BOX_W / 2)  # 567
    east_cf_x = CANVAS_WIDTH - west_cf_x - BOX_W  # 753
    east_r2_x = CANVAS_WIDTH - west_r2_x - BOX_W  # 924
    east_r1_x = CANVAS_WIDTH - BOX_W              # 1135

    # ── Y positions: distribute 4 R1 slots evenly, then center upward ────────
    _pad  = 20
    _slot = (CANVAS_HEIGHT - 2 * _pad) / 4        # 170 px per R1 slot
    r1_centers = [_pad + i * _slot + _slot / 2 for i in range(4)]
    # R2 center = midpoint between the two R1 centers that feed it
    r2_centers = [
        (r1_centers[0] + r1_centers[1]) / 2,      # 190
        (r1_centers[2] + r1_centers[3]) / 2,      # 530
    ]
    cf_center     = (r2_centers[0] + r2_centers[1]) / 2   # 360
    finals_center = cf_center                               # 360

    def _box_y(center: float) -> int:
        return int(center - BOX_H / 2)

    r1_y     = [_box_y(c) for c in r1_centers]    # [69, 239, 409, 579]
    r2_y     = [_box_y(c) for c in r2_centers]    # [154, 494]
    cf_y     = _box_y(cf_center)                   # 324
    finals_y = _box_y(finals_center)               # 324

    # ── Extract bracket data ─────────────────────────────────────────────────
    champ        = bracket.get("champion")
    champ_abbrev = champ["abbrev"] if champ else None

    # R1 display order [1v8, 4v5, 2v7, 3v6] (raw indices 0,3,1,2) keeps adjacent
    # pairs feeding the same R2 slot so midpoints align vertically.
    west_r1_raw     = bracket["west"][1]
    west_r1_ordered = [west_r1_raw[0], west_r1_raw[3], west_r1_raw[1], west_r1_raw[2]]
    west_r2_list    = bracket["west"][2]
    west_cf         = bracket["west"][3][0]

    east_r1_raw     = bracket["east"][1]
    east_r1_ordered = [east_r1_raw[0], east_r1_raw[3], east_r1_raw[1], east_r1_raw[2]]
    east_r2_list    = bracket["east"][2]
    east_cf         = bracket["east"][3][0]

    finals_matchup = bracket["finals"][4][0]

    # ── Position specs: (x, y, matchup_dict, is_finals) ──────────────────────
    box_specs = []
    for i, m in enumerate(west_r1_ordered):
        box_specs.append((west_r1_x, r1_y[i], m, False))
    for i, m in enumerate(west_r2_list):
        box_specs.append((west_r2_x, r2_y[i], m, False))
    box_specs.append((west_cf_x, cf_y,     west_cf,       False))
    box_specs.append((finals_x,  finals_y, finals_matchup, True))
    box_specs.append((east_cf_x, cf_y,     east_cf,       False))
    for i, m in enumerate(east_r2_list):
        box_specs.append((east_r2_x, r2_y[i], m, False))
    for i, m in enumerate(east_r1_ordered):
        box_specs.append((east_r1_x, r1_y[i], m, False))

    # ── Generate positioned matchup boxes ────────────────────────────────────
    html_boxes = "".join(
        (
            f'<div style="position:absolute;left:{x}px;top:{y}px;width:{BOX_W}px;z-index:1;'
            f'transform:scale(1.3);transform-origin:center center">'
            if is_finals else
            f'<div style="position:absolute;left:{x}px;top:{y}px;width:{BOX_W}px;z-index:1">'
        )
        + f'{_matchup_html(m, champion_abbrev=champ_abbrev, is_finals=is_finals, prob_mode=prob_mode, champ_probs=champ_probs)}'
        + f'</div>'
        for x, y, m, is_finals in box_specs
    )

    # ── SVG connector lines (rendered behind boxes via z-index:0) ────────────
    # BOX_H (105) positions each box but the actual rendered height is
    # 2 team-node rows (36px each) + 3px gap = 75px. Derive connector y-centres
    # from actual box tops so stubs exit the visual midpoint between the two team
    # rows, not the lower portion of the box.
    _ACTUAL_H = 2 * 36 + 3  # 75px
    _r1_cy  = [y + _ACTUAL_H / 2 for y in r1_y]    # actual centre of each R1 box
    _r2_cy  = [y + _ACTUAL_H / 2 for y in r2_y]    # actual centre of each R2 box
    _cf_cy  = cf_y     + _ACTUAL_H / 2
    _fin_cy = finals_y + _ACTUAL_H / 2

    _S = 'stroke="#6b7280" stroke-width="1.5" fill="none" opacity="1"'

    def _arm_right(src_rx: float, dst_lx: float, top_cy: float, bot_cy: float, dst_cy: float, box_height: float) -> str:
        """Bracket arm extending rightward: two stubs → vertical gather bar → bridge to dest."""
        gx = (src_rx + dst_lx)
        gy_top = dst_cy - box_height/3
        gy_bot = dst_cy + box_height/3
        return (
            f'<line x1="{src_rx:.1f}" y1="{top_cy:.1f}" x2="{gx:.1f}" y2="{top_cy:.1f}" {_S}/>'
            f'<line x1="{src_rx:.1f}" y1="{bot_cy:.1f}" x2="{gx:.1f}" y2="{bot_cy:.1f}" {_S}/>'
            f'<line x1="{gx:.1f}" y1="{top_cy:.1f}" x2="{gx:.1f}" y2="{gy_top:.1f}" {_S}/>'
            f'<line x1="{gx:.1f}" y1="{bot_cy:.1f}" x2="{gx:.1f}" y2="{gy_bot:.1f}" {_S}/>'
        )

    def _arm_left(src_lx: float, dst_rx: float, top_cy: float, bot_cy: float, dst_cy: float, box_height: float) -> str:
        """Bracket arm extending leftward: two stubs → vertical gather bar → bridge to dest."""
        gx = (src_lx - dst_rx)
        gy_top = dst_cy - box_height/3
        gy_bot = dst_cy + box_height/3
        return (
            f'<line x1="{src_lx:.1f}" y1="{top_cy:.1f}" x2="{dst_rx:.1f}" y2="{top_cy:.1f}" {_S}/>'
            f'<line x1="{src_lx:.1f}" y1="{bot_cy:.1f}" x2="{dst_rx:.1f}" y2="{bot_cy:.1f}" {_S}/>'
            f'<line x1="{dst_rx:.1f}" y1="{top_cy:.1f}" x2="{dst_rx:.1f}" y2="{gy_top:.1f}" {_S}/>'
            f'<line x1="{dst_rx:.1f}" y1="{bot_cy:.1f}" x2="{dst_rx:.1f}" y2="{gy_bot:.1f}" {_S}/>'
        )

    def _hline(x1: float, x2: float, y: float) -> str:
        return f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" {_S}/>'

    svg_lines = ""
    # West: R1→R2 (two pairs)
    svg_lines += _arm_right(west_r1_x + BOX_W, west_r2_x - BOX_W/2, _r1_cy[0], _r1_cy[1], _r2_cy[0], BOX_H)
    svg_lines += _arm_right(west_r1_x + BOX_W, west_r2_x - BOX_W/2, _r1_cy[2], _r1_cy[3], _r2_cy[1], BOX_H)
    # West: R2→CF
    svg_lines += _arm_right(west_r2_x + BOX_W, west_cf_x - BOX_W*1.3, _r2_cy[0], _r2_cy[1], _cf_cy, BOX_H)
    # West: CF→Finals (single horizontal)
    svg_lines += _hline(west_cf_x + BOX_W, finals_x - 0.3*BOX_W/2, _cf_cy)
    # East: R1→R2 (two pairs, arms grow leftward)
    svg_lines += _arm_left(east_r1_x, east_r2_x + BOX_W/2, _r1_cy[0], _r1_cy[1], _r2_cy[0], BOX_H)
    svg_lines += _arm_left(east_r1_x, east_r2_x + BOX_W/2, _r1_cy[2], _r1_cy[3], _r2_cy[1], BOX_H)
    # East: R2→CF
    svg_lines += _arm_left(east_r2_x, east_cf_x + BOX_W*0.2, _r2_cy[0], _r2_cy[1], _cf_cy, BOX_H)
    # East: CF→Finals (single horizontal)
    svg_lines += _hline(east_cf_x, finals_x + 1.15*BOX_W, _cf_cy)

    svg_el = (
        f'<svg style="position:absolute;top:0;left:0;width:{CANVAS_WIDTH}px;height:{CANVAS_HEIGHT}px;'
        f'z-index:0;pointer-events:none;overflow:visible">'
        f'{svg_lines}'
        f'</svg>'
    )

    # ── CSS scoped to team-node components (iframe has no parent page styles) ─
    champ_border = COLORS["champion_border"]
    _css = f"""
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{ background: {COLORS['background']}; overflow-x: auto; overflow-y: hidden;
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
      .matchup {{ display: flex; flex-direction: column; gap: 3px; width: 100%; }}
      .team-node {{
        display: flex; flex-direction: row; align-items: stretch;
        background: #0d1520; border: 1px solid #1e2d3d; border-radius: 4px;
        min-height: 36px; position: relative; width: 100%; overflow: hidden;
      }}
      .team-node:hover {{ border-color: #4a7aaa; }}
      .team-node.champion {{
        border-color: {champ_border};
        box-shadow: 0 0 8px 2px rgba(255,215,0,0.28);
      }}
      .team-node.finalist {{
        border-color: {champ_border};
        box-shadow: 0 0 8px 2px rgba(255,215,0,0.28);
      }}
      .team-pill {{
        display: flex; flex-direction: row; align-items: center; flex: 1; min-width: 0;
      }}
      .team-seed-badge {{
        width: 18px; min-width: 18px; align-self: stretch;
        display: flex; align-items: center; justify-content: center;
        font-size: 9px; font-weight: 900; color: rgba(255,255,255,0.85);
        background: rgba(0,0,0,0.28); flex-shrink: 0;
      }}
      .team-abbrev {{
        font-size: 10px; font-weight: 800; color: #ffffff;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        text-transform: uppercase; letter-spacing: 0.04em; padding: 0 4px 0 5px;
      }}
      .team-logo-area {{
        width: 30px; min-width: 30px; align-self: stretch;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
      }}
      .team-logo {{ width: 22px; height: 22px; object-fit: contain; }}
      .team-logo-fallback {{
        width: 22px; height: 22px; display: flex; align-items: center;
        justify-content: center; font-size: 7px; font-weight: 700; color: #8a9ab0;
      }}
      .team-prob {{
        font-size: 10px; font-weight: 700; color: #8aadcc; white-space: nowrap;
        flex-shrink: 0; align-self: center; padding: 0 5px; min-width: 28px;
        text-align: right;
      }}
      .team-node.champion .team-prob {{ color: {champ_border}; }}
      .team-node.finalist .team-prob {{ color: {champ_border}; }}
    """

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>{_css}</style>
</head>
<body>
  <div id="outer" style="width:100%;overflow:hidden;position:relative">
    <div id="canvas" style="position:relative;width:{CANVAS_WIDTH}px;height:{CANVAS_HEIGHT}px;transform-origin:top left">
      {svg_el}
      {html_boxes}
    </div>
  </div>
  <script>
    var CW = {CANVAS_WIDTH}, CH = {CANVAS_HEIGHT};
    var canvas = document.getElementById('canvas');
    var outer  = document.getElementById('outer');
    function applyScale() {{
      var w = outer.clientWidth || document.documentElement.clientWidth;
      if (w < 10) {{ setTimeout(applyScale, 50); return; }}
      var scale = w / CW;
      canvas.style.transform = 'scale(' + scale + ')';
      var newH = Math.ceil(CH * scale) + 4;
      outer.style.height = (newH - 4) + 'px';
      // st.components.v1.html() ignores setFrameHeight messages, so resize the
      // parent iframe element directly. Works because Streamlit runs same-origin.
      try {{
        var frames = window.parent.document.querySelectorAll('iframe');
        for (var i = 0; i < frames.length; i++) {{
          if (frames[i].contentWindow === window) {{
            frames[i].style.height = newH + 'px';
            frames[i].style.minHeight = newH + 'px';
            break;
          }}
        }}
      }} catch(e) {{}}
    }}
    applyScale();
    setTimeout(applyScale, 100);
    window.addEventListener('resize', applyScale);
  </script>
</body>
</html>"""


def _render_champ_prob_chart_html(
    champ_df: pd.DataFrame,
    seeds: dict[str, list[str]],
) -> str:
    """Return self-contained HTML for the championship probability bar chart.

    Bars are sorted descending by probability, colored West=blue / East=red,
    with team logo images along the x-axis.
    """
    west_set = set(seeds.get("west", []))
    sorted_df = champ_df.sort_values("championship_prob", ascending=False).reset_index(drop=True)

    max_prob = float(sorted_df["championship_prob"].max()) or 1.0
    MAX_BAR_H = 160

    bar_cells = ""
    logo_cells = ""
    for _, row in sorted_df.iterrows():
        team = str(row["team"])
        prob = float(row["championship_prob"])
        color = "#1D428A" if team in west_set else "#CE1141"
        bar_h = max(2, int(MAX_BAR_H * prob / max_prob))
        pct_label = f"{prob:.1%}" if prob < 0.005 else f"{prob:.0%}"
        url = logo_url(team)

        bar_cells += (
            f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;min-width:0;padding:0 1px">'
            f'<span style="font-size:9px;color:#ccc;margin-bottom:2px;white-space:nowrap">{pct_label}</span>'
            f'<div style="width:80%;height:{bar_h}px;background:{color};border-radius:2px 2px 0 0"></div>'
            f'</div>'
        )
        logo_cells += (
            f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;min-width:0;padding:0 1px">'
            f'<img src="{url}" style="width:22px;height:22px;object-fit:contain" '
            f'onerror="this.style.visibility=\'hidden\'">'
            f'<span style="font-size:8px;color:#777;margin-top:1px">{team}</span>'
            f'</div>'
        )

    return (
        f'<div style="font-family:sans-serif;background:transparent;width:100%">'
        f'<div style="display:flex;align-items:flex-end;width:100%;height:{MAX_BAR_H + 22}px;'
        f'border-bottom:1px solid #444">'
        f'{bar_cells}'
        f'</div>'
        f'<div style="display:flex;width:100%;margin-top:4px">'
        f'{logo_cells}'
        f'</div>'
        f'</div>'
    )


@st.cache_data
def _load_actual_champions(playoff_series_dir: str) -> dict[int, str]:
    """Return a dict mapping year -> actual champion abbreviation.

    Reads the raw per-year playoff series CSVs which carry explicit team_high /
    team_low columns.  Years with more than one finals row are resolved via a
    small hardcoded override for the early-era data quality issues (1980-1983).
    """
    # Hardcoded for 1980-1983 where the raw data has multiple finals rows
    _overrides: dict[int, str] = {1980: "LAL", 1981: "BOS", 1982: "LAL", 1983: "PHI"}

    champions: dict[int, str] = dict(_overrides)
    series_path = Path(playoff_series_dir)
    for csv_file in sorted(series_path.glob("*_nba_api.csv")):
        df = pd.read_csv(csv_file)
        year = int(df["season"].iloc[0])
        if year in _overrides:
            continue
        finals = df[df["round"] == "finals"]
        if len(finals) != 1:
            continue
        row = finals.iloc[0]
        champions[year] = row["team_high"] if int(row["higher_seed_wins"]) == 1 else row["team_low"]
    return champions


@st.cache_data
def _compute_model_performance(
    window: str,
    features: tuple[str, ...],
    series_dataset_path: str,
    training_windows_config: str,
    results_dir: str = "results/simulations",
) -> dict:
    """Fit logistic regression for the given window and return performance metrics.

    Returns dict with keys: pseudo_r2, auc, brier, n_obs, feat_stats (list of dicts).
    Results are cached so the fit only runs once per window/feature combination.
    """
    with open(training_windows_config) as f:
        tw_cfg = yaml.safe_load(f)

    window_row = next(w for w in tw_cfg["windows"] if w["name"] == window)
    start_year, end_year = window_row["start_year"], window_row["end_year"]

    df = pd.read_parquet(series_dataset_path)
    sub = df[(df["year"] >= start_year) & (df["year"] <= end_year)].dropna(
        subset=list(features) + ["higher_seed_wins"]
    )

    X = sm.add_constant(sub[list(features)])
    y = sub["higher_seed_wins"].astype(float)
    result = sm.Logit(y, X).fit(disp=0)

    probs = result.predict(X).values
    y_arr = y.values

    # AUC via trapezoidal rule (no sklearn dependency)
    order = np.argsort(probs)[::-1]
    y_s = y_arr[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]
    auc = float(np.abs(np.trapz(tpr, fpr)))

    brier = float(np.mean((probs - y_arr) ** 2))

    feat_stats = [
        {
            "Feature": feat,
            "Coef.": round(float(result.params[feat]), 4),
            "z": round(float(result.tvalues[feat]), 2),
            "p-value": float(result.pvalues[feat]),
        }
        for feat in features
    ]

    preds = (probs >= 0.5).astype(int)
    correct_series = int((preds == y_arr).sum())
    total_series = len(y_arr)

    actual_champions = _load_actual_champions(cfg["paths"]["playoff_series_dir"])

    # Compare simulated predicted_champion vs actual champion for each in-window year.
    sim_path = Path(results_dir)
    correct_champs = 0
    total_champs = 0
    for yr in range(start_year, end_year + 1):
        summary_path = sim_path / f"{yr}_{window}" / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as _f:
            sim_summary = json.load(_f)
        predicted = sim_summary.get("predicted_champion")
        actual = actual_champions.get(yr)
        if not predicted or actual is None:
            continue
        total_champs += 1
        if predicted == actual:
            correct_champs += 1

    return {
        "pseudo_r2": float(result.prsquared),
        "auc": auc,
        "brier": brier,
        "n_obs": len(sub),
        "feat_stats": feat_stats,
        "correct_series": correct_series,
        "total_series": total_series,
        "correct_champs": correct_champs,
        "total_champs": total_champs,
    }


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
    selected_window = st.selectbox(
        "Window",
        _windows,
        index=0,
        format_func=lambda w: f"{w} ({_WINDOW_SPANS.get(w, '')})" if _WINDOW_SPANS.get(w) else w,
    )
    selected_run = f"{selected_year}_{selected_window}"
    if selected_run not in available_runs:
        st.error(f"No results for {selected_year} / {selected_window}.")
        st.stop()

# Load data
sim = load_simulation_results(selected_run)
summary = sim["summary"]
adv_df = sim["round_advancement"]
champ_df = sim["championship_probs"]
champ_probs_dict: dict[str, float] = dict(zip(champ_df["team"], champ_df["championship_prob"]))
year: int = summary["year"]
window: str = summary["window"]

team_features = load_team_features(year)
try:
    spec = load_model_spec(window)
except FileNotFoundError:
    spec = None

_insample_all: list[dict] = []
for _w in ["full", "modern", "recent"]:
    try:
        _w_spec = load_model_spec(_w)
    except FileNotFoundError:
        continue
    _w_perf = _compute_model_performance(
        _w,
        tuple(_w_spec["features"]),
        cfg["paths"]["series_dataset_path"],
        cfg["paths"]["training_windows_config"],
        cfg["paths"]["results_dir"],
    )
    _insample_all.append({
        "window": _w,
        "span": _WINDOW_SPANS.get(_w, ""),
        **_w_perf,
    })

_item_style = (
    "display:block;width:100%;box-sizing:border-box;background:#1e1e1e;border:1px solid #333;"
    "border-radius:3px;padding:2px 6px;font-family:monospace;"
    "font-size:12px;color:#ffffff;margin:3px 0;"
)

with panel_col:
    st.markdown('<h1 style="font-size:1.6rem;margin:0rem 0 1rem 0">Model Specification</h1>', unsafe_allow_html=True)
    if spec is not None:
        _span = _WINDOW_SPANS.get(spec["window"], "")
        _span_str = f" ({_span})" if _span else ""
        st.markdown(f"**Training window:** {spec['window']}{_span_str}")
        st.markdown(f"**Observations (N):** {spec['n_obs']}")
        st.markdown("**Features:**")
        pills_html = "".join(
            f'<div style="{_item_style}">{feat} ({spec["coefficients"].get(feat, 0.0):+.4f})</div>'
            for feat in spec["features"]
        )
        pills_html += f'<div style="{_item_style}">intercept ({spec["intercept"]:+.4f})</div>'
        st.markdown(pills_html, unsafe_allow_html=True)
    else:
        st.warning(f"Model spec not found for window '{window}'.")

    # ── Model performance ─────────────────────────────────────────────────
    if spec is not None:
        st.markdown('<hr style="margin:0.5rem 0;border-color:#333">', unsafe_allow_html=True)
        st.markdown('<h1 style="font-size:1.6rem;margin:0 0 1rem 0">Model Performance</h1>', unsafe_allow_html=True)
        series_ds_path = cfg["paths"]["series_dataset_path"]
        tw_config_path = cfg["paths"]["training_windows_config"]
        perf = _compute_model_performance(
            window,
            tuple(spec["features"]),
            series_ds_path,
            tw_config_path,
        )

        feat_rows = ""
        for fs in perf["feat_stats"]:
            sig = "***" if fs["p-value"] < 0.001 else ("**" if fs["p-value"] < 0.01 else ("*" if fs["p-value"] < 0.05 else "·"))
            feat_rows += (
                f'<tr>'
                f'<td style="padding:3px 6px;color:#ccc;font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{fs["Feature"]}</td>'
                f'<td style="padding:3px 6px;text-align:right;color:#fff;font-size:10px">{fs["Coef."]:+.4f}</td>'
                f'<td style="padding:3px 6px;text-align:right;color:#fff;font-size:10px">{fs["z"]:.2f}</td>'
                f'<td style="padding:3px 6px;text-align:right;color:#fff;font-size:10px">{fs["p-value"]:.4f}</td>'
                f'<td style="padding:3px 6px;text-align:center;color:#ff9f40;font-size:10px">{sig}</td>'
                f'</tr>'
            )
        perf_block = (
            "".join(
                f'<div style="{_item_style}">{label}: {val}</div>'
                for label, val in [
                    ("McFadden R²", f"{perf['pseudo_r2']:.3f}"),
                    ("AUC-ROC",     f"{perf['auc']:.3f}"),
                    ("Brier Score", f"{perf['brier']:.4f}"),
                ]
            )
            + '<div style="height:0.5rem"></div>'
            + '<table style="width:100%;table-layout:fixed;border-collapse:collapse;background:#1a1a1a;border-radius:4px;overflow:hidden">'
            + '<thead><tr style="background:#222">'
            + '<th style="width:42%;padding:4px 6px;text-align:left;color:#90a4ae;font-size:10px;font-weight:600">Feature</th>'
            + '<th style="width:20%;padding:4px 6px;text-align:right;color:#90a4ae;font-size:10px;font-weight:600">Coef.</th>'
            + '<th style="width:12%;padding:4px 6px;text-align:right;color:#90a4ae;font-size:10px;font-weight:600">z</th>'
            + '<th style="width:18%;padding:4px 6px;text-align:right;color:#90a4ae;font-size:10px;font-weight:600">p</th>'
            + '<th style="width:8%;padding:4px 6px;text-align:center;color:#90a4ae;font-size:10px;font-weight:600"></th>'
            + '</tr></thead>'
            + f'<tbody>{feat_rows}</tbody>'
            + '</table>'
        )
        st.markdown(perf_block, unsafe_allow_html=True)

        if _insample_all:
            st.markdown('<hr style="margin:0.5rem 0;border-color:#333">', unsafe_allow_html=True)
            st.markdown('<h1 style="font-size:1.6rem;margin:0 0 0.5rem 0">In-Sample Fit</h1>', unsafe_allow_html=True)
            rows_html = ""
            for _r in _insample_all:
                cs, ts = _r["correct_series"], _r["total_series"]
                cc, tc = _r["correct_champs"], _r["total_champs"]
                champ_pct = f"({cc/tc:.0%})" if tc > 0 else "(—)"
                rows_html += (
                    f'<tr>'
                    f'<td style="padding:3px 6px;color:#ccc;font-size:10px;white-space:nowrap">'
                    f'{_r["window"].capitalize()} ({_r["span"]})</td>'
                    f'<td style="padding:3px 6px;text-align:right;color:#fff;font-size:10px">{cs}/{ts}</td>'
                    f'<td style="padding:3px 6px;color:#90a4ae;font-size:10px">({cs/ts:.0%})</td>'
                    f'<td style="padding:3px 6px;text-align:right;color:#fff;font-size:10px">{cc}/{tc}</td>'
                    f'<td style="padding:3px 6px;color:#90a4ae;font-size:10px">{champ_pct}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;background:#1a1a1a;border-radius:4px">'
                f'<thead><tr style="background:#222">'
                f'<th style="padding:4px 6px;text-align:left;color:#90a4ae;font-size:10px;font-weight:600">Window</th>'
                f'<th colspan="2" style="padding:4px 6px;text-align:left;color:#90a4ae;font-size:10px;font-weight:600">Series</th>'
                f'<th colspan="2" style="padding:4px 6px;text-align:left;color:#90a4ae;font-size:10px;font-weight:600">Championships</th>'
                f'</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                f'</table>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
with main_col:
    st.markdown('<h1 style="margin-top:0rem;margin-bottom:0.8rem">NBA Playoff Prediction Model</h1>', unsafe_allow_html=True)
    st.caption("Monte Carlo bracket simulation")

    n_sims = summary.get("n_sims", 0)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Year", str(year))
    c2.metric("Window", window.capitalize())
    _actual_champs_lookup = _load_actual_champions(cfg["paths"]["playoff_series_dir"])
    _actual_champ = summary.get("actual_champion") or _actual_champs_lookup.get(year) or "TBD"
    c3.metric("Predicted Champion", summary.get("predicted_champion") or "—")
    c4.metric("Actual Champion", _actual_champ)
    c5.metric("Simulations", f"{n_sims // 1000}k" if n_sims >= 1000 else str(n_sims))

    st.divider()

    prob_mode = st.radio(
        "Probability Mode",
        ["Matchup Win %", "Championship %"],
        horizontal=True,
    )

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
        components.html(
            _render_bracket_html_canvas(bracket, prob_mode=prob_mode, champ_probs=champ_probs_dict),
            height=400,
            scrolling=False,
        )

        # ── Championship probability bar chart + longest shots ──────────────
        st.markdown('<hr style="margin:6px 0;border-color:#333">', unsafe_allow_html=True)
        chart_col, _gap_col, shots_col = st.columns([3, 0.25, 1.2])

        with chart_col:
            st.subheader("Championship Probabilities")
            components.html(
                _render_champ_prob_chart_html(champ_df, seeds),
                height=250,
                scrolling=False,
            )

        with shots_col:
            st.subheader("Longest Shots")
            nonzero = (
                champ_df[champ_df["championship_prob"] > 0]
                .sort_values("championship_prob")
                .head(3)
            )
            for _, row in nonzero.iterrows():
                prob = float(row["championship_prob"])
                n_wins = int(round(prob * n_sims))
                st.markdown(
                    f'<div style="margin-bottom:12px">'
                    f'<div style="font-size:12px;color:#90a4ae;margin-bottom:1px">{row["team"]}</div>'
                    f'<div style="font-size:24px;font-weight:700;color:#ffffff;line-height:1.1">{prob:.3%}</div>'
                    f'<div style="font-size:11px;color:#ffffff;margin-top:2px">{n_wins} / {n_sims:,} sims</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

