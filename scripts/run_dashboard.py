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

import pandas as pd
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
from src.dashboard.html_renderer import render_bracket_html_canvas, render_champ_prob_chart_html
from src.dashboard.ui_layout import compute_model_performance, load_actual_champions

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
    w["name"]: f"{w['start_year']}–{w['end_year']}" for w in _tw_cfg["windows"]
}

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_GLOBAL_CSS = f"""
<style>
  /* ── Page background ─────────────────────────────────────────────────── */
  .stApp {{ background-color: {COLORS['background']}; }}
  section[data-testid="stSidebar"] {{ background-color: #121b2e; }}

  /* ── Left panel column ───────────────────────────────────────────────── */
  div[data-testid="stColumn"]:first-child {{
    position: relative !important;
    background-color: #1e2a45 !important;
    border-radius: 14px !important;
    padding: 12px 10px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35) !important;
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
    border-radius: 14px;
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
    color: #8fa3c1;
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
    background: #1e2a45;
    border: none;
    border-radius: 8px;
    min-height: 36px;
    position: relative;
    box-sizing: border-box;
    width: 100%;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }}
  .team-node:hover {{ box-shadow: 0 2px 12px rgba(90,180,255,0.22); }}
  .team-node.champion {{
    box-shadow: 0 0 12px 3px rgba(255,215,0,0.32);
    outline: 1px solid {COLORS['champion_border']};
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
    color: #8fa3c1;
  }}

  /* ── Win probability ─────────────────────────────────────────────────── */
  .team-prob {{
    font-size: 10px;
    font-weight: 700;
    color: #8fa3c1;
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
    border-right: 1.5px solid rgba(90,150,220,0.45);
  }}
  .conf-block.east .round-col-body .matchup::after {{
    content: '';
    position: absolute;
    left: -7px;
    top: 24%;
    height: 52%;
    width: 0;
    border-left: 1.5px solid rgba(90,150,220,0.45);
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
    border-right: 1.5px solid rgba(90,150,220,0.45);
  }}
  .conf-block.east .matchup-pair::after {{
    content: '';
    position: absolute;
    left: -7px;
    top: 25%;
    height: 50%;
    width: 0;
    border-left: 1.5px solid rgba(90,150,220,0.45);
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
    background: rgba(90,150,220,0.45);
  }}
  .conf-block.west .r3 .matchup::before {{
    content: '';
    position: absolute;
    right: -20px;
    top: 50%;
    transform: translateY(-50%);
    width: 13px;
    height: 1.5px;
    background: rgba(90,150,220,0.45);
  }}
  .conf-block.east .r1 .matchup-pair::before {{
    content: '';
    position: absolute;
    left: -32px;
    top: 50%;
    transform: translateY(-50%);
    width: 25px;
    height: 1.5px;
    background: rgba(90,150,220,0.45);
  }}
  .conf-block.east .r3 .matchup::before {{
    content: '';
    position: absolute;
    left: -20px;
    top: 50%;
    transform: translateY(-50%);
    width: 13px;
    height: 1.5px;
    background: rgba(90,150,220,0.45);
  }}

/* ── Text visibility on dark background ──────────────────────────────── */
  .stApp {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
  .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
  }}
  .stApp p {{ color: #c8d6e8 !important; }}
  [data-testid="stMetricValue"] {{ color: #ffffff !important; font-weight: 700 !important; }}
  [data-testid="stMetricLabel"] > div {{ color: #8fa3c1 !important; font-size: 0.78rem !important; }}
  [data-testid="stMarkdownContainer"] p {{ color: #c8d6e8 !important; }}
  .stCaption {{ color: #8fa3c1 !important; }}

  /* ── Streamlit divider ───────────────────────────────────────────────── */
  hr {{ border-color: #2a3a54 !important; }}

  /* ── Streamlit buttons (pill style, accent fill) ─────────────────────── */
  .stButton > button {{
    background-color: #253350 !important;
    color: #e8f0fb !important;
    border: 1px solid #2a3a54 !important;
    border-radius: 20px !important;
    padding: 6px 18px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    transition: background 0.15s, box-shadow 0.15s !important;
  }}
  .stButton > button:hover {{
    background-color: #5ab4ff !important;
    color: #0d1520 !important;
    border-color: #5ab4ff !important;
    box-shadow: 0 2px 10px rgba(90,180,255,0.3) !important;
  }}

  /* ── Streamlit radio buttons ─────────────────────────────────────────── */
  [data-testid="stRadio"] > div > label {{
    color: #c8d6e8 !important;
  }}
  [data-testid="stRadio"] > div > label[data-baseweb="radio"] span:first-child {{
    background-color: #1e2a45 !important;
    border-color: #2a3a54 !important;
  }}

  /* ── Streamlit selectbox ─────────────────────────────────────────────── */
  [data-testid="stSelectbox"] > div > div {{
    background-color: #1e2a45 !important;
    border: 1px solid #2a3a54 !important;
    border-radius: 8px !important;
    color: #e8f0fb !important;
  }}
  [data-testid="stSelectbox"] label {{
    color: #8fa3c1 !important;
    font-size: 0.82rem !important;
  }}

  /* ── Streamlit metric cards ──────────────────────────────────────────── */
  [data-testid="stMetric"] {{
    background-color: #1e2a45 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
  }}

  /* ── Streamlit native table / dataframe ──────────────────────────────── */
  [data-testid="stDataFrame"] thead tr th {{
    background-color: #253350 !important;
    color: #8fa3c1 !important;
    font-weight: 600 !important;
    border-bottom: 1px solid #2a3a54 !important;
  }}
  [data-testid="stDataFrame"] tbody tr:nth-child(even) td {{
    background-color: #1e2a45 !important;
  }}
  [data-testid="stDataFrame"] tbody tr:nth-child(odd) td {{
    background-color: #243050 !important;
  }}
  [data-testid="stDataFrame"] tbody tr td {{
    color: #e8f0fb !important;
    border: none !important;
  }}

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

  /* ── Hide heading anchor links ───────────────────────────────────────── */
  h1 a, h2 a, h3 a {{ display: none !important; }}
</style>
"""

st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)



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
    st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 style="font-size:1.6rem;margin:0 0 0.5rem 0">Configuration</h1>',
        unsafe_allow_html=True,
    )
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
    _w_perf = compute_model_performance(
        _w,
        tuple(_w_spec["features"]),
        cfg["paths"]["series_dataset_path"],
        cfg["paths"]["training_windows_config"],
        cfg["paths"]["playoff_series_dir"],
        cfg["paths"]["results_dir"],
    )
    _insample_all.append(
        {
            "window": _w,
            "span": _WINDOW_SPANS.get(_w, ""),
            **_w_perf,
        }
    )

_item_style = (
    "display:block;width:100%;box-sizing:border-box;background:#253350;border:none;"
    "border-radius:8px;padding:3px 8px;font-family:monospace;"
    "font-size:11px;color:#e8f0fb;margin:3px 0;"
    "box-shadow:0 2px 6px rgba(0,0,0,0.3);"
)

with panel_col:
    st.markdown(
        '<hr style="margin:0.3rem 0 0.5rem 0;border-color:#2a3a54">', unsafe_allow_html=True
    )
    st.markdown(
        '<h1 style="font-size:1.6rem;margin:0 0 0.5rem 0">Model Specification</h1>',
        unsafe_allow_html=True,
    )
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
        st.markdown(
            '<hr style="margin:1.5rem 0 0.5rem 0;border-color:#2a3a54">', unsafe_allow_html=True
        )
        st.markdown(
            '<h1 style="font-size:1.6rem;margin:0 0 0.5rem 0">Model Performance</h1>',
            unsafe_allow_html=True,
        )
        series_ds_path = cfg["paths"]["series_dataset_path"]
        tw_config_path = cfg["paths"]["training_windows_config"]
        perf = compute_model_performance(
            window,
            tuple(spec["features"]),
            series_ds_path,
            tw_config_path,
            cfg["paths"]["playoff_series_dir"],
        )

        feat_rows = ""
        for fs in perf["feat_stats"]:
            sig = (
                "***"
                if fs["p-value"] < 0.001
                else ("**" if fs["p-value"] < 0.01 else ("*" if fs["p-value"] < 0.05 else "·"))
            )
            feat_rows += (
                f'<tr style="border-bottom:1px solid #1e2a45">'
                f'<td style="padding:4px 6px;color:#8fa3c1;font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{fs["Feature"]}</td>'
                f'<td style="padding:4px 6px;text-align:right;color:#e8f0fb;font-size:10px">{fs["Coef."]:+.4f}</td>'
                f'<td style="padding:4px 6px;text-align:right;color:#e8f0fb;font-size:10px">{fs["z"]:.2f}</td>'
                f'<td style="padding:4px 6px;text-align:right;color:#e8f0fb;font-size:10px">{fs["p-value"]:.4f}</td>'
                f'<td style="padding:4px 6px;text-align:center;color:#e84d4d;font-size:10px">{sig}</td>'
                f'</tr>'
            )
        perf_block = (
            "".join(
                f'<div style="{_item_style}">{label}: {val}</div>'
                for label, val in [
                    ("McFadden R²", f"{perf['pseudo_r2']:.3f}"),
                    ("AUC-ROC", f"{perf['auc']:.3f}"),
                    ("Brier Score", f"{perf['brier']:.4f}"),
                ]
            )
            + '<div style="height:0.5rem"></div>'
            + '<table style="width:100%;table-layout:fixed;border-collapse:collapse;background:#1e2a45;border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.3)">'
            + '<thead><tr style="background:#253350;border-bottom:1px solid #2a3a54">'
            + '<th style="width:42%;padding:5px 6px;text-align:left;color:#8fa3c1;font-size:10px;font-weight:600;letter-spacing:0.05em">Feature</th>'
            + '<th style="width:20%;padding:5px 6px;text-align:right;color:#8fa3c1;font-size:10px;font-weight:600;letter-spacing:0.05em">Coef.</th>'
            + '<th style="width:12%;padding:5px 6px;text-align:right;color:#8fa3c1;font-size:10px;font-weight:600;letter-spacing:0.05em">z</th>'
            + '<th style="width:18%;padding:5px 6px;text-align:right;color:#8fa3c1;font-size:10px;font-weight:600;letter-spacing:0.05em">p</th>'
            + '<th style="width:8%;padding:5px 6px;text-align:center;color:#8fa3c1;font-size:10px;font-weight:600"></th>'
            + "</tr></thead>"
            + f"<tbody>{feat_rows}</tbody>"
            + "</table>"
        )
        st.markdown(perf_block, unsafe_allow_html=True)

        if _insample_all:
            st.markdown(
                '<hr style="margin:1.5rem 0 0.5rem 0;border-color:#2a3a54">', unsafe_allow_html=True
            )
            st.markdown(
                '<h1 style="font-size:1.6rem;margin:0 0 0.5rem 0">In-Sample Fit</h1>',
                unsafe_allow_html=True,
            )
            rows_html = ""
            for _r in _insample_all:
                cs, ts = _r["correct_series"], _r["total_series"]
                cc, tc = _r["correct_champs"], _r["total_champs"]
                champ_pct = f"({cc/tc:.0%})" if tc > 0 else "(—)"
                rows_html += (
                    f'<tr style="border-bottom:1px solid #1e2a45">'
                    f'<td style="padding:4px 6px;color:#8fa3c1;font-size:10px;white-space:nowrap">'
                    f'{_r["window"].capitalize()} ({_r["span"]})</td>'
                    f'<td style="padding:4px 6px;text-align:right;color:#e8f0fb;font-size:10px">{cs}/{ts}</td>'
                    f'<td style="padding:4px 6px;color:#8fa3c1;font-size:10px">({cs/ts:.0%})</td>'
                    f'<td style="padding:4px 6px;text-align:right;color:#e8f0fb;font-size:10px">{cc}/{tc}</td>'
                    f'<td style="padding:4px 6px;color:#8fa3c1;font-size:10px">{champ_pct}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<table style="width:100%;table-layout:fixed;border-collapse:collapse;background:#1e2a45;border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.3)">'
                f'<thead><tr style="background:#253350;border-bottom:1px solid #2a3a54">'
                f'<th style="width:36%;padding:5px 6px;text-align:left;color:#8fa3c1;font-size:10px;font-weight:600;letter-spacing:0.05em">Window</th>'
                f'<th style="width:16%;padding:5px 6px;text-align:right;color:#8fa3c1;font-size:10px;font-weight:600;letter-spacing:0.05em">Series</th>'
                f'<th style="width:12%;padding:5px 6px;text-align:left;color:#8fa3c1;font-size:10px;font-weight:600;letter-spacing:0.05em"></th>'
                f'<th style="width:24%;padding:5px 6px;text-align:right;color:#8fa3c1;font-size:10px;font-weight:600;letter-spacing:0.05em">Championships</th>'
                f'<th style="width:12%;padding:5px 6px;text-align:left;color:#8fa3c1;font-size:10px;font-weight:600;letter-spacing:0.05em"></th>'
                f"</tr></thead>"
                f"<tbody>{rows_html}</tbody>"
                f"</table>",
                unsafe_allow_html=True,
            )

    # ── Injury impact ──────────────────────────────────────────────────────
    _iter_path = Path(cfg["paths"]["results_dir"]) / selected_run / "iterations.parquet"
    if _iter_path.exists():
        _iter_df = pd.read_parquet(_iter_path)
        _inj_df = _iter_df.dropna(
            subset=["finalist_east_injuries", "finalist_west_injuries"]
        ).copy()
        if not _inj_df.empty:
            _inj_df["_east_inj"] = _inj_df["finalist_east_injuries"].astype(int)
            _inj_df["_west_inj"] = _inj_df["finalist_west_injuries"].astype(int)
            _inj_df["_total_inj"] = _inj_df["_east_inj"] + _inj_df["_west_inj"]
            _inj_df["_champ_is_east"] = _inj_df["champion"] == _inj_df["finalist_east"]
            _inj_df["_champ_inj"] = _inj_df["_east_inj"].where(
                _inj_df["_champ_is_east"], _inj_df["_west_inj"]
            )
            _inj_df["_loser_inj"] = _inj_df["_west_inj"].where(
                _inj_df["_champ_is_east"], _inj_df["_east_inj"]
            )

            _pct_any = (_inj_df["_total_inj"] > 0).mean()
            _one_sided = _inj_df[(_inj_df["_champ_inj"] == 0) != (_inj_df["_loser_inj"] == 0)]
            _healthy_won = ((_one_sided["_champ_inj"] == 0) & (_one_sided["_loser_inj"] > 0)).sum()
            _healthy_win_rate = _healthy_won / len(_one_sided) if len(_one_sided) > 0 else 0.0
            _pct_inj_champ = (_inj_df["_champ_inj"] > 0).mean()

            st.markdown(
                '<hr style="margin:1.5rem 0 0.5rem 0;border-color:#2a3a54">', unsafe_allow_html=True
            )
            st.markdown(
                '<h1 style="font-size:1.6rem;margin:0 0 0rem 0">Injury Impact*</h1>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p style="font-size:0.78rem;color:#8fa3c1;margin:0 0 0.5rem 0">* Applicable only to out-of-sample (2025–2026)</p>',
                unsafe_allow_html=True,
            )
            _inj_items = [
                f"{_pct_any:.0%} of Finals have 1+ injured star",
                f"Healthy finalist wins {_healthy_win_rate:.0%} of 1-sided injury matchups",
                f"Champion overcame an injury in {_pct_inj_champ:.0%} of all simulated Finals",
            ]
            st.markdown(
                "".join(f'<div style="{_item_style}">{item}</div>' for item in _inj_items),
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
with main_col:
    st.markdown(
        '<h1 style="margin-top:0rem;margin-bottom:0.1rem">NBA Playoff Prediction Model</h1>'
        '<p style="margin-top:0;margin-bottom:0.3rem;color:#8fa3c1;font-size:0.875rem">Monte Carlo bracket simulation</p>',
        unsafe_allow_html=True,
    )

    n_sims = summary.get("n_sims", 0)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Year", str(year))
    c2.metric("Window", window.capitalize())
    _actual_champs_lookup = load_actual_champions(cfg["paths"]["playoff_series_dir"])
    _actual_champ = summary.get("actual_champion") or _actual_champs_lookup.get(year) or "TBD"
    c3.metric("Predicted Champion", summary.get("predicted_champion") or "—")
    c4.metric("Actual Champion", _actual_champ)
    c5.metric("Simulations", f"{n_sims // 1000}k" if n_sims >= 1000 else str(n_sims))

    st.markdown(
        '<hr style="margin-top:2.3rem;margin-bottom:2rem;border-color:#2a3a54">',
        unsafe_allow_html=True,
    )

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
            render_bracket_html_canvas(bracket, COLORS, prob_mode=prob_mode, champ_probs=champ_probs_dict),
            height=400,
            scrolling=False,
        )

        # ── Championship probability bar chart + longest shots ──────────────
        st.markdown('<hr style="margin:2px 0;border-color:#333">', unsafe_allow_html=True)
        chart_col, _gap_col, shots_col = st.columns([3, 0.25, 1.2])

        with chart_col:
            st.subheader("Championship Probabilities")
            components.html(
                render_champ_prob_chart_html(champ_df, seeds),
                height=250,
                scrolling=False,
            )

        with shots_col:
            st.subheader("Longest Shots")
            nonzero = (
                champ_df[champ_df["championship_prob"] > 0].sort_values("championship_prob").head(3)
            )
            for _, row in nonzero.iterrows():
                prob = float(row["championship_prob"])
                n_wins = int(round(prob * n_sims))
                st.markdown(
                    f'<div style="margin-bottom:12px">'
                    f'<div style="font-size:12px;color:#8fa3c1;margin-bottom:1px">{row["team"]}</div>'
                    f'<div style="font-size:24px;font-weight:700;color:#ffffff;line-height:1.1">{prob:.3%}</div>'
                    f'<div style="font-size:11px;color:#ffffff;margin-top:2px">{n_wins} / {n_sims:,} sims</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
