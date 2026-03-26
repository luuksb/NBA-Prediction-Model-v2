#!/usr/bin/env python3
"""run_dashboard.py — CLI entry point for Module 5: Dashboard.

Launches the Streamlit dashboard. Reads only from results/.

Usage:
    python scripts/run_dashboard.py
    # or directly:
    streamlit run scripts/run_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(page_title="NBA Playoff Prediction Model", layout="wide")

st.title("NBA Playoff Prediction Model")
st.caption("Monte Carlo bracket simulation — 50,000 iterations")

RESULTS_DIR = Path("results/simulations")

# Year selector
available_dirs = sorted(RESULTS_DIR.glob("*/")) if RESULTS_DIR.exists() else []
if not available_dirs:
    st.warning("No simulation results found. Run `python scripts/run_bracket_sim.py` first.")
    st.stop()

year_window_options = [d.name for d in available_dirs]
selected = st.sidebar.selectbox("Year × Window", year_window_options)

result_dir = RESULTS_DIR / selected

# Championship probabilities
import json
import pandas as pd

summary_path = result_dir / "summary.json"
if summary_path.exists():
    with open(summary_path) as f:
        summary = json.load(f)
    st.subheader(f"Year: {summary['year']}  |  Window: {summary['window']}")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Champion", summary.get("predicted_champion", "—"))
    col2.metric("Actual Champion", summary.get("actual_champion") or "TBD")

champ_path = result_dir / "championship_probs.parquet"
if champ_path.exists():
    champ_df = pd.read_parquet(champ_path)
    st.subheader("Championship Probabilities")
    st.bar_chart(champ_df.set_index("team")["championship_prob"])

adv_path = result_dir / "round_advancement.parquet"
if adv_path.exists():
    adv_df = pd.read_parquet(adv_path)
    st.subheader("Round-by-Round Advancement Rates")
    pivot = adv_df.pivot(index="team", columns="round", values="advancement_prob")
    pivot.columns = [f"Round {r}" for r in pivot.columns]
    st.dataframe(pivot.sort_values("Round 1", ascending=False))
