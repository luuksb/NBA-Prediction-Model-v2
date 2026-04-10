"""src/dashboard/ui_layout.py — Streamlit-cached data functions for the dashboard.

Contains ``@st.cache_data`` functions that are expensive to recompute on each
Streamlit rerun.  Imported by ``scripts/run_dashboard.py``.

No cross-module imports outside of ``src.dashboard``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import yaml


@st.cache_data
def load_actual_champions(playoff_series_dir: str) -> dict[int, str]:
    """Return a dict mapping year → actual champion abbreviation.

    Reads the raw per-year playoff series CSVs which carry explicit team_high /
    team_low columns.  Years with more than one finals row are resolved via a
    small hardcoded override for early-era data quality issues (1980–1983).

    Args:
        playoff_series_dir: Path to the directory containing ``*_nba_api.csv``
            playoff series files.

    Returns:
        Dict mapping integer year to team abbreviation string.
    """
    # Hardcoded for 1980-1983 where the raw data has multiple finals rows
    _overrides: dict[int, str] = {1980: "LAL", 1981: "BOS", 1982: "LAL", 1983: "PHI", 2025: "OKC"}

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
def compute_model_performance(
    window: str,
    features: tuple[str, ...],
    series_dataset_path: str,
    training_windows_config: str,
    playoff_series_dir: str,
    results_dir: str = "results/simulations",
) -> dict:
    """Fit logistic regression for the given window and return performance metrics.

    Results are cached so the fit only runs once per window/feature combination.

    Args:
        window: Training window name (``"full"``, ``"modern"``, or ``"recent"``).
        features: Tuple of feature column names used by the model.
        series_dataset_path: Path to ``series_dataset.parquet``.
        training_windows_config: Path to ``configs/training_windows.yaml``.
        playoff_series_dir: Path to raw playoff series CSVs (for champion lookup).
        results_dir: Path to the simulations results directory.

    Returns:
        Dict with keys: ``pseudo_r2``, ``auc``, ``brier``, ``n_obs``,
        ``feat_stats``, ``correct_series``, ``total_series``,
        ``correct_champs``, ``total_champs``.
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

    actual_champions = load_actual_champions(playoff_series_dir)

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
