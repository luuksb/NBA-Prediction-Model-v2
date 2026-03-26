"""quality.py — Data quality checks on the assembled series dataset.

Runs per-feature DQ checks across all training windows, the validation year,
and the prediction year. Outputs a structured parquet report and a
human-readable text summary to data/quality_reports/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

QUALITY_DIR = Path("data/quality_reports")
FEATURES_CONFIG = Path("configs/features.yaml")
WINDOWS_CONFIG = Path("configs/training_windows.yaml")

# Configurable thresholds (override via configs if desired)
DEFAULT_THRESHOLDS = {
    "max_missingness_rate": 0.05,
    "max_outlier_rate": 0.02,
}


def _load_windows() -> dict:
    with open(WINDOWS_CONFIG) as f:
        return yaml.safe_load(f)


def _iqr_outlier_rate(series: pd.Series) -> float:
    """Return the fraction of values more than 3 IQRs from the median."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    mask = (series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)
    return float(mask.mean())


def run_feature_checks(
    df: pd.DataFrame,
    features: list[str],
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Run DQ checks for each feature across the full DataFrame slice.

    Args:
        df: Series-level DataFrame (already filtered to a window or year).
        features: Feature column names to check.
        thresholds: Override default missingness/outlier thresholds.

    Returns:
        DataFrame with one row per feature and columns:
        feature, n_rows, missingness_rate, mean, std, min, p25, median, p75, max,
        outlier_rate, year_coverage, pass_flag.
    """
    thresh = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    records = []
    for feat in features:
        if feat not in df.columns:
            records.append({"feature": feat, "pass_flag": False, "note": "column missing"})
            continue
        col = df[feat].dropna()
        n_rows = len(df)
        miss_rate = df[feat].isna().mean()
        year_coverage = int(df.loc[df[feat].notna(), "year"].nunique()) if "year" in df.columns else -1
        outlier_rate = _iqr_outlier_rate(col) if len(col) > 0 and pd.api.types.is_numeric_dtype(col) else np.nan

        pass_flag = (
            miss_rate <= thresh["max_missingness_rate"]
            and (np.isnan(outlier_rate) or outlier_rate <= thresh["max_outlier_rate"])
        )

        records.append({
            "feature": feat,
            "n_rows": n_rows,
            "missingness_rate": round(miss_rate, 4),
            "mean": round(col.mean(), 4) if len(col) > 0 else np.nan,
            "std": round(col.std(), 4) if len(col) > 0 else np.nan,
            "min": round(col.min(), 4) if len(col) > 0 else np.nan,
            "p25": round(col.quantile(0.25), 4) if len(col) > 0 else np.nan,
            "median": round(col.median(), 4) if len(col) > 0 else np.nan,
            "p75": round(col.quantile(0.75), 4) if len(col) > 0 else np.nan,
            "max": round(col.max(), 4) if len(col) > 0 else np.nan,
            "outlier_rate": round(outlier_rate, 4) if not np.isnan(outlier_rate) else np.nan,
            "year_coverage": year_coverage,
            "pass_flag": pass_flag,
        })
    return pd.DataFrame(records)


def run_quality_checks(
    df: pd.DataFrame,
    features: list[str] | None = None,
    output_dir: Path = QUALITY_DIR,
) -> Path:
    """Run DQ checks across all windows and save reports.

    Args:
        df: Full assembled series dataset.
        features: Feature names to check. Defaults to all active features.
        output_dir: Where to write reports.

    Returns:
        Path to the parquet report file.
    """
    if features is None:
        with open(FEATURES_CONFIG) as f:
            config = yaml.safe_load(f)
        features = [feat["name"] for feat in config["features"] if feat.get("active", True)]

    windows_cfg = _load_windows()
    slices: dict[str, pd.DataFrame] = {}

    for w in windows_cfg["windows"]:
        mask = (df["year"] >= w["start_year"]) & (df["year"] <= w["end_year"])
        slices[w["name"]] = df[mask]

    for special_year_key in ("validation_year", "prediction_year"):
        year = windows_cfg.get(special_year_key)
        if year is not None:
            slices[special_year_key] = df[df["year"] == year]

    all_checks = []
    for slice_name, slice_df in slices.items():
        checks = run_feature_checks(slice_df, features)
        checks.insert(0, "window", slice_name)
        all_checks.append(checks)

    report = pd.concat(all_checks, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dq_report.parquet"
    report.to_parquet(report_path, index=False)

    # Human-readable summary
    summary_path = output_dir / "dq_summary.txt"
    with open(summary_path, "w") as f:
        f.write("NBA Playoff Model — Data Quality Summary\n")
        f.write("=" * 60 + "\n\n")
        for window, group in report.groupby("window"):
            n_pass = group["pass_flag"].sum()
            n_total = len(group)
            f.write(f"Window: {window}  ({n_pass}/{n_total} features pass)\n")
            failures = group[~group["pass_flag"]]
            if not failures.empty:
                for _, row in failures.iterrows():
                    f.write(f"  FAIL  {row['feature']}: miss={row.get('missingness_rate','?')}, "
                            f"outlier={row.get('outlier_rate','?')}\n")
            f.write("\n")

    logger.info("DQ report written to %s", report_path)
    logger.info("DQ summary written to %s", summary_path)
    return report_path
