"""quality.py — Data quality checks on the assembled series dataset.

Runs DQ checks:
  - On all feature columns (not just those registered in features.yaml)
  - Across all configured training windows
  - For the validation year and prediction year

Required checks:
  - Missing values per column (missingness rate)
  - Duplicate series IDs
  - Feature distributions (mean, std, min/max, percentiles)
  - Class balance of the dependent variable (higher_seed_wins)
  - Obvious outliers (IQR method, ±3 IQRs from median)

Outputs:
  data/quality_reports/dq_report.parquet   — structured per-feature report
  data/quality_reports/dq_summary.txt      — human-readable summary table
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

# Configurable thresholds
DEFAULT_THRESHOLDS = {
    "max_missingness_rate": 0.05,
    "max_outlier_rate": 0.02,
}

TARGET_COL = "higher_seed_wins"
SERIES_ID_COL = "series_id"
YEAR_COL = "year"


# ── Helpers ────────────────────────────────────────────────────────────────────


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


# ── Individual checks ──────────────────────────────────────────────────────────


def check_duplicate_series_ids(df: pd.DataFrame) -> dict:
    """Check for duplicate series_id values in the dataset.

    Args:
        df: Full assembled series dataset.

    Returns:
        Dict with keys: n_total, n_unique, n_duplicates, duplicate_ids (list).
    """
    if SERIES_ID_COL not in df.columns:
        return {"error": f"Column '{SERIES_ID_COL}' not found."}

    n_total = len(df)
    n_unique = df[SERIES_ID_COL].nunique()
    n_duplicates = n_total - n_unique
    dupes = df[df.duplicated(subset=SERIES_ID_COL, keep=False)][SERIES_ID_COL].unique().tolist()

    result = {
        "n_total": n_total,
        "n_unique": n_unique,
        "n_duplicates": n_duplicates,
        "duplicate_ids": dupes[:10],  # cap at 10 for readability
    }
    if n_duplicates > 0:
        logger.warning(
            "check_duplicate_series_ids: %d duplicate series IDs found: %s",
            n_duplicates,
            dupes[:5],
        )
    else:
        logger.info("check_duplicate_series_ids: all %d series IDs are unique.", n_unique)
    return result


def check_class_balance(df: pd.DataFrame) -> dict:
    """Report class balance of the dependent variable.

    Args:
        df: Full assembled series dataset.

    Returns:
        Dict with keys: n_total, n_higher_seed_wins, n_higher_seed_losses,
        higher_seed_win_rate, is_balanced (True if win rate in [0.45, 0.75]).
    """
    if TARGET_COL not in df.columns:
        return {"error": f"Column '{TARGET_COL}' not found."}

    valid = df[TARGET_COL].dropna()
    n_total = len(valid)
    n_wins = int((valid == 1).sum())
    n_losses = int((valid == 0).sum())
    win_rate = n_wins / n_total if n_total > 0 else np.nan

    result = {
        "n_total": n_total,
        "n_higher_seed_wins": n_wins,
        "n_higher_seed_losses": n_losses,
        "higher_seed_win_rate": round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
        "is_balanced": bool(0.45 <= win_rate <= 0.75) if not np.isnan(win_rate) else False,
    }
    logger.info(
        "check_class_balance: higher_seed_win_rate=%.3f  (%d wins / %d total)",
        win_rate if not np.isnan(win_rate) else -1,
        n_wins,
        n_total,
    )
    return result


# ── Per-feature checks ─────────────────────────────────────────────────────────


def run_feature_checks(
    df: pd.DataFrame,
    features: list[str],
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Run DQ checks for each feature column.

    Args:
        df: Series-level DataFrame (already filtered to a window or year).
        features: Feature column names to check.
        thresholds: Override default missingness/outlier thresholds.

    Returns:
        DataFrame with one row per feature and columns:
        feature, n_rows, missingness_rate, mean, std, min, p25, median, p75, max,
        outlier_rate, year_coverage, pass_flag, note.
    """
    thresh = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    records = []
    for feat in features:
        if feat not in df.columns:
            records.append(
                {
                    "feature": feat,
                    "n_rows": len(df),
                    "pass_flag": False,
                    "note": "column missing",
                }
            )
            continue

        col = df[feat].dropna()
        n_rows = len(df)
        miss_rate = df[feat].isna().mean()
        year_coverage = (
            int(df.loc[df[feat].notna(), YEAR_COL].nunique()) if YEAR_COL in df.columns else -1
        )
        is_numeric = pd.api.types.is_numeric_dtype(df[feat])
        outlier_rate = _iqr_outlier_rate(col) if len(col) > 0 and is_numeric else np.nan

        pass_flag = miss_rate <= thresh["max_missingness_rate"] and (
            np.isnan(outlier_rate) or outlier_rate <= thresh["max_outlier_rate"]
        )

        records.append(
            {
                "feature": feat,
                "n_rows": n_rows,
                "missingness_rate": round(miss_rate, 4),
                "mean": round(float(col.mean()), 4) if len(col) > 0 and is_numeric else np.nan,
                "std": round(float(col.std()), 4) if len(col) > 0 and is_numeric else np.nan,
                "min": round(float(col.min()), 4) if len(col) > 0 and is_numeric else np.nan,
                "p25": round(float(col.quantile(0.25)), 4)
                if len(col) > 0 and is_numeric
                else np.nan,
                "median": round(float(col.median()), 4) if len(col) > 0 and is_numeric else np.nan,
                "p75": round(float(col.quantile(0.75)), 4)
                if len(col) > 0 and is_numeric
                else np.nan,
                "max": round(float(col.max()), 4) if len(col) > 0 and is_numeric else np.nan,
                "outlier_rate": round(outlier_rate, 4) if not np.isnan(outlier_rate) else np.nan,
                "year_coverage": year_coverage,
                "pass_flag": pass_flag,
                "note": "",
            }
        )

    return pd.DataFrame(records)


# ── Main entrypoint ────────────────────────────────────────────────────────────


def run_quality_checks(
    df: pd.DataFrame,
    features: list[str] | None = None,
    output_dir: Path = QUALITY_DIR,
) -> Path:
    """Run all DQ checks on the assembled dataset and write reports.

    Checks all delta feature columns in the dataset (not just those registered
    in features.yaml), plus global checks for duplicate series IDs and class balance.

    Args:
        df: Full assembled series dataset (one row per playoff series).
        features: Feature names to check. Defaults to all delta/feature columns
            found in the dataset.
        output_dir: Where to write reports.

    Returns:
        Path to the parquet report file.
    """
    # Determine feature columns: all delta_* columns + top3_gp_pct_delta
    if features is None:
        features = sorted(
            c for c in df.columns if c.startswith("delta_") or c == "top3_gp_pct_delta"
        )
        if not features:
            # Fall back to features.yaml if no delta columns found
            try:
                with open(FEATURES_CONFIG) as f:
                    config = yaml.safe_load(f)
                features = [feat["name"] for feat in config["features"] if feat.get("active", True)]
            except FileNotFoundError:
                features = [
                    c
                    for c in df.columns
                    if c
                    not in {
                        "series_id",
                        "year",
                        "season",
                        "higher_seed_wins",
                        "team_high",
                        "team_low",
                        "round",
                        "conference",
                        "seed_high",
                        "seed_low",
                    }
                ]

    windows_cfg = _load_windows()
    slices: dict[str, pd.DataFrame] = {}

    for w in windows_cfg["windows"]:
        mask = (df[YEAR_COL] >= w["start_year"]) & (df[YEAR_COL] <= w["end_year"])
        slices[w["name"]] = df[mask].copy()

    for special_year_key in ("validation_year", "prediction_year"):
        year = windows_cfg.get(special_year_key)
        if year is not None:
            slices[special_year_key] = df[df[YEAR_COL] == year].copy()

    # Per-feature checks across all windows (skip empty slices)
    all_checks: list[pd.DataFrame] = []
    for slice_name, slice_df in slices.items():
        if slice_df.empty:
            logger.info("Skipping window '%s' — no rows in dataset for this period.", slice_name)
            continue
        checks = run_feature_checks(slice_df, features)
        checks.insert(0, "window", slice_name)
        all_checks.append(checks)

    if not all_checks:
        logger.warning("No non-empty windows found; per-feature report will be empty.")
        report = pd.DataFrame()
    else:
        report = pd.concat(all_checks, ignore_index=True)

    # Global checks
    dup_check = check_duplicate_series_ids(df)
    balance_check = check_class_balance(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dq_report.parquet"
    report.to_parquet(report_path, index=False)

    # ── Human-readable summary ─────────────────────────────────────────────────
    summary_path = output_dir / "dq_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("NBA Playoff Model — Data Quality Summary\n")
        f.write("=" * 72 + "\n\n")

        # Dataset shape
        f.write(f"Dataset shape : {len(df)} rows × {len(df.columns)} columns\n")
        f.write(f"Feature count : {len(features)} delta features\n")
        f.write(f"Season range  : {int(df[YEAR_COL].min())}–{int(df[YEAR_COL].max())}\n")
        f.write("\n")

        # Duplicate series IDs
        f.write("── Duplicate series IDs ─────────────────────────────────────────\n")
        if dup_check.get("n_duplicates", 0) == 0:
            f.write(f"  PASS  {dup_check['n_unique']} unique series IDs (no duplicates)\n")
        else:
            f.write(
                f"  FAIL  {dup_check['n_duplicates']} duplicate IDs found: "
                f"{dup_check['duplicate_ids']}\n"
            )
        f.write("\n")

        # Class balance
        f.write("── Class balance (higher_seed_wins) ─────────────────────────────\n")
        wr = balance_check.get("higher_seed_win_rate")
        wr_str = f"{wr:.3f}" if isinstance(wr, float) and not np.isnan(wr) else "N/A"
        status = "PASS" if balance_check.get("is_balanced") else "NOTE"
        f.write(
            f"  {status}  higher_seed_win_rate = {wr_str}  "
            f"({balance_check.get('n_higher_seed_wins','?')} wins / "
            f"{balance_check.get('n_total','?')} total)\n"
        )
        f.write("\n")

        # Per-window feature checks
        for window, group in report.groupby("window"):
            n_pass = int(group["pass_flag"].sum())
            n_total = len(group)
            f.write(
                f"── Window: {window}  ({n_pass}/{n_total} features pass) "
                f"─────────────────────────────\n"
            )

            # Distribution table for this window (full dataset window only)
            if window == "full":
                header = (
                    f"  {'Feature':<45} {'Miss%':>6} {'Mean':>9} {'Std':>9} "
                    f"{'Min':>9} {'Median':>9} {'Max':>9} {'Outlier%':>9} Pass\n"
                )
                f.write(header)
                f.write("  " + "-" * (len(header) - 3) + "\n")
                for _, row in group.sort_values("feature").iterrows():
                    flag = "OK" if row["pass_flag"] else "FAIL"
                    miss = f"{row.get('missingness_rate', 0):.1%}"
                    mean_ = (
                        f"{row.get('mean', float('nan')):.2f}"
                        if pd.notna(row.get("mean"))
                        else "  —"
                    )
                    std_ = (
                        f"{row.get('std', float('nan')):.2f}" if pd.notna(row.get("std")) else "  —"
                    )
                    min_ = (
                        f"{row.get('min', float('nan')):.2f}" if pd.notna(row.get("min")) else "  —"
                    )
                    med_ = (
                        f"{row.get('median', float('nan')):.2f}"
                        if pd.notna(row.get("median"))
                        else "  —"
                    )
                    max_ = (
                        f"{row.get('max', float('nan')):.2f}" if pd.notna(row.get("max")) else "  —"
                    )
                    out_ = (
                        f"{row.get('outlier_rate', float('nan')):.1%}"
                        if pd.notna(row.get("outlier_rate"))
                        else "  —"
                    )
                    f.write(
                        f"  {row['feature']:<45} {miss:>6} {mean_:>9} {std_:>9} "
                        f"{min_:>9} {med_:>9} {max_:>9} {out_:>9} {flag}\n"
                    )
            else:
                # For other windows just list failures
                failures = group[~group["pass_flag"]]
                if failures.empty:
                    f.write("  All features pass.\n")
                else:
                    for _, row in failures.iterrows():
                        f.write(
                            f"  FAIL  {row['feature']}: "
                            f"miss={row.get('missingness_rate','?')}, "
                            f"outlier={row.get('outlier_rate','?')}\n"
                        )
            f.write("\n")

    logger.info("DQ report written to %s", report_path)
    logger.info("DQ summary written to %s", summary_path)
    return report_path
