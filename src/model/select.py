"""select.py — Lock in and serialize the chosen model specification.

After reviewing the leaderboard, this module serialises the chosen model to
results/model_selection/chosen_model.json so the simulation module can load
it without importing anything from src.model.

Also contains run_combinatorial_pipeline(), which fits all active-feature
combinations (sizes 2–5) across all training windows and prints 12 ranked
leaderboards (3 windows × 4 metrics, top 5 each) for human review.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import warnings
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.model.evaluate import build_window_metric_leaderboards, evaluate_model
from src.model.feature_sets import (
    filter_forbidden_pairs,
    generate_all_subsets,
    get_active_features,
    get_forbidden_pairs,
)
from src.model.fit import ModelSpec, fit_logit

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/model_selection")
CHOSEN_MODEL_PATH = RESULTS_DIR / "chosen_model.json"
_WINDOWS_CONFIG = Path("configs/training_windows.yaml")
_MODEL_SELECTION_CONFIG = Path("configs/model_selection.yaml")
_TARGET_COL = "higher_seed_wins"

# Combination size range for the combinatorial search (per task spec).
_COMBO_MIN_SIZE = 5
_COMBO_MAX_SIZE = 5

# Stop-condition thresholds.
_MAX_COMBINATIONS = 80_000
_MIN_WINDOW_SERIES = 30
_MAX_CONVERGENCE_FAILURE_RATE = 0.20

_METRIC_LABELS: dict[str, str] = {
    "mcfadden_r2": "McFadden Pseudo R²",
    "brier_score": "Brier Score       ",
    "auc_roc": "AUC-ROC           ",
    "bic": "BIC               ",
}


def save_chosen_model(spec: ModelSpec) -> Path:
    """Serialize the chosen model spec to JSON.

    Args:
        spec: Fitted model specification from fit.fit_logit().

    Returns:
        Path to the written JSON file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHOSEN_MODEL_PATH, "w") as f:
        json.dump(spec, f, indent=2)
    logger.info("Chosen model saved to %s", CHOSEN_MODEL_PATH)
    return CHOSEN_MODEL_PATH


def load_chosen_model() -> ModelSpec:
    """Load the chosen model spec from disk.

    Returns:
        ModelSpec dict loaded from results/model_selection/chosen_model.json.

    Raises:
        FileNotFoundError: If chosen_model.json does not exist.
    """
    if not CHOSEN_MODEL_PATH.exists():
        raise FileNotFoundError(f"{CHOSEN_MODEL_PATH} not found. Run model selection first.")
    with open(CHOSEN_MODEL_PATH) as f:
        return json.load(f)


def select_top_model(leaderboard: pd.DataFrame) -> ModelSpec:
    """Return the top model from the leaderboard as a ModelSpec.

    Note: this function returns the spec for inspection only. Call
    save_chosen_model() explicitly to lock it in.

    Args:
        leaderboard: Ranked leaderboard from evaluate.build_leaderboard().

    Returns:
        ModelSpec for the highest-ranked model.
    """
    row = leaderboard.iloc[0]
    return ModelSpec(
        features=row["features"],
        window=row["window"],
        intercept=row.get("intercept", 0.0),
        coefficients=row.get("coefficients", {}),
        n_obs=int(row.get("n_obs", 0)),
    )


def print_leaderboards(leaderboards: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Print all window × metric leaderboards to stdout for human review.

    Args:
        leaderboards: Output of evaluate.build_window_metric_leaderboards().
    """
    sep = "=" * 76
    thin = "-" * 76

    for window, metrics in leaderboards.items():
        with open(_WINDOWS_CONFIG) as f:
            windows_cfg = yaml.safe_load(f)["windows"]
        win_meta = next((w for w in windows_cfg if w["name"] == window), {})
        year_range = (
            f"{win_meta.get('start_year', '?')}-{win_meta.get('end_year', '?')}" if win_meta else ""
        )
        print(f"\n{sep}")
        print(f"  WINDOW: {window.upper()}  ({year_range})")
        print(sep)

        for metric, df in metrics.items():
            label = _METRIC_LABELS.get(metric, metric).strip()
            print(f"\n  Metric: {label}  (top {len(df)})")
            print(f"  {thin}")
            print(f"  {'Rank':<5}  {'#Feat':<5}  {'Score':>9}  Features")
            print(f"  {thin}")
            for rank, row in df.iterrows():
                feat_str = ", ".join(row["features"])
                score = row[metric]
                n_feat = int(row["n_features"])
                print(f"  {rank:<5}  {n_feat:<5}  {score:>9.4f}  {feat_str}")
        print()


def _validate_pipeline(
    df: pd.DataFrame,
    active: list[str],
    candidate_sets: list,
    windows: list[dict],
    min_size: int,
    max_size: int,
    raw_combos: int,
) -> None:
    """Validate stop conditions before committing to a long combinatorial fit.

    Raises RuntimeError if the combination count exceeds _MAX_COMBINATIONS or
    any training window has fewer than _MIN_WINDOW_SERIES series.

    Args:
        df: Full series-level dataset.
        active: Usable active features (already filtered to those in df).
        candidate_sets: Filtered combination list (forbidden pairs removed).
        windows: List of window config dicts from training_windows.yaml.
        min_size: Minimum feature combination size.
        max_size: Maximum feature combination size.
        raw_combos: Unfiltered combination count (before forbidden-pair removal).

    Raises:
        RuntimeError: If any stop condition is triggered.
    """
    total_combos = len(candidate_sets)
    excluded = raw_combos - total_combos
    n = len(active)

    print(
        f"\n[OK] Step 2 -- {total_combos:,} valid combinations generated "
        f"(sizes {min_size}-{max_size}, {excluded:,} excluded by forbidden-pair rules)."
    )
    if total_combos > _MAX_COMBINATIONS:
        raise RuntimeError(
            f"\n[STOP] {total_combos:,} valid combinations exceeds the limit of "
            f"{_MAX_COMBINATIONS:,}.\n"
            f"   Active features: {n}  |  Size range: {min_size}-{max_size}  |  "
            f"Excluded by forbidden pairs: {excluded:,}\n"
            "   Resolution options:\n"
            "     - Reduce max_size\n"
            "     - Deactivate features in configs/features.yaml\n"
            "     - Add forbidden pairs in configs/model_selection.yaml\n"
            "     - Raise _MAX_COMBINATIONS if the large search space is intentional"
        )

    for w in windows:
        mask = (df["year"] >= w["start_year"]) & (df["year"] <= w["end_year"])
        n_series = int(mask.sum())
        if n_series < _MIN_WINDOW_SERIES:
            raise RuntimeError(
                f"\n[STOP] Window '{w['name']}' "
                f"({w['start_year']}-{w['end_year']}) has only {n_series} series "
                f"(minimum required: {_MIN_WINDOW_SERIES})."
            )
        logger.info(
            "Window '%s' (%d–%d): %d series.", w["name"], w["start_year"], w["end_year"], n_series
        )


def _fit_all_models(
    df: pd.DataFrame,
    candidate_sets: list,
    windows: list[dict],
) -> tuple[list[dict], int, int]:
    """Fit every combination × window and return results with failure counts.

    Args:
        df: Full series-level dataset.
        candidate_sets: List of feature-name tuples to fit.
        windows: List of window config dicts from training_windows.yaml.

    Returns:
        Tuple of (metric_rows, convergence_failures, fit_attempts) where
        metric_rows is a list of evaluate_model() result dicts.

    Raises:
        RuntimeError: If the convergence failure rate exceeds
            _MAX_CONVERGENCE_FAILURE_RATE.
    """
    total_fits = len(candidate_sets) * len(windows)
    logger.info(
        "Fitting %d models (%d combos × %d windows)…",
        total_fits,
        len(candidate_sets),
        len(windows),
    )
    metric_rows: list[dict] = []
    convergence_failures = 0
    fit_attempts = 0

    for features in candidate_sets:
        for w in windows:
            fit_attempts += 1
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    spec = fit_logit(df, features, w["name"], w["start_year"], w["end_year"])
                    if any(issubclass(c.category, ConvergenceWarning) for c in caught):
                        convergence_failures += 1
                        logger.debug(
                            "Convergence warning: features=%s window=%s",
                            features,
                            w["name"],
                        )
                    window_df = df[
                        (df["year"] >= w["start_year"]) & (df["year"] <= w["end_year"])
                    ]
                    metric_rows.append(evaluate_model(spec, window_df))
                except (ValueError, np.linalg.LinAlgError) as exc:
                    convergence_failures += 1
                    logger.warning("Skipped features=%s window=%s: %s", features, w["name"], exc)

    failure_rate = convergence_failures / fit_attempts if fit_attempts else 0.0
    if failure_rate > _MAX_CONVERGENCE_FAILURE_RATE:
        raise RuntimeError(
            f"\n[STOP] Convergence failure rate is "
            f"{failure_rate:.1%} ({convergence_failures}/{fit_attempts}), "
            f"exceeding the {_MAX_CONVERGENCE_FAILURE_RATE:.0%} threshold.\n"
            "   Inspect for near-collinear features or sparse training windows."
        )
    logger.info(
        "Fitting complete: %d successful, %d skipped/failed (%.1f%%).",
        len(metric_rows),
        convergence_failures,
        failure_rate * 100,
    )
    return metric_rows, convergence_failures, fit_attempts


def _save_and_print_leaderboards(
    metric_rows: list[dict],
    active: list[str],
    min_size: int,
    max_size: int,
    top_n: int,
    windows: list[dict],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Persist all model results, build leaderboards, print, and save to text.

    Args:
        metric_rows: List of evaluate_model() result dicts.
        active: Active feature list (used to build the run tag).
        min_size: Minimum combination size (used for run tag).
        max_size: Maximum combination size (used for run tag).
        top_n: Number of top models per leaderboard to retain.
        windows: List of window config dicts (used for count in summary line).

    Returns:
        Nested dict {window_name: {metric_name: DataFrame}} of top-N models.

    Raises:
        RuntimeError: If metric_rows is empty.
    """
    if not metric_rows:
        raise RuntimeError("No models were successfully fitted — cannot build leaderboards.")

    run_tag = f"{len(active)}feat_size{min_size}-{max_size}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_models_path = RESULTS_DIR / f"all_models_{run_tag}.parquet"
    pd.DataFrame(metric_rows).to_parquet(all_models_path, index=False)
    logger.info("All model results saved to %s", all_models_path)
    print(f"\n[OK] Step 4 -- All model results saved to {all_models_path}")

    leaderboards = build_window_metric_leaderboards(metric_rows, top_n=top_n)

    buf = io.StringIO()
    _old_stdout = sys.stdout
    sys.stdout = buf
    print_leaderboards(leaderboards)
    sys.stdout = _old_stdout
    leaderboard_text = buf.getvalue()
    print(leaderboard_text, end="")

    leaderboards_path = RESULTS_DIR / f"leaderboards_{run_tag}.txt"
    leaderboards_path.write_text(leaderboard_text, encoding="utf-8")
    logger.info("Leaderboards saved to %s", leaderboards_path)
    print(
        f"[OK] Step 5 -- {len(windows)} windows x 4 metrics = "
        f"{len(windows) * 4} leaderboards printed above and saved to {leaderboards_path}. "
        "No model has been auto-selected."
    )
    return leaderboards


def run_combinatorial_pipeline(
    df: pd.DataFrame,
    min_size: int = _COMBO_MIN_SIZE,
    max_size: int = _COMBO_MAX_SIZE,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run the full combinatorial logit model selection pipeline.

    Fits every active-feature combination of size ``min_size``–``max_size``
    across all training windows, then builds and prints 12 ranked leaderboards
    (3 windows × 4 metrics, top 5 per leaderboard).  Does NOT auto-select or
    save any model — output is for human review only.

    Stop conditions (raise RuntimeError before fitting begins):
    - Total combinations > 10,000
    - Any training window has fewer than 30 series

    Stop condition (raise RuntimeError after fitting):
    - Convergence failure rate > 20 %

    Args:
        df: Full series-level dataset (output of the data pipeline).
        min_size: Minimum feature combination size (default 2).
        max_size: Maximum feature combination size (default 5).

    Returns:
        Nested dict ``{window_name: {metric_name: DataFrame}}`` containing
        the top-5 models per leaderboard, for further inspection.

    Raises:
        RuntimeError: If any stop condition is triggered.
    """
    with open(_WINDOWS_CONFIG) as f:
        windows = yaml.safe_load(f)["windows"]
    with open(_MODEL_SELECTION_CONFIG) as f:
        ms_cfg = yaml.safe_load(f)
    top_n: int = ms_cfg["top_n_models"]

    # ── Step 1: Confirm active feature list ───────────────────────────────────
    active_cfg = get_active_features()
    missing = [f for f in active_cfg if f not in df.columns]
    if missing:
        logger.warning(
            "Excluding %d active feature(s) not present in the dataset "
            "(producing module may not have run yet): %s",
            len(missing),
            missing,
        )
        print(
            f"\n[WARN] {len(missing)} active feature(s) absent from dataset "
            f"(skipped): {missing}"
        )
    active = [f for f in active_cfg if f in df.columns]
    print(f"\n[OK] Step 1 -- Active features ({len(active)} usable):")
    for feat in active:
        print(f"   - {feat}")

    # ── Step 2: Generate combinations, apply forbidden-pair filter, stop check ──
    n = len(active)
    raw_combos = sum(comb(n, k) for k in range(min_size, min(max_size, n) + 1))
    logger.info("Raw combination count: C(%d, %d..%d) = %d", n, min_size, max_size, raw_combos)
    candidate_sets = generate_all_subsets(active, max_size=max_size, min_size=min_size)
    forbidden_pairs = get_forbidden_pairs()
    candidate_sets = filter_forbidden_pairs(candidate_sets, forbidden_pairs)
    logger.info(
        "After forbidden-pair filtering: %d valid combinations (%d excluded).",
        len(candidate_sets),
        raw_combos - len(candidate_sets),
    )
    _validate_pipeline(df, active, candidate_sets, windows, min_size, max_size, raw_combos)

    # ── Step 3: Fit all combinations × windows ────────────────────────────────
    metric_rows, _, _ = _fit_all_models(df, candidate_sets, windows)

    # ── Steps 4 & 5: Build and print leaderboards ─────────────────────────────
    return _save_and_print_leaderboards(metric_rows, active, min_size, max_size, top_n, windows)
