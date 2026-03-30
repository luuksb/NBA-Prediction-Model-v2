"""assemble.py — Orchestrate the data pipeline and build the final series dataset.

Pipeline stages:
  1. load_base_series()  — reads all playoff_series/*.csv into one base DataFrame
  2. run_all_steps()     — runs every feature-engineering step, saves intermediates
  3. _join_intermediates() — merges all intermediate parquets on (season, series_id)
  4. compute_deltas()    — converts *_high / *_low column pairs → delta_* columns
  5. save_final_dataset() — writes data/final/series_dataset.parquet

Final dataset columns:
  series_id, year (= season), higher_seed_wins,
  team_high, team_low, round, conference,
  and all delta_* feature columns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PLAYOFF_SERIES_DIR = RAW_DIR / "playoff_series"
INTERMEDIATE_DIR = Path("data/intermediate")
FINAL_DIR = Path("data/final")
FEATURES_CONFIG = Path("configs/features.yaml")

FIRST_SEASON = 1980
LAST_SEASON = 2024


# ── Base series ────────────────────────────────────────────────────────────────

def load_base_series(
    raw_dir: Path = RAW_DIR,
    first_season: int = FIRST_SEASON,
    last_season: int = LAST_SEASON,
) -> pd.DataFrame:
    """Load all playoff series CSVs and return a base DataFrame.

    Each row represents one historical playoff series. The series_id is already
    in YYYY_TM1_TM2 format where TM1 is the higher seed.

    Args:
        raw_dir: Root raw data directory.
        first_season: Earliest season to include (inclusive).
        last_season: Latest season to include (inclusive).

    Returns:
        DataFrame with columns:
          series_id, season, team_high, team_low,
          higher_seed_wins, round, conference, seed_high, seed_low.
    """
    series_dir = raw_dir / "playoff_series"
    frames: list[pd.DataFrame] = []
    for path in sorted(series_dir.glob("*_nba_api.csv")):
        year = int(path.stem.split("_")[0])
        if year < first_season or year > last_season:
            continue
        df = pd.read_csv(
            path,
            usecols=[
                "season", "series_id", "round", "conference",
                "team_high", "team_low", "seed_high", "seed_low",
                "higher_seed_wins",
            ],
        )
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No playoff series CSVs found in {series_dir} "
            f"for seasons {first_season}–{last_season}."
        )

    base = pd.concat(frames, ignore_index=True)

    # Validate: series_id should be unique (one row per series)
    n_dupes = base.duplicated(subset="series_id").sum()
    if n_dupes > 0:
        logger.warning(
            "load_base_series: %d duplicate series_id values found — "
            "keeping first occurrence.",
            n_dupes,
        )
        base = base.drop_duplicates(subset="series_id", keep="first")

    # Validate: higher_seed_wins must be 0 or 1
    invalid = ~base["higher_seed_wins"].isin([0, 1])
    if invalid.any():
        logger.warning(
            "load_base_series: %d rows with unexpected higher_seed_wins values.",
            invalid.sum(),
        )

    logger.info(
        "load_base_series: %d series loaded (%d–%d), %d seasons.",
        len(base),
        int(base["season"].min()),
        int(base["season"].max()),
        int(base["season"].nunique()),
    )
    return base.reset_index(drop=True)


# ── Step runner ────────────────────────────────────────────────────────────────

def _save_intermediate(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved intermediate: %s  (%d rows, %d cols)", path, len(df), len(df.columns))


def run_all_steps(
    base: pd.DataFrame,
    intermediate_dir: Path = INTERMEDIATE_DIR,
) -> None:
    """Run all feature-engineering steps and save outputs to intermediate_dir.

    Steps run in order:
      1. team_ratings     — delta team efficiency stats
      2. playoff_experience — cumulative series wins / experience
      3. player_availability — top-3 GP% delta
      4. player_ratings   — star BPM / PER / USG + availability weights
      5. coach_experience — coach series win %

    Each step receives the base series DataFrame and returns it augmented with
    new columns. Only the keys (series_id, season) plus the step's new columns
    are saved to the intermediate parquet.

    Args:
        base: Output of load_base_series().
        intermediate_dir: Directory to write per-step parquet files.
    """
    # Import steps lazily so this module stays importable without all deps
    from src.data.steps import (  # noqa: F401
        team_ratings,
        playoff_experience,
        player_availability,
        player_ratings,
        coach_experience,
        home_court,
    )

    key_cols = ["season", "series_id"]

    steps = [
        ("team_ratings", team_ratings.run),
        ("playoff_experience", playoff_experience.run),
        ("player_availability", player_availability.run),
        ("player_ratings", player_ratings.run),
        ("coach_experience", coach_experience.run),
        ("home_court", home_court.run),
    ]

    for step_name, step_fn in steps:
        logger.info("Running step: %s", step_name)
        augmented = step_fn(base)

        # Identify new columns added by this step
        new_cols = [c for c in augmented.columns if c not in base.columns]
        if not new_cols:
            logger.warning("Step %s added no new columns — skipping save.", step_name)
            continue

        intermediate = augmented[key_cols + new_cols].copy()
        _save_intermediate(intermediate, step_name, intermediate_dir)
        logger.info(
            "Step %s: added %d columns (%s…)",
            step_name, len(new_cols), new_cols[:3],
        )


# ── Intermediate joiner ────────────────────────────────────────────────────────

def _join_intermediates(
    base: pd.DataFrame,
    intermediate_dir: Path = INTERMEDIATE_DIR,
) -> pd.DataFrame:
    """Join all intermediate parquets onto the base series DataFrame.

    Args:
        base: Base series DataFrame (from load_base_series).
        intermediate_dir: Directory containing per-step parquet files.

    Returns:
        Merged DataFrame with base metadata + all step features.
    """
    parquet_files = sorted(intermediate_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No intermediate parquets found in {intermediate_dir}. "
            "Run run_all_steps() first."
        )

    join_keys = ["season", "series_id"]
    result = base.copy()

    for path in parquet_files:
        logger.info("Joining intermediate: %s", path.name)
        step_df = pd.read_parquet(path)
        result = result.merge(step_df, on=join_keys, how="left", suffixes=("", "_dup"))
        dup_cols = [c for c in result.columns if c.endswith("_dup")]
        if dup_cols:
            logger.warning("Duplicate columns dropped after joining %s: %s", path.name, dup_cols)
            result = result.drop(columns=dup_cols)

    # Validate row count
    if len(result) != len(base):
        raise RuntimeError(
            f"STOP CONDITION: merge produced {len(result)} rows but base has "
            f"{len(base)} rows. Investigate the join keys."
        )

    logger.info(
        "_join_intermediates: %d rows, %d columns after all joins.",
        len(result), len(result.columns),
    )
    return result


# ── Delta computation ──────────────────────────────────────────────────────────

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all *_high / *_low column pairs into delta_* columns.

    For every column named X_high that has a corresponding X_low, computes:
        delta_X = X_high − X_low

    Columns that already start with 'delta_' are left as-is.
    After computing all deltas, the original *_high and *_low columns are dropped.

    Special case: top3_gp_pct_delta is already a delta and kept unchanged.

    Args:
        df: DataFrame containing *_high and *_low column pairs.

    Returns:
        df with delta_* columns replacing the *_high / *_low columns.
    """
    df = df.copy()
    high_cols = [c for c in df.columns if c.endswith("_high")]
    paired_bases = [c[: -len("_high")] for c in high_cols if c[: -len("_high")] + "_low" in df.columns]

    cols_to_drop: list[str] = []
    for base_name in paired_bases:
        high_col = f"{base_name}_high"
        low_col = f"{base_name}_low"
        delta_col = f"delta_{base_name}"

        if delta_col not in df.columns:
            h = pd.to_numeric(df[high_col], errors="coerce")
            l = pd.to_numeric(df[low_col], errors="coerce")
            df[delta_col] = h - l

        cols_to_drop.extend([high_col, low_col])

    df = df.drop(columns=cols_to_drop, errors="ignore")

    n_deltas = sum(1 for c in df.columns if c.startswith("delta_") or c == "top3_gp_pct_delta")
    logger.info(
        "compute_deltas: %d *_high/*_low pairs converted; %d delta features total.",
        len(paired_bases), n_deltas,
    )
    return df


# ── Public assembly entrypoint ────────────────────────────────────────────────

def assemble_dataset(
    raw_dir: Path = RAW_DIR,
    intermediate_dir: Path = INTERMEDIATE_DIR,
) -> pd.DataFrame:
    """Run the full data pipeline and return the final series-level DataFrame.

    Steps:
      1. Load base series data from playoff_series/*.csv.
      2. Run all feature-engineering steps and save to intermediate/.
      3. Join all intermediates onto base.
      4. Compute delta_* features from *_high / *_low pairs.
      5. Rename season → year; retain metadata + delta features only.

    Returns:
        DataFrame with one row per historical playoff series. Columns:
          series_id, year, higher_seed_wins, team_high, team_low,
          round, conference, seed_high, seed_low,
          and all delta_* / top3_gp_pct_delta feature columns.
    """
    base = load_base_series(raw_dir=raw_dir)
    run_all_steps(base, intermediate_dir=intermediate_dir)
    build_team_season_features(base, intermediate_dir=intermediate_dir)
    joined = _join_intermediates(base, intermediate_dir=intermediate_dir)
    final = compute_deltas(joined)

    # Rename season → year (data contract)
    final = final.rename(columns={"season": "year"})

    # Drop any residual non-feature numeric columns that aren't metadata
    # Drop any delta columns that are entirely NaN — these are artefacts from
    # non-numeric metadata columns (e.g. delta_team from team string subtraction)
    # or columns never populated in the raw data (e.g. delta_seed when seeds are NaN).
    all_nan_cols = [c for c in final.columns if c.startswith("delta_") and final[c].isna().all()]
    if all_nan_cols:
        logger.warning(
            "Dropping %d all-NaN delta columns: %s", len(all_nan_cols), all_nan_cols
        )
        final = final.drop(columns=all_nan_cols)

    keep_meta = {"series_id", "year", "higher_seed_wins", "team_high", "team_low",
                 "round", "conference"}
    feature_cols = [c for c in final.columns if c not in keep_meta]
    final = final[sorted(keep_meta & set(final.columns)) + sorted(feature_cols)]

    logger.info(
        "assemble_dataset: final shape %d rows × %d cols  "
        "(%d feature columns).",
        len(final), len(final.columns), len(feature_cols),
    )
    return final


def save_final_dataset(df: pd.DataFrame, output_dir: Path = FINAL_DIR) -> Path:
    """Persist the assembled dataset to data/final/series_dataset.parquet and .xlsx.

    Args:
        df: Assembled series-level DataFrame.
        output_dir: Target directory (created if absent).

    Returns:
        Path to the written parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "series_dataset.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved final dataset to %s  (%d rows, %d cols)", out_path, len(df), len(df.columns))

    xlsx_path = output_dir / "series_dataset.xlsx"
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    logger.info("Saved final dataset to %s", xlsx_path)

    return out_path


def build_team_season_features(
    base: pd.DataFrame,
    intermediate_dir: Path = INTERMEDIATE_DIR,
    output_dir: Path = FINAL_DIR,
) -> pd.DataFrame:
    """Build a per-team feature table for use by the bracket simulation module.

    Produces one row per (year, team) with raw (non-delta) values for all
    active features that have per-team source data. The simulation module reads
    this file and computes deltas on-the-fly for any matchup:
        delta_X = features[high_seed]['X'] - features[low_seed]['X']

    Feature sources:
    - team_ratings features: obtained via team_ratings.build_team_stats(), one
      row per (season, team) from the raw CSVs.
    - player_ratings, playoff_experience, coach_experience: pivoted from their
      intermediates' *_high / *_low columns using Round 1 series appearances.
    - player_availability (top3_gp_pct_delta): skipped — already a delta with
      no per-team raw values stored in its intermediate.

    Args:
        base: Output of load_base_series() — provides season, series_id,
            team_high, team_low, and round for the pivot step.
        intermediate_dir: Directory containing per-step parquet files.
        output_dir: Target directory (created if absent).

    Returns:
        DataFrame with columns: year, team, <all extractable active features>.
        Also saves to output_dir/team_season_features.parquet.
    """
    from src.data.steps import team_ratings as tr_mod

    with open(FEATURES_CONFIG) as f:
        config = yaml.safe_load(f)

    # Build mapping: raw_feature_name -> producing_step
    step_to_raw_features: dict[str, list[str]] = {}
    for feat in config["features"]:
        if not feat.get("active", True):
            continue
        step = feat.get("producing_step", "")
        if not step.startswith("steps/"):
            continue  # skip injury_module features
        step_name = step.replace("steps/", "")
        if feat.get("series_level", False):
            continue  # already a delta, no per-team raw values to pivot
        raw_name = feat["name"].removeprefix("delta_")
        step_to_raw_features.setdefault(step_name, []).append(raw_name)

    seasons = sorted(base["season"].unique().tolist())

    # ── 1. Team ratings: call build_team_stats() directly ─────────────────────
    tr_raw_features = step_to_raw_features.get("team_ratings", [])
    team_stats = tr_mod.build_team_stats(seasons)
    available_tr = [f for f in tr_raw_features if f in team_stats.columns]
    if len(available_tr) < len(tr_raw_features):
        missing = set(tr_raw_features) - set(available_tr)
        logger.warning(
            "build_team_season_features: team_ratings columns not found: %s", missing
        )
    result = team_stats[["season", "team"] + available_tr].copy()

    # ── 2. Intermediates with _high/_low columns ───────────────────────────────
    # Use Round 1 rows so each team appears exactly once per season.
    r1 = base[base["round"] == "first_round"][["series_id", "season", "team_high", "team_low"]].copy()

    intermediates_to_pivot = {
        "player_ratings": step_to_raw_features.get("player_ratings", []),
        "playoff_experience": step_to_raw_features.get("playoff_experience", []),
        "coach_experience": step_to_raw_features.get("coach_experience", []),
    }

    for step_name, raw_features in intermediates_to_pivot.items():
        if not raw_features:
            continue
        path = intermediate_dir / f"{step_name}.parquet"
        if not path.exists():
            logger.warning(
                "build_team_season_features: intermediate not found: %s — skipping.", path
            )
            continue

        step_df = pd.read_parquet(path)
        merged = r1.merge(step_df, on=["season", "series_id"], how="left")

        # Determine which raw features have _high/_low in this intermediate
        found_raw = [f for f in raw_features if f"{f}_high" in step_df.columns]
        if not found_raw:
            logger.warning(
                "build_team_season_features: no *_high/*_low columns found for %s in %s",
                raw_features, step_name,
            )
            continue

        high_cols = [f"{f}_high" for f in found_raw]
        low_cols = [f"{f}_low" for f in found_raw]
        rename_high = {f"{f}_high": f for f in found_raw}
        rename_low = {f"{f}_low": f for f in found_raw}

        high_df = (
            merged[["season", "team_high"] + high_cols]
            .rename(columns={"team_high": "team", **rename_high})
        )
        low_df = (
            merged[["season", "team_low"] + low_cols]
            .rename(columns={"team_low": "team", **rename_low})
        )

        combined = (
            pd.concat([high_df, low_df], ignore_index=True)
            .drop_duplicates(subset=["season", "team"], keep="first")
        )

        result = result.merge(combined, on=["season", "team"], how="left")

    result = result.rename(columns={"season": "year"})

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "team_season_features.parquet"
    result.to_parquet(out_path, index=False)
    logger.info(
        "build_team_season_features: %d rows × %d cols saved to %s",
        len(result), len(result.columns), out_path,
    )
    return result


def load_active_features() -> list[str]:
    """Return active feature names from configs/features.yaml.

    Returns:
        Sorted list of active feature name strings.
    """
    with open(FEATURES_CONFIG) as f:
        config = yaml.safe_load(f)
    return [feat["name"] for feat in config["features"] if feat.get("active", True)]
