"""assemble.py — Join all intermediate step outputs into the final series dataset.

Reads intermediate parquet files produced by steps/ and joins them into
data/final/series_dataset.parquet (one row per historical playoff series).

Columns in the final dataset are registered in configs/features.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

INTERMEDIATE_DIR = Path("data/intermediate")
FINAL_DIR = Path("data/final")
FEATURES_CONFIG = Path("configs/features.yaml")


def load_active_features() -> list[str]:
    """Return the list of feature names that are active in configs/features.yaml.

    Returns:
        Sorted list of active feature name strings.
    """
    with open(FEATURES_CONFIG) as f:
        config = yaml.safe_load(f)
    return [feat["name"] for feat in config["features"] if feat.get("active", True)]


def assemble_dataset(intermediate_dir: Path = INTERMEDIATE_DIR) -> pd.DataFrame:
    """Join all intermediate step outputs into the final series-level DataFrame.

    Each step writes a parquet file to data/intermediate/. This function reads
    all of them and joins on (year, series_id) to produce one row per series.

    Args:
        intermediate_dir: Directory containing per-step parquet outputs.

    Returns:
        DataFrame with one row per playoff series and all feature columns.
    """
    parquet_files = sorted(intermediate_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {intermediate_dir}. "
            "Run the data pipeline steps first."
        )

    dfs = []
    for path in parquet_files:
        logger.info("Loading intermediate file: %s", path)
        dfs.append(pd.read_parquet(path))

    # All intermediate frames must share the join keys.
    join_keys = ["year", "series_id"]
    base, *rest = dfs
    for df in rest:
        base = base.merge(df, on=join_keys, how="outer", suffixes=("", "_dup"))
        dup_cols = [c for c in base.columns if c.endswith("_dup")]
        if dup_cols:
            logger.warning("Duplicate columns detected and dropped: %s", dup_cols)
            base = base.drop(columns=dup_cols)

    return compute_deltas(base)


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Replace paired _high/_low columns with a single _delta column each.

    For every column named ``<base>_high`` that has a matching ``<base>_low``
    column, computes ``<base>_delta = <base>_high - <base>_low`` and drops the
    originals.  Unmatched ``_high`` or ``_low`` columns are left untouched.

    Args:
        df: DataFrame that may contain ``_high``/``_low`` column pairs.

    Returns:
        New DataFrame with delta columns replacing matched pairs.
    """
    high_bases = {c[:-5] for c in df.columns if c.endswith("_high")}
    low_bases = {c[:-4] for c in df.columns if c.endswith("_low")}
    paired_bases = high_bases & low_bases

    if not paired_bases:
        return df

    result = df.copy()
    for base in sorted(paired_bases):
        result[f"{base}_delta"] = result[f"{base}_high"] - result[f"{base}_low"]
        result = result.drop(columns=[f"{base}_high", f"{base}_low"])
        logger.debug("Computed delta column: %s_delta", base)

    return result


def save_final_dataset(df: pd.DataFrame, output_dir: Path = FINAL_DIR) -> Path:
    """Persist the assembled dataset to data/final/series_dataset.parquet.

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
    return out_path
