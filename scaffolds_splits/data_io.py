"""Input loading and label preparation for scaffold split pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scaffolds_splits.monotonic import MonotonicFilterConfig, filter_monotonic_profiles

REQUIRED_COLUMNS = ("chembl_id", "canonical_smiles")


def _ensure_required_columns(df: pd.DataFrame, dataset_name: str) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: missing required columns: {missing}. "
            f"Required={list(REQUIRED_COLUMNS)}"
        )


def _compute_pchembl_if_needed(df: pd.DataFrame) -> pd.Series:
    if "pchembl_value" in df.columns:
        pchembl = pd.to_numeric(df["pchembl_value"], errors="coerce")
    else:
        pchembl = pd.Series(np.nan, index=df.index, dtype="float64")

    if pchembl.isna().any() and "standard_value" in df.columns:
        std = pd.to_numeric(df["standard_value"], errors="coerce")
        recover_mask = pchembl.isna() & std.gt(0)
        if recover_mask.any():
            pchembl.loc[recover_mask] = 9.0 - np.log10(std.loc[recover_mask])

    return pchembl


def _ensure_binary_label(
    df: pd.DataFrame,
    threshold_pchembl: float,
    dataset_name: str,
) -> pd.DataFrame:
    out = df.copy()

    if "label" in out.columns:
        label = pd.to_numeric(out["label"], errors="coerce")
        valid = label.isin([0, 1])
        if valid.any():
            out = out.loc[valid].copy()
            out["label"] = label.loc[valid].astype(np.int8)
            # Keep pchembl for transparency if available/recoverable.
            out["pchembl_value"] = _compute_pchembl_if_needed(out)
            return out

    pchembl = _compute_pchembl_if_needed(out)
    valid = pchembl.notna()
    dropped = int((~valid).sum())
    if dropped > 0:
        print(
            f"[{dataset_name}] dropping {dropped} rows without valid pChEMBL/standard_value "
            "for label construction"
        )

    out = out.loc[valid].copy()
    out["pchembl_value"] = pchembl.loc[valid].astype(float)
    out["label"] = (out["pchembl_value"] >= threshold_pchembl).astype(np.int8)
    return out


def load_dataset(
    path: str,
    dataset_name: str,
    threshold_pchembl: float = 6.0,
    max_rows: Optional[int] = None,
    remove_monotonic_kinases: bool = True,
    remove_monotonic_compounds: bool = True,
) -> pd.DataFrame:
    """Load TSV dataset and ensure `label` exists as binary target."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{dataset_name}: dataset file not found: {file_path}")

    print(f"[{dataset_name}] loading: {file_path}")
    df = pd.read_csv(file_path, sep="\t", nrows=max_rows, low_memory=False)
    _ensure_required_columns(df, dataset_name)

    # Minimal row-level sanity filters.
    df = df.dropna(subset=["chembl_id", "canonical_smiles"]).copy()
    df["chembl_id"] = df["chembl_id"].astype(str)
    df["canonical_smiles"] = df["canonical_smiles"].astype(str)

    df = _ensure_binary_label(df, threshold_pchembl=threshold_pchembl, dataset_name=dataset_name)

    mono_cfg = MonotonicFilterConfig(
        remove_monotonic_kinases=remove_monotonic_kinases,
        remove_monotonic_compounds=remove_monotonic_compounds,
    )
    df, mono_report = filter_monotonic_profiles(df, config=mono_cfg)
    if mono_report["removed_rows_total"] > 0:
        print(
            f"[{dataset_name}] monotonic filter removed {mono_report['removed_rows_total']:,} rows "
            f"(kinases={mono_report['removed_monotonic_kinases']:,}, "
            f"compounds={mono_report['removed_monotonic_compounds']:,})"
        )
    else:
        print(f"[{dataset_name}] monotonic filter removed 0 rows")

    if df.empty:
        raise ValueError(f"{dataset_name}: no rows left after preprocessing")

    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    print(
        f"[{dataset_name}] rows={len(df):,} compounds={df['chembl_id'].nunique():,} "
        f"labels: pos={pos:,} neg={neg:,}"
    )
    return df.reset_index(drop=True)
