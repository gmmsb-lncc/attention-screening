"""Monotonic-profile filtering utilities.

A profile is considered monotonic when all labels are the same:
- kinase monotonic: all rows for a kinase are 0 or 1
- compound monotonic: for compounds tested against >1 kinase, all rows are 0 or 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class MonotonicFilterConfig:
    remove_monotonic_kinases: bool = True
    remove_monotonic_compounds: bool = True
    min_kinases_for_compound: int = 2
    eps: float = 1e-12


def _is_monotonic_rate(rate: pd.Series, eps: float) -> pd.Series:
    return (rate <= eps) | (rate >= (1.0 - eps))


def filter_monotonic_profiles(
    df: pd.DataFrame,
    config: MonotonicFilterConfig,
    kinase_col: str = "target_kinase",
    compound_col: str = "chembl_id",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove monotonic kinases/compounds and return filtering report."""
    out = df.copy()

    report: Dict[str, int] = {
        "rows_before": int(len(out)),
        "rows_after": 0,
        "removed_rows_total": 0,
        "removed_monotonic_kinases": 0,
        "removed_rows_monotonic_kinases": 0,
        "removed_monotonic_compounds": 0,
        "removed_rows_monotonic_compounds": 0,
        "removed_pan_active_compounds": 0,
        "removed_pan_inactive_compounds": 0,
    }

    if config.remove_monotonic_kinases:
        if kinase_col not in out.columns:
            raise ValueError(
                f"Monotonic kinase filter requires column '{kinase_col}', but it is missing"
            )

        rates = out.groupby(kinase_col, dropna=False)[label_col].mean()
        mono_kinases = set(rates[_is_monotonic_rate(rates, config.eps)].index)
        if mono_kinases:
            n_before = len(out)
            out = out[~out[kinase_col].isin(mono_kinases)].copy()
            report["removed_monotonic_kinases"] = int(len(mono_kinases))
            report["removed_rows_monotonic_kinases"] = int(n_before - len(out))

    if config.remove_monotonic_compounds:
        if kinase_col not in out.columns:
            raise ValueError(
                f"Monotonic compound filter requires column '{kinase_col}', but it is missing"
            )

        kin_counts = out.groupby(compound_col, dropna=False)[kinase_col].nunique()
        eligible = set(kin_counts[kin_counts >= config.min_kinases_for_compound].index)

        if eligible:
            subset = out[out[compound_col].isin(eligible)]
            rates = subset.groupby(compound_col, dropna=False)[label_col].mean()
            mono_compounds = set(rates[_is_monotonic_rate(rates, config.eps)].index)

            if mono_compounds:
                pan_active = int((rates.loc[list(mono_compounds)] >= (1.0 - config.eps)).sum())
                pan_inactive = int(len(mono_compounds) - pan_active)

                n_before = len(out)
                out = out[~out[compound_col].isin(mono_compounds)].copy()

                report["removed_monotonic_compounds"] = int(len(mono_compounds))
                report["removed_rows_monotonic_compounds"] = int(n_before - len(out))
                report["removed_pan_active_compounds"] = pan_active
                report["removed_pan_inactive_compounds"] = pan_inactive

    report["rows_after"] = int(len(out))
    report["removed_rows_total"] = int(report["rows_before"] - report["rows_after"])
    return out.reset_index(drop=True), report
