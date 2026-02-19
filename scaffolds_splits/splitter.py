"""Build train/val/test splits using scaffold-level subset selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationSelectionConfig:
    target_val_fraction: float = 0.10
    seed: int = 42
    restarts: int = 48
    class_penalty: float = 5.0


def _state_loss(
    val_compounds: float,
    val_pos: float,
    val_neg: float,
    total_compounds: float,
    target_val_fraction: float,
    remaining_pos: float,
    remaining_neg: float,
    class_penalty: float,
) -> float:
    val_fraction = val_compounds / max(total_compounds, 1e-12)
    err = abs(val_fraction - target_val_fraction)

    train_pos = remaining_pos - val_pos
    train_neg = remaining_neg - val_neg

    missing = (
        int(val_pos <= 0)
        + int(val_neg <= 0)
        + int(train_pos <= 0)
        + int(train_neg <= 0)
    )
    return float(err + class_penalty * missing)


def select_validation_scaffolds(
    stats_df: pd.DataFrame,
    test_scaffolds: Set[str],
    total_unique_compounds: int,
    config: ValidationSelectionConfig,
    excluded_scaffolds: Set[str] | None = None,
) -> List[str]:
    """Select validation scaffolds from remaining scaffolds for one dataset."""
    excluded = set(test_scaffolds)
    if excluded_scaffolds:
        excluded |= set(excluded_scaffolds)
    remaining = stats_df[~stats_df["scaffold"].isin(excluded)].copy()
    if remaining.empty:
        raise RuntimeError("No scaffolds left after test allocation; cannot build train/val")

    scaffolds = remaining["scaffold"].to_numpy()
    n_comp = remaining["unique_compounds"].to_numpy(dtype=float)
    n_pos = remaining["rows_pos"].to_numpy(dtype=float)
    n_neg = remaining["rows_neg"].to_numpy(dtype=float)

    rem_pos_total = float(n_pos.sum())
    rem_neg_total = float(n_neg.sum())
    target_compounds = float(config.target_val_fraction * total_unique_compounds)

    best_sel = None
    best_loss = float("inf")

    for restart in range(config.restarts):
        rng = np.random.default_rng(config.seed + 104729 * restart)
        selected = np.zeros(len(scaffolds), dtype=bool)

        val_comp = 0.0
        val_pos = 0.0
        val_neg = 0.0
        current_loss = _state_loss(
            val_comp,
            val_pos,
            val_neg,
            total_unique_compounds,
            config.target_val_fraction,
            rem_pos_total,
            rem_neg_total,
            config.class_penalty,
        )

        # Fast randomized greedy pass: near O(n) per restart.
        # Priority = larger scaffolds first, with tiny jitter for diversification.
        jitter = rng.uniform(0.0, 1e-6, size=len(scaffolds))
        order = np.argsort(-(n_comp + jitter))
        floor_target = 0.98 * target_compounds

        for idx in order:
            new_comp = val_comp + float(n_comp[idx])
            before = abs(val_comp - target_compounds)
            after = abs(new_comp - target_compounds)
            must_fill = val_comp < floor_target
            if must_fill or after <= before:
                selected[idx] = True
                val_comp = new_comp
                val_pos += float(n_pos[idx])
                val_neg += float(n_neg[idx])

        # If still below target, top-up with candidates that best reduce deficit.
        while val_comp < target_compounds and np.any(~selected):
            remaining_idx = np.where(~selected)[0]
            if remaining_idx.size == 0:
                break
            best_add_pos = int(
                np.argmin(np.abs((val_comp + n_comp[remaining_idx]) - target_compounds))
            )
            idx = int(remaining_idx[best_add_pos])
            selected[idx] = True
            val_comp += float(n_comp[idx])
            val_pos += float(n_pos[idx])
            val_neg += float(n_neg[idx])

        # Enforce class coverage on validation set.
        if val_pos <= 0:
            cand = np.where((~selected) & (n_pos > 0))[0]
            if cand.size > 0:
                idx = int(cand[np.argmax(n_pos[cand])])
                selected[idx] = True
                val_comp += float(n_comp[idx])
                val_pos += float(n_pos[idx])
                val_neg += float(n_neg[idx])

        if val_neg <= 0:
            cand = np.where((~selected) & (n_neg > 0))[0]
            if cand.size > 0:
                idx = int(cand[np.argmax(n_neg[cand])])
                selected[idx] = True
                val_comp += float(n_comp[idx])
                val_pos += float(n_pos[idx])
                val_neg += float(n_neg[idx])

        # Lightweight pruning to improve target fit while keeping class validity.
        selected_idx = np.where(selected)[0]
        if selected_idx.size > 0:
            prune_order = selected_idx[np.argsort(n_comp[selected_idx])]
            max_prune = min(len(prune_order), 1500)
            for idx in prune_order[:max_prune]:
                next_pos = val_pos - float(n_pos[idx])
                next_neg = val_neg - float(n_neg[idx])
                next_train_pos = rem_pos_total - next_pos
                next_train_neg = rem_neg_total - next_neg
                if next_pos <= 0 or next_neg <= 0 or next_train_pos <= 0 or next_train_neg <= 0:
                    continue
                next_comp = val_comp - float(n_comp[idx])
                if abs(next_comp - target_compounds) <= abs(val_comp - target_compounds):
                    selected[idx] = False
                    val_comp = next_comp
                    val_pos = next_pos
                    val_neg = next_neg

        current_loss = _state_loss(
            val_comp,
            val_pos,
            val_neg,
            total_unique_compounds,
            config.target_val_fraction,
            rem_pos_total,
            rem_neg_total,
            config.class_penalty,
        )

        if current_loss < best_loss:
            best_loss = current_loss
            best_sel = selected.copy()

    if best_sel is None:
        raise RuntimeError("Validation scaffold selection failed")

    selected_scaffolds = scaffolds[best_sel].tolist()
    selected_scaffolds.sort()
    return selected_scaffolds


def select_test_scaffolds(
    stats_df: pd.DataFrame,
    total_unique_compounds: int,
    config: ValidationSelectionConfig,
    excluded_scaffolds: Set[str] | None = None,
) -> List[str]:
    """Select test scaffolds for a dataset (dataset-specific target fraction)."""
    excluded = excluded_scaffolds or set()
    return select_validation_scaffolds(
        stats_df=stats_df,
        test_scaffolds=set(),
        total_unique_compounds=total_unique_compounds,
        config=ValidationSelectionConfig(
            target_val_fraction=config.target_val_fraction,
            seed=config.seed,
            restarts=config.restarts,
            class_penalty=config.class_penalty,
        ),
        excluded_scaffolds=excluded,
    )


def apply_split(
    df_with_scaffold: pd.DataFrame,
    test_scaffolds: Set[str],
    val_scaffolds: Set[str],
) -> Dict[str, pd.DataFrame]:
    """Create train/val/test row splits from scaffold assignments."""
    test_mask = df_with_scaffold["scaffold"].isin(test_scaffolds)
    val_mask = (~test_mask) & df_with_scaffold["scaffold"].isin(val_scaffolds)
    train_mask = (~test_mask) & (~val_mask)

    train_df = df_with_scaffold.loc[train_mask].copy().reset_index(drop=True)
    val_df = df_with_scaffold.loc[val_mask].copy().reset_index(drop=True)
    test_df = df_with_scaffold.loc[test_mask].copy().reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
