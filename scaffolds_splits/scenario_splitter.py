"""Scenario-specific train/validation splitting with class-distribution control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

SCENARIO_ORDER = ("S1", "S2", "S3", "S4", "Sc")
SCENARIO_NAMES = {
    "S1": "random",
    "S2": "compound",
    "S3": "kinase",
    "S4": "new_compound_new_kinase",
    "Sc": "scaffold",
}


@dataclass(frozen=True)
class ScenarioSplitConfig:
    val_fraction_in_pool: float
    seed: int = 42
    restarts: int = 64
    s4_restarts: int = 192
    class_penalty: float = 10.0
    class_rate_weight: float = 2.0
    min_val_groups: int = 15
    monotonic_group_penalty: float = 1.0


@dataclass(frozen=True)
class ScenarioSplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    dropped_df: pd.DataFrame
    metrics: Dict[str, float]


def _class_counts(df: pd.DataFrame) -> Tuple[int, int, int]:
    total = int(len(df))
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    return total, pos, neg


def _class_rate(df: pd.DataFrame) -> float:
    n = max(len(df), 1)
    return float((df["label"] == 1).sum() / n)


def _class_missing_penalty(pos: int, neg: int) -> int:
    return int(pos <= 0) + int(neg <= 0)


def _split_loss(
    train_total: int,
    train_pos: int,
    train_neg: int,
    val_total: int,
    val_pos: int,
    val_neg: int,
    target_val_fraction: float,
    pool_pos_rate: float,
    class_penalty: float,
    class_rate_weight: float,
    drop_fraction: float = 0.0,
    monotonic_val_rows: int = 0,
    n_val_groups: int = 0,
    min_val_groups: int = 0,
    monotonic_group_penalty: float = 0.0,
) -> float:
    kept_total = train_total + val_total
    if kept_total <= 0:
        return 1e9

    val_fraction = val_total / kept_total
    err_fraction = abs(val_fraction - target_val_fraction)

    train_rate = train_pos / max(train_total, 1)
    val_rate = val_pos / max(val_total, 1)
    err_rate = abs(train_rate - pool_pos_rate) + abs(val_rate - pool_pos_rate)

    missing = _class_missing_penalty(train_pos, train_neg) + _class_missing_penalty(val_pos, val_neg)

    # Penalize monotonic groups (all-positive or all-negative) in validation.
    mono_frac = monotonic_val_rows / max(val_total, 1)
    mono_pen = monotonic_group_penalty * mono_frac

    # Penalize low diversity (too few groups in validation).
    diversity_pen = 0.0
    if min_val_groups > 0 and n_val_groups < min_val_groups:
        diversity_pen = 0.5 * (min_val_groups - n_val_groups) / max(min_val_groups, 1)

    return float(
        err_fraction
        + class_rate_weight * err_rate
        + class_penalty * missing
        + 0.5 * drop_fraction
        + mono_pen
        + diversity_pen
    )


def _select_validation_groups(
    df: pd.DataFrame,
    group_col: str,
    cfg: ScenarioSplitConfig,
) -> Set[str]:
    grouped = df.groupby(group_col, dropna=False).agg(
        rows_total=("label", "size"),
        rows_pos=("label", lambda x: int((x == 1).sum())),
    )
    grouped["rows_neg"] = grouped["rows_total"] - grouped["rows_pos"]
    grouped = grouped.reset_index()

    groups = grouped[group_col].astype(str).to_numpy()
    n_rows = grouped["rows_total"].to_numpy(dtype=float)
    n_pos = grouped["rows_pos"].to_numpy(dtype=float)
    n_neg = grouped["rows_neg"].to_numpy(dtype=float)

    # Precompute which groups are monotonic (all-positive or all-negative).
    is_monotonic = ((n_pos == 0) | (n_neg == 0)).astype(float)
    mono_rows = is_monotonic * n_rows  # rows in monotonic groups

    total_rows = float(len(df))
    total_pos = float((df["label"] == 1).sum())
    total_neg = total_rows - total_pos
    pool_pos_rate = total_pos / max(total_rows, 1.0)

    target_rows = cfg.val_fraction_in_pool * total_rows
    floor_target = 0.98 * target_rows

    def _compute_loss(sel, v_rows, v_pos, v_neg):
        """Helper to compute loss with all penalty terms."""
        t_rows = total_rows - v_rows
        t_pos = total_pos - v_pos
        t_neg = total_neg - v_neg
        v_mono = float(mono_rows[sel].sum()) if sel.any() else 0.0
        v_ngrp = int(sel.sum())
        return _split_loss(
            train_total=int(t_rows), train_pos=int(t_pos), train_neg=int(t_neg),
            val_total=int(v_rows), val_pos=int(v_pos), val_neg=int(v_neg),
            target_val_fraction=cfg.val_fraction_in_pool,
            pool_pos_rate=pool_pos_rate,
            class_penalty=cfg.class_penalty,
            class_rate_weight=cfg.class_rate_weight,
            monotonic_val_rows=int(v_mono),
            n_val_groups=v_ngrp,
            min_val_groups=cfg.min_val_groups,
            monotonic_group_penalty=cfg.monotonic_group_penalty,
        )

    best_sel = None
    best_loss = float("inf")
    no_improve_count = 0

    for restart in range(cfg.restarts):
        rng = np.random.default_rng(cfg.seed + 3571 * restart)
        selected = np.zeros(len(groups), dtype=bool)

        val_rows = 0.0
        val_pos = 0.0
        val_neg = 0.0

        jitter = rng.uniform(0.0, 1e-6, size=len(groups))
        order = np.argsort(-(n_rows + jitter))

        for idx in order:
            new_rows = val_rows + float(n_rows[idx])
            new_pos = val_pos + float(n_pos[idx])
            new_train_rows = total_rows - new_rows
            new_train_pos = total_pos - new_pos

            before_rate_err = abs((val_pos / max(val_rows, 1.0)) - pool_pos_rate) + abs(
                ((total_pos - val_pos) / max(total_rows - val_rows, 1.0)) - pool_pos_rate
            )
            after_rate_err = abs((new_pos / max(new_rows, 1.0)) - pool_pos_rate) + abs(
                (new_train_pos / max(new_train_rows, 1.0)) - pool_pos_rate
            )
            before = abs(val_rows - target_rows) / max(total_rows, 1.0) + cfg.class_rate_weight * before_rate_err
            after = abs(new_rows - target_rows) / max(total_rows, 1.0) + cfg.class_rate_weight * after_rate_err
            must_fill = val_rows < floor_target
            if must_fill or after <= before:
                selected[idx] = True
                val_rows = new_rows
                val_pos = new_pos
                val_neg += float(n_neg[idx])

        while val_rows < target_rows and np.any(~selected):
            remaining = np.where(~selected)[0]
            if remaining.size == 0:
                break
            cand_rows = val_rows + n_rows[remaining]
            cand_pos = val_pos + n_pos[remaining]
            cand_train_rows = total_rows - cand_rows
            cand_train_pos = total_pos - cand_pos
            cand_rate_err = np.abs((cand_pos / np.maximum(cand_rows, 1.0)) - pool_pos_rate) + np.abs(
                (cand_train_pos / np.maximum(cand_train_rows, 1.0)) - pool_pos_rate
            )
            cand_score = (
                np.abs(cand_rows - target_rows) / max(total_rows, 1.0)
                + cfg.class_rate_weight * cand_rate_err
            )
            best_add = int(np.argmin(cand_score + rng.uniform(0.0, 1e-9, size=remaining.size)))
            idx = int(remaining[best_add])
            selected[idx] = True
            val_rows += float(n_rows[idx])
            val_pos += float(n_pos[idx])
            val_neg += float(n_neg[idx])

        # Ensure both classes in val if feasible.
        # Use SMALLEST group with the needed class to minimize overshoot.
        if val_pos <= 0:
            cand = np.where((~selected) & (n_pos > 0))[0]
            if cand.size > 0:
                idx = int(cand[np.argmin(n_rows[cand])])
                selected[idx] = True
                val_rows += float(n_rows[idx])
                val_pos += float(n_pos[idx])
                val_neg += float(n_neg[idx])

        if val_neg <= 0:
            cand = np.where((~selected) & (n_neg > 0))[0]
            if cand.size > 0:
                idx = int(cand[np.argmin(n_rows[cand])])
                selected[idx] = True
                val_rows += float(n_rows[idx])
                val_pos += float(n_pos[idx])
                val_neg += float(n_neg[idx])

        current_loss = _compute_loss(selected, val_rows, val_pos, val_neg)

        # Lightweight pruning for tighter target while keeping class support.
        selected_idx = np.where(selected)[0]
        if selected_idx.size > 0:
            prune_order = selected_idx[np.argsort(n_rows[selected_idx])]
            max_prune = min(len(prune_order), 1500)
            for idx in prune_order[:max_prune]:
                next_pos = val_pos - float(n_pos[idx])
                next_neg = val_neg - float(n_neg[idx])
                chk_train_pos = total_pos - next_pos
                chk_train_neg = total_neg - next_neg
                if next_pos <= 0 or next_neg <= 0 or chk_train_pos <= 0 or chk_train_neg <= 0:
                    continue
                next_rows = val_rows - float(n_rows[idx])
                sel_tmp = selected.copy()
                sel_tmp[idx] = False
                next_loss = _compute_loss(sel_tmp, next_rows, next_pos, next_neg)
                if next_loss <= current_loss:
                    selected[idx] = False
                    val_rows = next_rows
                    val_pos = next_pos
                    val_neg = next_neg
                    current_loss = next_loss

        loss = _compute_loss(selected, val_rows, val_pos, val_neg)

        if loss < best_loss:
            best_loss = loss
            best_sel = selected.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping: stop if no improvement for 16 consecutive restarts.
        if no_improve_count >= 16 and restart >= 16:
            break

    if best_sel is None:
        raise RuntimeError(f"Failed to select validation groups for column '{group_col}'")

    return set(groups[best_sel].tolist())


def _split_random_stratified(df: pd.DataFrame, cfg: ScenarioSplitConfig) -> ScenarioSplitResult:
    rng = np.random.default_rng(cfg.seed)
    labels = df["label"].to_numpy(dtype=int)
    all_idx = np.arange(len(df))

    val_idx_list: List[int] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0].copy()
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)
        if n_cls <= 1:
            continue
        n_val = int(round(cfg.val_fraction_in_pool * n_cls))
        n_val = max(1, min(n_cls - 1, n_val))
        val_idx_list.extend(cls_idx[:n_val].tolist())

    val_idx = np.array(sorted(set(val_idx_list)), dtype=int)
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[val_idx] = False

    train_df = df.loc[train_mask].copy().reset_index(drop=True)
    val_df = df.loc[val_idx].copy().reset_index(drop=True)
    dropped_df = df.iloc[0:0].copy()

    train_total, train_pos, train_neg = _class_counts(train_df)
    val_total, val_pos, val_neg = _class_counts(val_df)

    metrics = {
        "scenario_val_fraction_target": float(cfg.val_fraction_in_pool),
        "effective_val_fraction": float(val_total / max(train_total + val_total, 1)),
        "dropped_rows": 0.0,
        "loss": float(
            _split_loss(
                train_total,
                train_pos,
                train_neg,
                val_total,
                val_pos,
                val_neg,
                cfg.val_fraction_in_pool,
                _class_rate(df),
                cfg.class_penalty,
                cfg.class_rate_weight,
                drop_fraction=0.0,
            )
        ),
    }
    return ScenarioSplitResult(train_df=train_df, val_df=val_df, dropped_df=dropped_df, metrics=metrics)


def _split_by_group(df: pd.DataFrame, group_col: str, cfg: ScenarioSplitConfig) -> ScenarioSplitResult:
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' required for grouped scenario")

    val_groups = _select_validation_groups(df, group_col=group_col, cfg=cfg)
    val_mask = df[group_col].astype(str).isin(val_groups)

    train_df = df.loc[~val_mask].copy().reset_index(drop=True)
    val_df = df.loc[val_mask].copy().reset_index(drop=True)
    dropped_df = df.iloc[0:0].copy()

    train_total, train_pos, train_neg = _class_counts(train_df)
    val_total, val_pos, val_neg = _class_counts(val_df)

    metrics = {
        "scenario_val_fraction_target": float(cfg.val_fraction_in_pool),
        "effective_val_fraction": float(val_total / max(train_total + val_total, 1)),
        "n_val_groups": float(len(val_groups)),
        "dropped_rows": 0.0,
        "loss": float(
            _split_loss(
                train_total,
                train_pos,
                train_neg,
                val_total,
                val_pos,
                val_neg,
                cfg.val_fraction_in_pool,
                _class_rate(df),
                cfg.class_penalty,
                cfg.class_rate_weight,
                drop_fraction=0.0,
            )
        ),
    }
    return ScenarioSplitResult(train_df=train_df, val_df=val_df, dropped_df=dropped_df, metrics=metrics)


def _split_s4_double_disjoint(df: pd.DataFrame, cfg: ScenarioSplitConfig) -> ScenarioSplitResult:
    if "target_kinase" not in df.columns:
        raise ValueError("S4 requires column 'target_kinase'")

    comp_codes, comp_uniques = pd.factorize(df["chembl_id"].astype(str), sort=False)
    kin_codes, kin_uniques = pd.factorize(df["target_kinase"].astype(str), sort=False)

    labels = df["label"].to_numpy(dtype=int)
    total_rows = len(df)
    pool_pos_rate = float((labels == 1).sum() / max(total_rows, 1))
    target = cfg.val_fraction_in_pool
    base = float(np.clip(np.sqrt(max(target, 1e-6)), 0.05, 0.8))

    best_masks = None
    best_loss = float("inf")
    best_metrics = None

    for restart in range(cfg.s4_restarts):
        rng = np.random.default_rng(cfg.seed + 8713 * restart)

        p_comp = float(np.clip(base * rng.uniform(0.75, 1.25), 0.05, 0.9))
        p_kin = float(np.clip(base * rng.uniform(0.75, 1.25), 0.05, 0.9))

        comp_is_val = rng.random(len(comp_uniques)) < p_comp
        kin_is_val = rng.random(len(kin_uniques)) < p_kin

        row_comp_val = comp_is_val[comp_codes]
        row_kin_val = kin_is_val[kin_codes]

        val_mask = row_comp_val & row_kin_val
        train_mask = (~row_comp_val) & (~row_kin_val)
        keep_mask = train_mask | val_mask
        drop_mask = ~keep_mask

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if train_idx.size == 0 or val_idx.size == 0:
            continue

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        train_total = int(train_idx.size)
        train_pos = int((train_labels == 1).sum())
        train_neg = int((train_labels == 0).sum())
        val_total = int(val_idx.size)
        val_pos = int((val_labels == 1).sum())
        val_neg = int((val_labels == 0).sum())

        drop_fraction = float(drop_mask.sum() / max(total_rows, 1))

        loss = _split_loss(
            train_total=train_total,
            train_pos=train_pos,
            train_neg=train_neg,
            val_total=val_total,
            val_pos=val_pos,
            val_neg=val_neg,
            target_val_fraction=target,
            pool_pos_rate=pool_pos_rate,
            class_penalty=cfg.class_penalty,
            class_rate_weight=cfg.class_rate_weight,
            drop_fraction=drop_fraction,
        )

        if loss < best_loss:
            best_loss = loss
            best_masks = (train_mask, val_mask, drop_mask)
            best_metrics = {
                "scenario_val_fraction_target": float(target),
                "effective_val_fraction": float(val_total / max(train_total + val_total, 1)),
                "drop_fraction": drop_fraction,
                "dropped_rows": float(drop_mask.sum()),
                "p_comp": p_comp,
                "p_kin": p_kin,
                "loss": float(loss),
            }

    if best_masks is None:
        raise RuntimeError("Failed to build S4 split with non-empty train/val")

    train_mask, val_mask, drop_mask = best_masks
    train_df = df.loc[train_mask].copy().reset_index(drop=True)
    val_df = df.loc[val_mask].copy().reset_index(drop=True)
    dropped_df = df.loc[drop_mask].copy().reset_index(drop=True)

    return ScenarioSplitResult(
        train_df=train_df,
        val_df=val_df,
        dropped_df=dropped_df,
        metrics=best_metrics or {},
    )


def split_train_val_by_scenario(
    remainder_df: pd.DataFrame,
    scenario_code: str,
    config: ScenarioSplitConfig,
) -> ScenarioSplitResult:
    """Create train/val split for one scenario using a fixed test remainder pool."""
    if scenario_code not in SCENARIO_NAMES:
        raise ValueError(
            f"Unknown scenario_code='{scenario_code}'. Supported={list(SCENARIO_NAMES.keys())}"
        )

    if remainder_df.empty:
        raise ValueError("Remainder dataset is empty; cannot split train/val")

    if scenario_code == "S1":
        return _split_random_stratified(remainder_df, config)
    if scenario_code == "S2":
        return _split_by_group(remainder_df, group_col="chembl_id", cfg=config)
    if scenario_code == "S3":
        return _split_by_group(remainder_df, group_col="target_kinase", cfg=config)
    if scenario_code == "Sc":
        return _split_by_group(remainder_df, group_col="scaffold", cfg=config)
    return _split_s4_double_disjoint(remainder_df, cfg=config)


def validate_scenario_split(scenario_code: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    if train_df.empty or val_df.empty:
        raise ValueError(f"{scenario_code}: train or val is empty")

    for name, df in (("train", train_df), ("val", val_df)):
        labels = set(df["label"].astype(int).unique().tolist())
        if not {0, 1}.issubset(labels):
            raise ValueError(f"{scenario_code}: split '{name}' must contain labels 0 and 1")

    if scenario_code == "S2":
        a = set(train_df["chembl_id"].astype(str).unique())
        b = set(val_df["chembl_id"].astype(str).unique())
        if a & b:
            raise ValueError("S2: compound overlap between train and val")

    if scenario_code == "S3":
        a = set(train_df["target_kinase"].astype(str).unique())
        b = set(val_df["target_kinase"].astype(str).unique())
        if a & b:
            raise ValueError("S3: kinase overlap between train and val")

    if scenario_code == "Sc":
        a = set(train_df["scaffold"].astype(str).unique())
        b = set(val_df["scaffold"].astype(str).unique())
        if a & b:
            raise ValueError("Sc: scaffold overlap between train and val")

    if scenario_code == "S4":
        comp_a = set(train_df["chembl_id"].astype(str).unique())
        comp_b = set(val_df["chembl_id"].astype(str).unique())
        if comp_a & comp_b:
            raise ValueError("S4: compound overlap between train and val")

        kin_a = set(train_df["target_kinase"].astype(str).unique())
        kin_b = set(val_df["target_kinase"].astype(str).unique())
        if kin_a & kin_b:
            raise ValueError("S4: kinase overlap between train and val")


def build_split_distribution_row(
    dataset_name: str,
    scenario_code: str,
    split_name: str,
    split_df: pd.DataFrame,
    total_rows_dataset: int,
    dropped_rows: int,
) -> Dict[str, float]:
    rows, pos_rows, neg_rows = _class_counts(split_df)
    pos_pct = 100.0 * pos_rows / max(rows, 1)
    neg_pct = 100.0 * neg_rows / max(rows, 1)

    return {
        "dataset": dataset_name,
        "scenario": scenario_code,
        "scenario_name": SCENARIO_NAMES.get(scenario_code, "unknown"),
        "split": split_name,
        "rows": int(rows),
        "fraction_rows_dataset": rows / max(total_rows_dataset, 1),
        "pos_rows": int(pos_rows),
        "neg_rows": int(neg_rows),
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "unique_compounds": int(split_df["chembl_id"].nunique()),
        "unique_kinases": int(split_df["target_kinase"].nunique()) if "target_kinase" in split_df.columns else 0,
        "unique_scaffolds": int(split_df["scaffold"].nunique()) if "scaffold" in split_df.columns else 0,
        "dropped_rows_scenario": int(dropped_rows),
    }
