"""Universal test scaffold selection across human and non-human datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UniversalSelectionConfig:
    target_test_fraction: float = 0.10
    seed: int = 42
    restarts: int = 64
    weight_human: float = 8.0
    weight_non_human: float = 3.0
    weight_ratio: float = 0.1
    class_penalty: float = 5.0
    class_rate_weight: float = 2.0


@dataclass(frozen=True)
class UniversalSelectionResult:
    test_scaffolds: List[str]
    metrics: Dict[str, float]


def _missing_classes_scalar(pos_h: float, neg_h: float, pos_n: float, neg_n: float) -> int:
    return int(pos_h <= 0) + int(neg_h <= 0) + int(pos_n <= 0) + int(neg_n <= 0)


def _loss(
    sum_h: np.ndarray,
    sum_n: np.ndarray,
    pos_h: np.ndarray,
    neg_h: np.ndarray,
    pos_n: np.ndarray,
    neg_n: np.ndarray,
    total_h: float,
    total_n: float,
    target: float,
    ratio_target: float,
    cfg: UniversalSelectionConfig,
    pool_rate_h: float = 0.5,
    pool_rate_n: float = 0.5,
) -> np.ndarray:
    eps = 1e-12
    sum_h = np.asarray(sum_h, dtype=float)
    sum_n = np.asarray(sum_n, dtype=float)
    pos_h = np.asarray(pos_h, dtype=float)
    neg_h = np.asarray(neg_h, dtype=float)
    pos_n = np.asarray(pos_n, dtype=float)
    neg_n = np.asarray(neg_n, dtype=float)

    frac_h = sum_h / max(total_h, eps)
    frac_n = sum_n / max(total_n, eps)

    err_h = np.abs(frac_h - target)
    err_n = np.abs(frac_n - target)

    ratio = (sum_h + eps) / (sum_n + eps)
    err_ratio = np.abs(np.log(ratio / max(ratio_target, eps)))

    missing = (pos_h <= 0).astype(float) + (neg_h <= 0).astype(float) + (pos_n <= 0).astype(float) + (neg_n <= 0).astype(float)

    # Class rate preservation: penalize test sets whose positive rate
    # diverges from the overall pool rate.
    rows_h = pos_h + neg_h
    rows_n = pos_n + neg_n
    rate_h = np.where(rows_h > 0, pos_h / rows_h, 0.5)
    rate_n = np.where(rows_n > 0, pos_n / rows_n, 0.5)
    err_rate = np.abs(rate_h - pool_rate_h) + np.abs(rate_n - pool_rate_n)

    return (
        cfg.weight_human * err_h
        + cfg.weight_non_human * err_n
        + cfg.weight_ratio * err_ratio
        + cfg.class_penalty * missing
        + cfg.class_rate_weight * err_rate
    )


def _run_single_restart(
    n_h: np.ndarray,
    n_n: np.ndarray,
    pos_h: np.ndarray,
    neg_h: np.ndarray,
    pos_n: np.ndarray,
    neg_n: np.ndarray,
    total_h: int,
    total_n: int,
    target: float,
    ratio_target: float,
    cfg: UniversalSelectionConfig,
    rng: np.random.Generator,
    pool_rate_h: float = 0.5,
    pool_rate_n: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    m = len(n_h)
    selected = np.zeros(m, dtype=bool)

    sum_h = 0.0
    sum_n = 0.0
    sum_pos_h = 0.0
    sum_neg_h = 0.0
    sum_pos_n = 0.0
    sum_neg_n = 0.0

    current_loss = float(
        _loss(sum_h, sum_n, sum_pos_h, sum_neg_h, sum_pos_n, sum_neg_n, total_h, total_n, target, ratio_target, cfg, pool_rate_h, pool_rate_n)
    )

    # Greedy addition phase.
    for _ in range(m):
        remaining = np.where(~selected)[0]
        if remaining.size == 0:
            break

        cand_loss = _loss(
            sum_h + n_h[remaining],
            sum_n + n_n[remaining],
            sum_pos_h + pos_h[remaining],
            sum_neg_h + neg_h[remaining],
            sum_pos_n + pos_n[remaining],
            sum_neg_n + neg_n[remaining],
            total_h,
            total_n,
            target,
            ratio_target,
            cfg,
            pool_rate_h,
            pool_rate_n,
        )
        best_pos = int(np.argmin(cand_loss + rng.uniform(0.0, 1e-9, size=remaining.size)))
        idx = int(remaining[best_pos])
        best_loss = float(cand_loss[best_pos])

        missing_now = _missing_classes_scalar(sum_pos_h, sum_neg_h, sum_pos_n, sum_neg_n)
        below_target = (sum_h / total_h) < target or (sum_n / total_n) < target

        if best_loss < current_loss - 1e-12 or missing_now > 0 or below_target:
            selected[idx] = True
            sum_h += float(n_h[idx])
            sum_n += float(n_n[idx])
            sum_pos_h += float(pos_h[idx])
            sum_neg_h += float(neg_h[idx])
            sum_pos_n += float(pos_n[idx])
            sum_neg_n += float(neg_n[idx])
            current_loss = best_loss
        else:
            break

    # Optional topping-up to reach target if still below.
    while ((sum_h / total_h) < target or (sum_n / total_n) < target) and np.any(~selected):
        remaining = np.where(~selected)[0]
        deficit_score = (
            np.maximum(0.0, target - (sum_h + n_h[remaining]) / total_h)
            + np.maximum(0.0, target - (sum_n + n_n[remaining]) / total_n)
            + 0.05 * np.abs(
                np.log(
                    ((sum_h + n_h[remaining]) + 1e-12)
                    / ((sum_n + n_n[remaining]) + 1e-12)
                    / max(ratio_target, 1e-12)
                )
            )
        )
        add_idx = int(remaining[int(np.argmin(deficit_score + rng.uniform(0.0, 1e-9, size=remaining.size)))])
        selected[add_idx] = True
        sum_h += float(n_h[add_idx])
        sum_n += float(n_n[add_idx])
        sum_pos_h += float(pos_h[add_idx])
        sum_neg_h += float(neg_h[add_idx])
        sum_pos_n += float(pos_n[add_idx])
        sum_neg_n += float(neg_n[add_idx])
        current_loss = float(
            _loss(sum_h, sum_n, sum_pos_h, sum_neg_h, sum_pos_n, sum_neg_n, total_h, total_n, target, ratio_target, cfg, pool_rate_h, pool_rate_n)
        )

    # Prune phase.
    changed = True
    while changed:
        changed = False
        for idx in rng.permutation(np.where(selected)[0]):
            new_loss = float(
                _loss(
                    sum_h - n_h[idx],
                    sum_n - n_n[idx],
                    sum_pos_h - pos_h[idx],
                    sum_neg_h - neg_h[idx],
                    sum_pos_n - pos_n[idx],
                    sum_neg_n - neg_n[idx],
                    total_h,
                    total_n,
                    target,
                    ratio_target,
                    cfg,
                    pool_rate_h,
                    pool_rate_n,
                )
            )
            if new_loss < current_loss - 1e-12:
                selected[idx] = False
                sum_h -= float(n_h[idx])
                sum_n -= float(n_n[idx])
                sum_pos_h -= float(pos_h[idx])
                sum_neg_h -= float(neg_h[idx])
                sum_pos_n -= float(pos_n[idx])
                sum_neg_n -= float(neg_n[idx])
                current_loss = new_loss
                changed = True

    metrics = {
        "test_compounds_human": float(sum_h),
        "test_compounds_non_human": float(sum_n),
        "test_fraction_human": float(sum_h / total_h),
        "test_fraction_non_human": float(sum_n / total_n),
        "test_ratio_human_non_human": float((sum_h + 1e-12) / (sum_n + 1e-12)),
        "pos_rows_human": float(sum_pos_h),
        "neg_rows_human": float(sum_neg_h),
        "pos_rows_non_human": float(sum_pos_n),
        "neg_rows_non_human": float(sum_neg_n),
        "selected_scaffolds": float(selected.sum()),
        "loss": float(current_loss),
    }
    return selected, metrics, current_loss


def select_universal_test_scaffolds(
    human_stats: pd.DataFrame,
    non_human_stats: pd.DataFrame,
    total_unique_human: int,
    total_unique_non_human: int,
    config: UniversalSelectionConfig,
    unknown_scaffold: str = "UNKNOWN",
) -> UniversalSelectionResult:
    """Select universal test scaffolds shared between human and non-human datasets."""
    if config.target_test_fraction <= 0 or config.target_test_fraction >= 1:
        raise ValueError("target_test_fraction must be in (0, 1)")

    merged = human_stats.merge(non_human_stats, on="scaffold", suffixes=("_h", "_n"))
    merged = merged[merged["scaffold"] != unknown_scaffold].copy()
    if merged.empty:
        raise ValueError("No shared non-UNKNOWN scaffolds found between datasets")

    n_h = merged["unique_compounds_h"].to_numpy(dtype=float)
    n_n = merged["unique_compounds_n"].to_numpy(dtype=float)
    pos_h = merged["rows_pos_h"].to_numpy(dtype=float)
    neg_h = merged["rows_neg_h"].to_numpy(dtype=float)
    pos_n = merged["rows_pos_n"].to_numpy(dtype=float)
    neg_n = merged["rows_neg_n"].to_numpy(dtype=float)

    gamma_h = float(n_h.sum() / total_unique_human)
    gamma_n = float(n_n.sum() / total_unique_non_human)
    target_effective = min(config.target_test_fraction, gamma_h, gamma_n)

    if target_effective <= 0:
        raise ValueError("Effective target test fraction is zero; cannot create universal test set")

    ratio_target = float(total_unique_human / total_unique_non_human)

    # Pool class rates for class-rate preservation in loss.
    total_pos_h = float(pos_h.sum())
    total_pos_n = float(pos_n.sum())
    total_rows_h = float((pos_h + neg_h).sum())
    total_rows_n = float((pos_n + neg_n).sum())
    pool_rate_h = total_pos_h / max(total_rows_h, 1e-12)
    pool_rate_n = total_pos_n / max(total_rows_n, 1e-12)

    best_selected = None
    best_metrics = None
    best_loss = float("inf")

    for restart in range(config.restarts):
        rng = np.random.default_rng(config.seed + restart * 7919)
        perm = rng.permutation(len(merged))

        sel_perm, metrics_perm, loss_perm = _run_single_restart(
            n_h=n_h[perm],
            n_n=n_n[perm],
            pos_h=pos_h[perm],
            neg_h=neg_h[perm],
            pos_n=pos_n[perm],
            neg_n=neg_n[perm],
            total_h=total_unique_human,
            total_n=total_unique_non_human,
            target=target_effective,
            ratio_target=ratio_target,
            cfg=config,
            rng=rng,
            pool_rate_h=pool_rate_h,
            pool_rate_n=pool_rate_n,
        )

        sel = np.zeros_like(sel_perm)
        sel[perm] = sel_perm

        missing_classes = (
            int(metrics_perm["pos_rows_human"] <= 0)
            + int(metrics_perm["neg_rows_human"] <= 0)
            + int(metrics_perm["pos_rows_non_human"] <= 0)
            + int(metrics_perm["neg_rows_non_human"] <= 0)
        )
        feasible = missing_classes == 0 and int(sel.sum()) > 0

        if feasible and loss_perm < best_loss:
            best_selected = sel
            best_loss = loss_perm
            best_metrics = metrics_perm

    if best_selected is None:
        raise RuntimeError(
            "Failed to build a feasible universal test scaffold set. "
            "Try increasing --restarts or reducing target test fraction."
        )

    chosen = merged.loc[best_selected, "scaffold"].tolist()
    chosen.sort()

    metrics = {
        **best_metrics,
        "shared_scaffolds": float(len(merged)),
        "chosen_scaffolds": float(len(chosen)),
        "target_test_fraction": float(config.target_test_fraction),
        "effective_target_test_fraction": float(target_effective),
        "gamma_human": gamma_h,
        "gamma_non_human": gamma_n,
        "ratio_target_human_non_human": ratio_target,
    }

    return UniversalSelectionResult(test_scaffolds=chosen, metrics=metrics)
