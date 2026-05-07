"""Effect size: Hedges' g paired (J(4)) primary, unpaired (J(8)) cross-check.

Implements Guideline 3.3.2 of Ash/Wognum 2025 with small-sample
correction (Hedges 1981). Two flavours:

- Paired Cohen's d_z: nu = n - 1; with n = 5, J(4) = 1 - 3/15 = 0.8000
  exactly under the Lakens 2013 / Borenstein approximation
  J(nu) = 1 - 3/(4*nu - 1).
  This is the statistically correct form for our 5-seed paired design
  (same proteins, different inits) and is the primary reported value.
- Unpaired Hedges' g (two-sample form): nu = 2(n - 1), J(8) approximately
  0.9032. Reported for cross-check only.

The choice of J(4) vs J(8) is documented in
docs/01-methodology/statistical_protocol.md (D3 footnote).

CLI:
    python -m scripts.statistical_analysis.effect_size \\
        --corpus non_human \\
        --out results/statistical/non_human/effect_size.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

from . import (
    HEDGES_NU_PAIRED, HEDGES_NU_UNPAIRED, MODELS, PRIMARY_METRICS, SEEDS,
    data_loader, hedges_J,
)


def _per_seed_metric(seed_data: dict, metric: str) -> float:
    y = seed_data["y_true"]
    p = seed_data["y_prob"]
    pred = (p >= seed_data["threshold"]).astype(np.int64)
    if metric == "mcc":
        return float(matthews_corrcoef(y, pred)) if y.std() > 0 else 0.0
    if metric == "auroc":
        return float(roc_auc_score(y, p)) if len(set(y)) == 2 else 0.5
    if metric == "auprc":
        return float(average_precision_score(y, p)) if len(set(y)) == 2 else float(np.mean(y))
    if metric == "f1":
        return float(f1_score(y, pred, zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    if metric == "precision":
        return float(precision_score(y, pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y, pred, zero_division=0))
    raise ValueError(f"Unknown metric {metric!r}")


def _hedges_paired(per_seed_a: np.ndarray,
                  per_seed_b: np.ndarray) -> dict:
    """Paired Cohen's d_z with J(n - 1) correction."""
    delta = per_seed_a - per_seed_b
    n = len(delta)
    sd = float(np.std(delta, ddof=1)) if n > 1 else 0.0
    mean = float(np.mean(delta))
    d_z = mean / sd if sd > 0 else 0.0
    J = hedges_J(HEDGES_NU_PAIRED) if n >= 2 else 1.0
    return {
        "mean_delta": mean,
        "sd_delta": sd,
        "d_z": float(d_z),
        "J_nu": float(J),
        "g_paired": float(J * d_z),
        "nu": HEDGES_NU_PAIRED,
        "n_seeds": int(n),
    }


def _hedges_unpaired(per_seed_a: np.ndarray,
                    per_seed_b: np.ndarray) -> dict:
    """Two-sample Hedges' g with J(2(n - 1)) correction (cross-check)."""
    n = len(per_seed_a)
    if n < 2:
        return {"d": 0.0, "J_nu": 1.0, "g_unpaired": 0.0, "nu": 0, "n_seeds": int(n)}
    s_a = float(np.std(per_seed_a, ddof=1))
    s_b = float(np.std(per_seed_b, ddof=1))
    s_pooled = float(np.sqrt(((n - 1) * s_a ** 2 + (n - 1) * s_b ** 2)
                            / (2 * n - 2)))
    if s_pooled == 0:
        d = 0.0
    else:
        d = float((np.mean(per_seed_a) - np.mean(per_seed_b)) / s_pooled)
    J = hedges_J(HEDGES_NU_UNPAIRED)
    return {
        "d": d,
        "J_nu": float(J),
        "g_unpaired": float(J * d),
        "nu": HEDGES_NU_UNPAIRED,
        "n_seeds": int(n),
    }


def effect_size_panel(corpus: str, models=None, seeds=None) -> dict:
    if models is None:
        models = MODELS
    if seeds is None:
        seeds = SEEDS
    panel = data_loader.load_panel(corpus, models, seeds)

    # Compute per-seed metric vectors per model.
    metric_vecs: dict[tuple[str, str], np.ndarray] = {}
    for model, seeds_data in panel.items():
        for metric in PRIMARY_METRICS:
            metric_vecs[(model, metric)] = np.array(
                [_per_seed_metric(s, metric) for s in seeds_data],
                dtype=np.float64,
            )

    by_metric = {}
    for metric in PRIMARY_METRICS:
        by_metric[metric] = {}
        per_model_mean = {m: float(np.mean(metric_vecs[(m, metric)])) for m in models}
        per_model_std = {m: float(np.std(metric_vecs[(m, metric)], ddof=1))
                         for m in models}
        by_metric[metric]["per_model"] = {
            m: {"mean": per_model_mean[m], "std": per_model_std[m],
                "values": metric_vecs[(m, metric)].tolist()}
            for m in models
        }
        pairs = {}
        for a, b in itertools.permutations(models, 2):
            paired = _hedges_paired(metric_vecs[(a, metric)],
                                    metric_vecs[(b, metric)])
            unpaired = _hedges_unpaired(metric_vecs[(a, metric)],
                                        metric_vecs[(b, metric)])
            pairs[f"{a}__vs__{b}"] = {
                "model_a": a,
                "model_b": b,
                "paired": paired,
                "unpaired": unpaired,
            }
        by_metric[metric]["pairs"] = pairs

    return {
        "corpus": corpus,
        "models": list(models),
        "seeds": list(seeds),
        "primary_metrics": list(PRIMARY_METRICS),
        "hedges_J_paired_J4": hedges_J(HEDGES_NU_PAIRED),
        "hedges_J_unpaired_J8": hedges_J(HEDGES_NU_UNPAIRED),
        "primary_form": "paired_J4",
        "by_metric": by_metric,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("human", "non_human", "all"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = effect_size_panel(args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[effect_size] corpus={args.corpus} "
          f"J(4)={result['hedges_J_paired_J4']:.4f} (primary), "
          f"J(8)={result['hedges_J_unpaired_J8']:.4f} (cross-check)")
    for metric in PRIMARY_METRICS:
        print(f"  metric={metric}")
        per_model = result["by_metric"][metric]["per_model"]
        for m, v in per_model.items():
            print(f"    {m:10s} mean={v['mean']:+.4f} std={v['std']:.4f}")
        print(f"    paired g (DT-Kinase vs others):")
        for key, p in result["by_metric"][metric]["pairs"].items():
            if p["model_a"] != "dtkinase":
                continue
            g = p["paired"]["g_paired"]
            print(f"      vs {p['model_b']:10s} g_paired={g:+.4f} "
                  f"(d_z={p['paired']['d_z']:+.4f})")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
