"""Post-hoc classification metrics (Ash/Wognum 2025 G3 operational layer).

Computes downstream-relevant metrics that translate predicted scores
into operational decisions for kinase virtual screening:

- precision@recall=0.8: max precision achievable while retaining
  >= 80 percent of true actives. Reflects "how clean is the hit list
  if we want broad recall".
- recall@precision=0.8: max recall achievable while keeping precision
  >= 80 percent. Reflects "how many actives can we find if we cap
  the false-positive rate".
- TNR@recall=0.9: true-negative rate when we keep recall >= 0.9.
  Reflects how aggressively the model can discard inactives without
  sacrificing actives.

These three points are computed per (model, corpus, seed) and
aggregated as mean +/- std over the 5 seeds.

CLI:
    python -m scripts.statistical_analysis.posthoc_classification \\
        --corpus non_human \\
        --out results/statistical/non_human/posthoc.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from . import MODELS, SEEDS, data_loader


def _precision_at_recall(y_true: np.ndarray, y_prob: np.ndarray,
                        recall_target: float) -> dict:
    """Return max precision among thresholds giving recall >= target."""
    p, r, t = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve appends a final (precision=1, recall=0)
    # dummy point with no associated threshold; truncate.
    p = p[:-1]
    r = r[:-1]
    mask = r >= recall_target
    if not np.any(mask):
        return {"precision": float("nan"), "recall": float("nan"),
                "threshold": float("nan")}
    idx = int(np.argmax(p[mask]))
    sub_p = p[mask][idx]
    sub_r = r[mask][idx]
    sub_t = t[mask][idx]
    return {"precision": float(sub_p), "recall": float(sub_r),
            "threshold": float(sub_t)}


def _recall_at_precision(y_true: np.ndarray, y_prob: np.ndarray,
                        precision_target: float) -> dict:
    p, r, t = precision_recall_curve(y_true, y_prob)
    p = p[:-1]
    r = r[:-1]
    mask = p >= precision_target
    if not np.any(mask):
        return {"precision": float("nan"), "recall": float("nan"),
                "threshold": float("nan")}
    idx = int(np.argmax(r[mask]))
    sub_p = p[mask][idx]
    sub_r = r[mask][idx]
    sub_t = t[mask][idx]
    return {"precision": float(sub_p), "recall": float(sub_r),
            "threshold": float(sub_t)}


def _tnr_at_recall(y_true: np.ndarray, y_prob: np.ndarray,
                  recall_target: float) -> dict:
    """TNR (specificity) at the threshold giving recall >= target."""
    fpr, tpr, t = roc_curve(y_true, y_prob)
    # tpr is the recall (sensitivity); tnr = 1 - fpr.
    mask = tpr >= recall_target
    if not np.any(mask):
        return {"tnr": float("nan"), "recall": float("nan"),
                "threshold": float("nan")}
    # Among thresholds with recall >= target, pick the one with highest
    # TNR (i.e., lowest FPR).
    sub_fpr = fpr[mask]
    sub_tpr = tpr[mask]
    sub_t = t[mask]
    idx = int(np.argmin(sub_fpr))
    return {"tnr": float(1.0 - sub_fpr[idx]),
            "recall": float(sub_tpr[idx]),
            "threshold": float(sub_t[idx])}


def _summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


def posthoc_panel(corpus: str,
                 recall_target: float = 0.8,
                 precision_target: float = 0.8,
                 tnr_recall_target: float = 0.9) -> dict:
    panel = data_loader.load_panel(corpus, MODELS, SEEDS)

    by_model = {}
    for model, seeds_data in panel.items():
        per_seed = []
        for sd in seeds_data:
            y, p = sd["y_true"], sd["y_prob"]
            entry = {
                "seed": sd["seed"],
                "p_at_r": _precision_at_recall(y, p, recall_target),
                "r_at_p": _recall_at_precision(y, p, precision_target),
                "tnr_at_r": _tnr_at_recall(y, p, tnr_recall_target),
            }
            per_seed.append(entry)
        summary = {
            "p_at_r": _summarize([e["p_at_r"]["precision"] for e in per_seed]),
            "r_at_p": _summarize([e["r_at_p"]["recall"] for e in per_seed]),
            "tnr_at_r": _summarize([e["tnr_at_r"]["tnr"] for e in per_seed]),
        }
        by_model[model] = {"per_seed": per_seed, "summary": summary}

    return {
        "corpus": corpus,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "targets": {
            "recall_target_for_p_at_r": recall_target,
            "precision_target_for_r_at_p": precision_target,
            "recall_target_for_tnr_at_r": tnr_recall_target,
        },
        "by_model": by_model,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("human", "non_human", "all"))
    ap.add_argument("--recall-target", type=float, default=0.8)
    ap.add_argument("--precision-target", type=float, default=0.8)
    ap.add_argument("--tnr-recall-target", type=float, default=0.9)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = posthoc_panel(args.corpus, args.recall_target,
                          args.precision_target, args.tnr_recall_target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[posthoc] corpus={args.corpus}")
    for model, payload in result["by_model"].items():
        s = payload["summary"]
        print(f"  {model:10s} "
              f"p@r={args.recall_target:.1f}: {s['p_at_r']['mean']:.4f}+/-{s['p_at_r']['std']:.4f} | "
              f"r@p={args.precision_target:.1f}: {s['r_at_p']['mean']:.4f}+/-{s['r_at_p']['std']:.4f} | "
              f"tnr@r={args.tnr_recall_target:.1f}: {s['tnr_at_r']['mean']:.4f}+/-{s['tnr_at_r']['std']:.4f}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
