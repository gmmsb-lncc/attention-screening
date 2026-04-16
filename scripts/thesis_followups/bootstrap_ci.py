#!/usr/bin/env python3
"""Bootstrap CI + paired statistical tests for DT-Kinase v7 benchmark.

Replaces the "x +/- sigma" reporting in Chapter 5 with median + 95% CI +
paired Wilcoxon p-value, addressing reviewer concern about n=5 seeds and
absent significance tests.

Expects metrics.json files with keys 'raw_logits', 'raw_labels', and
'threshold' from the v7 test-set evaluation. One JSON per (model, seed).

Usage:
    # Single-model CI:
    python3 scripts/thesis_followups/bootstrap_ci.py \\
        --paths "results/benchmark_non_human_8M/seed_*/metrics.json" \\
        --model-name DT-Kinase-v7 \\
        --n-bootstrap 1000

    # Pairwise comparison (same test set, shared bootstrap samples):
    python3 scripts/thesis_followups/bootstrap_ci.py \\
        --paths "results/.../dt_kinase/seed_*.json" "results/.../graphban/seed_*.json" \\
        --compare dt_kinase graphban

    # Calibration ablation (Platt vs Temperature from #5):
    python3 scripts/thesis_followups/bootstrap_ci.py \\
        --paths "results/thesis_followups/platt_vs_temperature/non_human/platt/seed_*/metrics.json" \\
                "results/thesis_followups/platt_vs_temperature/non_human/temperature/seed_*/metrics.json" \\
        --compare platt temperature

Outputs LaTeX table fragments ready to paste into Chapter 5.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)


METRICS = ["mcc", "auroc", "f1", "accuracy", "precision", "recall"]


def load_predictions(glob_pattern: str) -> list[dict]:
    """Load per-seed metrics.json files matching a glob."""
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    out = []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        if "raw_logits" not in d or "raw_labels" not in d:
            raise ValueError(f"{p} lacks raw_logits/raw_labels — rerun eval")
        out.append({
            "path": p,
            "logits": np.asarray(d["raw_logits"], dtype=np.float64),
            "labels": np.asarray(d["raw_labels"], dtype=np.int64),
            "threshold": float(d.get("threshold", 0.5)),
            "seed": d.get("seed", int(Path(p).parent.name.split("_")[-1])),
        })
    return out


def compute_metrics(logits: np.ndarray, labels: np.ndarray,
                   threshold: float) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    return {
        "mcc": matthews_corrcoef(labels, preds),
        "auroc": roc_auc_score(labels, probs) if len(set(labels)) == 2 else 0.5,
        "f1": f1_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def bootstrap_ci(seeds_data: list[dict], n_boot: int, rng: np.random.Generator
                ) -> dict[str, tuple[float, float, float]]:
    """Paired bootstrap across test samples (same resampling mask per seed).

    Returns dict: metric -> (median, lo_95, hi_95) averaged over seeds.
    """
    n = len(seeds_data[0]["labels"])
    for s in seeds_data:
        assert len(s["labels"]) == n, "Test sets must align across seeds"

    metric_samples = {m: [] for m in METRICS}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        per_seed = []
        for s in seeds_data:
            m = compute_metrics(s["logits"][idx], s["labels"][idx], s["threshold"])
            per_seed.append(m)
        mean_over_seeds = {k: np.mean([ps[k] for ps in per_seed]) for k in METRICS}
        for k in METRICS:
            metric_samples[k].append(mean_over_seeds[k])

    result = {}
    for k in METRICS:
        arr = np.array(metric_samples[k])
        result[k] = (
            float(np.median(arr)),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        )
    return result


def paired_delta(seeds_a: list[dict], seeds_b: list[dict], n_boot: int,
                rng: np.random.Generator) -> dict[str, dict]:
    """Paired bootstrap of delta = A - B, with Wilcoxon over per-seed deltas."""
    n = len(seeds_a[0]["labels"])
    assert all(np.array_equal(a["labels"], b["labels"])
               for a, b in zip(seeds_a, seeds_b)), \
        "Paired test requires identical label arrays across models"

    delta_samples = {m: [] for m in METRICS}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k in METRICS:
            a_vals = [compute_metrics(a["logits"][idx], a["labels"][idx],
                                     a["threshold"])[k] for a in seeds_a]
            b_vals = [compute_metrics(bs["logits"][idx], bs["labels"][idx],
                                     bs["threshold"])[k] for bs in seeds_b]
            delta_samples[k].append(np.mean(a_vals) - np.mean(b_vals))

    # Per-seed deltas (no bootstrap) for Wilcoxon signed-rank
    per_seed_deltas = {m: [] for m in METRICS}
    for a, b in zip(seeds_a, seeds_b):
        ma = compute_metrics(a["logits"], a["labels"], a["threshold"])
        mb = compute_metrics(b["logits"], b["labels"], b["threshold"])
        for k in METRICS:
            per_seed_deltas[k].append(ma[k] - mb[k])

    result = {}
    for k in METRICS:
        arr = np.array(delta_samples[k])
        per_seed = np.array(per_seed_deltas[k])
        if len(per_seed) >= 3 and np.any(per_seed != 0):
            try:
                w_stat, p_val = stats.wilcoxon(per_seed, alternative="two-sided")
            except ValueError:
                w_stat, p_val = 0.0, 1.0
        else:
            w_stat, p_val = 0.0, 1.0

        result[k] = {
            "median_delta": float(np.median(arr)),
            "ci_95": (float(np.percentile(arr, 2.5)),
                      float(np.percentile(arr, 97.5))),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(p_val),
            "per_seed": per_seed.tolist(),
        }
    return result


def print_single_table(name: str, ci: dict) -> None:
    print(f"\n% {name} — bootstrap 95% CI (n_boot=1000, n_seeds={{...}})")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Metric & Median & CI$_{2.5}$ & CI$_{97.5}$ \\")
    print(r"\midrule")
    for k in METRICS:
        med, lo, hi = ci[k]
        print(f"{k.upper():<10} & {med:.4f} & {lo:.4f} & {hi:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_paired_table(name_a: str, name_b: str, cmp: dict) -> None:
    print(f"\n% Paired comparison: {name_a} - {name_b}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Metric & $\Delta$ median & CI$_{95}$ & Wilcoxon $p$ & Significant? \\")
    print(r"\midrule")
    for k in METRICS:
        d = cmp[k]
        sig = "Yes" if d["wilcoxon_p"] < 0.05 else "No"
        lo, hi = d["ci_95"]
        print(f"{k.upper():<10} & {d['median_delta']:+.4f} "
              f"& [{lo:+.4f}, {hi:+.4f}] "
              f"& {d['wilcoxon_p']:.4f} & {sig} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True,
                    help="Glob patterns; one per model (order matches --compare).")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="Model names matching --paths (required if >1 path).")
    ap.add_argument("--model-name", default="model",
                    help="Used when only one --paths entry is given.")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    loaded = [load_predictions(p) for p in args.paths]

    if len(loaded) == 1:
        ci = bootstrap_ci(loaded[0], args.n_bootstrap, rng)
        print_single_table(args.model_name, ci)
        return

    if args.compare is None or len(args.compare) != len(loaded):
        raise SystemExit("--compare must list one name per --paths entry")

    # Single-model CIs
    for name, data in zip(args.compare, loaded):
        print_single_table(name, bootstrap_ci(data, args.n_bootstrap, rng))

    # Pairwise (first vs each subsequent)
    base_name, base_data = args.compare[0], loaded[0]
    for name, data in zip(args.compare[1:], loaded[1:]):
        cmp = paired_delta(base_data, data, args.n_bootstrap, rng)
        print_paired_table(base_name, name, cmp)


if __name__ == "__main__":
    main()
