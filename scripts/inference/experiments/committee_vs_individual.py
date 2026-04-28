#!/usr/bin/env python3
"""Empirical test: committee (4-model ensemble) vs individual models.

For each corpus (Non-Human, Human, All) and each model
(DT-Kinase, DrugBAN, GraphBAN, ConPLex):

  1. Load raw_predictions.npz across 5 canonical seeds
     {42, 123, 456, 789, 1024}.
  2. Compute prob = mean_5_seeds(prob_per_seed) per pair.
  3. Compute threshold = mean_5_seeds(threshold_per_seed) from
     calibration sidecars.
  4. Compute pred = (prob >= threshold).
  5. Score against ground truth from universal_test.tsv.

Then build the committee:

  6. committee_prob[i] = mean_m( prob_5seed[m, i] )
  7. committee_thr    = mean_m( thr_5seed[m] )
  8. committee_pred   = (committee_prob >= committee_thr)

Compute MCC, AUROC, F1, Accuracy for each of the 5 systems.

Statistical inference:

  9. Per-pair paired bootstrap (B = 10000 resamples) on the test set:
     - For each resample, recompute MCC for each system.
     - Delta_MCC = MCC(committee) - MCC(individual_m) per resample.
     - IC95 = [percentile 2.5, percentile 97.5] of the resampled deltas.
     - The 95% CI tells whether the committee is detectably better, worse,
       or indistinguishable from each individual model under the same
       resampling-induced uncertainty.

Output:
  - results/inference/committee_experiment/<corpus>_metrics.csv
  - results/inference/committee_experiment/<corpus>_paired_bootstrap.csv
  - results/inference/committee_experiment/REPORT.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                              f1_score, accuracy_score)


import os

REPO = Path(__file__).resolve().parents[3]
SEEDS = [42, 123, 456, 789, 1024]
CORPORA = ["non_human", "human", "all"]
MODELS = ["dtkinase", "drugban", "graphban", "conplex"]
OUT_DIR = REPO / "results" / "inference" / "committee_experiment"
N_BOOT = int(os.environ.get("BENCHMARK_BOOTSTRAP_B", "10000"))


def model_paths(model: str, corpus: str, seed: int) -> tuple[Path, Path]:
    """Return (npz_path, calibration_json_path) for a (model, corpus, seed) cell."""
    if model == "dtkinase":
        if corpus == "non_human":
            base = REPO / "results" / "benchmark_non_human_8M_13_05_2026_v3" \
                        / "test" / "level4_cnn_8M" / "non_human"
        elif corpus == "human":
            base = REPO / "results" / "benchmark_human_8M_13_05_2026" \
                        / "test" / "level4_cnn_8M" / "human"
        else:
            base = REPO / "results" / "all" / "benchmark_all_8M_13_04_2026" \
                        / "test" / "level4_cnn_8M" / "all"
        return (base / f"seed_{seed}" / "raw_predictions.npz",
                base / f"seed_{seed}" / "level4_cnn_calibration.json")
    if model == "drugban":
        base = REPO / "DrugBAN" / "results_universal" / "results_universal" / corpus
        return (base / f"seed_{seed}" / "raw_predictions.npz",
                base / f"seed_{seed}" / "drugban_calibration.json")
    if model == "graphban":
        base = REPO / "GraphBAN" / "results_universal" / corpus
        return (base / f"seed_{seed}" / "raw_predictions.npz",
                base / f"seed_{seed}" / "graphban_calibration.json")
    if model == "conplex":
        base = REPO / "ConPLex" / "results_universal" / corpus
        return (base / f"seed_{seed}" / "raw_predictions.npz",
                base / f"seed_{seed}" / "conplex_calibration.json")
    raise ValueError(f"unknown model: {model}")


def load_5seed(model: str, corpus: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (mean_prob over seeds, y_true, mean_threshold)."""
    probs_per_seed = []
    thresholds = []
    y_true_ref = None
    for s in SEEDS:
        npz_path, cal_path = model_paths(model, corpus, s)
        d = np.load(npz_path)
        # Baselines store val_y_*/test_y_*; DT-Kinase stores y_*
        prob_key = "test_y_prob" if "test_y_prob" in d.files else "y_prob"
        true_key = "test_y_true" if "test_y_true" in d.files else "y_true"
        y_true = d[true_key].astype(int)
        probs_per_seed.append(d[prob_key].astype(np.float64))
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            assert np.array_equal(y_true_ref, y_true), \
                f"y_true mismatch in {model}/{corpus}/seed_{s}"
        cal = json.loads(cal_path.read_text())
        thresholds.append(float(cal["threshold"]))

    prob_mean = np.mean(np.stack(probs_per_seed), axis=0)
    thr_mean = float(np.mean(thresholds))
    return prob_mean, y_true_ref, thr_mean


def system_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    """Return MCC, AUROC, F1, Accuracy for a single system."""
    pred = (prob >= threshold).astype(int)
    return {
        "mcc": matthews_corrcoef(y_true, pred),
        "auroc": roc_auc_score(y_true, prob),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
    }


def _mcc_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Vectorized MCC computation; ~50× faster than sklearn for tight loop."""
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    num = tp * tn - fp * fn
    den_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den_sq == 0:
        return 0.0
    return float(num / np.sqrt(float(den_sq)))


def paired_bootstrap_delta(y_true: np.ndarray,
                            prob_a: np.ndarray, thr_a: float,
                            prob_b: np.ndarray, thr_b: float,
                            n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap CI95 for MCC(a) - MCC(b) over per-pair resampling.

    Resampling unit is the test pair (with replacement). For each resample,
    MCC is recomputed for both systems; the delta is the bootstrap statistic.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    pred_a = (prob_a >= thr_a).astype(int)
    pred_b = (prob_b >= thr_b).astype(int)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        deltas[b] = _mcc_fast(yt, pred_a[idx]) - _mcc_fast(yt, pred_b[idx])
        if (b + 1) % 500 == 0:
            print(f"      bootstrap {b+1}/{n_boot} (Δ_running_mean={deltas[:b+1].mean():+.4f})",
                  flush=True)
    return {
        "delta_mean": float(deltas.mean()),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "frac_positive": float((deltas > 0).mean()),
    }


def run_corpus(corpus: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n{'='*70}")
    print(f"  Corpus: {corpus}")
    print(f"{'='*70}")

    # Load 5-seed-averaged predictions for each model.
    model_probs: dict[str, np.ndarray] = {}
    model_thrs: dict[str, float] = {}
    y_true = None
    for m in MODELS:
        prob, yt, thr = load_5seed(m, corpus)
        model_probs[m] = prob
        model_thrs[m] = thr
        if y_true is None:
            y_true = yt
        else:
            assert np.array_equal(y_true, yt), f"y_true mismatch across models: {m}"
        print(f"  {m:9s}: prob_mean = {prob.mean():.4f}  thr = {thr:.4f}  n = {len(prob)}")

    # Build committee: mean of 4 model probabilities; mean of 4 thresholds.
    committee_prob = np.mean(np.stack([model_probs[m] for m in MODELS]), axis=0)
    committee_thr = float(np.mean([model_thrs[m] for m in MODELS]))
    print(f"  {'committee':9s}: prob_mean = {committee_prob.mean():.4f}  thr = {committee_thr:.4f}")

    # Compute metrics per system.
    rows = []
    for m in MODELS:
        met = system_metrics(y_true, model_probs[m], model_thrs[m])
        rows.append({"system": m, **met})
    met = system_metrics(y_true, committee_prob, committee_thr)
    rows.append({"system": "committee", **met})
    metrics_df = pd.DataFrame(rows).set_index("system")
    print("\nMetrics:")
    print(metrics_df.round(4).to_string())

    # Paired bootstrap: committee vs each individual model.
    boot_rows = []
    print("\nPaired bootstrap (B=10000) — committee vs each individual model:")
    for m in MODELS:
        delta = paired_bootstrap_delta(
            y_true,
            committee_prob, committee_thr,
            model_probs[m], model_thrs[m],
            n_boot=N_BOOT, seed=42,
        )
        verdict = ("committee leads ▲" if delta["ci_lo"] > 0 else
                   "committee trails ▼" if delta["ci_hi"] < 0 else
                   "indistinguishable ⊘")
        boot_rows.append({
            "comparison": f"committee - {m}",
            "delta_mean": delta["delta_mean"],
            "ci_lo": delta["ci_lo"],
            "ci_hi": delta["ci_hi"],
            "frac_positive": delta["frac_positive"],
            "verdict": verdict,
        })
        print(f"  vs {m:9s}: Δ={delta['delta_mean']:+.4f}  "
              f"CI95=[{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}]  "
              f"P(Δ>0)={delta['frac_positive']:.3f}  {verdict}")
    boot_df = pd.DataFrame(boot_rows)

    return metrics_df, boot_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_all = {}
    boot_all = {}
    for corpus in CORPORA:
        try:
            mdf, bdf = run_corpus(corpus)
        except FileNotFoundError as e:
            print(f"\n  WARNING: skipping {corpus} ({e})", file=sys.stderr)
            continue
        mdf.to_csv(OUT_DIR / f"{corpus}_metrics.csv")
        bdf.to_csv(OUT_DIR / f"{corpus}_paired_bootstrap.csv", index=False)
        metrics_all[corpus] = mdf
        boot_all[corpus] = bdf

    # Markdown report
    lines = ["# Committee vs Individual Models — Empirical Test", ""]
    lines.append(f"**Protocol**: 5-seed averaged probabilities and thresholds "
                 f"per model; committee = mean of 4 models. Bootstrap B=10000, "
                 f"per-pair resampling, IC95 = percentile [2.5, 97.5].")
    lines.append("")

    for corpus in CORPORA:
        if corpus not in metrics_all:
            continue
        lines.append(f"## Corpus: {corpus}")
        lines.append("")
        lines.append("### Metrics (5 systems)")
        lines.append("")
        lines.append(metrics_all[corpus].round(4).to_markdown())
        lines.append("")
        lines.append("### Paired bootstrap: committee vs individual model")
        lines.append("")
        lines.append(boot_all[corpus].round(4).to_markdown(index=False))
        lines.append("")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"\n  → wrote {OUT_DIR / 'REPORT.md'} (and per-corpus CSVs)")


if __name__ == "__main__":
    main()
