#!/usr/bin/env python3
"""Second batch of alternative committee aggregation rules vs soft-mean.

5 additional rules, complementing committee_aggregation_alts.py:

  6. weighted_logit — mean of w_m·logit(p_m), then sigmoid; w_m ∝ val_MCC.
                      Combines advantages of logit-mean (non-linear space)
                      + weighted_mcc (val-based trust). Stacking test.
  7. median         — element-wise median across the 4 model probs.
                      Robust to one outlier model. Tax & Duin 2000.
  8. trimmed_mean   — drop max+min per pair, mean of remaining 2.
                      More aggressive outlier rejection (winsorized at 25%).
  9. max            — element-wise max prob; "any-model-confident" rule.
                      Permissive companion to product-of-experts.
 10. harmonic_mean  — n / Σ(1/p_m); power mean with k=−1. Conservative
                      companion to product-of-experts (k=0). Penalizes
                      pairs where any model has low confidence.

Threshold convention:
  - weighted_logit: sigmoid(weighted-mean-of-logits-of-thresholds)
  - median/trimmed/max/harmonic: same aggregation rule applied to per-model
    thresholds. Keeps method self-consistent.

All compared against canonical soft_mean via paired block-bootstrap by
protein (B = 10000), IC95 percentile.

Output:
  results/inference/committee_aggregation_alts2/
      <corpus>_metrics.csv
      <corpus>_paired_bootstrap.csv
      REPORT.md
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                              f1_score, accuracy_score)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from committee_vs_individual import (  # noqa: E402
    load_5seed, dedupe_predictions, load_test_keys, _mcc_fast, model_paths,
)

REPO = Path(__file__).resolve().parents[3]
SEEDS = [42, 123, 456, 789, 1024]
CORPORA = ["non_human", "human", "all"]
MODELS = ["dtkinase", "drugban", "graphban", "conplex"]

OUT_DIR = REPO / "results" / "inference" / "committee_aggregation_alts2"
N_BOOT = int(os.environ.get("BENCHMARK_BOOTSTRAP_B", "10000"))
EPS = 1e-6


def load_val_mcc(model: str, corpus: str) -> float:
    scores = []
    for s in SEEDS:
        _, cal_path = model_paths(model, corpus, s)
        cal = json.loads(cal_path.read_text())
        scores.append(float(cal.get("val_score", cal.get("val_mcc", 0.0))))
    return float(np.mean(scores))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def aggregate_weighted_logit(probs: list[np.ndarray], thrs: list[float],
                              val_mccs: list[float]) -> tuple[np.ndarray, float]:
    w = np.clip(np.array(val_mccs, dtype=float), 0.0, None)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()
    z = np.sum(w[:, None] * np.stack([logit(p) for p in probs]), axis=0)
    score = sigmoid(z)
    z_thr = float(np.sum(w * np.array([logit(np.array([t]))[0] for t in thrs])))
    thr = float(sigmoid(np.array([z_thr]))[0])
    return score, thr


def aggregate_median(probs: list[np.ndarray], thrs: list[float]) -> tuple[np.ndarray, float]:
    score = np.median(np.stack(probs), axis=0)
    thr = float(np.median(thrs))
    return score, thr


def aggregate_trimmed_mean(probs: list[np.ndarray],
                            thrs: list[float]) -> tuple[np.ndarray, float]:
    """Mean after dropping element-wise max and min across models."""
    P = np.stack(probs)                 # (M, n)
    sorted_P = np.sort(P, axis=0)
    if sorted_P.shape[0] >= 4:
        middle = sorted_P[1:-1]         # drop top + bottom; for M=4 → mean of 2
    else:
        middle = sorted_P
    score = middle.mean(axis=0)
    sorted_T = np.sort(thrs)
    if len(sorted_T) >= 4:
        thr = float(np.mean(sorted_T[1:-1]))
    else:
        thr = float(np.mean(sorted_T))
    return score, thr


def aggregate_max(probs: list[np.ndarray], thrs: list[float]) -> tuple[np.ndarray, float]:
    score = np.max(np.stack(probs), axis=0)
    thr = float(np.max(thrs))
    return score, thr


def aggregate_harmonic_mean(probs: list[np.ndarray],
                             thrs: list[float]) -> tuple[np.ndarray, float]:
    """n / Σ(1/p_m). Power mean k=-1; conservative — any low-prob model
    drags the score down sharply.
    """
    P = np.stack([np.clip(p, EPS, 1.0) for p in probs])
    score = P.shape[0] / np.sum(1.0 / P, axis=0)
    T = np.array([max(t, EPS) for t in thrs])
    thr = float(len(T) / np.sum(1.0 / T))
    return score, thr


def metric_block(y_true, score, thr):
    pred = (score >= thr).astype(int)
    return {
        "mcc": matthews_corrcoef(y_true, pred),
        "auroc": roc_auc_score(y_true, score),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "n_pos_pred": int(pred.sum()),
    }


def paired_bootstrap_pp(y_true, pred_a, pred_b, blocks, n_boot, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=np.float64)
    if blocks is not None:
        unique_blocks, inverse = np.unique(blocks, return_inverse=True)
        block_to_pairs = [np.where(inverse == k)[0] for k in range(len(unique_blocks))]
    for b in range(n_boot):
        if blocks is None:
            idx = rng.integers(0, n, size=n)
        else:
            block_ids = rng.integers(0, len(unique_blocks), size=len(unique_blocks))
            idx = np.concatenate([block_to_pairs[k] for k in block_ids])
        yt = y_true[idx]
        deltas[b] = _mcc_fast(yt, pred_a[idx]) - _mcc_fast(yt, pred_b[idx])
    return {
        "delta_mean": float(deltas.mean()),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "frac_positive": float((deltas > 0).mean()),
    }


def run_corpus(corpus: str):
    print(f"\n{'='*70}\n  Corpus: {corpus}\n{'='*70}")

    model_probs, model_thrs, y_true = {}, {}, None
    for m in MODELS:
        prob, yt, thr = load_5seed(m, corpus)
        model_probs[m] = prob
        model_thrs[m] = thr
        if y_true is None:
            y_true = yt
        else:
            assert np.array_equal(y_true, yt)

    val_mccs = {m: load_val_mcc(m, corpus) for m in MODELS}
    print(f"  val MCC weights: {dict((k, round(v,4)) for k,v in val_mccs.items())}")

    # Dedupe.
    keys, seq_ids = load_test_keys(corpus)
    y_true_d, seq_ids_d, key_order = None, None, None
    new_probs = {}
    for m in MODELS:
        p_d, y_d, k_d = dedupe_predictions(model_probs[m], y_true, keys)
        new_probs[m] = p_d
        if y_true_d is None:
            y_true_d = y_d
            key_order = k_d
            df = pd.DataFrame({"key": keys, "seq_id": seq_ids})
            seq_map = df.drop_duplicates("key").set_index("key")["seq_id"]
            seq_ids_d = seq_map.loc[k_d].to_numpy()
        else:
            assert np.array_equal(y_true_d, y_d)
    print(f"  dedupe: n {len(y_true)} → {len(y_true_d)}")
    y_true = y_true_d
    seq_ids = seq_ids_d
    model_probs = new_probs

    probs_list = [model_probs[m] for m in MODELS]
    thrs_list  = [model_thrs[m]  for m in MODELS]
    val_list   = [val_mccs[m]    for m in MODELS]

    # Build all systems.
    systems = {}
    soft_score = np.mean(np.stack(probs_list), axis=0)
    soft_thr = float(np.mean(thrs_list))
    systems["soft_mean"] = (soft_score, soft_thr)

    s, t = aggregate_weighted_logit(probs_list, thrs_list, val_list)
    systems["weighted_logit"] = (s, t)

    s, t = aggregate_median(probs_list, thrs_list)
    systems["median"] = (s, t)

    s, t = aggregate_trimmed_mean(probs_list, thrs_list)
    systems["trimmed_mean"] = (s, t)

    s, t = aggregate_max(probs_list, thrs_list)
    systems["max"] = (s, t)

    s, t = aggregate_harmonic_mean(probs_list, thrs_list)
    systems["harmonic_mean"] = (s, t)

    rows = []
    for m in MODELS:
        rows.append({"system": m, **metric_block(y_true, model_probs[m], model_thrs[m])})
    for name, (s, t) in systems.items():
        rows.append({"system": name, **metric_block(y_true, s, t)})
    metrics_df = pd.DataFrame(rows).set_index("system")
    print("\nMetrics:")
    print(metrics_df.round(4).to_string())

    soft_pred = (soft_score >= soft_thr).astype(int)
    boot_rows = []
    print(f"\nPaired block-bootstrap (B={N_BOOT}, by protein):")
    for name, (s, t) in systems.items():
        if name == "soft_mean":
            continue
        pred = (s >= t).astype(int)
        d = paired_bootstrap_pp(y_true, pred, soft_pred, seq_ids, N_BOOT)
        verdict = (f"{name} leads ▲" if d["ci_lo"] > 0 else
                   "soft_mean leads ▼" if d["ci_hi"] < 0 else "indistinguishable ⊘")
        boot_rows.append({"comparison": f"{name} - soft_mean", **d, "verdict": verdict})
        print(f"  {name:18s}: Δ={d['delta_mean']:+.4f}  "
              f"CI95=[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  "
              f"P(Δ>0)={d['frac_positive']:.3f}  {verdict}")

    boot_df = pd.DataFrame(boot_rows)
    diag = {"n_dedup": int(len(y_true)),
            "n_proteins": int(len(np.unique(seq_ids))),
            "val_mccs": val_mccs}
    return metrics_df, boot_df, diag


def _md_table(df):
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_all, boot_all, diag_all = {}, {}, {}
    for c in CORPORA:
        m, b, d = run_corpus(c)
        m.to_csv(OUT_DIR / f"{c}_metrics.csv")
        b.to_csv(OUT_DIR / f"{c}_paired_bootstrap.csv", index=False)
        metrics_all[c], boot_all[c], diag_all[c] = m, b, d

    lines = ["# Aggregation alternatives (batch 2): weighted_logit, median, "
             "trimmed_mean, max, harmonic_mean", ""]
    lines.append(f"**Bootstrap**: B={N_BOOT}, block-by-protein, IC95 percentile.")
    lines.append("")
    for c in CORPORA:
        d = diag_all[c]
        lines.append(f"## {c}")
        lines.append(f"n_dedup={d['n_dedup']}, n_proteins={d['n_proteins']}.")
        lines.append("Val MCC weights: " +
                     ", ".join(f"{k}={v:.4f}" for k, v in d['val_mccs'].items()))
        lines.append("")
        lines.append("### Per-system metrics")
        lines.append(_md_table(metrics_all[c].round(4).reset_index()))
        lines.append("")
        lines.append("### Paired block-bootstrap vs soft_mean")
        lines.append(_md_table(boot_all[c].round(4)))
        lines.append("")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"\n→ wrote {OUT_DIR}/REPORT.md")


if __name__ == "__main__":
    main()
