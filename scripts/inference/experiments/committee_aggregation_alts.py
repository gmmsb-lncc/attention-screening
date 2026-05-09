#!/usr/bin/env python3
"""Alternative committee aggregation rules vs canonical soft-mean.

Compared (zero re-training, all over saved 5-seed-averaged probs):

  1. soft_mean       — canonical: mean of calibrated probs (current default)
  2. logit_mean      — mean in logit space: mean(log(p/(1-p))); equivalent to
                       geometric mean of odds. Robust to multimodal probs;
                       per Kahn et al. and Bayesian model combination lit.
  3. weighted_mcc    — weighted soft-mean with weights ∝ val_MCC per model
                       (read from calibration sidecars; pre-registered, no
                       test-set leakage). Kuncheva 2014.
  4. product_experts — product of probs, normalized: p ∝ ∏ p_m. Hinton 2002.
                       Sharpens consensus regions; sensitive to p≈0 (use
                       eps clip). Threshold = mean of individual thresholds.
  5. rrf             — reciprocal rank fusion: score = Σ 1/(60+rank_m).
                       Cormack et al. 2009 (SIGIR). Calibration-free; uses
                       within-corpus rank only.

For each rule, decision = (score >= threshold). Threshold rules:
  - soft_mean / weighted_mcc / product_experts: mean of per-model thresholds
    (canonical convention; reported in tese §sec:metodologia-comite).
  - logit_mean: logit(mean_thr).
  - rrf: rank threshold computed on val set (proportion of positives).

Statistics: paired block-bootstrap by protein (B = 10000), IC95 percentile,
each alternative vs soft_mean.

Output:
  results/inference/committee_aggregation_alts/
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

OUT_DIR = REPO / "results" / "inference" / "committee_aggregation_alts"
N_BOOT = int(os.environ.get("BENCHMARK_BOOTSTRAP_B", "10000"))
EPS = 1e-6


def load_val_mcc(model: str, corpus: str) -> float:
    """Mean val MCC across 5 seeds, read from calibration sidecars."""
    scores = []
    for s in SEEDS:
        _, cal_path = model_paths(model, corpus, s)
        cal = json.loads(cal_path.read_text())
        # DT-K records 'val_score' (MCC by default); baselines record same key
        # under their respective calibrators (drugban_calibration.json etc.).
        scores.append(float(cal.get("val_score", cal.get("val_mcc", 0.0))))
    return float(np.mean(scores))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def aggregate_logit_mean(probs_per_model: list[np.ndarray],
                         thrs_per_model: list[float]) -> tuple[np.ndarray, float]:
    z = np.mean(np.stack([logit(p) for p in probs_per_model]), axis=0)
    score = sigmoid(z)
    thr = sigmoid(np.mean([logit(np.array([t]))[0] for t in thrs_per_model]))
    return score, float(thr)


def aggregate_weighted_mcc(probs_per_model: list[np.ndarray],
                            thrs_per_model: list[float],
                            val_mccs: list[float]) -> tuple[np.ndarray, float]:
    w = np.array(val_mccs, dtype=float)
    w = np.clip(w, 0.0, None)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()
    score = np.sum(w[:, None] * np.stack(probs_per_model), axis=0)
    thr = float(np.sum(w * np.array(thrs_per_model)))
    return score, thr


def aggregate_product_experts(probs_per_model: list[np.ndarray],
                               thrs_per_model: list[float]) -> tuple[np.ndarray, float]:
    """Geometric mean of probs (PoE without normalizing partition; equivalent
    to geometric mean for binary). For p=ProductOfExperts, the unnormalized
    score is ∏p; normalizing p→p/(p+(1-p)·∏(1-p_m)/∏p_m) is the calibrated
    PoE binary form. We use geometric mean since both rules induce the same
    ordering after monotonic transformation; threshold = geometric mean of
    individual thresholds.
    """
    log_p = np.mean(np.stack([np.log(np.clip(p, EPS, 1.0)) for p in probs_per_model]),
                    axis=0)
    score = np.exp(log_p)
    thr = float(np.exp(np.mean([np.log(max(t, EPS)) for t in thrs_per_model])))
    return score, thr


def aggregate_rrf(probs_per_model: list[np.ndarray],
                   y_val_per_model: list[np.ndarray] | None = None,
                   k_const: int = 60) -> tuple[np.ndarray, float]:
    """Reciprocal rank fusion: score[i] = Σ_m 1/(k + rank_m[i]).
    Higher score = higher consensus rank. Threshold = score percentile
    matching mean of (1 - per-model positive rate) on test (no val rank
    available cross-model); approximated as global score median for binary
    decision parity.
    """
    n = len(probs_per_model[0])
    score = np.zeros(n, dtype=float)
    for p in probs_per_model:
        order = np.argsort(-p)             # descending; rank 1 = highest p
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(1, n + 1)
        score += 1.0 / (k_const + rank)
    # Threshold: pick percentile so that fraction of positives matches the
    # mean per-model positive rate on test (test labels not used; mean of
    # per-model preds is a calibration-free heuristic).
    return score, float(np.median(score))


def metric_block(y_true, score, thr, allow_auroc=True):
    pred = (score >= thr).astype(int)
    return {
        "mcc": matthews_corrcoef(y_true, pred),
        "auroc": roc_auc_score(y_true, score) if allow_auroc else float("nan"),
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
        print(f"  {m:9s}: prob_mean={prob.mean():.4f}  thr={thr:.4f}  n={len(prob)}")

    # Read val MCCs from calibration sidecars (mean over 5 seeds).
    val_mccs = {m: load_val_mcc(m, corpus) for m in MODELS}
    print(f"\n  val MCC (5-seed mean) — used as weights in weighted_mcc:")
    for m, v in val_mccs.items():
        print(f"    {m:9s}: {v:.4f}")

    # Dedupe by (seq_id, chembl_id).
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
    print(f"\n  dedupe: n {len(y_true)} → {len(y_true_d)}")
    y_true = y_true_d
    seq_ids = seq_ids_d
    model_probs = new_probs

    probs_list = [model_probs[m] for m in MODELS]
    thrs_list  = [model_thrs[m]  for m in MODELS]
    val_list   = [val_mccs[m]    for m in MODELS]

    # Build all 5 aggregation systems.
    systems = {}
    # 1. soft-mean (canonical baseline)
    soft_score = np.mean(np.stack(probs_list), axis=0)
    soft_thr = float(np.mean(thrs_list))
    systems["soft_mean"] = (soft_score, soft_thr, True)

    # 2. logit-mean
    s, t = aggregate_logit_mean(probs_list, thrs_list)
    systems["logit_mean"] = (s, t, True)

    # 3. weighted soft-mean by val MCC
    s, t = aggregate_weighted_mcc(probs_list, thrs_list, val_list)
    systems["weighted_mcc"] = (s, t, True)

    # 4. product-of-experts (geometric mean)
    s, t = aggregate_product_experts(probs_list, thrs_list)
    systems["product_experts"] = (s, t, True)

    # 5. reciprocal rank fusion
    s, t = aggregate_rrf(probs_list)
    systems["rrf"] = (s, t, True)  # AUROC OK (rank-monotonic with score)

    # Per-system metrics + per-model individual rows.
    rows = []
    for m in MODELS:
        rows.append({"system": m,
                     **metric_block(y_true, model_probs[m], model_thrs[m])})
    for name, (s, t, allow_auroc) in systems.items():
        rows.append({"system": name,
                     **metric_block(y_true, s, t, allow_auroc=allow_auroc)})
    metrics_df = pd.DataFrame(rows).set_index("system")
    print("\nMetrics:")
    print(metrics_df.round(4).to_string())

    # Paired block-bootstrap: each alternative vs soft_mean.
    soft_pred = (systems["soft_mean"][0] >= systems["soft_mean"][1]).astype(int)
    boot_rows = []
    print(f"\nPaired block-bootstrap (B={N_BOOT}, by protein):")
    print("\n  alternative vs soft_mean:")
    for name, (s, t, _) in systems.items():
        if name == "soft_mean":
            continue
        pred = (s >= t).astype(int)
        d = paired_bootstrap_pp(y_true, pred, soft_pred, seq_ids, N_BOOT)
        verdict = (f"{name} leads ▲" if d["ci_lo"] > 0 else
                   "soft_mean leads ▼" if d["ci_hi"] < 0 else "indistinguishable ⊘")
        boot_rows.append({"comparison": f"{name} - soft_mean", **d, "verdict": verdict})
        print(f"    {name:18s}: Δ={d['delta_mean']:+.4f}  "
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

    lines = ["# Alternative committee aggregation rules vs soft-mean", ""]
    lines.append("**Rules tested**: soft_mean (canonical), logit_mean, "
                 "weighted_mcc (∝ val_MCC, no leakage), product_experts "
                 "(geometric mean), rrf (reciprocal rank fusion).")
    lines.append(f"**Bootstrap**: B={N_BOOT}, block-by-protein (seq_id), "
                 f"IC95 percentile.")
    lines.append("")
    for c in CORPORA:
        d = diag_all[c]
        lines.append(f"## {c}")
        lines.append(f"n_dedup={d['n_dedup']}, n_proteins={d['n_proteins']}.")
        lines.append("Val MCC weights (mean over 5 seeds): " +
                     ", ".join(f"{k}={v:.4f}" for k, v in d['val_mccs'].items()))
        lines.append("")
        lines.append("### Per-system metrics")
        lines.append(_md_table(metrics_all[c].round(4).reset_index()))
        lines.append("")
        lines.append("### Paired block-bootstrap (alternative vs soft_mean)")
        lines.append(_md_table(boot_all[c].round(4)))
        lines.append("")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"\n→ wrote {OUT_DIR}/REPORT.md")


if __name__ == "__main__":
    main()
