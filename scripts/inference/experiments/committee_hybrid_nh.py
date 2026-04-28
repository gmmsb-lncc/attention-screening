#!/usr/bin/env python3
"""Hybrid Non-Human committee: cross-corpus baselines + in-domain DT-Kinase.

Hypothesis: the Non-Human test set has only 1702 pairs vs 41441 in `all`,
so models trained on `all` (24× more data, including all NH pairs) and
evaluated on the NH test split should outperform their NH-only-trained
counterparts when measured on NH test (Anexo A cross-matrix shows this
holds for off-diagonal evaluations).

This script builds an NH committee with:
  - DT-Kinase: NH-trained (canonical; no cross-matrix npz available — DT-K
    cross-matrix metrics.json has raw_logits=null for all seeds)
  - DrugBAN:   all-trained, evaluated on NH test (cross-matrix all_to_non_human)
  - GraphBAN:  all-trained, evaluated on NH test (cross-matrix all_to_non_human)
  - ConPLex:   all-trained, evaluated on NH test (cross-matrix all_to_non_human)

Threshold per baseline model is derived on-the-fly from val_y_prob /
val_y_true (universal_val, 69401 pairs) using each model's native
protocol: F1-opt for DrugBAN/GraphBAN (F1-optimal threshold), MCC-opt
for ConPLex (consistent with its native calibration).

DT-Kinase predictions and threshold are reused from the canonical
in-domain inference (results/benchmark_non_human_8M_13_05_2026_v3/...).

Pipeline same as committee_vs_individual.py: dedupe, block-bootstrap by
protein, B=2000, paired bootstrap delta.

Output:
  results/inference/committee_hybrid_nh/{metrics,paired_bootstrap}.csv
  results/inference/committee_hybrid_nh/REPORT.md
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

REPO = Path(__file__).resolve().parents[3]
SEEDS = [42, 123, 456, 789, 1024]
CORPUS = "non_human"
OUT_DIR = REPO / "results" / "inference" / "committee_hybrid_nh"
N_BOOT = int(os.environ.get("BENCHMARK_BOOTSTRAP_B", "2000"))


def _mcc_fast(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    num = tp * tn - fp * fn
    den_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den_sq == 0:
        return 0.0
    return float(num / np.sqrt(float(den_sq)))


def _f1_fast(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def find_threshold(y_true, prob, metric="f1", grid_n=200):
    grid = np.linspace(0.05, 0.95, grid_n)
    if metric == "f1":
        scores = np.array([_f1_fast(y_true, (prob >= t).astype(int)) for t in grid])
    else:
        scores = np.array([_mcc_fast(y_true, (prob >= t).astype(int)) for t in grid])
    return float(grid[scores.argmax()])


def load_baseline_5seed_cross(model: str, threshold_metric: str):
    """Load 5-seed cross-matrix predictions for `all_to_non_human` cell."""
    base = REPO / "results" / "cross_matrix" / model / "all_to_non_human"
    test_probs, val_probs = [], []
    test_y, val_y = None, None
    thrs = []
    for s in SEEDS:
        p = base / f"seed_{s}" / "raw_predictions.npz"
        if not p.exists():
            sys.exit(f"missing: {p}")
        d = np.load(p)
        tp = d["test_y_prob"].astype(np.float64)
        ty = d["test_y_true"].astype(int)
        vp = d["val_y_prob"].astype(np.float64)
        vy = d["val_y_true"].astype(int)
        test_probs.append(tp)
        val_probs.append(vp)
        if test_y is None:
            test_y, val_y = ty, vy
        thrs.append(find_threshold(vy, vp, metric=threshold_metric))
    test_prob_mean = np.mean(np.stack(test_probs), axis=0)
    val_prob_mean = np.mean(np.stack(val_probs), axis=0)
    thr_mean = float(np.mean(thrs))
    return test_prob_mean, test_y, val_prob_mean, val_y, thr_mean


def load_dtkinase_5seed_indomain():
    """Load DT-Kinase NH-trained 5-seed predictions (in-domain canonical)."""
    base = REPO / "results" / "benchmark_non_human_8M_13_05_2026_v3" \
                / "test" / "level4_cnn_8M" / "non_human"
    probs, thrs = [], []
    y = None
    for s in SEEDS:
        npz = base / f"seed_{s}" / "raw_predictions.npz"
        cal = base / f"seed_{s}" / "level4_cnn_calibration.json"
        d = np.load(npz)
        probs.append(d["y_prob"].astype(np.float64))
        if y is None:
            y = d["y_true"].astype(int)
        c = json.loads(cal.read_text())
        thrs.append(float(c["threshold"]))
    return np.mean(np.stack(probs), axis=0), y, float(np.mean(thrs))


def load_test_keys(corpus: str):
    tsv = REPO / "scaffolds_splits" / "output" / f"{corpus}_test.tsv"
    df = pd.read_csv(tsv, sep="\t", usecols=["chembl_id", "seq_id"])
    keys = (df["seq_id"].astype(str) + "__" + df["chembl_id"].astype(str)).to_numpy()
    return keys, df["seq_id"].astype(str).to_numpy()


def dedupe(prob, y, keys):
    df = pd.DataFrame({"key": keys, "prob": prob, "y": y})
    g = df.groupby("key", sort=True, as_index=False).agg(
        prob=("prob", "mean"), y=("y", "max"))
    return g["prob"].to_numpy(), g["y"].to_numpy().astype(int), g["key"].to_numpy()


def system_metrics(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "mcc": matthews_corrcoef(y_true, pred),
        "auroc": roc_auc_score(y_true, prob),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
    }


def paired_bootstrap_delta(y_true, prob_a, thr_a, prob_b, thr_b,
                            n_boot, seed, blocks):
    rng = np.random.default_rng(seed)
    pred_a = (prob_a >= thr_a).astype(int)
    pred_b = (prob_b >= thr_b).astype(int)
    deltas = np.empty(n_boot)
    unique_blocks, inverse = np.unique(blocks, return_inverse=True)
    block_to_pairs = [np.where(inverse == k)[0] for k in range(len(unique_blocks))]
    for b in range(n_boot):
        block_ids = rng.integers(0, len(unique_blocks), size=len(unique_blocks))
        idx = np.concatenate([block_to_pairs[k] for k in block_ids])
        yt = y_true[idx]
        deltas[b] = _mcc_fast(yt, pred_a[idx]) - _mcc_fast(yt, pred_b[idx])
        if (b + 1) % 500 == 0:
            print(f"      boot {b+1}/{n_boot} (Δ={deltas[:b+1].mean():+.4f})",
                  flush=True)
    return {
        "delta_mean": float(deltas.mean()),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "frac_positive": float((deltas > 0).mean()),
    }


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  Hybrid NH committee — cross-corpus baselines + in-domain DT-K\n{'='*70}")

    # 1. Load each model's 5-seed predictions on NH test set.
    print("\nLoading models:")
    dtk_prob, y_true, dtk_thr = load_dtkinase_5seed_indomain()
    print(f"  dtkinase  (NH-trained, in-domain):    n={len(dtk_prob)}, thr={dtk_thr:.4f}")
    drug_prob, drug_y, drug_val_prob, drug_val_y, drug_thr = \
        load_baseline_5seed_cross("drugban", "f1")
    print(f"  drugban   (all-trained, all→NH):      n={len(drug_prob)}, thr_F1={drug_thr:.4f}")
    grph_prob, grph_y, grph_val_prob, grph_val_y, grph_thr = \
        load_baseline_5seed_cross("graphban", "f1")
    print(f"  graphban  (all-trained, all→NH):      n={len(grph_prob)}, thr_F1={grph_thr:.4f}")
    conp_prob, conp_y, conp_val_prob, conp_val_y, conp_thr = \
        load_baseline_5seed_cross("conplex", "mcc")
    print(f"  conplex   (all-trained, all→NH):      n={len(conp_prob)}, thr_MCC={conp_thr:.4f}")

    # Sanity: y_true must match across all 4 sources for the NH test set.
    assert np.array_equal(y_true, drug_y), "y mismatch dtk vs drugban"
    assert np.array_equal(y_true, grph_y), "y mismatch dtk vs graphban"
    assert np.array_equal(y_true, conp_y), "y mismatch dtk vs conplex"

    model_probs = {"dtkinase": dtk_prob, "drugban": drug_prob,
                   "graphban": grph_prob, "conplex": conp_prob}
    model_thrs = {"dtkinase": dtk_thr, "drugban": drug_thr,
                  "graphban": grph_thr, "conplex": conp_thr}

    # 2. Dedupe + block bootstrap setup.
    keys, seq_ids = load_test_keys(CORPUS)
    assert len(keys) == len(y_true)

    new_probs = {}
    y_d = None
    for m, p in model_probs.items():
        pd_, yd_, kd_ = dedupe(p, y_true, keys)
        new_probs[m] = pd_
        if y_d is None:
            y_d = yd_
            df = pd.DataFrame({"key": keys, "seq_id": seq_ids})
            seq_map = df.drop_duplicates("key").set_index("key")["seq_id"]
            seq_ids_d = seq_map.loc[kd_].to_numpy()
    print(f"\n  dedupe: n {len(y_true)} → {len(y_d)} "
          f"({100*(len(y_true)-len(y_d))/len(y_true):.1f}% collapsed)")
    print(f"  block bootstrap: {len(np.unique(seq_ids_d))} unique proteins")
    y_true = y_d
    model_probs = new_probs
    blocks = seq_ids_d

    # 3. Build committees.
    MODELS_4 = ["dtkinase", "drugban", "graphban", "conplex"]
    com_prob = np.mean(np.stack([model_probs[m] for m in MODELS_4]), axis=0)
    com_thr = float(np.mean([model_thrs[m] for m in MODELS_4]))
    print(f"\n  committee (4 hybrid): prob_mean={com_prob.mean():.4f} thr={com_thr:.4f}")

    # 4. Compute metrics.
    rows = []
    for m in MODELS_4:
        rows.append({"system": f"{m}_hybrid", **system_metrics(y_true, model_probs[m], model_thrs[m])})
    rows.append({"system": "committee_hybrid", **system_metrics(y_true, com_prob, com_thr)})
    metrics_df = pd.DataFrame(rows).set_index("system")
    print("\nMetrics:")
    print(metrics_df.round(4).to_string())

    # 5. Paired bootstrap.
    print(f"\nPaired bootstrap (B={N_BOOT}, block-by-protein):")
    boot_rows = []
    for m in MODELS_4:
        delta = paired_bootstrap_delta(
            y_true, com_prob, com_thr,
            model_probs[m], model_thrs[m],
            n_boot=N_BOOT, seed=42, blocks=blocks)
        verdict = ("committee leads ▲" if delta["ci_lo"] > 0 else
                   "committee trails ▼" if delta["ci_hi"] < 0 else
                   "indistinguishable ⊘")
        boot_rows.append({"comparison": f"committee_hybrid - {m}_hybrid",
                          **delta, "verdict": verdict})
        print(f"  vs {m:9s}: Δ={delta['delta_mean']:+.4f} "
              f"CI95=[{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}] {verdict}")
    boot_df = pd.DataFrame(boot_rows)

    # 6. Save outputs.
    metrics_df.to_csv(OUT_DIR / "metrics.csv")
    boot_df.to_csv(OUT_DIR / "paired_bootstrap.csv", index=False)

    lines = ["# Hybrid NH Committee — Cross-Corpus Baselines + In-Domain DT-Kinase", ""]
    lines.append("**Setup**: NH test set evaluated by:")
    lines.append("- DT-Kinase: NH-trained (in-domain canonical)")
    lines.append("- DrugBAN:   all-trained, evaluated on NH test (cross-matrix all→NH)")
    lines.append("- GraphBAN:  all-trained, evaluated on NH test (cross-matrix all→NH)")
    lines.append("- ConPLex:   all-trained, evaluated on NH test (cross-matrix all→NH)")
    lines.append("")
    lines.append(f"**Protocol**: 5-seed mean prob; threshold derived per-seed on universal_val "
                 f"(F1-opt for DrugBAN/GraphBAN, MCC-opt for ConPLex, then averaged); "
                 f"committee = 4-model mean. Dedupe ON, block bootstrap by protein, B={N_BOOT}.")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(_md_table(metrics_df.round(4).reset_index()))
    lines.append("")
    lines.append("## Paired bootstrap (committee_hybrid vs each member)")
    lines.append("")
    lines.append(_md_table(boot_df.round(4)))
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"\n  → wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
