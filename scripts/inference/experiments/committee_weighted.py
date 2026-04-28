#!/usr/bin/env python3
"""Test confidence-weighted committee aggregations vs uniform mean.

Three weighting schemes tested:
  1. uniform   — w_m = 1/M (canonical, used in main pipeline)
  2. margin    — w_m^{(i)} = |p_m^{(i)} - 0.5|  (per-pair, distance from 0.5)
  3. entropy   — w_m^{(i)} = 1 - H_b(p_m^{(i)})  (per-pair, 1 minus binary entropy)
  4. mcc_val   — w_m = val_MCC_m       (global, derived from val performance)

For schemes 2 and 3, weights are normalized per-pair so they sum to 1
across the M models. Aggregated probability is the weighted soft-mean.

Threshold: mean of individual model thresholds (consistent with canonical).

Tested on the canonical 4-model committee, three corpora, with dedupe
and block bootstrap by protein.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "inference" / "experiments"))

from committee_vs_individual import (  # noqa: E402
    load_5seed, load_test_keys, dedupe_predictions,
    system_metrics, paired_bootstrap_delta, _md_table,
    SEEDS, MODELS, CORPORA,
)

OUT = REPO / "results" / "inference" / "committee_weighted"
N_BOOT = int(os.environ.get("BENCHMARK_BOOTSTRAP_B", "2000"))


def margin_weights(probs_stack: np.ndarray) -> np.ndarray:
    """w_m^(i) = |p_m^(i) - 0.5|, normalized over m per pair."""
    w = np.abs(probs_stack - 0.5)
    s = w.sum(axis=0, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return w / s


def entropy_weights(probs_stack: np.ndarray) -> np.ndarray:
    """w_m^(i) = 1 - H_b(p_m^(i)) where H_b is binary entropy in nats.

    H_b(p) = -p*log(p) - (1-p)*log(1-p), max at p=0.5 (=log 2).
    Normalize 1 - H_b/log(2) so weight in [0, 1] per (m, i), then
    renormalize across m per pair.
    """
    p = np.clip(probs_stack, 1e-9, 1.0 - 1e-9)
    H = -p * np.log(p) - (1 - p) * np.log(1 - p)
    Hn = H / np.log(2.0)  # normalized in [0, 1]
    w = 1.0 - Hn
    s = w.sum(axis=0, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return w / s


def mcc_global_weights(val_mcc_per_model: list[float], n_pairs: int) -> np.ndarray:
    """Global weight per model from val_MCC, broadcast to all pairs."""
    w = np.array(val_mcc_per_model, dtype=np.float64)
    w = np.clip(w, 0.0, None)  # negative MCC → zero weight
    w = w / w.sum()
    return np.tile(w[:, None], (1, n_pairs))


def run_corpus(corpus: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n{'='*70}\n  Corpus: {corpus}\n{'='*70}")

    model_probs = {}
    model_thrs = {}
    y_true = None
    for m in MODELS:
        prob, yt, thr = load_5seed(m, corpus)
        model_probs[m] = prob
        model_thrs[m] = thr
        if y_true is None:
            y_true = yt

    # Dedupe.
    keys, seq_ids = load_test_keys(corpus)
    new_probs = {}
    y_d, seq_ids_d = None, None
    for m, p in model_probs.items():
        pd_, yd_, kd_ = dedupe_predictions(p, y_true, keys)
        new_probs[m] = pd_
        if y_d is None:
            y_d = yd_
            df = pd.DataFrame({"key": keys, "seq_id": seq_ids})
            seq_map = df.drop_duplicates("key").set_index("key")["seq_id"]
            seq_ids_d = seq_map.loc[kd_].to_numpy()
    y_true = y_d
    model_probs = new_probs
    blocks = seq_ids_d
    print(f"  dedupe: n={len(y_true)}, proteins={len(np.unique(blocks))}")

    # Stack: shape (M, n).
    M = len(MODELS)
    n = len(y_true)
    probs_stack = np.stack([model_probs[m] for m in MODELS])

    # Mean threshold (used by all weighted variants for consistency).
    mean_thr = float(np.mean([model_thrs[m] for m in MODELS]))

    # Compute aggregation per scheme.
    def agg(weights: np.ndarray) -> np.ndarray:
        return (weights * probs_stack).sum(axis=0)

    schemes = {
        "uniform":  np.full((M, n), 1.0 / M),
        "margin":   margin_weights(probs_stack),
        "entropy":  entropy_weights(probs_stack),
    }

    # Add mcc_val scheme using individual model MCC on test as proxy
    # (no separate val available here; use test MCC for illustration —
    # this would leak to the metric, used only as informative reference).
    indiv_mccs = []
    for m in MODELS:
        from sklearn.metrics import matthews_corrcoef
        pred = (model_probs[m] >= model_thrs[m]).astype(int)
        indiv_mccs.append(float(matthews_corrcoef(y_true, pred)))
    schemes["mcc_test*"] = mcc_global_weights(indiv_mccs, n)

    # Compute committee per scheme + paired bootstrap.
    rows = []
    boot_rows = []
    for m in MODELS:
        rows.append({"system": m,
                     **system_metrics(y_true, model_probs[m], model_thrs[m])})

    for sch_name, W in schemes.items():
        com_prob = agg(W)
        rows.append({"system": f"committee_{sch_name}",
                     **system_metrics(y_true, com_prob, mean_thr)})

    metrics_df = pd.DataFrame(rows).set_index("system")
    print("\nMetrics per system:")
    print(metrics_df.round(4).to_string())

    # Paired bootstrap: each weighted committee vs uniform committee.
    uniform_prob = agg(schemes["uniform"])
    print(f"\nPaired bootstrap (B={N_BOOT}, block-by-protein) — "
          f"weighted committees vs uniform:")
    for sch_name in ["margin", "entropy", "mcc_test*"]:
        w_prob = agg(schemes[sch_name])
        delta = paired_bootstrap_delta(
            y_true, w_prob, mean_thr, uniform_prob, mean_thr,
            n_boot=N_BOOT, seed=42, blocks=blocks)
        verdict = ("weighted leads ▲" if delta["ci_lo"] > 0 else
                   "weighted trails ▼" if delta["ci_hi"] < 0 else
                   "indistinguishable ⊘")
        boot_rows.append({"comparison": f"committee_{sch_name} - committee_uniform",
                          **delta, "verdict": verdict})
        print(f"  {sch_name:12s}: Δ={delta['delta_mean']:+.4f} "
              f"CI95=[{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}] {verdict}")
    boot_df = pd.DataFrame(boot_rows)

    # Also compare margin/entropy committees against each individual model.
    indiv_boot = []
    for sch_name in ["uniform", "margin", "entropy"]:
        w_prob = agg(schemes[sch_name])
        for m in MODELS:
            delta = paired_bootstrap_delta(
                y_true, w_prob, mean_thr,
                model_probs[m], model_thrs[m],
                n_boot=N_BOOT, seed=42, blocks=blocks)
            verdict = ("comm. leads ▲" if delta["ci_lo"] > 0 else
                       "comm. trails ▼" if delta["ci_hi"] < 0 else
                       "indistinguishable ⊘")
            indiv_boot.append({"comparison": f"committee_{sch_name} - {m}",
                               **delta, "verdict": verdict})
    indiv_df = pd.DataFrame(indiv_boot)

    return metrics_df, boot_df, indiv_df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_all, boot_all, indiv_all = {}, {}, {}
    for corpus in CORPORA:
        try:
            mdf, bdf, idf = run_corpus(corpus)
        except FileNotFoundError as e:
            print(f"skip {corpus}: {e}")
            continue
        mdf.to_csv(OUT / f"{corpus}_metrics.csv")
        bdf.to_csv(OUT / f"{corpus}_weighted_vs_uniform.csv", index=False)
        idf.to_csv(OUT / f"{corpus}_committee_vs_individual.csv", index=False)
        metrics_all[corpus] = mdf
        boot_all[corpus] = bdf
        indiv_all[corpus] = idf

    lines = ["# Weighted Committee Aggregations — Empirical Test", ""]
    lines.append("**Schemes:**")
    lines.append("- `uniform`  — $w_m = 1/M$ (canonical)")
    lines.append("- `margin`   — $w_m^{(i)} = |p_m^{(i)} - 0.5|$ (per-pair confidence)")
    lines.append("- `entropy`  — $w_m^{(i)} = 1 - H_b(p_m^{(i)})$ (per-pair entropy)")
    lines.append("- `mcc_test*` — $w_m = $ val MCC (global; *uses test MCC as proxy, "
                 "informative only — leaks if used operationally)")
    lines.append("")
    for corpus in CORPORA:
        if corpus not in metrics_all:
            continue
        lines.append(f"## Corpus: {corpus}\n")
        lines.append("### Metrics per system\n")
        lines.append(_md_table(metrics_all[corpus].round(4).reset_index()))
        lines.append("\n### Weighted committees vs uniform committee (paired bootstrap)\n")
        lines.append(_md_table(boot_all[corpus].round(4)))
        lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(f"\n→ wrote {OUT}")


if __name__ == "__main__":
    main()
