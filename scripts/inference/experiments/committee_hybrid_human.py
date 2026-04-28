#!/usr/bin/env python3
"""Hybrid Human committee: cross-corpus baselines + in-domain DT-Kinase.

Same pattern as committee_hybrid_nh.py but for the Human test set.
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

from committee_hybrid_nh import (  # noqa: E402
    load_baseline_5seed_cross as _load_cross_nh,  # path-pattern-locked to all_to_non_human
    load_test_keys, dedupe, system_metrics, paired_bootstrap_delta,
    _md_table, find_threshold,
)

SEEDS = [42, 123, 456, 789, 1024]
N_BOOT = int(os.environ.get("BENCHMARK_BOOTSTRAP_B", "2000"))
OUT = REPO / "results" / "inference" / "committee_hybrid_human"


def load_baseline_5seed_cross_human(model: str, threshold_metric: str):
    """Same as load_baseline_5seed_cross but for all_to_human cell."""
    base = REPO / "results" / "cross_matrix" / model / "all_to_human"
    test_probs, val_probs, thrs = [], [], []
    test_y, val_y = None, None
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
    return (np.mean(np.stack(test_probs), axis=0), test_y,
            np.mean(np.stack(val_probs), axis=0), val_y,
            float(np.mean(thrs)))


def load_dtkinase_5seed_human():
    base = REPO / "results" / "benchmark_human_8M_13_05_2026" \
                / "test" / "level4_cnn_8M" / "human"
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  Hybrid Human committee\n{'='*70}")

    dtk_prob, y_true, dtk_thr = load_dtkinase_5seed_human()
    print(f"  dtkinase  (Human-trained): n={len(dtk_prob)}, thr={dtk_thr:.4f}")
    drug_prob, drug_y, _, _, drug_thr = load_baseline_5seed_cross_human("drugban", "f1")
    grph_prob, grph_y, _, _, grph_thr = load_baseline_5seed_cross_human("graphban", "f1")
    conp_prob, conp_y, _, _, conp_thr = load_baseline_5seed_cross_human("conplex", "mcc")
    print(f"  drugban   (all→Human):     thr_F1={drug_thr:.4f}")
    print(f"  graphban  (all→Human):     thr_F1={grph_thr:.4f}")
    print(f"  conplex   (all→Human):     thr_MCC={conp_thr:.4f}")

    assert np.array_equal(y_true, drug_y)
    assert np.array_equal(y_true, grph_y)
    assert np.array_equal(y_true, conp_y)

    mp = {"dtkinase": dtk_prob, "drugban": drug_prob,
          "graphban": grph_prob, "conplex": conp_prob}
    mt = {"dtkinase": dtk_thr, "drugban": drug_thr,
          "graphban": grph_thr, "conplex": conp_thr}

    keys, seq_ids = load_test_keys("human")
    new_mp, y_d, seq_ids_d = {}, None, None
    for m, p in mp.items():
        pd_, yd_, kd_ = dedupe(p, y_true, keys)
        new_mp[m] = pd_
        if y_d is None:
            y_d = yd_
            df = pd.DataFrame({"key": keys, "seq_id": seq_ids})
            seq_map = df.drop_duplicates("key").set_index("key")["seq_id"]
            seq_ids_d = seq_map.loc[kd_].to_numpy()
    y_true, mp, blocks = y_d, new_mp, seq_ids_d
    print(f"  dedupe: n={len(y_true)}, proteins={len(np.unique(blocks))}")

    MOD = ["dtkinase", "drugban", "graphban", "conplex"]
    com_prob = np.mean(np.stack([mp[m] for m in MOD]), axis=0)
    com_thr = float(np.mean([mt[m] for m in MOD]))

    rows = [{"system": f"{m}_hybrid", **system_metrics(y_true, mp[m], mt[m])} for m in MOD]
    rows.append({"system": "committee_hybrid", **system_metrics(y_true, com_prob, com_thr)})
    metrics_df = pd.DataFrame(rows).set_index("system")
    print("\nMetrics:")
    print(metrics_df.round(4).to_string())

    print(f"\nPaired bootstrap (B={N_BOOT}, block-by-protein):")
    boot_rows = []
    for m in MOD:
        delta = paired_bootstrap_delta(
            y_true, com_prob, com_thr, mp[m], mt[m],
            n_boot=N_BOOT, seed=42, blocks=blocks)
        verdict = ("committee leads ▲" if delta["ci_lo"] > 0 else
                   "committee trails ▼" if delta["ci_hi"] < 0 else
                   "indistinguishable ⊘")
        boot_rows.append({"comparison": f"committee_hybrid - {m}_hybrid",
                          **delta, "verdict": verdict})
        print(f"  vs {m:9s}: Δ={delta['delta_mean']:+.4f} "
              f"CI95=[{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}] {verdict}")
    boot_df = pd.DataFrame(boot_rows)

    metrics_df.to_csv(OUT / "metrics.csv")
    boot_df.to_csv(OUT / "paired_bootstrap.csv", index=False)

    lines = ["# Hybrid Human Committee — Cross-Corpus Baselines + In-Domain DT-Kinase\n"]
    lines.append("**Setup**: Human test set. DT-K Human-trained; baselines all-trained eval'd on Human test.")
    lines.append(f"\nB={N_BOOT}, dedupe ON, block bootstrap by protein.\n")
    lines.append("## Metrics\n")
    lines.append(_md_table(metrics_df.round(4).reset_index()))
    lines.append("\n## Paired bootstrap\n")
    lines.append(_md_table(boot_df.round(4)))
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(f"\n→ wrote {OUT}")


if __name__ == "__main__":
    main()
