#!/usr/bin/env python3
"""3-model hybrid NH committee variant: drops ConPLex_all (which regresses
substantially under all→NH cross-corpus transfer, MCC 0.347 vs 0.462
in-domain). Tests whether removing the cross-corpus regressor improves
the hybrid committee on NH test.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "inference" / "experiments"))

# Reuse helpers from the 4-model hybrid script.
from committee_hybrid_nh import (  # noqa: E402
    load_dtkinase_5seed_indomain, load_baseline_5seed_cross,
    load_test_keys, dedupe, system_metrics, paired_bootstrap_delta,
    _md_table,
)


N_BOOT = int(os.environ.get("BENCHMARK_BOOTSTRAP_B", "2000"))
OUT = REPO / "results" / "inference" / "committee_hybrid_nh_3model"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  3-model hybrid NH committee (no ConPLex)\n{'='*70}")

    dtk_prob, y_true, dtk_thr = load_dtkinase_5seed_indomain()
    drug_prob, _, _, _, drug_thr = load_baseline_5seed_cross("drugban", "f1")
    grph_prob, _, _, _, grph_thr = load_baseline_5seed_cross("graphban", "f1")

    keys, seq_ids = load_test_keys("non_human")
    mp = {"dtkinase": dtk_prob, "drugban": drug_prob, "graphban": grph_prob}
    mt = {"dtkinase": dtk_thr, "drugban": drug_thr, "graphban": grph_thr}

    new_mp = {}
    y_d, seq_ids_d = None, None
    for m, p in mp.items():
        pd_, yd_, kd_ = dedupe(p, y_true, keys)
        new_mp[m] = pd_
        if y_d is None:
            y_d = yd_
            df = pd.DataFrame({"key": keys, "seq_id": seq_ids})
            seq_map = df.drop_duplicates("key").set_index("key")["seq_id"]
            seq_ids_d = seq_map.loc[kd_].to_numpy()
    y_true = y_d
    mp = new_mp
    blocks = seq_ids_d
    print(f"  dedupe: n={len(y_true)}, proteins={len(np.unique(blocks))}")

    MOD = ["dtkinase", "drugban", "graphban"]
    com_prob = np.mean(np.stack([mp[m] for m in MOD]), axis=0)
    com_thr = float(np.mean([mt[m] for m in MOD]))
    print(f"  committee_hybrid_3: prob={com_prob.mean():.4f} thr={com_thr:.4f}")

    rows = []
    for m in MOD:
        rows.append({"system": f"{m}_hybrid",
                     **system_metrics(y_true, mp[m], mt[m])})
    rows.append({"system": "committee_hybrid_3",
                 **system_metrics(y_true, com_prob, com_thr)})
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
        boot_rows.append({"comparison": f"committee_hybrid_3 - {m}_hybrid",
                          **delta, "verdict": verdict})
        print(f"  vs {m:9s}: Δ={delta['delta_mean']:+.4f} "
              f"CI95=[{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}] {verdict}")
    boot_df = pd.DataFrame(boot_rows)

    metrics_df.to_csv(OUT / "metrics.csv")
    boot_df.to_csv(OUT / "paired_bootstrap.csv", index=False)

    lines = ["# 3-model Hybrid NH Committee (no ConPLex)", ""]
    lines.append("**Setup**: NH test set evaluated by:")
    lines.append("- DT-Kinase: NH-trained (in-domain)")
    lines.append("- DrugBAN:   all-trained, evaluated on NH test")
    lines.append("- GraphBAN:  all-trained, evaluated on NH test")
    lines.append("- ConPLex:   **EXCLUDED** (regresses under all→NH transfer)")
    lines.append("")
    lines.append(f"Dedupe ON, block bootstrap by protein, B={N_BOOT}.")
    lines.append("")
    lines.append("## Metrics\n")
    lines.append(_md_table(metrics_df.round(4).reset_index()))
    lines.append("\n## Paired bootstrap\n")
    lines.append(_md_table(boot_df.round(4)))
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(f"\n  → wrote {OUT}")


if __name__ == "__main__":
    main()
