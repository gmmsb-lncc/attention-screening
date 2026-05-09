#!/usr/bin/env python3
"""Hard-vote committee with each of the 4 models as tie-breaker.

Builds 4 hard-vote variants:
  hardvote_<arbiter>: votes>=3 → 1; votes<=1 → 0; votes==2 → pred_<arbiter>

Each variant is pre-registered (arbiter chosen before seeing test labels;
no leakage). Compared head-to-head per corpus + against soft-mean.

Output:
  results/inference/committee_hardvote_arbiters/
      <corpus>_metrics.csv
      <corpus>_paired_bootstrap.csv
      REPORT.md
"""
from __future__ import annotations

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
    load_5seed, dedupe_predictions, load_test_keys, _mcc_fast,
)

REPO = Path(__file__).resolve().parents[3]
SEEDS = [42, 123, 456, 789, 1024]
CORPORA = ["non_human", "human", "all"]

MODELS = ["dtkinase", "drugban", "graphban", "conplex"]
ARBITERS = MODELS[:]  # all 4 candidates

OUT_DIR = REPO / "results" / "inference" / "committee_hardvote_arbiters"
N_BOOT = int(os.environ.get("BENCHMARK_BOOTSTRAP_B", "10000"))


def hardvote(model_preds: dict[str, np.ndarray], arbiter: str) -> np.ndarray:
    stack = np.stack([model_preds[m] for m in MODELS], axis=0)
    votes = stack.sum(axis=0)
    pred = np.empty_like(votes, dtype=int)
    pred[votes >= 3] = 1
    pred[votes <= 1] = 0
    tie = (votes == 2)
    pred[tie] = model_preds[arbiter][tie]
    return pred


def metrics_hard(y_true, pred):
    return {
        "mcc": matthews_corrcoef(y_true, pred),
        "auroc": float("nan"),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "n_pos_pred": int(pred.sum()),
    }


def metrics_soft(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    return {
        "mcc": matthews_corrcoef(y_true, pred),
        "auroc": roc_auc_score(y_true, prob),
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


def verdict_from(d):
    if d["ci_lo"] > 0:
        return "A leads ▲"
    if d["ci_hi"] < 0:
        return "B leads ▼"
    return "indistinguishable ⊘"


def run_corpus(corpus):
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

    # Per-model hard predictions using calibrated thresholds.
    model_preds = {m: (model_probs[m] >= model_thrs[m]).astype(int) for m in MODELS}

    # Soft-mean baseline.
    soft_prob = np.mean(np.stack([model_probs[m] for m in MODELS]), axis=0)
    soft_thr = float(np.mean([model_thrs[m] for m in MODELS]))
    soft_pred = (soft_prob >= soft_thr).astype(int)

    # Tie diagnostics.
    votes = np.stack([model_preds[m] for m in MODELS], axis=0).sum(axis=0)
    tie_mask = (votes == 2)
    tie_n = int(tie_mask.sum())

    # Build all 4 hard-vote variants.
    hard_preds = {a: hardvote(model_preds, a) for a in ARBITERS}

    # Per-system metrics.
    rows = []
    for m in MODELS:
        rows.append({"system": m, **metrics_soft(y_true, model_probs[m], model_thrs[m])})
    rows.append({"system": "soft_mean", **metrics_soft(y_true, soft_prob, soft_thr)})
    for a in ARBITERS:
        rows.append({"system": f"hardvote_{a}", **metrics_hard(y_true, hard_preds[a])})
    metrics_df = pd.DataFrame(rows).set_index("system")
    print("\nMetrics:")
    print(metrics_df.round(4).to_string())

    # Tie-break analysis: how each arbiter votes on ties.
    print(f"\n  ties (votes==2): {tie_n}/{len(y_true)} ({100*tie_n/len(y_true):.1f}%)")
    print("  arbiter pos / neg / pos_correct on ties:")
    tie_y = y_true[tie_mask]
    for a in ARBITERS:
        a_pred_tie = model_preds[a][tie_mask]
        tp_tie = int(((a_pred_tie == 1) & (tie_y == 1)).sum())
        fp_tie = int(((a_pred_tie == 1) & (tie_y == 0)).sum())
        tn_tie = int(((a_pred_tie == 0) & (tie_y == 0)).sum())
        fn_tie = int(((a_pred_tie == 0) & (tie_y == 1)).sum())
        acc_tie = (tp_tie + tn_tie) / max(1, tie_n)
        print(f"    {a:9s}: pos={a_pred_tie.sum():4d}  neg={(1-a_pred_tie).sum():4d}  "
              f"acc_on_ties={acc_tie:.3f}  (tp={tp_tie} fn={fn_tie} fp={fp_tie} tn={tn_tie})")

    # Paired block-bootstrap: each hardvote variant vs soft-mean,
    # plus pairwise hardvote_a vs hardvote_b.
    boot_rows = []
    print(f"\nPaired block-bootstrap (B={N_BOOT}, by protein):")
    print("\n  hardvote_<arbiter> vs soft_mean:")
    for a in ARBITERS:
        d = paired_bootstrap_pp(y_true, hard_preds[a], soft_pred, seq_ids, N_BOOT)
        verdict = ("hardvote_" + a + " leads ▲" if d["ci_lo"] > 0 else
                   "soft_mean leads ▼" if d["ci_hi"] < 0 else "indistinguishable ⊘")
        boot_rows.append({"comparison": f"hardvote_{a} - soft_mean", **d, "verdict": verdict})
        print(f"    {a:9s}: Δ={d['delta_mean']:+.4f}  "
              f"CI95=[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  "
              f"P(Δ>0)={d['frac_positive']:.3f}  {verdict}")

    print("\n  hardvote_a vs hardvote_b (a vs b ordering by ARBITERS list):")
    for i, a in enumerate(ARBITERS):
        for b in ARBITERS[i+1:]:
            d = paired_bootstrap_pp(y_true, hard_preds[a], hard_preds[b], seq_ids, N_BOOT)
            verdict = (f"hardvote_{a} leads ▲" if d["ci_lo"] > 0 else
                       f"hardvote_{b} leads ▼" if d["ci_hi"] < 0 else "indistinguishable ⊘")
            boot_rows.append({"comparison": f"hardvote_{a} - hardvote_{b}", **d, "verdict": verdict})
            print(f"    {a:9s} vs {b:9s}: Δ={d['delta_mean']:+.4f}  "
                  f"CI95=[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  "
                  f"P(Δ>0)={d['frac_positive']:.3f}  {verdict}")

    boot_df = pd.DataFrame(boot_rows)
    diag = {"n_dedup": int(len(y_true)),
            "n_proteins": int(len(np.unique(seq_ids))),
            "ties": tie_n,
            "tie_frac": float(tie_n / len(y_true))}
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

    lines = ["# Hard-vote committee with all 4 candidate tie-breakers", ""]
    lines.append(f"**Setup**: 5-seed averaged probs + thresholds; majority rule "
                 f"(>=3 pos / <=1 neg); ties (==2) broken by candidate arbiter "
                 f"∈ {{dtkinase, drugban, graphban, conplex}}, each pre-registered "
                 f"(no test-set leakage).")
    lines.append(f"**Bootstrap**: B={N_BOOT}, block-by-protein (seq_id), IC95 percentile.")
    lines.append("")
    for c in CORPORA:
        d = diag_all[c]
        lines.append(f"## {c}")
        lines.append(f"n_dedup={d['n_dedup']}, n_proteins={d['n_proteins']}, "
                     f"ties={d['ties']} ({100*d['tie_frac']:.1f}% of pairs).")
        lines.append("")
        lines.append("### Per-system metrics")
        lines.append(_md_table(metrics_all[c].round(4).reset_index()))
        lines.append("")
        lines.append("### Paired block-bootstrap")
        lines.append(_md_table(boot_all[c].round(4)))
        lines.append("")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"\n→ wrote {OUT_DIR}/REPORT.md")


if __name__ == "__main__":
    main()
