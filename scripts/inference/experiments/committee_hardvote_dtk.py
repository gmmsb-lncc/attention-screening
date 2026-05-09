#!/usr/bin/env python3
"""Hard-vote committee with DT-Kinase tie-break vs soft-mean baseline.

Decision rule (per pair):
  votes = sum_m  (prob_m >= threshold_m).astype(int)        # 0..4
  if votes >= 3:  pred = 1
  if votes <= 1:  pred = 0
  if votes == 2:  pred = pred_dtkinase                      # tie-break a priori

Rationale: response to thesis defense question on whether soft-mean is the
right aggregation. Hard-vote is the intuitive "majority decides" rule;
ties broken by DT-Kinase because the model exhibits balanced precision/
recall on the human corpus (Cap. 5, Tab. baselines-desempenho), which is
the operational target. The choice is pre-registered (independent of
test labels), so no data leakage.

Comparison: paired block-bootstrap by protein (B = 10000) on
  delta_MCC = MCC(hardvote_dtk) - MCC(soft_mean)
across the 3 corpora. Reports CI95 + frac_positive + verdict.

Output:
  results/inference/committee_hardvote_dtk/
      <corpus>_metrics.csv
      <corpus>_paired_bootstrap.csv
      REPORT.md

Reuses load_5seed / dedupe / find_optimal_threshold helpers from
committee_vs_individual.py (loaded as module).
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

# Re-use the canonical 5-seed loader + dedupe machinery.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from committee_vs_individual import (  # noqa: E402
    load_5seed, dedupe_predictions, load_test_keys, _mcc_fast,
)

REPO = Path(__file__).resolve().parents[3]
SEEDS = [42, 123, 456, 789, 1024]
CORPORA = ["non_human", "human", "all"]

MODELS = ["dtkinase", "drugban", "graphban", "conplex"]
TIE_BREAKER = "dtkinase"
assert TIE_BREAKER in MODELS, "tie-breaker must be a committee member"

OUT_DIR = REPO / "results" / "inference" / "committee_hardvote_dtk"
N_BOOT = int(os.environ.get("BENCHMARK_BOOTSTRAP_B", "10000"))


def hardvote_dtk(model_preds: dict[str, np.ndarray]) -> np.ndarray:
    """Majority vote (>=3 = positive, <=1 = negative); DT-K breaks 2-2 ties."""
    stack = np.stack([model_preds[m] for m in MODELS], axis=0)  # (4, n)
    votes = stack.sum(axis=0)                                   # (n,)
    pred = np.empty_like(votes, dtype=int)
    pred[votes >= 3] = 1
    pred[votes <= 1] = 0
    tie = (votes == 2)
    pred[tie] = model_preds[TIE_BREAKER][tie]
    return pred


def metrics_hard(y_true: np.ndarray, pred: np.ndarray) -> dict:
    """MCC, F1, Accuracy on hard predictions; AUROC undefined → NaN."""
    return {
        "mcc": matthews_corrcoef(y_true, pred),
        "auroc": float("nan"),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
    }


def metrics_soft(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    return {
        "mcc": matthews_corrcoef(y_true, pred),
        "auroc": roc_auc_score(y_true, prob),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
    }


def paired_bootstrap_pred_pred(y_true: np.ndarray,
                                pred_a: np.ndarray, pred_b: np.ndarray,
                                blocks: np.ndarray | None,
                                n_boot: int = 10000, seed: int = 0) -> dict:
    """Bootstrap CI95 for MCC(a) - MCC(b) when both are hard predictions."""
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
        if (b + 1) % 1000 == 0:
            print(f"      bootstrap {b+1}/{n_boot}", flush=True)
    return {
        "delta_mean": float(deltas.mean()),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "frac_positive": float((deltas > 0).mean()),
    }


def run_corpus(corpus: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print(f"\n{'='*70}\n  Corpus: {corpus}\n{'='*70}")

    # 5-seed-averaged probs + thresholds per model.
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
            assert np.array_equal(y_true, yt), f"y_true mismatch: {m}"
        print(f"  {m:9s}: prob_mean={prob.mean():.4f}  thr={thr:.4f}  n={len(prob)}")

    # Dedupe by (seq_id, chembl_id) — same protocol as committee_vs_individual.
    keys, seq_ids = load_test_keys(corpus)
    assert len(keys) == len(y_true), f"tsv n={len(keys)} != npz n={len(y_true)}"

    y_true_d, seq_ids_d, key_order = None, None, None
    new_probs: dict[str, np.ndarray] = {}
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
            assert np.array_equal(y_true_d, y_d), f"dedupe y mismatch for {m}"
    n_before, n_after = len(y_true), len(y_true_d)
    print(f"\n  dedupe: n {n_before} → {n_after}")
    y_true = y_true_d
    seq_ids = seq_ids_d
    model_probs = new_probs

    # Hard predictions per model (using each model's calibrated threshold).
    model_preds = {m: (model_probs[m] >= model_thrs[m]).astype(int) for m in MODELS}

    # Soft-mean baseline (canonical).
    soft_prob = np.mean(np.stack([model_probs[m] for m in MODELS]), axis=0)
    soft_thr = float(np.mean([model_thrs[m] for m in MODELS]))
    soft_pred = (soft_prob >= soft_thr).astype(int)

    # Hard-vote with DT-K tie-break.
    hard_pred = hardvote_dtk(model_preds)

    # Tie diagnostics.
    votes = np.stack([model_preds[m] for m in MODELS], axis=0).sum(axis=0)
    tie_count = int((votes == 2).sum())
    tie_dt_pos = int(((votes == 2) & (model_preds[TIE_BREAKER] == 1)).sum())
    tie_dt_neg = tie_count - tie_dt_pos
    print(f"  ties (votes==2): {tie_count}/{len(y_true)} "
          f"({100*tie_count/len(y_true):.1f}%); "
          f"DT-K casts +1 in {tie_dt_pos}, 0 in {tie_dt_neg}")

    # Per-system metrics table.
    rows = []
    for m in MODELS:
        rows.append({"system": m, **metrics_soft(y_true, model_probs[m], model_thrs[m])})
    rows.append({"system": "soft_mean", **metrics_soft(y_true, soft_prob, soft_thr)})
    rows.append({"system": "hardvote_dtk", **metrics_hard(y_true, hard_pred)})
    metrics_df = pd.DataFrame(rows).set_index("system")
    print("\nMetrics:")
    print(metrics_df.round(4).to_string())

    # Paired block-bootstrap: hardvote_dtk vs soft_mean and vs each individual.
    print(f"\nPaired bootstrap (B={N_BOOT}, block-by-protein):")
    boot_rows = []

    delta = paired_bootstrap_pred_pred(
        y_true, hard_pred, soft_pred, blocks=seq_ids,
        n_boot=N_BOOT, seed=42,
    )
    verdict = ("hardvote leads ▲" if delta["ci_lo"] > 0 else
               "hardvote trails ▼" if delta["ci_hi"] < 0 else
               "indistinguishable ⊘")
    boot_rows.append({"comparison": "hardvote_dtk - soft_mean", **delta, "verdict": verdict})
    print(f"  vs soft_mean: Δ={delta['delta_mean']:+.4f}  "
          f"CI95=[{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}]  "
          f"P(Δ>0)={delta['frac_positive']:.3f}  {verdict}")

    for m in MODELS:
        delta = paired_bootstrap_pred_pred(
            y_true, hard_pred, model_preds[m], blocks=seq_ids,
            n_boot=N_BOOT, seed=42,
        )
        verdict = ("hardvote leads ▲" if delta["ci_lo"] > 0 else
                   "hardvote trails ▼" if delta["ci_hi"] < 0 else
                   "indistinguishable ⊘")
        boot_rows.append({"comparison": f"hardvote_dtk - {m}", **delta, "verdict": verdict})
        print(f"  vs {m:9s}: Δ={delta['delta_mean']:+.4f}  "
              f"CI95=[{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}]  "
              f"P(Δ>0)={delta['frac_positive']:.3f}  {verdict}")

    boot_df = pd.DataFrame(boot_rows)
    diag = {
        "n_dedup": int(len(y_true)),
        "n_proteins": int(len(np.unique(seq_ids))),
        "ties": tie_count,
        "tie_dt_pos": tie_dt_pos,
        "tie_dt_neg": tie_dt_neg,
        "tie_frac": float(tie_count / len(y_true)),
    }
    return metrics_df, boot_df, diag


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_all, boot_all, diag_all = {}, {}, {}
    for c in CORPORA:
        m, b, d = run_corpus(c)
        m.to_csv(OUT_DIR / f"{c}_metrics.csv")
        b.to_csv(OUT_DIR / f"{c}_paired_bootstrap.csv", index=False)
        metrics_all[c], boot_all[c], diag_all[c] = m, b, d

    lines = ["# Hard-vote committee + DT-Kinase tie-break vs soft-mean", ""]
    lines.append(f"**Protocol**: 5-seed averaged probs/thresholds per model; "
                 f"hard predictions per model via per-model calibrated threshold; "
                 f"majority rule (>=3 positive, <=1 negative); ties (==2) broken "
                 f"by DT-Kinase (pre-registered, no test-set leakage).")
    lines.append(f"**Bootstrap**: B={N_BOOT}, block-by-protein (seq_id), IC95 percentile.")
    lines.append("")
    for c in CORPORA:
        d = diag_all[c]
        lines.append(f"## {c}")
        lines.append(f"n_dedup={d['n_dedup']}, n_proteins={d['n_proteins']}, "
                     f"ties={d['ties']} ({100*d['tie_frac']:.1f}% of pairs); "
                     f"DT-K breaks {d['tie_dt_pos']} as positive / {d['tie_dt_neg']} as negative.")
        lines.append("")
        lines.append("### Per-system metrics")
        lines.append(_md_table(metrics_all[c].round(4).reset_index()))
        lines.append("")
        lines.append("### Paired block-bootstrap (hardvote_dtk reference)")
        lines.append(_md_table(boot_all[c].round(4)))
        lines.append("")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"\n→ wrote {OUT_DIR}/REPORT.md")


if __name__ == "__main__":
    main()
