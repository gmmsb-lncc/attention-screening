#!/usr/bin/env python3
"""A3 -- Error-correlation matrix + ECE per model (CPU, saved logits only).

Reuses load_corpus() from committee_canonical_unified (aligned, deduped,
per-seed probs for the 4 models). For each corpus:
  1. VALIDATION: pooled per-model MCC vs results/inference/committee_experiment
     /<corpus>_metrics.csv. If these don't match, alignment is wrong -> abort
     trust (do not write any number into the thesis).
  2. 4x4 error-correlation matrix (Pearson on signed residual p - y).
  3. Mean off-diagonal correlation per model (lower = more orthogonal).
  4. ECE per model (10 equal-width bins) on pooled probs.

No training, no GPU. Pure arithmetic over raw_predictions.npz.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "inference" / "experiments"))
# Import primitives only (NOT committee_canonical_unified, which runs main() on import).
from committee_per_seed_poe import (  # noqa: E402
    MODELS, load_per_seed, load_universal_test_keys,
)
from committee_vs_individual import dedupe_predictions, SEEDS  # noqa: E402

CORPORA = ["non_human", "human", "all"]


def load_corpus(corpus):
    """Replica de committee_canonical_unified.load_corpus (aligned, deduped)."""
    keys, _ = load_universal_test_keys(corpus)
    raw = {m: load_per_seed(m, corpus) for m in MODELS}
    y_ref = raw["dtkinase"][SEEDS[0]][1]
    per_seed = {}
    y_dedup = None
    key_dedup = None
    for m in MODELS:
        per_seed[m] = {}
        for s in SEEDS:
            p_d, y_d, k_d = dedupe_predictions(raw[m][s][0], y_ref, keys)
            per_seed[m][s] = (p_d, raw[m][s][2])
            if y_dedup is None:
                y_dedup, key_dedup = y_d, k_d
    seq_ids = np.array([k.split("__")[0] for k in key_dedup])
    return per_seed, y_dedup, seq_ids


def pooled_per_model(per_seed):
    """Arithmetic-mean probabilities and thresholds over the 5 seeds."""
    probs, thrs = {}, {}
    for m in MODELS:
        stack = np.stack([per_seed[m][s][0] for s in SEEDS], axis=0)
        probs[m] = np.mean(stack, axis=0)
        thrs[m] = float(np.mean([per_seed[m][s][1] for s in SEEDS]))
    return probs, thrs


def ece(y, p, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        e += (m.sum() / len(p)) * abs(conf - acc)
    return e


def load_validation(corpus):
    f = REPO / "results" / "inference" / "committee_experiment" / f"{corpus}_metrics.csv"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().strip().splitlines()[1:]:
        parts = line.split(",")
        out[parts[0]] = float(parts[1])
    return out


def main():
    for corpus in CORPORA:
        per_seed, y, seq_ids = load_corpus(corpus)
        probs, thrs = pooled_per_model(per_seed)
        print(f"\n=== {corpus}  (n={len(y)}, proteins={len(np.unique(seq_ids))}) ===")

        # 1. validation
        ref = load_validation(corpus)
        print("[validate] pooled per-model MCC vs committee_experiment CSV:")
        ok = True
        for m in MODELS:
            mcc = matthews_corrcoef(y, (probs[m] >= thrs[m]).astype(int))
            r = ref.get(m if m != "dtkinase" else "dtkinase", None)
            tag = ""
            if r is not None:
                d = abs(mcc - r)
                tag = f" | csv={r:.4f} d={d:.4f} {'OK' if d < 0.02 else 'MISMATCH'}"
                ok = ok and d < 0.05
            print(f"    {m:9s} MCC={mcc:.4f}{tag}")
        print(f"  alignment trust: {'OK' if ok else 'SUSPECT -- do not use'}")

        # 2. residual correlation
        resid = np.stack([probs[m] - y for m in MODELS], axis=0)
        C = np.corrcoef(resid)
        print("[error-correlation matrix] Pearson(p_i - y, p_j - y):")
        print("           " + "  ".join(f"{m:>8s}" for m in MODELS))
        for i, m in enumerate(MODELS):
            print(f"    {m:9s} " + "  ".join(f"{C[i, j]:8.3f}" for j in range(len(MODELS))))

        # 3. mean off-diagonal per model
        print("[mean off-diagonal correlation] (lower = more orthogonal):")
        for i, m in enumerate(MODELS):
            off = np.mean([C[i, j] for j in range(len(MODELS)) if j != i])
            print(f"    {m:9s} {off:.3f}")

        # 4. ECE
        print("[ECE] (pooled probs, 10 bins):")
        for m in MODELS:
            print(f"    {m:9s} {ece(y, probs[m]):.4f}")


if __name__ == "__main__":
    main()
