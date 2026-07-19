#!/usr/bin/env python3
"""Unified canonical committee statistics: per-seed vs block-by-protein.

Resolves the estimator tangle flagged in the thesis audit (2026-05-30).
Computes the 4-model PoE committee vs each individual model under BOTH:

  Method A (per-seed):   committee MCC = mean over 5 seeds of per-seed PoE MCC
                         (matches the reported 0,518/0,532/0,541);
                         Delta CI = paired bootstrap over the 5 seeds (B=1e4).

  Method B (block-by-protein): committee prob = geom-mean over models of the
                         seed-averaged calibrated prob (single prediction set);
                         Delta CI = cluster bootstrap resampling seq_id
                         clusters (B=2000), matching Anexo B Sec. B.5.

Prints, per corpus: MCC per system (both methods), the 12 paired Delta with
IC95 + verdict, and the Holm-Bonferroni (m=12) table for each method.

No GPU. Reuses raw_predictions.npz already on disk.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "inference" / "experiments"))
from committee_per_seed_poe import (  # noqa: E402
    MODELS, geometric_mean, load_per_seed, load_universal_test_keys,
)
from committee_vs_individual import dedupe_predictions, SEEDS  # noqa: E402

EPS = 1e-12
B_SEED = 10000
B_BLOCK = 2000
CORPORA = ["non_human", "human", "all"]
RNG = np.random.RandomState(42)


def mcc_thr(y, p, thr):
    return matthews_corrcoef(y, (p >= thr).astype(int))


def load_corpus(corpus):
    """Return per-seed dedup probs/thr per model + y + seq_ids (aligned)."""
    keys, _ = load_universal_test_keys(corpus)
    raw = {m: load_per_seed(m, corpus) for m in MODELS}
    y_ref = raw["dtkinase"][SEEDS[0]][1]
    per_seed = {}        # per_seed[m][seed] = (prob_dedup, thr)
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


def method_a(per_seed, y):
    """Per-seed: returns committee MCC mean, and per-seed MCC matrix."""
    sys_mcc = {m: [] for m in MODELS + ["committee"]}
    for s in SEEDS:
        probs = {m: per_seed[m][s][0] for m in MODELS}
        thr = {m: per_seed[m][s][1] for m in MODELS}
        for m in MODELS:
            sys_mcc[m].append(mcc_thr(y, probs[m], thr[m]))
        p_poe = geometric_mean(np.stack([probs[m] for m in MODELS]), 0)
        t_poe = float(geometric_mean(np.array([thr[m] for m in MODELS])))
        sys_mcc["committee"].append(mcc_thr(y, p_poe, t_poe))
    return {k: np.array(v) for k, v in sys_mcc.items()}


def pooled_predictions(per_seed):
    """Seed-average calibrated prob + thr per model; PoE committee."""
    probs, thrs = {}, {}
    for m in MODELS:
        probs[m] = np.mean(np.stack([per_seed[m][s][0] for s in SEEDS]), 0)
        thrs[m] = float(np.mean([per_seed[m][s][1] for s in SEEDS]))
    p_poe = geometric_mean(np.stack([probs[m] for m in MODELS]), 0)
    t_poe = float(geometric_mean(np.array([thrs[m] for m in MODELS])))
    probs["committee"], thrs["committee"] = p_poe, t_poe
    return probs, thrs


def holm(deltas):
    """deltas: list of (label, d_mean, lo, hi, p_uni). Returns sorted+Holm."""
    rows = sorted(deltas, key=lambda r: r[4])
    m = len(rows)
    out = []
    for k, (lab, d, lo, hi, p) in enumerate(rows, 1):
        pholm = min(1.0, max((m - j) * rows[j][4] for j in range(k)))
        out.append((k, lab, d, lo, hi, p, pholm, "Y" if pholm < 0.05 else "N"))
    return out


def verdict(lo, hi):
    return "LEAD" if lo > 0 else ("LOSS" if hi < 0 else "tie")


for corpus in CORPORA:
    print(f"\n{'#'*72}\n# CORPUS: {corpus}\n{'#'*72}")
    per_seed, y, seq_ids = load_corpus(corpus)
    print(f"  dedup pairs: {len(y)}  | proteins: {len(np.unique(seq_ids))}")

    # ---- Method A: per-seed ----
    a = method_a(per_seed, y)
    print("\n[A] per-seed PoE MCC (mean +/- sd):")
    for m in MODELS + ["committee"]:
        print(f"    {m:11s} {a[m].mean():.4f} +/- {a[m].std():.4f}")
    a_deltas = []
    for m in MODELS:
        d = a["committee"] - a[m]
        bs = np.array([RNG.choice(d, 5, replace=True).mean()
                       for _ in range(B_SEED)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = max((bs <= 0).mean(), 1.0 / B_SEED)
        a_deltas.append((f"{corpus}:committee-{m}", d.mean(), lo, hi, p))
    print("[A] paired bootstrap over seeds (B=1e4):")
    for lab, d, lo, hi, p in a_deltas:
        print(f"    {lab:26s} d={d:+.4f} [{lo:+.4f},{hi:+.4f}] {verdict(lo,hi)}")

    # ---- Method B: pooled prob + block bootstrap by protein ----
    probs, thrs = pooled_predictions(per_seed)
    print("\n[B] pooled-prob PoE MCC (single prediction set):")
    for m in MODELS + ["committee"]:
        print(f"    {m:11s} {mcc_thr(y, probs[m], thrs[m]):.4f}")
    # cluster indices
    uniq = np.unique(seq_ids)
    idx_by_prot = {p_: np.where(seq_ids == p_)[0] for p_ in uniq}
    b_deltas = []
    for m in MODELS:
        ds = []
        for _ in range(B_BLOCK):
            samp = RNG.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([idx_by_prot[p_] for p_ in samp])
            yy = y[idx]
            if yy.min() == yy.max():
                continue
            dc = mcc_thr(yy, probs["committee"][idx], thrs["committee"])
            dm = mcc_thr(yy, probs[m][idx], thrs[m])
            ds.append(dc - dm)
        ds = np.array(ds)
        lo, hi = np.percentile(ds, [2.5, 97.5])
        p = max((ds <= 0).mean(), 1.0 / B_BLOCK)
        d0 = (mcc_thr(y, probs["committee"], thrs["committee"])
              - mcc_thr(y, probs[m], thrs[m]))
        b_deltas.append((f"{corpus}:committee-{m}", d0, lo, hi, p))
    print("[B] cluster bootstrap by protein (B=2000):")
    for lab, d, lo, hi, p in b_deltas:
        print(f"    {lab:26s} d={d:+.4f} [{lo:+.4f},{hi:+.4f}] {verdict(lo,hi)}")

    # stash for global Holm
    if corpus == CORPORA[0]:
        all_a, all_b = [], []
    all_a += a_deltas
    all_b += b_deltas

print(f"\n{'='*72}\n# GLOBAL Holm-Bonferroni (m=12)\n{'='*72}")
for name, dd in [("A per-seed", all_a), ("B block-by-protein", all_b)]:
    print(f"\n--- Method {name} ---")
    leads = sum(1 for _, d, lo, hi, p in dd if lo > 0)
    ties = sum(1 for _, d, lo, hi, p in dd if lo <= 0 <= hi)
    loss = sum(1 for _, d, lo, hi, p in dd if hi < 0)
    print(f"verdict: {leads} leads, {ties} ties, {loss} losses")
    for k, lab, d, lo, hi, p, ph, sv in holm(dd):
        print(f"  {k:2d} {lab:26s} d={d:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"p={p:.4f} pHolm={ph:.4f} {sv}")
