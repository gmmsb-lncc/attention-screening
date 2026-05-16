#!/usr/bin/env python3
"""Per-seed PoE committee metrics with full sigma over 5 seeds.

Canonical aggregation: Product-of-Experts (PoE) = geometric mean of
calibrated per-model probabilities. Threshold = geometric mean of per-model
calibrated thresholds. Matches `aggregate.py` canonical convention.

Protocol (Anexo B §B.5, Anexo D §sec:agreg-protocolo):
  - For each corpus in {non_human, human, all} (configurable via --corpus):
    * For each seed in {42, 123, 456, 789, 1024}:
      - Load 5-seed npz raw predictions for the 4 models
      - Align with universal_test.tsv filtered by dataset_source
      - Dedupe by (seq_id, chembl_id)
      - Aggregate via PoE: prob_committee_seed = (prod_m prob_m)^(1/4)
      - Threshold per seed: thr_committee_seed = (prod_m thr_m)^(1/4)
      - Compute MCC, AUROC, F1, Accuracy, Precision, Recall
  - Aggregate mean ± σ across 5 seeds for committee + 4 individuals.

CLI:
  --corpus {non_human,human,all}       single-corpus run (parallel dispatch)
  --corpora non_human,human,all        comma-separated list (default order)
  --out-dir PATH                       output dir (default: per-corpus subdir)

Output (per corpus):
  results/inference/committee_per_seed_poe/<corpus>/
    per_seed_metrics.csv     6 metrics × 5 seeds × 5 systems
    summary_mean_std.csv     mean ± σ per system
    REPORT.md                tabular summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                              f1_score, accuracy_score,
                              precision_score, recall_score)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "inference" / "experiments"))
from committee_vs_individual import (  # type: ignore  # noqa: E402
    model_paths, dedupe_predictions, SEEDS,
)

MODELS = ["dtkinase", "drugban", "graphban", "conplex"]
CORPORA_DEFAULT = ["non_human", "human", "all"]  # NH → Human → All
UNIVERSAL_TEST_TSV = REPO / "scaffolds_splits" / "output" / "universal_test.tsv"

EPS = 1e-12  # numerical floor for log-domain ops


def metrics_full(y_true: np.ndarray, y_prob: np.ndarray,
                 threshold: float) -> dict:
    """All 6 metrics: MCC, AUROC, F1, Accuracy, Precision, Recall."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def geometric_mean(probs: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable geometric mean via log-domain averaging.

    probs: shape (n_models, n_samples). axis: axis to reduce over.
    """
    clipped = np.clip(probs, EPS, 1.0 - EPS)
    return np.exp(np.mean(np.log(clipped), axis=axis))


def load_universal_test_keys(corpus: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (keys, seq_ids) from universal_test.tsv filtered by source."""
    df = pd.read_csv(UNIVERSAL_TEST_TSV, sep="\t",
                     usecols=["chembl_id", "seq_id", "dataset_source"])
    if corpus != "all":
        df = df[df["dataset_source"] == corpus].reset_index(drop=True)
    keys = (df["seq_id"].astype(str) + "__" +
            df["chembl_id"].astype(str)).to_numpy()
    return keys, df["seq_id"].astype(str).to_numpy()


def load_per_seed(model: str, corpus: str) -> dict:
    """Return {seed: (prob, y_true, threshold)} for 5 canonical seeds."""
    out = {}
    for s in SEEDS:
        npz_path, cal_path = model_paths(model, corpus, s)
        d = np.load(npz_path)
        prob_key = "test_y_prob" if "test_y_prob" in d.files else "y_prob"
        true_key = "test_y_true" if "test_y_true" in d.files else "y_true"
        prob = d[prob_key].astype(np.float64)
        y = d[true_key].astype(int)
        thr = float(json.loads(cal_path.read_text())["threshold"])
        out[s] = (prob, y, thr)
    return out


def per_seed_corpus_poe(corpus: str) -> pd.DataFrame:
    """Compute per-seed metrics for 4 models + Committee PoE."""
    print(f"\n{'='*70}\n=== Corpus: {corpus} (PoE per-seed) ===\n{'='*70}")
    raw = {m: load_per_seed(m, corpus) for m in MODELS}
    y_ref = raw["dtkinase"][SEEDS[0]][1]
    n_npz = len(y_ref)

    keys, seq_ids = load_universal_test_keys(corpus)
    assert len(keys) == n_npz, \
        f"{corpus}: universal_test filtered has {len(keys)} rows but npz has {n_npz}"

    # y_true must match across seeds and models
    for m in MODELS:
        for s in SEEDS:
            assert np.array_equal(raw[m][s][1], y_ref), \
                f"y mismatch {m}/seed_{s}"

    rows = []
    for s in SEEDS:
        per_model_thr = {m: raw[m][s][2] for m in MODELS}

        # Dedupe per model
        dedup_probs = {}
        y_dedup = None
        for m in MODELS:
            p_d, y_d, _ = dedupe_predictions(raw[m][s][0], y_ref, keys)
            if y_dedup is None:
                y_dedup = y_d
            else:
                assert np.array_equal(y_dedup, y_d), \
                    f"dedupe inconsistent {m}/seed_{s}"
            dedup_probs[m] = p_d

        if s == SEEDS[0]:
            print(f"  post-dedupe: {len(y_dedup)} pairs (was {n_npz})")

        # 4 individuals
        for m in MODELS:
            rows.append({"corpus": corpus, "seed": s, "system": m,
                         **metrics_full(y_dedup, dedup_probs[m],
                                        per_model_thr[m])})

        # Committee PoE: geometric mean over 4 models per sample
        probs_stack = np.stack([dedup_probs[m] for m in MODELS], axis=0)
        p_poe = geometric_mean(probs_stack, axis=0)
        thr_stack = np.array([per_model_thr[m] for m in MODELS])
        thr_poe = float(geometric_mean(thr_stack, axis=0))
        rows.append({"corpus": corpus, "seed": s, "system": "committee_poe",
                     **metrics_full(y_dedup, p_poe, thr_poe)})

        print(f"  seed {s}: PoE MCC={rows[-1]['mcc']:.4f}, "
              f"thr={thr_poe:.4f}")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± σ per (corpus, system) over 5 seeds."""
    agg_cols = ["mcc", "auroc", "f1", "accuracy", "precision", "recall"]
    out = (df.groupby(["corpus", "system"], as_index=False)
             .agg({c: ["mean", "std"] for c in agg_cols}))
    out.columns = ["_".join(c).rstrip("_") for c in out.columns]
    return out


def write_report(corpus: str, summary: pd.DataFrame, out_dir: Path) -> None:
    order = ["dtkinase", "drugban", "graphban", "conplex", "committee_poe"]
    lines = [f"# Per-seed PoE committee — corpus `{corpus}`", ""]
    lines.append("**Protocol**: PoE = geometric mean of 4 calibrated probs "
                 "per seed; thr = geometric mean of 4 thresholds per seed. "
                 "Dedupe by `(seq_id, chembl_id)`. Aggregated mean ± σ "
                 "over 5 canonical seeds {42, 123, 456, 789, 1024}.")
    lines.append("")
    lines.append("| system | MCC | AUROC | F1 | Accuracy | Precision | Recall |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for sysname in order:
        row = summary[(summary["corpus"] == corpus) &
                      (summary["system"] == sysname)]
        if row.empty:
            continue
        r = row.iloc[0]
        label = sysname
        if sysname == "committee_poe":
            label = "**Committee PoE**"
        lines.append(
            f"| {label} "
            f"| {r.mcc_mean:.4f} ± {r.mcc_std:.4f} "
            f"| {r.auroc_mean:.4f} ± {r.auroc_std:.4f} "
            f"| {r.f1_mean:.4f} ± {r.f1_std:.4f} "
            f"| {r.accuracy_mean:.4f} ± {r.accuracy_std:.4f} "
            f"| {r.precision_mean:.4f} ± {r.precision_std:.4f} "
            f"| {r.recall_mean:.4f} ± {r.recall_std:.4f} |"
        )
    lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines))


def run_corpus(corpus: str, base_out: Path) -> None:
    out_dir = base_out / corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    df = per_seed_corpus_poe(corpus)
    df.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    print(f"wrote {out_dir / 'per_seed_metrics.csv'}")

    summary = summarize(df)
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False)
    print(f"wrote {out_dir / 'summary_mean_std.csv'}")

    write_report(corpus, summary, out_dir)
    print(f"wrote {out_dir / 'REPORT.md'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-seed PoE committee metrics with sigma.")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--corpus",
                     choices=CORPORA_DEFAULT,
                     help="single corpus (parallel dispatch). "
                          "E.g. machine A: --corpus non_human; "
                          "machine B: --corpus all.")
    grp.add_argument("--corpora",
                     help="comma-separated list, in order. "
                          f"Default: {','.join(CORPORA_DEFAULT)}.")
    ap.add_argument("--out-dir",
                    type=Path,
                    default=REPO / "results" / "inference" / "committee_per_seed_poe",
                    help="base output dir (per-corpus subdir created).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.corpus:
        targets: Iterable[str] = [args.corpus]
    elif args.corpora:
        targets = [c.strip() for c in args.corpora.split(",") if c.strip()]
        for c in targets:
            if c not in CORPORA_DEFAULT:
                raise SystemExit(f"unknown corpus: {c}; "
                                 f"valid: {CORPORA_DEFAULT}")
    else:
        targets = CORPORA_DEFAULT  # NH → Human → All canonical order

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Out dir: {args.out_dir}")
    print(f"Targets: {list(targets)}")

    for corpus in targets:
        run_corpus(corpus, args.out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
