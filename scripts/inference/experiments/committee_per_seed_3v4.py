"""Per-seed metrics + 3-model vs 4-model committee comparison.

Canonical protocol (Anexo B §B.5):
  - For each corpus and seed in {42, 123, 456, 789, 1024}:
    * load 5-seed npz raw predictions for the 4 models
    * align with universal_test.tsv filtered by dataset_source
    * dedupe by (seq_id, chembl_id) — collapses repeated measurement rows
    * compute MCC/AUROC/F1/Accuracy per system
  - Aggregate mean ± σ across 5 seeds.
  - Block bootstrap by seq_id (B=10000) for committee_3 vs committee_4.

Systems evaluated:
  - 4 individuals (dtkinase, drugban, graphban, conplex)
  - committee_4 (mean of 4 models at the seed)
  - committee_3 (DT-K + DrugBAN + ConPLex, no GraphBAN)

Output: results/inference/committee_per_seed/{REPORT.md,
        per_seed_metrics.csv, summary_mean_std.csv,
        committee_3v4_bootstrap.csv}.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "inference" / "experiments"))
from committee_vs_individual import (  # type: ignore  # noqa: E402
    model_paths, load_threshold, system_metrics, dedupe_predictions,
    paired_bootstrap_delta, SEEDS,
)

CORPORA = ["non_human", "human", "all"]
N_BOOT = 10_000
UNIVERSAL_TEST_TSV = REPO / "scaffolds_splits" / "output" / "universal_test.tsv"
OUT_DIR = REPO / "results" / "inference" / "committee_per_seed"


def load_universal_test_keys(corpus: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (keys, seq_ids) from universal_test.tsv filtered by dataset_source.

    Universal-source filtering matches what the training pipeline saw
    (`benchmark/levels/matrix_utils.py:283-292`), so the returned arrays
    are aligned row-by-row with each model's `raw_predictions.npz`.
    """
    df = pd.read_csv(UNIVERSAL_TEST_TSV, sep="\t",
                     usecols=["chembl_id", "seq_id", "dataset_source"])
    if corpus != "all":
        df = df[df["dataset_source"] == corpus].reset_index(drop=True)
    keys = (df["seq_id"].astype(str) + "__" + df["chembl_id"].astype(str)).to_numpy()
    return keys, df["seq_id"].astype(str).to_numpy()


def load_per_seed(model: str, corpus: str) -> dict:
    """Return {seed: (prob, y_true, threshold)} for the 5 canonical seeds."""
    out = {}
    for s in SEEDS:
        npz_path, cal_path = model_paths(model, corpus, s)
        d = np.load(npz_path)
        prob_key = "test_y_prob" if "test_y_prob" in d.files else "y_prob"
        true_key = "test_y_true" if "test_y_true" in d.files else "y_true"
        prob = d[prob_key].astype(np.float64)
        y = d[true_key].astype(int)
        thr = load_threshold(cal_path)
        out[s] = (prob, y, thr)
    return out


def per_seed_corpus(corpus: str) -> tuple[pd.DataFrame, dict, dict, np.ndarray, np.ndarray]:
    """Compute per-seed metrics + per-seed committee probs/y/thr/seq_ids."""
    models = ["dtkinase", "drugban", "graphban", "conplex"]
    raw = {m: load_per_seed(m, corpus) for m in models}
    y_ref = raw["dtkinase"][SEEDS[0]][1]
    n_npz = len(y_ref)

    # Load aligned keys + seq_ids from universal_test filtered by source.
    keys, seq_ids = load_universal_test_keys(corpus)
    assert len(keys) == n_npz, \
        f"{corpus}: universal_test filtered has {len(keys)} rows but npz has {n_npz}"
    print(f"\n=== {corpus}: {n_npz} pre-dedupe → ?? post-dedupe ===")

    # y_true must match across seeds AND models within corpus
    for m in models:
        for s in SEEDS:
            assert np.array_equal(raw[m][s][1], y_ref), \
                f"y mismatch {m}/seed_{s}"

    # seq_id per dedup group (pick first since all rows of same key share seq_id)
    df_keys = pd.DataFrame({"key": keys, "seq_id": seq_ids})
    seq_ids_dedup = (df_keys.groupby("key", sort=True, as_index=False)
                            .first()["seq_id"].to_numpy())

    rows = []
    com3_per_seed: dict = {}
    com4_per_seed: dict = {}
    for s in SEEDS:
        per_model_thr = {m: raw[m][s][2] for m in models}

        # Dedupe each model's prob aligned to keys; y/keys identical across.
        dedup_probs = {}
        y_dedup = None; keys_dedup = None
        for m in models:
            p_d, y_d, k_d = dedupe_predictions(raw[m][s][0], y_ref, keys)
            if y_dedup is None:
                y_dedup, keys_dedup = y_d, k_d
            else:
                assert np.array_equal(y_dedup, y_d) and np.array_equal(keys_dedup, k_d), \
                    f"dedupe inconsistent {m}/seed_{s}"
            dedup_probs[m] = p_d

        if s == SEEDS[0]:
            print(f"  post-dedupe: {len(y_dedup)} pairs (was {n_npz})")

        # Per-model metrics
        for m in models:
            t = per_model_thr[m]
            rows.append({"corpus": corpus, "seed": s, "system": m,
                         **system_metrics(y_dedup, dedup_probs[m], t)})

        # 4-model committee
        p4 = np.mean([dedup_probs[m] for m in models], axis=0)
        t4 = float(np.mean([per_model_thr[m] for m in models]))
        rows.append({"corpus": corpus, "seed": s, "system": "committee_4",
                     **system_metrics(y_dedup, p4, t4)})

        # 3-model committee (no GraphBAN)
        m3 = ["dtkinase", "drugban", "conplex"]
        p3 = np.mean([dedup_probs[m] for m in m3], axis=0)
        t3 = float(np.mean([per_model_thr[m] for m in m3]))
        rows.append({"corpus": corpus, "seed": s, "system": "committee_3_no_graphban",
                     **system_metrics(y_dedup, p3, t3)})

        com3_per_seed[s] = (p3, y_dedup, t3)
        com4_per_seed[s] = (p4, y_dedup, t4)

    return pd.DataFrame(rows), com3_per_seed, com4_per_seed, y_dedup, seq_ids_dedup


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    bootstrap_rows = []
    for corpus in CORPORA:
        df, com3, com4, y_ref, seq_ids_dedup = per_seed_corpus(corpus)
        all_rows.append(df)

        # Pooled-seed paired bootstrap (committee_3 − committee_4) with
        # block bootstrap by seq_id (canonical Anexo B §B.5).
        p3_pooled = np.mean([com3[s][0] for s in SEEDS], axis=0)
        p4_pooled = np.mean([com4[s][0] for s in SEEDS], axis=0)
        t3_pooled = float(np.mean([com3[s][2] for s in SEEDS]))
        t4_pooled = float(np.mean([com4[s][2] for s in SEEDS]))

        boot = paired_bootstrap_delta(
            y_ref, p3_pooled, t3_pooled, p4_pooled, t4_pooled,
            n_boot=N_BOOT, blocks=seq_ids_dedup,
        )
        bootstrap_rows.append({
            "corpus": corpus,
            "comparison": "committee_3 − committee_4",
            "delta_mean": boot["delta_mean"],
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
            "frac_positive": boot["frac_positive"],
            "verdict": ("3-model leads" if boot["ci_lo"] > 0
                        else "4-model leads" if boot["ci_hi"] < 0
                        else "indistinguishable"),
        })

    big = pd.concat(all_rows, ignore_index=True)
    big.to_csv(OUT_DIR / "per_seed_metrics.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'per_seed_metrics.csv'}")

    summary = (big.groupby(["corpus", "system"], as_index=False)
                  .agg(mcc_mean=("mcc","mean"),    mcc_std=("mcc","std"),
                       auroc_mean=("auroc","mean"),auroc_std=("auroc","std"),
                       f1_mean=("f1","mean"),      f1_std=("f1","std"),
                       acc_mean=("accuracy","mean"),acc_std=("accuracy","std")))
    summary.to_csv(OUT_DIR / "summary_mean_std.csv", index=False)
    print(f"wrote {OUT_DIR / 'summary_mean_std.csv'}")

    bdf = pd.DataFrame(bootstrap_rows)
    bdf.to_csv(OUT_DIR / "committee_3v4_bootstrap.csv", index=False)
    print(f"wrote {OUT_DIR / 'committee_3v4_bootstrap.csv'}")

    # ===================== REPORT =====================
    lines = []
    lines.append("# Per-seed committee metrics + 3-model vs 4-model comparison")
    lines.append("")
    lines.append("**Protocol** (canonical, Anexo B §B.5): per-seed "
                 "evaluation; alignment via `universal_test.tsv` filtered "
                 "by `dataset_source`; dedupe by `(seq_id, chembl_id)`; "
                 "threshold per system = mean of constituent models' "
                 "calibration thresholds at that seed. Block bootstrap "
                 f"by `seq_id` for committee_3 vs committee_4, B={N_BOOT}.")
    lines.append("")

    for corpus in CORPORA:
        lines.append(f"## Corpus: {corpus}")
        lines.append("")
        lines.append("### Mean ± σ over 5 seeds")
        lines.append("")
        lines.append("| system | MCC | AUROC | F1 | Accuracy |")
        lines.append("| --- | --- | --- | --- | --- |")
        order = ["dtkinase","drugban","graphban","conplex",
                 "committee_3_no_graphban","committee_4"]
        for sysname in order:
            r = summary[(summary["corpus"]==corpus) &
                        (summary["system"]==sysname)].iloc[0]
            label = sysname
            if sysname == "committee_3_no_graphban":
                label = "**committee_3 (DT-K + DrugBAN + ConPLex)**"
            elif sysname == "committee_4":
                label = "**committee_4 (+ GraphBAN)**"
            lines.append(
                f"| {label} | "
                f"{r.mcc_mean:.4f} ± {r.mcc_std:.4f} | "
                f"{r.auroc_mean:.4f} ± {r.auroc_std:.4f} | "
                f"{r.f1_mean:.4f} ± {r.f1_std:.4f} | "
                f"{r.acc_mean:.4f} ± {r.acc_std:.4f} |"
            )
        lines.append("")

        b = bdf[bdf["corpus"]==corpus].iloc[0]
        lines.append(f"### Paired bootstrap (committee_3 − committee_4, "
                     f"5-seed pooled, B={N_BOOT})")
        lines.append("")
        lines.append("| Δ MCC mean | CI95 lo | CI95 hi | frac. > 0 | verdict |")
        lines.append("| --- | --- | --- | --- | --- |")
        lines.append(
            f"| {b.delta_mean:+.4f} | {b.ci_lo:+.4f} | {b.ci_hi:+.4f} "
            f"| {b.frac_positive:.3f} | {b.verdict} |"
        )
        lines.append("")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"\nwrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
