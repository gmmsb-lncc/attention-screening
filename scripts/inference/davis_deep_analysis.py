"""Deep analysis of committee imatinib predictions vs DAVIS Kd ground truth.

Generates:
  - sensitivity table: AUROC across Kd cutoffs {30, 100, 1000, 3000, 10000} nM
  - per-tier breakdown: strong (Kd<100 nM) / mid (100-3000) / weak (>3000) recall
  - plots: roc_per_model.pdf, score_vs_kd_scatter.pdf, score_distribution.pdf
  - canonical targets table with DAVIS Kd
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)

CANONICAL = {
    "P00519": "ABL1", "P42684": "ABL2", "P10721": "KIT",
    "P16234": "PDGFRA", "P09619": "PDGFRB", "P07333": "CSF1R",
    "Q08345": "DDR1", "Q16832": "DDR2", "P06239": "LCK",
}


def davis_to_gene(name: str) -> str:
    s = name.split("(")[0]; s = s.split("-")[0]; s = re.sub(r"p$", "", s)
    return s.upper()


def load_merged(consensus: Path, davis_kd: Path, mapping: Path) -> pd.DataFrame:
    cons = pd.read_csv(consensus); cons["uniprot"] = cons["uniprot"].astype(str)
    davis = pd.read_csv(davis_kd); mp = pd.read_csv(mapping, sep="\t")
    davis["gene"] = davis["davis_kinase"].apply(davis_to_gene)
    by_gene = (davis.groupby("gene", as_index=False)
               .agg(Kd_nM=("Kd_nM","min"), n_variants=("davis_kinase","count")))
    mp["gene"] = mp["gene"].str.upper()
    by_gene = by_gene.merge(mp[["uniprot","gene"]], on="gene", how="left")
    return cons.merge(by_gene[["uniprot","Kd_nM","gene"]], on="uniprot", how="inner")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", type=Path, required=True)
    ap.add_argument("--davis-kd", type=Path,
                    default=Path("data/external/davis/imatinib_kd.csv"))
    ap.add_argument("--mapping", type=Path,
                    default=Path("data/reference/kinome_human_mapping.tsv"))
    ap.add_argument("--out-dir", type=Path,
                    default=None,
                    help="default: parent of --consensus")
    args = ap.parse_args()
    out_dir = args.out_dir or args.consensus.parent

    merged = load_merged(args.consensus, args.davis_kd, args.mapping)
    print(f"merged rows: {len(merged)}")

    # Score columns
    score_cols = ["prob_mean"]
    for col in merged.columns:
        if col.startswith("prob_") and col not in {"prob_mean","prob_std"} \
           and not col.startswith("prob_std_"):
            score_cols.append(col)
    name_of = {"prob_mean":"COMMITTEE",
               "prob_dtkinase":"DT-Kinase",
               "prob_drugban":"DrugBAN",
               "prob_conplex":"ConPLex",
               "prob_graphban":"GraphBAN"}
    color_of = {"COMMITTEE":"black", "DT-Kinase":"#1f77b4",
                "DrugBAN":"#d62728", "ConPLex":"#2ca02c", "GraphBAN":"#ff7f0e"}

    # ====== Sensitivity to Kd cutoff ======
    cutoffs = [30, 100, 300, 1000, 3000, 10000]
    rows = []
    kd = merged["Kd_nM"].to_numpy()
    for col in score_cols:
        s = merged[col].to_numpy()
        for c in cutoffs:
            y = (kd <= c).astype(int)
            n_pos = int(y.sum()); n_neg = int((1-y).sum())
            row = {"model": name_of.get(col, col), "cutoff_nM": c,
                   "n_binders": n_pos, "n_non_binders": n_neg}
            if n_pos > 0 and n_neg > 0:
                row["auroc"] = float(roc_auc_score(y, s))
                row["auprc"] = float(average_precision_score(y, s))
            else:
                row["auroc"] = float("nan"); row["auprc"] = float("nan")
            rows.append(row)
    sens = pd.DataFrame(rows)
    print("\n=== AUROC sensitivity across Kd cutoffs ===")
    pivot_auroc = sens.pivot(index="model", columns="cutoff_nM", values="auroc")
    print(pivot_auroc.round(3).to_string())
    print("\nN binders per cutoff:", {c: int((kd<=c).sum()) for c in cutoffs})
    sens.to_csv(out_dir / "davis_sensitivity.csv", index=False)

    # ====== Per-tier breakdown ======
    tiers = [("strong (≤100 nM)", kd <= 100),
             ("mid (100-3000 nM)", (kd > 100) & (kd <= 3000)),
             ("weak (3000-10000 nM)", (kd > 3000) & (kd < 10000)),
             ("non-binder (=10000 nM)", kd == 10000)]
    print("\n=== Per-tier breakdown (n kinases per tier) ===")
    for label, mask in tiers:
        print(f"  {label}: {int(mask.sum())}")

    # Per-tier rank distribution: mean rank of each tier per model
    print("\n=== Mean rank by tier (lower = strong binders ranked higher) ===")
    n = len(merged)
    tier_ranks = []
    for col in score_cols:
        ranked = merged.copy()
        ranked["rank"] = ranked[col].rank(ascending=False, method="min").astype(int)
        for label, mask in tiers:
            if mask.sum() == 0: continue
            mr = ranked.loc[mask, "rank"].mean()
            mh = ranked.loc[mask, "rank"].median()
            tier_ranks.append({"model": name_of.get(col, col),
                               "tier": label, "n": int(mask.sum()),
                               "mean_rank": float(mr),
                               "median_rank": float(mh)})
    tr = pd.DataFrame(tier_ranks)
    pivot_tr = tr.pivot(index="model", columns="tier", values="median_rank")
    print(pivot_tr.round(1).to_string())
    tr.to_csv(out_dir / "davis_tier_ranks.csv", index=False)

    # ====== Canonical targets table ======
    canon = merged[merged["uniprot"].isin(CANONICAL)].copy()
    canon["canon_gene"] = canon["uniprot"].map(CANONICAL)
    canon = canon.sort_values("Kd_nM")
    print("\n=== 9 canonical imatinib targets in DAVIS ∩ FASTA ===")
    cols_show = ["uniprot","canon_gene","Kd_nM"] + score_cols
    print(canon[cols_show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    canon.to_csv(out_dir / "davis_canonical.csv", index=False)

    # ====== Plot 1: ROC curves at Kd ≤ 3000 nM ======
    fig, ax = plt.subplots(figsize=(7, 6))
    y = (kd <= 3000).astype(int)
    for col in score_cols:
        s = merged[col].to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        nm = name_of.get(col, col)
        ax.plot(fpr, tpr, label=f"{nm} (AUROC={auc:.3f})",
                color=color_of.get(nm, "gray"), linewidth=2)
    ax.plot([0,1],[0,1],"--", color="gray", alpha=0.5, label="random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC vs DAVIS imatinib (Kd ≤ 3000 nM, n={int(y.sum())} binders / {len(y)} total)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_per_model.png", dpi=150)
    plt.savefig(out_dir / "roc_per_model.pdf")
    plt.close()
    print(f"\nwrote {out_dir / 'roc_per_model.pdf'}")

    # ====== Plot 2: scatter score vs -log10 Kd ======
    fig, axes = plt.subplots(1, len(score_cols), figsize=(4.2*len(score_cols), 4.2),
                             sharey=True)
    if len(score_cols) == 1: axes = [axes]
    minus_logKd = -np.log10(np.maximum(kd, 0.1))   # higher = stronger binder
    for ax, col in zip(axes, score_cols):
        s = merged[col].to_numpy()
        nm = name_of.get(col, col)
        # Mark canonical targets
        is_canon = merged["uniprot"].isin(CANONICAL).to_numpy()
        ax.scatter(minus_logKd[~is_canon], s[~is_canon], s=15, alpha=0.4,
                   color="gray", label="other")
        ax.scatter(minus_logKd[is_canon], s[is_canon], s=60, alpha=0.9,
                   color="red", edgecolor="black", linewidth=0.5,
                   label="canonical imatinib targets")
        ax.set_title(nm); ax.set_xlabel("−log₁₀(Kd / nM)")
        ax.grid(alpha=0.3)
        if ax is axes[0]: ax.set_ylabel("model probability")
    axes[-1].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "score_vs_kd_scatter.png", dpi=150)
    plt.savefig(out_dir / "score_vs_kd_scatter.pdf")
    plt.close()
    print(f"wrote {out_dir / 'score_vs_kd_scatter.pdf'}")

    # ====== Plot 3: probability distribution by tier ======
    fig, axes = plt.subplots(1, len(score_cols), figsize=(4.2*len(score_cols), 4.2),
                             sharey=True)
    if len(score_cols) == 1: axes = [axes]
    for ax, col in zip(axes, score_cols):
        s = merged[col].to_numpy()
        nm = name_of.get(col, col)
        data = []
        labels = []
        for label, mask in tiers:
            if mask.sum() == 0: continue
            data.append(s[mask]); labels.append(f"{label}\n(n={int(mask.sum())})")
        ax.boxplot(data, labels=labels, showfliers=True, widths=0.6)
        ax.set_title(nm); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        if ax is axes[0]: ax.set_ylabel("model probability")
    plt.tight_layout()
    plt.savefig(out_dir / "score_distribution_by_tier.png", dpi=150)
    plt.savefig(out_dir / "score_distribution_by_tier.pdf")
    plt.close()
    print(f"wrote {out_dir / 'score_distribution_by_tier.pdf'}")

    print(f"\nAll outputs under {out_dir}")


if __name__ == "__main__":
    main()
