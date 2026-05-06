"""Evaluate committee imatinib predictions against Davis et al. 2011 Kd ground
truth (Nat Biotechnol 29:1046, comprehensive kinome scan).

Inputs:
    --consensus path/to/consensus.csv      (UniProt-keyed committee output)
    --davis-kd  data/external/davis/imatinib_kd.csv   (442 kinases × Kd_nM)
    --mapping   data/reference/kinome_human_mapping.tsv  (UniProt → gene)

Outputs metrics per model + committee:
    AUROC, AUPRC over Kd-derived binary labels (cutoff configurable)
    Enrichment Factor at top-1%, 5%, 10%
    Recall@K of canonical imatinib targets
    Spearman correlation: model_score vs -log10(Kd)
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef)
from scipy.stats import spearmanr


CANONICAL = {
    "P00519": "ABL1", "P42684": "ABL2", "P10721": "KIT",
    "P16234": "PDGFRA", "P09619": "PDGFRB", "P07333": "CSF1R",
    "Q08345": "DDR1", "Q16832": "DDR2", "P06239": "LCK",
}


def davis_to_gene(name: str) -> str:
    """DAVIS keys like 'ABL1(F317L)', 'ABL1p', 'CDK7-cyclinH-MNAT1' →
    canonical gene symbol used as primary key.

    The dataset uses gene symbols with parenthesised mutation suffixes,
    a trailing 'p' for phosphorylated forms, and hyphenated complexes.
    Strategy: take the first symbol before any '(' / hyphen / 'p$'.
    """
    s = name.split("(")[0]
    s = s.split("-")[0]
    s = re.sub(r"p$", "", s)
    return s.upper()


def build_davis_lookup(davis_df: pd.DataFrame, mapping_df: pd.DataFrame
                       ) -> pd.DataFrame:
    """Return per-gene min-Kd table joined to UniProt accession.

    DAVIS measures multiple variants per gene (mutants, phospho forms);
    we collapse by gene, taking min Kd as the most-favourable binding
    measurement, which is the convention in selectivity benchmarks.
    """
    davis_df = davis_df.copy()
    davis_df["gene"] = davis_df["davis_kinase"].apply(davis_to_gene)
    by_gene = (davis_df.groupby("gene", as_index=False)
                       .agg(Kd_nM=("Kd_nM", "min"),
                            n_variants=("davis_kinase", "count")))
    # Map gene symbol -> UniProt via mapping.tsv (gene column)
    map_df = mapping_df.copy()
    map_df["gene"] = map_df["gene"].str.upper()
    merged = by_gene.merge(map_df[["uniprot", "gene"]], on="gene", how="left")
    return merged


def metrics_for(scores: np.ndarray, kd: np.ndarray, kd_cutoff_nM: float,
                top_pct_list=(1, 5, 10)) -> dict:
    """Compute AUROC/AUPRC/EF/Spearman given continuous scores and Kd values.

    Binders are defined as Kd <= kd_cutoff_nM. Missing Kd entries (NaN) and
    the DAVIS saturation value (10000 nM, "no binding") are treated as
    non-binders unless explicitly excluded upstream.
    """
    valid = ~np.isnan(scores) & ~np.isnan(kd)
    if valid.sum() < 10:
        return {"n_valid": int(valid.sum())}
    s, k = scores[valid], kd[valid]
    y = (k <= kd_cutoff_nM).astype(int)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    out = {
        "n_valid": int(valid.sum()),
        "n_binders": n_pos,
        "n_non_binders": n_neg,
        "kd_cutoff_nM": kd_cutoff_nM,
    }
    if n_pos == 0 or n_neg == 0:
        out["auroc"] = float("nan"); out["auprc"] = float("nan")
        return out
    out["auroc"] = float(roc_auc_score(y, s))
    out["auprc"] = float(average_precision_score(y, s))
    rho, _ = spearmanr(s, -np.log10(np.maximum(k, 1e-3)))
    out["spearman"] = float(rho)
    # Enrichment factors
    order = np.argsort(-s)  # descending
    n = len(s)
    base_rate = n_pos / n
    for pct in top_pct_list:
        k_top = max(1, int(np.ceil(n * pct / 100)))
        hit_rate = y[order[:k_top]].sum() / k_top
        out[f"EF_{pct}pct"] = float(hit_rate / base_rate) if base_rate > 0 else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", type=Path, required=True)
    ap.add_argument("--davis-kd",  type=Path,
                    default=Path("data/external/davis/imatinib_kd.csv"))
    ap.add_argument("--mapping",   type=Path,
                    default=Path("data/reference/kinome_human_mapping.tsv"))
    ap.add_argument("--kd-cutoff", type=float, default=3000.0,
                    help="Kd threshold (nM) for binder/non-binder")
    args = ap.parse_args()

    cons = pd.read_csv(args.consensus)
    cons["uniprot"] = cons["uniprot"].astype(str)
    davis = pd.read_csv(args.davis_kd)
    mapping = pd.read_csv(args.mapping, sep="\t")

    lookup = build_davis_lookup(davis, mapping)

    # Diagnostics: how many DAVIS genes mapped to a UniProt in our FASTA?
    n_mapped = lookup["uniprot"].notna().sum()
    n_total = len(lookup)
    print(f"DAVIS gene→UniProt mapping: {n_mapped}/{n_total} mapped to FASTA")
    n_in_cons = lookup[lookup["uniprot"].isin(cons["uniprot"])].shape[0]
    print(f"  ...of which {n_in_cons} are in committee consensus")

    merged = cons.merge(lookup[["uniprot", "Kd_nM", "gene"]],
                        on="uniprot", how="inner")
    print(f"  joined rows: {len(merged)} (n with Kd ground truth)")

    # Score columns to evaluate
    score_cols = ["prob_mean"]
    for col in cons.columns:
        if col.startswith("prob_") and col not in {"prob_mean", "prob_std"} \
           and not col.startswith("prob_std_"):
            score_cols.append(col)

    # Pretty per-model name
    name_of = {"prob_mean": "COMMITTEE",
               "prob_dtkinase": "DT-Kinase",
               "prob_drugban":  "DrugBAN",
               "prob_conplex":  "ConPLex",
               "prob_graphban": "GraphBAN"}

    # ===================== Metrics table =====================
    rows = []
    for col in score_cols:
        scores = merged[col].to_numpy(dtype=float)
        kd = merged["Kd_nM"].to_numpy(dtype=float)
        m = metrics_for(scores, kd, args.kd_cutoff)
        m["model"] = name_of.get(col, col)
        rows.append(m)
    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[["model", "n_valid", "n_binders", "n_non_binders",
                             "auroc", "auprc", "spearman",
                             "EF_1pct", "EF_5pct", "EF_10pct"]]
    print("\n=== Metrics vs DAVIS (Kd cutoff =",
          f"{args.kd_cutoff:.0f} nM, binders={metrics_df['n_binders'].iloc[0]}"
          f", non-binders={metrics_df['n_non_binders'].iloc[0]}) ===")
    print(metrics_df.to_string(index=False))

    out_csv = args.consensus.parent / "metrics_vs_davis.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # ===================== Recall@K of canonical targets =====================
    canon_in = merged[merged["uniprot"].isin(CANONICAL)].copy()
    canon_in["canon_gene"] = canon_in["uniprot"].map(CANONICAL)
    print(f"\n=== Canonical imatinib targets in consensus ∩ DAVIS "
          f"({len(canon_in)}/{len(CANONICAL)}) ===")
    print(canon_in[["uniprot", "canon_gene", "Kd_nM"] + score_cols]
          .sort_values("Kd_nM").to_string(index=False))

    # Recall@K table
    recall_rows = []
    for col in score_cols:
        ranked = merged.sort_values(col, ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        for K in (10, 20, 50, 100):
            n_canon_in_topK = ranked.head(K)["uniprot"].isin(CANONICAL).sum()
            recall_rows.append({
                "model":  name_of.get(col, col),
                "K":      K,
                "canon_in_topK": int(n_canon_in_topK),
                "canon_total": int((merged["uniprot"].isin(CANONICAL)).sum()),
            })
    rdf = pd.DataFrame(recall_rows)
    print("\n=== Recall@K of canonical imatinib targets (in consensus ∩ DAVIS) ===")
    pivot = rdf.pivot(index="model", columns="K", values="canon_in_topK")
    pivot.columns = [f"top-{c}" for c in pivot.columns]
    print(pivot.to_string())


if __name__ == "__main__":
    main()
