"""Aggregate per-model scores into committee consensus.

Reads N CSV files, one per model (dtkinase/drugban/graphban/conplex),
each with columns:
    uniprot, chembl_id, prob, pred, threshold

Emits consensus.csv with:
    pair_id, uniprot, chembl_id,
    prob_dtkinase, prob_drugban, prob_graphban, prob_conplex,
    pred_dtkinase, pred_drugban, pred_graphban, pred_conplex,
    prob_mean, prob_std, agreement_count, tier, rank_fusion, confidence

Aggregation rules:
    prob_mean       = mean of 4 calibrated probabilities
    prob_std        = std (used as inverse confidence)
    agreement_count = # models predicting 1 (each w/ its own threshold)
    confidence      = 1 - prob_std
    rank_fusion     = sum of per-model ranks (Borda count, lower = better)
    tier            = STRONG (4) | LIKELY (3) | UNCERTAIN (2) | UNLIKELY (≤1)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


MODELS = ("dtkinase", "drugban", "graphban", "conplex")


TIER_TABLES = {
    # Per Anexo B Tabela B.2 (n=4) and Tabela B.6 (n<4 rescale).
    4: {4: "STRONG", 3: "LIKELY", 2: "UNCERTAIN", 1: "UNLIKELY", 0: "UNLIKELY"},
    3: {3: "STRONG", 2: "LIKELY", 1: "UNCERTAIN", 0: "UNLIKELY"},
    2: {2: "STRONG", 1: "LIKELY",                 0: "UNLIKELY"},
}


def assign_tier(agreement: int, n_models: int) -> str:
    """Map (agreement, n_models) to tier label via the explicit Anexo B tables.

    Direct lookup (no ratio math): the per-n thresholds in Anexo B are
    not strictly proportional (1/4 = UNLIKELY but 1/3 = UNCERTAIN), so a
    formulaic mapping is harder to verify than a table. The committee
    requires n >= 2 (load_model_scores enforces this).
    """
    if n_models not in TIER_TABLES:
        raise ValueError(f"unsupported committee size n={n_models}; valid: 2, 3, 4")
    return TIER_TABLES[n_models][int(agreement)]


def load_model_scores(scores_dir: Path) -> dict[str, pd.DataFrame]:
    """Load scores_<model>.csv for each baseline; warn if missing.

    Each file is deduplicated by (uniprot, chembl_id) BEFORE the outer-join
    in merge_scores. Without this, an outer-join between four CSVs that
    each contain N duplicate rows for the same (uniprot, chembl_id) key
    produces an N**4 cartesian explosion. The dedupe rule is to average
    duplicate probabilities (and OR-fold predictions): if the same key
    appears multiple times in one model's CSV — e.g. because the user
    submitted a redundant pairs.tsv or a baseline emitted predictions per
    label-disambiguation row — the row reflects the per-key consensus of
    that model rather than an arbitrary first-occurrence pick.
    """
    out = {}
    for m in MODELS:
        path = scores_dir / f"scores_{m}.csv"
        if not path.exists():
            print(f"warning: scores_{m}.csv missing — committee will skip {m}")
            continue
        df = pd.read_csv(path)
        required = {"uniprot", "chembl_id", "prob", "pred", "threshold"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        # Dedupe by (uniprot, chembl_id): average prob, max pred, first thr.
        n_before = len(df)
        df = (df.groupby(["uniprot", "chembl_id"], as_index=False)
                .agg(prob=("prob", "mean"),
                     pred=("pred", "max"),
                     threshold=("threshold", "first")))
        n_after = len(df)
        if n_after < n_before:
            print(f"  {m}: deduped {n_before} → {n_after} rows "
                  f"({n_before - n_after} duplicate keys collapsed)")

        out[m] = df.rename(columns={
            "prob": f"prob_{m}", "pred": f"pred_{m}", "threshold": f"thr_{m}",
        })
    if len(out) < 2:
        raise RuntimeError(
            f"need at least 2 model score files for committee; got {list(out)}"
        )
    return out


def merge_scores(model_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join on (uniprot, chembl_id) — pairs missing from a model become NaN.

    Inputs are guaranteed deduplicated by load_model_scores, so the result
    has at most one row per (uniprot, chembl_id) key.
    """
    keys = ["uniprot", "chembl_id"]
    df = None
    for m, d in model_dfs.items():
        df = d if df is None else df.merge(d, on=keys, how="outer")
    return df


def aggregate(df: pd.DataFrame, models_present: list[str]) -> pd.DataFrame:
    prob_cols = [f"prob_{m}" for m in models_present]
    pred_cols = [f"pred_{m}" for m in models_present]

    probs = df[prob_cols].to_numpy(dtype=float)
    preds = df[pred_cols].to_numpy(dtype=float)

    df["prob_mean"] = np.nanmean(probs, axis=1)
    df["prob_std"]  = np.nanstd(probs, axis=1)
    df["confidence"] = 1.0 - df["prob_std"]
    df["agreement_count"] = np.nansum(preds, axis=1).astype(int)

    # Rank fusion: per-model rank by descending probability (lower rank = better).
    # Average ranks across models, ignoring NaN.
    ranks = np.full_like(probs, np.nan)
    for j, col in enumerate(prob_cols):
        col_probs = probs[:, j]
        valid = ~np.isnan(col_probs)
        order = np.argsort(-col_probs[valid])  # descending
        ranks_valid = np.empty_like(order, dtype=float)
        ranks_valid[order] = np.arange(1, len(order) + 1)
        ranks[valid, j] = ranks_valid
    df["rank_fusion"] = np.nansum(ranks, axis=1)

    n_models = len(models_present)
    df["tier"] = df["agreement_count"].map(
        lambda c: assign_tier(int(c), n_models)
    )

    df["pair_id"] = df["uniprot"].astype(str) + "__" + df["chembl_id"].astype(str)
    return df


def order_columns(df: pd.DataFrame, models_present: list[str]) -> pd.DataFrame:
    base = ["pair_id", "uniprot", "chembl_id"]
    per_model = [
        col for m in models_present
        for col in (f"prob_{m}", f"pred_{m}", f"thr_{m}")
    ]
    committee = ["prob_mean", "prob_std", "confidence",
                 "agreement_count", "tier", "rank_fusion"]
    return df[base + per_model + committee]


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate per-model scores into committee.")
    ap.add_argument("--scores-dir", type=Path, required=True,
                    help="dir containing scores_<model>.csv files")
    ap.add_argument("--out", type=Path, required=True,
                    help="output consensus.csv path")
    ap.add_argument("--top-k", type=int, default=0,
                    help="if > 0, also write top-K by prob_mean to <out>.top.csv")
    args = ap.parse_args()

    model_dfs = load_model_scores(args.scores_dir)
    models_present = list(model_dfs.keys())
    merged = merge_scores(model_dfs)
    out = aggregate(merged, models_present)
    out = order_columns(out, models_present)

    out_sorted = out.sort_values(
        by=["prob_mean", "agreement_count", "confidence"],
        ascending=[False, False, False],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_sorted.to_csv(args.out, index=False)
    print(f"wrote {len(out_sorted)} rows → {args.out}")
    print(f"models in committee: {models_present}")
    print(out_sorted["tier"].value_counts().to_string())

    if args.top_k > 0:
        top = out_sorted.head(args.top_k)
        top_path = args.out.with_suffix(".top.csv")
        top.to_csv(top_path, index=False)
        print(f"wrote top-{args.top_k} → {top_path}")


if __name__ == "__main__":
    main()
