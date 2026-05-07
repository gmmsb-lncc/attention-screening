"""RM-ANOVA + Tukey HSD complementary verification (D2 of protocol).

Implements Guideline 2 of Ash/Wognum 2025 as a parallel inference
layer to the paired bootstrap (which remains the primary test per
deviation D2).

Per metric: repeated-measures ANOVA with seed as the subject and
model as the within-factor. Inter-metric multiplicity is controlled by
Bonferroni on the ANOVA p-value (m = 4 primary metrics by default,
alpha' = alpha / m). For metrics passing the Bonferroni-adjusted ANOVA,
Tukey HSD pairwise intervals are reported (FWER-controlled within the
metric by construction).

With n = 5 seeds per cell the test is acknowledged to be underpowered
(see deviation D1 in statistical_protocol.md); the layer is still
reported because (i) Tukey HSD gives FWER-controlled simultaneous CIs
that are more interpretable than the bootstrap deltas, and (ii) the
intersection of "ANOVA-significant" and "bootstrap CI excludes zero"
is a stronger joint signal than either alone.

CLI:
    python -m scripts.statistical_analysis.anova_tukey \\
        --corpus non_human \\
        --out results/statistical/non_human/anova_tukey.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from . import MODELS, PRIMARY_METRICS, SEEDS, data_loader


def _per_seed_metric(seed_data: dict, metric: str) -> float:
    y = seed_data["y_true"]
    p = seed_data["y_prob"]
    pred = (p >= seed_data["threshold"]).astype(np.int64)
    if metric == "mcc":
        return float(matthews_corrcoef(y, pred)) if y.std() > 0 else 0.0
    if metric == "auroc":
        return float(roc_auc_score(y, p)) if len(set(y)) == 2 else 0.5
    if metric == "auprc":
        return float(average_precision_score(y, p)) if len(set(y)) == 2 else float(np.mean(y))
    if metric == "f1":
        return float(f1_score(y, pred, zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    if metric == "precision":
        return float(precision_score(y, pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y, pred, zero_division=0))
    raise ValueError(metric)


def _build_long_df(panel: dict, metric: str) -> pd.DataFrame:
    rows = []
    for model, seeds_data in panel.items():
        for sd in seeds_data:
            rows.append({
                "subject": int(sd["seed"]),
                "model": model,
                "value": _per_seed_metric(sd, metric),
            })
    return pd.DataFrame(rows)


def _run_anova(df: pd.DataFrame) -> dict:
    """Repeated-measures ANOVA: subject = seed, factor = model."""
    aov = AnovaRM(df, depvar="value", subject="subject",
                  within=["model"]).fit()
    table = aov.anova_table
    F = float(table.loc["model", "F Value"])
    p = float(table.loc["model", "Pr > F"])
    df_num = float(table.loc["model", "Num DF"])
    df_den = float(table.loc["model", "Den DF"])
    return {
        "F": F, "p": p,
        "df_numerator": df_num, "df_denominator": df_den,
    }


def _run_tukey(df: pd.DataFrame, alpha: float = 0.05) -> list[dict]:
    res = pairwise_tukeyhsd(endog=df["value"].to_numpy(),
                            groups=df["model"].to_numpy(),
                            alpha=alpha)
    summary = res.summary().data
    header = summary[0]
    rows = []
    for r in summary[1:]:
        entry = dict(zip(header, r))
        rows.append({
            "group1": str(entry["group1"]),
            "group2": str(entry["group2"]),
            "meandiff": float(entry["meandiff"]),
            "p_adj": float(entry["p-adj"]),
            "lower": float(entry["lower"]),
            "upper": float(entry["upper"]),
            "reject": bool(entry["reject"]),
        })
    return rows


def anova_tukey_panel(corpus: str, metrics: list[str],
                     bonferroni_m: int) -> dict:
    panel = data_loader.load_panel(corpus, MODELS, SEEDS)
    by_metric = {}
    for metric in metrics:
        df = _build_long_df(panel, metric)
        anova_summary = _run_anova(df)
        anova_summary["p_bonf"] = float(min(1.0,
                                            bonferroni_m * anova_summary["p"]))
        tukey = _run_tukey(df, alpha=0.05)
        per_model_summary = (df.groupby("model")["value"]
                             .agg(["mean", "std"]).to_dict("index"))
        by_metric[metric] = {
            "anova": anova_summary,
            "tukey": tukey,
            "per_model": {
                m: {"mean": float(v["mean"]), "std": float(v["std"])}
                for m, v in per_model_summary.items()
            },
        }

    return {
        "corpus": corpus,
        "metrics": list(metrics),
        "bonferroni_m": int(bonferroni_m),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "by_metric": by_metric,
        "note": (
            "Complementary to paired bootstrap (D2). Underpowered with "
            "n=5 seeds (see D1)."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("human", "non_human", "all"))
    ap.add_argument("--metrics", nargs="+", default=list(PRIMARY_METRICS))
    ap.add_argument("--bonferroni-m", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = anova_tukey_panel(args.corpus, args.metrics, args.bonferroni_m)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[anova_tukey] corpus={args.corpus} bonferroni_m={args.bonferroni_m}")
    for metric in args.metrics:
        a = result["by_metric"][metric]["anova"]
        sig = "*" if a["p_bonf"] < 0.05 else " "
        print(f"  {metric:6s} ANOVA F={a['F']:.3f} p={a['p']:.4f} "
              f"p_bonf={a['p_bonf']:.4f}{sig}")
        for t in result["by_metric"][metric]["tukey"]:
            mark = "REJECT" if t["reject"] else "      "
            print(f"    {t['group1']:10s} vs {t['group2']:10s} "
                  f"diff={t['meandiff']:+.4f} p_adj={t['p_adj']:.4f} "
                  f"CI=[{t['lower']:+.4f}, {t['upper']:+.4f}] {mark}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
