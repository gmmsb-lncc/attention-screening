#!/usr/bin/env python3
"""Aggregate per-family MCC across 5 seeds, emit LaTeX table for thesis."""

from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "per_family_mcc"
SEEDS = [42, 123, 456, 789, 1024]
MODELS = ["dtkinase", "drugban", "graphban", "conplex"]
CORPORA = ["non_human", "human", "all"]
MODEL_LABEL = {
    "dtkinase": "DT-Kinase",
    "drugban":  "DrugBAN",
    "graphban": "GraphBAN",
    "conplex":  "ConPLex",
}
CORPUS_LABEL = {
    "non_human": "Não-Humano",
    "human":     "Human",
    "all":       "All",
}
FAMILY_ORDER = ["TK", "CMGC", "AGC", "CAMK", "CK1", "STE", "TKL", "Atypical", "Other", "Other_parasite", "Pseudokinase", "unmapped"]


def aggregate():
    rows = []
    for corpus in CORPORA:
        for model in MODELS:
            dfs = []
            for s in SEEDS:
                pth = RES / f"{corpus}_{model}_seed{s}.csv"
                if not pth.exists():
                    continue
                df = pd.read_csv(pth)
                df["seed"] = s
                dfs.append(df)
            if not dfs:
                continue
            cat = pd.concat(dfs, ignore_index=True)
            agg = cat.groupby("family")["mcc"].agg(["mean", "std", "count"]).reset_index()
            n_per_seed = cat.groupby("family")["n"].first().to_dict()
            agg["n"] = agg["family"].map(n_per_seed)
            agg["model"] = model
            agg["corpus"] = corpus
            rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def to_latex(agg: pd.DataFrame) -> str:
    """Emit one table per corpus, columns = models, rows = families."""
    out = []
    for corpus in CORPORA:
        sub = agg[agg["corpus"] == corpus]
        # pivot: family rows, model cols (mean ± std)
        wide = sub.pivot_table(index="family", columns="model",
                               values=["mean", "std", "n"], aggfunc="first")
        out.append(f"\n\\subsection*{{Corpus {CORPUS_LABEL[corpus]}}}\n")
        out.append("\\begin{tabular}{lrrrrr}")
        out.append("\\toprule")
        out.append("Família & $n$ & DT-Kinase & DrugBAN & GraphBAN & ConPLex \\\\")
        out.append("\\midrule")
        ordered = [f for f in FAMILY_ORDER if f in wide.index]
        for fam in ordered:
            n = int(wide.loc[fam, ("n", "dtkinase")]) if ("n","dtkinase") in wide.columns and not pd.isna(wide.loc[fam, ("n","dtkinase")]) else "n/a"
            cells = []
            for m in MODELS:
                if ("mean", m) in wide.columns and not pd.isna(wide.loc[fam, ("mean", m)]):
                    mu = wide.loc[fam, ("mean", m)]
                    sd = wide.loc[fam, ("std",  m)]
                    cells.append(f"${mu:.3f} \\pm {sd:.3f}$".replace(".", "{,}"))
                else:
                    cells.append("\\textit{n/a}")
            label = fam.replace("_", "\\_")
            out.append(f"{label} & ${n}$ & " + " & ".join(cells) + " \\\\")
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
    return "\n".join(out)


def main():
    agg = aggregate()
    out_csv = RES / "aggregated_5seed.csv"
    agg.to_csv(out_csv, index=False)
    print(f"Aggregated: {out_csv}")
    print(agg.pivot_table(index=["corpus", "family"], columns="model",
                          values="mean", aggfunc="first").round(3).to_string())
    tex = to_latex(agg)
    out_tex = RES / "per_family_table.tex"
    out_tex.write_text(tex)
    print(f"\nLaTeX: {out_tex}")


if __name__ == "__main__":
    main()
