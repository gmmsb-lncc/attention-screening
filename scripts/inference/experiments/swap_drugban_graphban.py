#!/usr/bin/env python3
"""Head-to-head: 3-model committee with DrugBAN vs 3-model with GraphBAN.

   committee_with_DrugBAN  = DT-Kinase + DrugBAN  + ConPLex
   committee_with_GraphBAN = DT-Kinase + GraphBAN + ConPLex

Protocol: dedupe + block bootstrap by protein, B = 10000, IC95 percentile.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "inference" / "experiments"))

from committee_vs_individual import (  # type: ignore
    load_5seed, system_metrics, paired_bootstrap_delta,
    dedupe_predictions, load_test_keys,
)

CORPORA = ["non_human", "human", "all"]
N_BOOT = 10_000
OUT_DIR = REPO / "results" / "inference" / "swap_drugban_graphban"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    metrics: dict[str, dict] = {}

    for corpus in CORPORA:
        print(f"\n=== {corpus} ===")
        models = ["dtkinase", "drugban", "graphban", "conplex"]
        probs: dict[str, np.ndarray] = {}
        thrs: dict[str, float] = {}
        y_true: np.ndarray | None = None
        for m in models:
            p, y, t = load_5seed(m, corpus)
            probs[m] = p; thrs[m] = t
            y_true = y if y_true is None else y_true

        keys, _ = load_test_keys(corpus)
        n0 = len(y_true)

        # Dedupe
        probs_d: dict[str, np.ndarray] = {}
        y_true_d: np.ndarray | None = None
        for m in models:
            p_d, y_d, k_d = dedupe_predictions(probs[m], y_true, keys)
            probs_d[m] = p_d
            if y_true_d is None:
                y_true_d = y_d
                seq_ids_d = np.array([k.split("__")[0] for k in k_d])

        n1 = len(y_true_d)
        print(f"  dedupe: {n0} → {n1}")

        # Two competing 3-model committees
        with_dban = ["dtkinase", "drugban", "conplex"]
        with_gban = ["dtkinase", "graphban", "conplex"]

        prob_dban = np.mean([probs_d[m] for m in with_dban], axis=0)
        thr_dban = np.mean([thrs[m] for m in with_dban])

        prob_gban = np.mean([probs_d[m] for m in with_gban], axis=0)
        thr_gban = np.mean([thrs[m] for m in with_gban])

        m_dban = system_metrics(y_true_d, prob_dban, thr_dban)
        m_gban = system_metrics(y_true_d, prob_gban, thr_gban)

        metrics[corpus] = dict(
            with_drugban=m_dban,
            with_graphban=m_gban,
            delta_mcc=m_dban["mcc"] - m_gban["mcc"],
        )

        # Paired bootstrap: a = with_DrugBAN, b = with_GraphBAN
        d = paired_bootstrap_delta(
            y_true=y_true_d,
            prob_a=prob_dban, thr_a=thr_dban,
            prob_b=prob_gban, thr_b=thr_gban,
            n_boot=N_BOOT, seed=42,
            blocks=seq_ids_d,
        )
        fp = d["frac_positive"]
        p_two = max(2 * min(fp, 1 - fp), 1.0 / N_BOOT)
        rows.append(dict(
            corpus=corpus,
            mcc_with_drugban=m_dban["mcc"],
            mcc_with_graphban=m_gban["mcc"],
            delta_mcc=d["delta_mean"],
            ci_lo=d["ci_lo"], ci_hi=d["ci_hi"],
            frac_positive=fp,
            p_two=p_two,
        ))

    # REPORT
    lines: list[str] = []
    lines.append("# Substituição DrugBAN ↔ GraphBAN no comitê 3-modelos\n")
    lines.append("Comparação direta entre dois comitês de 3 modelos que diferem "
                 "apenas no membro BAN:\n")
    lines.append("- **A: with DrugBAN** = DT-Kinase + **DrugBAN** + ConPLex")
    lines.append("- **B: with GraphBAN** = DT-Kinase + **GraphBAN** + ConPLex\n")
    lines.append(f"Protocolo: dedupe + block bootstrap por proteína, "
                 f"$B = {N_BOOT}$, IC95 percentílico, limiar canônico $\\overline{{\\tau}}$.\n")

    lines.append("## Métricas absolutas (3 corpora)\n")
    lines.append("| Corpus | MCC (with DrugBAN) | MCC (with GraphBAN) | "
                 "Δ absoluto | AUROC (DBAN/GBAN) | F1 (DBAN/GBAN) |")
    lines.append("|---|---|---|---|---|---|")
    for corpus in CORPORA:
        m = metrics[corpus]
        lines.append(
            f"| {corpus} | {m['with_drugban']['mcc']:.4f} | "
            f"{m['with_graphban']['mcc']:.4f} | {m['delta_mcc']:+.4f} | "
            f"{m['with_drugban']['auroc']:.4f} / {m['with_graphban']['auroc']:.4f} | "
            f"{m['with_drugban']['f1']:.4f} / {m['with_graphban']['f1']:.4f} |"
        )
    lines.append("")

    lines.append("## Comparação pareada head-to-head\n")
    lines.append("Δ > 0 favorece **with-DrugBAN** (substituição GraphBAN→DrugBAN melhora). "
                 "Δ < 0 favorece **with-GraphBAN**.\n")
    lines.append("| Corpus | Δ_MCC (DBAN − GBAN) | IC 95\\% | $P(\\Delta > 0)$ | "
                 "$p_{\\text{bilateral}}$ | Veredito |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        p_disp = f"{r['p_two']:.4f}" if r['p_two'] >= 1e-4 else "$10^{-4}$"
        if r["ci_lo"] > 0:
            verdict = "with-DrugBAN lidera ▲"
        elif r["ci_hi"] < 0:
            verdict = "with-GraphBAN lidera ▼"
        else:
            verdict = "indistinguível ⊘"
        lines.append(
            f"| {r['corpus']} | {r['delta_mcc']:+.4f} | {ci} | "
            f"{r['frac_positive']:.4f} | {p_disp} | {verdict} |"
        )
    lines.append("")

    lines.append("## Resumo\n")
    n_dban = sum(1 for r in rows if r["ci_lo"] > 0)
    n_gban = sum(1 for r in rows if r["ci_hi"] < 0)
    n_tie = 3 - n_dban - n_gban
    lines.append(f"- with-DrugBAN lidera: **{n_dban}/3 corpora**")
    lines.append(f"- with-GraphBAN lidera: {n_gban}/3 corpora")
    lines.append(f"- empate: {n_tie}/3 corpora\n")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    pd.DataFrame(rows).to_csv(OUT_DIR / "swap_results.csv", index=False)
    print(f"\n  → wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
