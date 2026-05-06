#!/usr/bin/env python3
"""Formal validation of the 3-model committee (DT-Kinase + DrugBAN + ConPLex)
following Annex B §B.5 protocol: dedupe + block bootstrap by protein +
Holm-Bonferroni multiple testing correction.

Outputs results/inference/committee_no_graphban_holm/REPORT.md with:
  - Table A: Holm-Bonferroni m=9 (committee_no_graphban vs 3 individuals × 3 corpora)
  - Table B: Direct head-to-head paired bootstrap, committee_no_graphban vs
             4-model canonical committee (3 corpora).

Reuses helpers from committee_vs_individual.py.
"""
from __future__ import annotations

import json
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
ALPHA = 0.05
OUT_DIR = REPO / "results" / "inference" / "committee_no_graphban_holm"


def per_corpus_setup(corpus: str) -> dict:
    """Load 5-seed probs/thrs for all 4 models, dedupe + block bootstrap setup."""
    print(f"\n=== {corpus} ===")
    models_full = ["dtkinase", "drugban", "graphban", "conplex"]

    probs_per_model: dict[str, np.ndarray] = {}
    thrs_per_model: dict[str, float] = {}
    y_true: np.ndarray | None = None
    for m in models_full:
        prob, yt, thr = load_5seed(m, corpus)
        probs_per_model[m] = prob
        thrs_per_model[m] = thr
        y_true = yt if y_true is None else y_true

    keys, seq_ids = load_test_keys(corpus)
    n0 = len(y_true)

    # Dedupe by (seq_id, chembl_id) — Refinement I.
    seq_ids_d: np.ndarray | None = None
    y_true_d: np.ndarray | None = None
    probs_d: dict[str, np.ndarray] = {}
    for m in models_full:
        p_d, y_d, k_d = dedupe_predictions(probs_per_model[m], y_true, keys)
        probs_d[m] = p_d
        if y_true_d is None:
            y_true_d = y_d
            # Realign seq_ids to dedupe-sorted keys.
            key_to_seq = dict(zip(keys, seq_ids))
            seq_ids_d = np.array([key_to_seq[k.split("__")[0] + "__" + k.split("__")[1]]
                                  if "__" in k
                                  else key_to_seq.get(k, "")
                                  for k in k_d])
            # Simpler: rebuild from k_d
            seq_ids_d = np.array([k.split("__")[0] for k in k_d])
        else:
            assert np.array_equal(y_true_d, y_d), f"dedupe mismatch {m}"
    n1 = len(y_true_d)
    print(f"  dedupe: {n0} → {n1}  (uniques)")

    # Committees
    full_prob = np.mean([probs_d[m] for m in models_full], axis=0)
    full_thr = np.mean([thrs_per_model[m] for m in models_full])

    no_g_prob = np.mean([probs_d[m] for m in ["dtkinase", "drugban", "conplex"]], axis=0)
    no_g_thr = np.mean([thrs_per_model[m] for m in ["dtkinase", "drugban", "conplex"]])

    return dict(
        y_true=y_true_d, seq_ids=seq_ids_d,
        probs_d=probs_d, thrs=thrs_per_model,
        full_prob=full_prob, full_thr=full_thr,
        no_g_prob=no_g_prob, no_g_thr=no_g_thr,
        n_unique=n1,
    )


def holm_bonferroni(p_values: list[float]) -> tuple[list[float], list[bool]]:
    """Apply Holm-Bonferroni step-down correction. Returns (p_holm, survive)."""
    m = len(p_values)
    order = np.argsort(p_values)
    p_holm = np.zeros(m)
    running = 0.0
    for k, idx in enumerate(order):
        candidate = (m - k) * p_values[idx]
        running = max(running, candidate)
        p_holm[idx] = min(running, 1.0)
    survive = (p_holm < ALPHA).tolist()
    return p_holm.tolist(), survive


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all comparisons across corpora.
    rows_indiv: list[dict] = []
    rows_canonical: list[dict] = []
    metrics_summary: dict[str, dict] = {}

    for corpus in CORPORA:
        d = per_corpus_setup(corpus)
        y = d["y_true"]
        seq_ids = d["seq_ids"]

        # Individual + canonical metrics for the report.
        sys_metrics: dict[str, dict] = {}
        for m in ["dtkinase", "drugban", "graphban", "conplex"]:
            sys_metrics[m] = system_metrics(y, d["probs_d"][m], d["thrs"][m])
        sys_metrics["committee_4model"] = system_metrics(
            y, d["full_prob"], d["full_thr"]
        )
        sys_metrics["committee_3model_no_graphban"] = system_metrics(
            y, d["no_g_prob"], d["no_g_thr"]
        )
        metrics_summary[corpus] = sys_metrics

        # Block bootstrap setup
        unique_blocks, inverse = np.unique(seq_ids, return_inverse=True)
        block_to_pairs = [np.where(inverse == k)[0] for k in range(len(unique_blocks))]

        # Table A rows: 3-model vs each of 3 individuals.
        for m in ["dtkinase", "drugban", "conplex"]:
            d_boot = paired_bootstrap_delta(
                y_true=y, prob_a=d["no_g_prob"], prob_b=d["probs_d"][m],
                thr_a=d["no_g_thr"], thr_b=d["thrs"][m],
                n_boot=N_BOOT, seed=42,
                blocks=seq_ids,
            )
            p_uni = max(1 - d_boot["frac_positive"], 1.0 / N_BOOT)
            rows_indiv.append(dict(
                corpus=corpus, comparison=f"3-model − {m}",
                delta_mean=d_boot["delta_mean"],
                ci_lo=d_boot["ci_lo"], ci_hi=d_boot["ci_hi"],
                frac_positive=d_boot["frac_positive"],
                p_uni=p_uni,
            ))

        # Table B: 3-model (no GraphBAN) vs 4-model canonical.
        d_boot = paired_bootstrap_delta(
            y_true=y, prob_a=d["no_g_prob"], prob_b=d["full_prob"],
            thr_a=d["no_g_thr"], thr_b=d["full_thr"],
            n_boot=N_BOOT, seed=42,
            blocks=seq_ids,
        )
        # Two-sided test (sem-GraphBAN may lose to canonical):
        # use min(frac_positive, 1-frac_positive) doubled.
        fp = d_boot["frac_positive"]
        p_two = 2 * min(fp, 1 - fp)
        p_two = max(p_two, 1.0 / N_BOOT)
        rows_canonical.append(dict(
            corpus=corpus,
            comparison="3-model (sem GraphBAN) − 4-model (canônico)",
            delta_mean=d_boot["delta_mean"],
            ci_lo=d_boot["ci_lo"], ci_hi=d_boot["ci_hi"],
            frac_positive=d_boot["frac_positive"],
            p_two=p_two,
        ))

    # Holm-Bonferroni m=9 for Table A
    p_uni_list = [r["p_uni"] for r in rows_indiv]
    p_holm, survive = holm_bonferroni(p_uni_list)
    for r, ph, sv in zip(rows_indiv, p_holm, survive):
        r["p_holm"] = ph
        r["survive"] = "✓" if sv else "✗"

    # Sort by p_uni for Holm display order
    rows_indiv_sorted = sorted(rows_indiv, key=lambda r: r["p_uni"])

    # Build REPORT.md
    lines: list[str] = []
    lines.append("# Comitê 3-modelos sem GraphBAN — Validação Formal\n")
    lines.append("**Composição**: DT-Kinase + DrugBAN + ConPLex.\n")
    lines.append("**Protocolo**: idêntico ao Anexo B §B.5 da tese — desduplicação por "
                 "`(seq_id, chembl_id)`, *block bootstrap* por proteína, "
                 f"$B = {N_BOOT}$ reamostragens, IC95 percentílico, "
                 f"limiar canônico $\\overline{{\\tau}}$ (média dos limiares "
                 f"individuais), Holm–Bonferroni com $\\alpha = {ALPHA}$.\n")

    # Per-corpus metrics summary
    lines.append("## Métricas por sistema (3 corpora)\n")
    for corpus in CORPORA:
        lines.append(f"### Corpus: {corpus}\n")
        lines.append("| system | mcc | auroc | f1 | accuracy |")
        lines.append("|---|---|---|---|---|")
        for sys_name, met in metrics_summary[corpus].items():
            lines.append(
                f"| {sys_name} | {met['mcc']:.4f} | {met['auroc']:.4f} | "
                f"{met['f1']:.4f} | {met['accuracy']:.4f} |"
            )
        lines.append("")

    # Table A: Holm m=9
    lines.append(f"## Tabela A — Holm–Bonferroni $m = 9$ "
                 f"(3-model vs 3 individuais × 3 corpora)\n")
    lines.append("| $k$ | Corpus | Comparação | $\\Delta_{\\text{MCC}}$ | "
                 "IC 95\\% | $p_{\\text{uni}}$ | $p_{\\text{Holm}}$ | "
                 f"Sobrevive $\\alpha = {ALPHA}$ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k, r in enumerate(rows_indiv_sorted, 1):
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        p_uni_disp = (f"{r['p_uni']:.4f}" if r["p_uni"] >= 1e-4
                      else f"$10^{{-4}}$")
        lines.append(
            f"| {k} | {r['corpus']} | {r['comparison']} | "
            f"{r['delta_mean']:+.4f} | {ci} | {p_uni_disp} | "
            f"{r['p_holm']:.4f} | {r['survive']} |"
        )
    lines.append("")

    # Table B: head-to-head
    lines.append("## Tabela B — Comparação direta: 3-model (sem GraphBAN) vs "
                 "4-model (canônico)\n")
    lines.append("Teste pareado bilateral. $\\Delta > 0$ favorece 3-model; "
                 "$\\Delta < 0$ favorece canônico.\n")
    lines.append("| Corpus | $\\Delta_{\\text{MCC}}$ | IC 95\\% | "
                 "$P(\\Delta > 0)$ | $p_{\\text{bilateral}}$ | Veredito |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows_canonical:
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        p_two_disp = (f"{r['p_two']:.4f}" if r["p_two"] >= 1e-4
                      else f"$10^{{-4}}$")
        if r["ci_lo"] > 0:
            verdict = "3-model lidera ▲"
        elif r["ci_hi"] < 0:
            verdict = "canônico lidera ▼"
        else:
            verdict = "indistinguível ⊘"
        lines.append(
            f"| {r['corpus']} | {r['delta_mean']:+.4f} | {ci} | "
            f"{r['frac_positive']:.4f} | {p_two_disp} | {verdict} |"
        )
    lines.append("")

    # Summary verdicts
    n_lead_indiv = sum(1 for r in rows_indiv if r["ci_lo"] > 0)
    n_tie_indiv = sum(1 for r in rows_indiv if r["ci_lo"] <= 0 <= r["ci_hi"])
    n_holm_survive = sum(1 for r in rows_indiv if r["survive"] == "✓")
    lines.append("## Resumo\n")
    lines.append(f"- **3-model vs individuais (Tabela A):** "
                 f"{n_lead_indiv}/9 lidera (IC95 lo > 0), "
                 f"{n_tie_indiv}/9 indistinguível, "
                 f"{n_holm_survive}/9 sobrevive Holm–Bonferroni "
                 f"$\\alpha_{{\\text{{family}}}}={ALPHA}$.")

    n_canon_lead = sum(1 for r in rows_canonical if r["ci_lo"] > 0)
    n_canon_tie = sum(1 for r in rows_canonical if r["ci_lo"] <= 0 <= r["ci_hi"])
    n_canon_lose = sum(1 for r in rows_canonical if r["ci_hi"] < 0)
    lines.append(f"- **3-model vs 4-model canônico (Tabela B):** "
                 f"{n_canon_lead}/3 lidera, {n_canon_tie}/3 empate, "
                 f"{n_canon_lose}/3 canônico vence.\n")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))

    pd.DataFrame(rows_indiv).to_csv(OUT_DIR / "holm_indiv.csv", index=False)
    pd.DataFrame(rows_canonical).to_csv(OUT_DIR / "vs_canonical.csv", index=False)

    print(f"\n  → wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
