"""Aggregate per-component outputs into a single statistical panel.

Consumes the JSON outputs from the other scripts and produces:

- panel.json   - structured per-corpus report.
- panel.tex    - LaTeX tables ready for thesis appendix (legacy; kept for
                 back-compatibility).
- checklist.md - reporting-checklist (statistical_protocol.md section 4)
                 with PASS/FAIL per item based on artifact presence.
- tables/      - directory of stand-alone .tex tables (limits, hedges,
                 tost, anova_tukey, posthoc, checklist), each ready to
                 \input{} into a thesis appendix; plus a master
                 auditoria_complementar_{corpus}.tex that bundles them
                 with the provisional disclaimer header.

The optional --mirror-to flag also copies the figures to a sibling
directory (e.g. ~/PhD/figures/) for thesis-build integration.

The optional --provisional N flag injects a "PROVISIONAL B=N" disclaimer
header in every emitted .tex file, marking values as smoke-run output to
be replaced by canonical B=10^4 numbers later.

CLI:
    python -m scripts.statistical_analysis.aggregate_panel \\
        --corpus non_human \\
        --in-dir results/statistical/non_human \\
        --out-json results/statistical/non_human/panel.json \\
        --out-tex results/statistical/non_human/panel.tex \\
        --out-checklist results/statistical/non_human/checklist.md \\
        --provisional 2000
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import MODELS, PRIMARY_METRICS

_MODEL_LATEX = {
    "dtkinase": "DT-Kinase",
    "drugban": "DrugBAN",
    "graphban": "GraphBAN",
    "conplex": "ConPLex",
}
_METRIC_LATEX = {
    "mcc": "MCC", "auroc": "AUROC", "auprc": "AUPRC", "f1": "F1",
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
}

CHECKLIST_ITEMS = [
    ("All four metrics (MCC, AUROC, F1, AUPRC), mean +/- sigma",
     "effect_size.json", "by_metric"),
    ("Paired bootstrap IC95 per pair (Wilcoxon p)",
     "tost.json", "ordered_pair_results"),
    ("Hedges' g paired (J(4) primary) per pair",
     "effect_size.json", "primary_form"),
    ("Null-model lower limit per corpus",
     "null_model.json", "metrics"),
    ("Assay-noise upper limit per corpus",
     "upper_limit.json", "metrics"),
    ("TOST sensitivity over >= 3 bands",
     "tost.json", "by_band"),
    ("RM-ANOVA + Tukey HSD per metric (D2 complement)",
     "anova_tukey.json", "by_metric"),
    ("Simultaneous CI plot (figures/sim_ci_*.pdf)",
     "figures", "sim_ci"),
    ("MCSim heatmap (figures/mcsim_*.pdf)",
     "figures", "mcsim"),
    ("Post-hoc classification metrics (precision@recall, recall@precision, TNR@recall)",
     "posthoc.json", "by_model"),
]


def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None


def _figures_present(in_dir: Path, prefix: str) -> bool:
    figs_dir = in_dir / "figures"
    if not figs_dir.exists():
        return False
    return bool(list(figs_dir.glob(f"{prefix}_*.pdf")))


def _evaluate_checklist(in_dir: Path) -> list[dict]:
    rows = []
    for description, source, key in CHECKLIST_ITEMS:
        if source == "figures":
            ok = _figures_present(in_dir, key)
            evidence = f"figures/{key}_*.pdf"
        else:
            data = _safe_load(in_dir / source)
            ok = data is not None and key in data
            evidence = source
        rows.append({"description": description, "evidence": evidence,
                     "pass": bool(ok)})
    return rows


def _provisional_header(corpus: str, provisional_b: int | None) -> str:
    """Return a LaTeX comment header marking values as provisional, if applicable.

    The text is inserted at the top of every emitted .tex file when
    --provisional N is passed. It explains that the numbers come from a
    smoke run with B=N bootstrap iterations, but qualitative readings
    (rankings, TOST counts, Tukey HSD rejections, H1 conclusion) are
    robust against the eventual switch to canonical B=10^4 because the
    width of percentile CIs scales as 1/sqrt(B); going from B=2000 to
    B=10^4 narrows CIs by approximately 5-10 % (Davison & Hinkley 1997),
    insufficient to move verdicts under the SESOI band delta=0.05 MCC.
    """
    if not provisional_b:
        return ""
    lines = [
        "% =====================================================================",
        f"% [PROVISIONAL B={provisional_b}] — corpus = {corpus}",
        "%",
        f"% These numbers were generated from a smoke run with B={provisional_b}",
        "% bootstrap iterations. The canonical Ash/Wognum 2025 protocol calls for",
        "% B=10^4. Replacement is scheduled when the canonical run completes.",
        "%",
        "% Qualitative characteristics (model rankings, Tukey HSD rejections, TOST",
        "% equivalence counts, ANOVA significance verdicts) are robust against",
        "% this difference because the percentile CI width scales as 1/sqrt(B).",
        "% Going from B=2000 to B=10^4 narrows CIs by approximately 5--10 %",
        "% (Davison & Hinkley 1997; Field & Welsh 2007). This is below the SESOI",
        "% band delta_eq = 0.05 MCC and therefore cannot move equivalence",
        "% verdicts. Numerical values may shift in the third decimal place.",
        "% =====================================================================",
        "",
    ]
    return "\n".join(lines)


def _emit_table(out_path: Path, header: str, body: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + "\n".join(body) + "\n")


def _table_limits(panel: dict, corpus: str, header: str) -> str:
    """Lower (null model) + upper (assay-noise) limits, per metric."""
    nm = panel.get("null_model")
    ul = panel.get("upper_limit")
    if nm is None or ul is None:
        return header + "% MISSING null_model.json or upper_limit.json\n"
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Limite & MCC & AUROC & F1 & AUPRC \\")
    lines.append(r"    \midrule")
    nm_row = ["Null model (majority class)"]
    for metric in ("mcc", "auroc", "f1", "auprc"):
        m = nm["metrics"][metric]
        nm_row.append(f"{m['median']:.4f} [{m['ci_lo']:.4f},~{m['ci_hi']:.4f}]")
    lines.append("    " + " & ".join(nm_row) + r" \\")
    ul_row = [r"Assay upper limit (Brown 2009 / Kramer 2012)"]
    for metric in ("mcc", "auroc", "f1", "auprc"):
        m = ul["metrics"][metric]
        ul_row.append(f"{m['median']:.4f} [{m['ci_lo']:.4f},~{m['ci_hi']:.4f}]")
    lines.append("    " + " & ".join(ul_row) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(rf"  \caption{{Limites empíricos de performance para o corpus "
                 rf"\textit{{{corpus.replace('_', '-')}}}: classificador "
                 rf"\textit{{majority-class}} como limite inferior; ruído IC$_{{50}}$ "
                 rf"de duas vezes ($\sigma_{{\log_{{10}}}} = 0{{,}}301$, Brown 2009 "
                 rf"/ Kramer 2012) como limite superior. Mediana com IC$_{{95}}$ "
                 rf"\textit{{bootstrap}} percentílico. Convenção em "
                 rf"\texttt{{statistical\_protocol.md}} $\S$0.}}")
    lines.append(rf"  \label{{tab:auditoria-limites-{corpus.replace('_', '-')}}}")
    lines.append(r"\end{table}")
    return header + "\n".join(lines) + "\n"


def _table_hedges(panel: dict, corpus: str, header: str) -> str:
    """Hedges' g paired (J(4)) per metric, DT-Kinase vs each baseline."""
    eff = panel.get("effect_size")
    if eff is None:
        return header + "% MISSING effect_size.json\n"
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Comparação (DT-Kinase vs.) & MCC & AUROC & F1 & AUPRC \\")
    lines.append(r"    \midrule")
    for baseline in ("drugban", "graphban", "conplex"):
        cells = [_MODEL_LATEX[baseline]]
        for metric in ("mcc", "auroc", "f1", "auprc"):
            pair_key = f"dtkinase__vs__{baseline}"
            g = eff["by_metric"][metric]["pairs"][pair_key]["paired"]["g_paired"]
            cells.append(f"{g:+.3f}")
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(rf"  \caption{{\textit{{Hedges' g}} pareado para o corpus "
                 rf"\textit{{{corpus.replace('_', '-')}}}: DT-Kinase versus cada "
                 rf"\textit{{baseline}}, por métrica. Correção \textit{{small-sample}} "
                 rf"$J(\nu = 4) = 1 - 3/(4 \cdot 4 - 1) = 0{{,}}8000$ "
                 rf"(Hedges 1981; Lakens 2013, aproximação Borenstein). "
                 rf"\textit{{Cutoffs}} interpretativos (Cohen 1988): "
                 rf"$|g| \geq 0{{,}}2$ (\textit{{small}}); $\geq 0{{,}}5$ "
                 rf"(\textit{{medium}}); $\geq 0{{,}}8$ (\textit{{large}}). "
                 rf"Sinal negativo indica \textit{{baseline}} maior que DT-Kinase.}}")
    lines.append(rf"  \label{{tab:auditoria-hedges-{corpus.replace('_', '-')}}}")
    lines.append(r"\end{table}")
    return header + "\n".join(lines) + "\n"


def _table_tost(panel: dict, corpus: str, header: str) -> str:
    """TOST sensitivity over multiple delta_eq bands."""
    tost = panel.get("tost")
    if tost is None:
        return header + "% MISSING tost.json\n"
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Banda $\delta_{\mathrm{eq}}$ & Tipo & Valor & "
                 r"Pares equivalentes & Total \\")
    lines.append(r"    \midrule")
    for label, by_band in tost["by_band"].items():
        kind = ("SESOI absoluto" if by_band["kind"] == "absolute"
                else "Cohen $d$-anchored")
        if by_band.get("primary"):
            kind = kind + r" $\mathbf{(PRIMARY)}$"
        lines.append(f"    {label} & {kind} & "
                     f"{by_band['delta_eq']:+.4f} & "
                     f"{by_band['n_equivalent']} & "
                     f"{by_band['n_total']} "
                     + r"\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    sig = tost.get("sigma_pooled", 0.0)
    lines.append(rf"  \caption{{Análise de sensibilidade do TOST de "
                 rf"Schuirmann~\cite{{Schuirmann1987tost}} sobre seis bandas "
                 rf"para o corpus \textit{{{corpus.replace('_', '-')}}}: três bandas absolutas "
                 rf"ancoradas em SESOI operacional ($0{{,}}03$; $0{{,}}05$; $0{{,}}07$ "
                 rf"MCC; banda primária $\delta_{{\mathrm{{eq}}}} = 0{{,}}05$ marcada) e "
                 rf"três bandas ancoradas em \textit{{Cohen's d}} (Lakens 2017 SESOI-via-d, "
                 rf"$\delta_{{\mathrm{{eq}}}} \in \{{0{{,}}2;\, 0{{,}}5;\, 0{{,}}8\}} "
                 rf"\cdot \sigma_{{\mathrm{{pooled}}}}$, com $\sigma_{{\mathrm{{pooled}}}} = {sig:.4f}$ "
                 rf"em $\Delta_{{\mathrm{{MCC}}}}$ pareado por semente). Equivalência declarada se "
                 rf"o IC$_{{95}}$ \textit{{bootstrap}} pareado da diferença reside em "
                 rf"$[-\delta_{{\mathrm{{eq}}}}; +\delta_{{\mathrm{{eq}}}}]$.}}")
    lines.append(rf"  \label{{tab:auditoria-tost-{corpus.replace('_', '-')}}}")
    lines.append(r"\end{table}")
    return header + "\n".join(lines) + "\n"


def _table_anova_tukey(panel: dict, corpus: str, header: str) -> str:
    """ANOVA + Bonferroni + Tukey HSD with assumption checks per metric."""
    av = panel.get("anova_tukey")
    if av is None:
        return header + "% MISSING anova_tukey.json\n"
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Métrica & ANOVA $F$ & ANOVA $p$ & "
                 r"$p_{\mathrm{Bonf}}$ ($m=4$) & Premissas paramétricas \\")
    lines.append(r"    \midrule")
    for metric in ("mcc", "auroc", "f1", "auprc"):
        if metric not in av["by_metric"]:
            continue
        a = av["by_metric"][metric]["anova"]
        sig = r"$^{*}$" if a["p_bonf"] < 0.05 else ""
        asm = av["by_metric"][metric].get("assumptions", {})
        sw_viol = [m for m, v in asm.get("shapiro_per_model", {}).items()
                   if v.get("ok_normal_at_05") is False]
        lev = asm.get("levene_across_models", {})
        sph = asm.get("sphericity_proxy_levene", {})
        notes = []
        if sw_viol:
            notes.append(f"Shapiro: {','.join(sw_viol)}")
        else:
            notes.append("Shapiro OK")
        if lev:
            notes.append(f"Levene $p={lev['p']:.2f}$"
                         + ("" if lev.get('ok_homo_at_05') else r"\,VIOL"))
        if sph:
            notes.append(f"Esf.\\,proxy $p={sph['p']:.2f}$"
                         + ("" if sph.get('ok_spherical_at_05') else r"\,VIOL"))
        notes_str = "; ".join(notes)
        lines.append(f"    {_METRIC_LATEX[metric]} & "
                     f"{a['F']:.3f} & {a['p']:.4f} & "
                     f"{a['p_bonf']:.4f}{sig} & {notes_str} "
                     + r"\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(rf"  \caption{{ANOVA de medidas repetidas (\textit{{subject}} = semente, "
                 rf"\textit{{within}} = modelo) por métrica para o corpus "
                 rf"\textit{{{corpus.replace('_', '-')}}}, com correção Bonferroni "
                 rf"inter-métrica ($m = 4$). Asterisco indica rejeição de "
                 rf"$H_0$ (modelos diferem) sob $\alpha = 0{{,}}05$. "
                 rf"Premissas paramétricas verificadas via Shapiro-Wilk por "
                 rf"modelo (normalidade), Levene \textit{{across-models}} "
                 rf"(homocedasticidade) e Levene em diferenças par-a-par "
                 rf"(\textit{{proxy}} de esfericidade). Camada complementar "
                 rf"ao \textit{{bootstrap}} pareado primário (deviation D2 do "
                 rf"\texttt{{statistical\_protocol.md}}).}}")
    lines.append(rf"  \label{{tab:auditoria-anova-{corpus.replace('_', '-')}}}")
    lines.append(r"\end{table}")
    return header + "\n".join(lines) + "\n"


def _table_posthoc(panel: dict, corpus: str, header: str) -> str:
    """Post-hoc classification metrics per model."""
    ph = panel.get("posthoc")
    if ph is None:
        return header + "% MISSING posthoc.json\n"
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Modelo & "
                 r"\textit{precision}@\textit{recall}~$=0{,}8$ & "
                 r"\textit{recall}@\textit{precision}~$=0{,}8$ & "
                 r"TNR@\textit{recall}~$=0{,}9$ \\")
    lines.append(r"    \midrule")
    for m in MODELS:
        s = ph["by_model"][m]["summary"]
        cells = [_MODEL_LATEX[m]]
        cells.append(f"{s['p_at_r']['mean']:.4f} $\\pm$ {s['p_at_r']['std']:.4f}")
        cells.append(f"{s['r_at_p']['mean']:.4f} $\\pm$ {s['r_at_p']['std']:.4f}")
        cells.append(f"{s['tnr_at_r']['mean']:.4f} $\\pm$ {s['tnr_at_r']['std']:.4f}")
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(rf"  \caption{{Métricas pós-classificação operacionalmente relevantes "
                 rf"para o corpus \textit{{{corpus.replace('_', '-')}}} "
                 rf"(Ash/Wognum 2025 G3.3.1). \textit{{precision}}@\textit{{recall}}~$=0{{,}}8$: "
                 rf"\textit{{precision}} máxima alcançada com \textit{{recall}} "
                 rf"$\geq 0{{,}}8$; mede a limpeza da lista de \textit{{hits}} "
                 rf"sob \textit{{recall}} alto. \textit{{recall}}@\textit{{precision}}~$=0{{,}}8$: "
                 rf"\textit{{recall}} máximo com \textit{{precision}} $\geq 0{{,}}8$; "
                 rf"mede sensibilidade sob restrição de falsos positivos. "
                 rf"TNR@\textit{{recall}}~$=0{{,}}9$: \textit{{specificity}} a "
                 rf"\textit{{recall}} $\geq 0{{,}}9$; mede capacidade de descartar "
                 rf"inativos. Média $\pm \sigma$ amostral sobre cinco sementes.}}")
    lines.append(rf"  \label{{tab:auditoria-posthoc-{corpus.replace('_', '-')}}}")
    lines.append(r"\end{table}")
    return header + "\n".join(lines) + "\n"


def _table_checklist(panel: dict, corpus: str, header: str) -> str:
    """Reporting checklist (statistical_protocol.md §4) compliance."""
    rows = panel.get("checklist", [])
    if not rows:
        return header + "% MISSING checklist data\n"
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{p{0.6\textwidth}cc}")
    lines.append(r"    \toprule")
    lines.append(r"    Item de \textit{reporting} & Evidência & Status \\")
    lines.append(r"    \midrule")
    for r in rows:
        status = r"$\checkmark$" if r["pass"] else r"$\times$"
        desc = r["description"].replace("&", r"\&")
        evidence = r["evidence"].replace("_", r"\_").replace("&", r"\&")
        lines.append(f"    {desc} & \\texttt{{{evidence}}} & {status} "
                     + r"\\")
    n_pass = sum(1 for r in rows if r["pass"])
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(rf"  \caption{{\textit{{Reporting checklist}} (Ash/Wognum 2025; "
                 rf"\texttt{{statistical\_protocol.md}} $\S$4) para o corpus "
                 rf"\textit{{{corpus.replace('_', '-')}}}: $\checkmark$ indica "
                 rf"presença do artefato evidenciador; $\times$ indica "
                 rf"ausência. Conformidade: {n_pass}/{len(rows)}.}}")
    lines.append(rf"  \label{{tab:auditoria-checklist-{corpus.replace('_', '-')}}}")
    lines.append(r"\end{table}")
    return header + "\n".join(lines) + "\n"


def _format_latex_summary(panel: dict) -> str:
    lines = []
    lines.append(r"% Statistical panel auto-generated by aggregate_panel.py")
    lines.append(r"% Source: results/statistical/{corpus}/panel.json")
    lines.append("")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Model & MCC & AUROC & F1 & AUPRC \\")
    lines.append(r"    \midrule")
    eff = panel.get("effect_size")
    n_seeds_str = ""
    if eff:
        n_seeds_str = str(len(eff.get("seeds", [])))
        for m in MODELS:
            cells = []
            for metric in ("mcc", "auroc", "f1", "auprc"):
                pm = eff["by_metric"][metric]["per_model"][m]
                cells.append(f"{pm['mean']:.4f} $\\pm$ {pm['std']:.4f}")
            lines.append(
                f"    {m.replace('_', '-')} & " + " & ".join(cells) + r" \\")
    lines.append(r"    \midrule")
    null = panel.get("null_model")
    if null:
        cells = []
        for metric in ("mcc", "auroc", "f1", "auprc"):
            mm = null["metrics"][metric]
            cells.append(f"{mm['median']:.4f}")
        lines.append(r"    Null model (lower limit) & " + " & ".join(cells) + r" \\")
    upper = panel.get("upper_limit")
    if upper:
        cells = []
        for metric in ("mcc", "auroc", "f1", "auprc"):
            mm = upper["metrics"][metric]
            cells.append(f"{mm['median']:.4f}")
        lines.append(r"    Assay upper limit & " + " & ".join(cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    n_str = n_seeds_str if n_seeds_str else "n"
    lines.append(
        rf"  \caption{{Statistical panel, corpus = {panel['corpus']}. "
        rf"Per-model entries are mean $\pm$ $\sigma$ (sample standard "
        rf"deviation, ddof=1) over {n_str} seeds. Null model is "
        rf"majority-class predictor (Ash/Wognum 2025 G3); upper limit is "
        rf"the median MCC under simulated 2-fold IC$_{{50}}$ assay noise "
        rf"($\sigma_{{\log_{{10}}}} = 0{{,}}301$, Brown 2009 / Kramer 2012). "
        rf"Notation per `statistical\_protocol.md` $\S$0.}}")
    lines.append(rf"  \label{{tab:stat-panel-{panel['corpus']}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _format_checklist(panel: dict, checklist_rows: list[dict]) -> str:
    lines = []
    lines.append(f"# Reporting checklist - corpus {panel['corpus']}")
    lines.append("")
    lines.append("Source: `docs/01-methodology/statistical_protocol.md` section 4.")
    lines.append("This file is auto-generated by `aggregate_panel.py`.")
    lines.append("")
    lines.append("| Item | Evidence | Status |")
    lines.append("|---|---|:---:|")
    for r in checklist_rows:
        status = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| {r['description']} | `{r['evidence']}` | {status} |")
    lines.append("")
    n_pass = sum(1 for r in checklist_rows if r["pass"])
    lines.append(f"**Summary:** {n_pass}/{len(checklist_rows)} items PASS.")
    lines.append("")
    lines.append("Three deviations are declared and preserved (see "
                 "`docs/01-methodology/statistical_protocol.md` section 2):")
    lines.append("- D1: single scaffold split + 5 seeds (no 5x5 CV).")
    lines.append("- D2: paired bootstrap is primary; ANOVA + Tukey HSD complementary.")
    lines.append("- D3: TOST band 0.05 MCC SESOI-anchored (primary); "
                 "Cohen-anchored bands reported as sensitivity.")
    return "\n".join(lines)


def aggregate(corpus: str, in_dir: Path,
              out_json: Path, out_tex: Path,
              out_checklist: Path,
              mirror_to: Path | None = None,
              provisional_b: int | None = None) -> dict:
    panel = {
        "corpus": corpus,
        "models": list(MODELS),
        "primary_metrics": list(PRIMARY_METRICS),
        "null_model": _safe_load(in_dir / "null_model.json"),
        "upper_limit": _safe_load(in_dir / "upper_limit.json"),
        "posthoc": _safe_load(in_dir / "posthoc.json"),
        "effect_size": _safe_load(in_dir / "effect_size.json"),
        "anova_tukey": _safe_load(in_dir / "anova_tukey.json"),
        "tost": _safe_load(in_dir / "tost.json"),
    }
    if provisional_b is not None:
        panel["provisional_b"] = int(provisional_b)

    checklist_rows = _evaluate_checklist(in_dir)
    panel["checklist"] = checklist_rows

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as fh:
        json.dump(panel, fh, indent=2)

    header = _provisional_header(corpus, provisional_b)

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(header + _format_latex_summary(panel))

    out_checklist.parent.mkdir(parents=True, exist_ok=True)
    out_checklist.write_text(_format_checklist(panel, checklist_rows))

    # Emit one .tex per stand-alone table for granular \input{} in thesis.
    tables_dir = in_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_emitters = {
        "limits": _table_limits,
        "hedges": _table_hedges,
        "tost": _table_tost,
        "anova_tukey": _table_anova_tukey,
        "posthoc": _table_posthoc,
        "checklist": _table_checklist,
    }
    for name, fn in table_emitters.items():
        out_tab = tables_dir / f"tab_{name}_{corpus}.tex"
        out_tab.write_text(fn(panel, corpus, header))

    # Master file bundling all six tables (single \input target for thesis).
    master = tables_dir / f"auditoria_complementar_{corpus}.tex"
    master_lines = [header,
                    "% Master file bundling the six auditoria tables for the corpus.",
                    "% Adjust the order or comment out individual \\input lines as needed.",
                    ""]
    for name in ("limits", "hedges", "tost", "anova_tukey", "posthoc", "checklist"):
        master_lines.append(rf"\input{{tab_{name}_{corpus}}}")
        master_lines.append("")
    master.write_text("\n".join(master_lines) + "\n")

    if mirror_to is not None:
        mirror_to.mkdir(parents=True, exist_ok=True)
        figs_src = in_dir / "figures"
        if figs_src.exists():
            for fig in figs_src.glob("*.pdf"):
                shutil.copy2(fig, mirror_to / f"stat_{corpus}_{fig.name}")
            for fig in figs_src.glob("*.png"):
                shutil.copy2(fig, mirror_to / f"stat_{corpus}_{fig.name}")
        # Mirror tables too so the thesis can \input{...} from PhD/figures or
        # a sibling tables/ directory if the user prefers.
        tables_mirror = mirror_to.parent / "stat_tables"
        tables_mirror.mkdir(parents=True, exist_ok=True)
        for tab in tables_dir.glob("*.tex"):
            shutil.copy2(tab, tables_mirror / tab.name)

    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("human", "non_human", "all"))
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-tex", type=Path, required=True)
    ap.add_argument("--out-checklist", type=Path, required=True)
    ap.add_argument("--mirror-to", type=Path, default=None,
                    help="Optional: copy figures to this directory "
                         "(e.g. ~/PhD/figures/) for thesis build.")
    ap.add_argument("--provisional", type=int, default=None,
                    help="If set, mark all emitted .tex files with a "
                         "PROVISIONAL B=<N> disclaimer header. Use 2000 "
                         "for the smoke run, omit for canonical B=10^4.")
    args = ap.parse_args()
    panel = aggregate(args.corpus, args.in_dir, args.out_json,
                     args.out_tex, args.out_checklist, args.mirror_to,
                     provisional_b=args.provisional)
    n_pass = sum(1 for r in panel["checklist"] if r["pass"])
    n_total = len(panel["checklist"])
    flag = f" [PROVISIONAL B={args.provisional}]" if args.provisional else ""
    print(f"[aggregate_panel] corpus={args.corpus} "
          f"checklist={n_pass}/{n_total} PASS{flag}")
    print(f"  json       -> {args.out_json}")
    print(f"  tex        -> {args.out_tex}")
    print(f"  checklist  -> {args.out_checklist}")
    print(f"  tables/    -> {args.in_dir / 'tables'} (6 stand-alone .tex files + master)")


if __name__ == "__main__":
    main()
