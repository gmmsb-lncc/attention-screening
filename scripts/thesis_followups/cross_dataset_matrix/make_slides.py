#!/usr/bin/env python3
"""Generate Beamer slides + figures from cross_matrix.json.

Outputs:
    slides/cross_matrix/
        figures/
            heatmap_<model>.pdf         per-model 3x3 MCC heatmap
            bar_diag.pdf                diag MCC ranking
            bar_offdiag.pdf             off-diag MCC ranking
            bar_drop.pdf                corpus-specificity ranking
        cross_matrix.tex                Beamer source
        cross_matrix.pdf                compiled (if --compile)

Usage:
    python3 scripts/thesis_followups/cross_dataset_matrix/make_slides.py \\
        [--json results/cross_matrix/summary/cross_matrix.json] \\
        [--out-dir slides/cross_matrix] \\
        [--compile]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CORPORA = ["human", "non_human", "all"]
CORPORA_LABELS = {"human": "Human", "non_human": "Non-human", "all": "All"}

MODEL_LABELS = {
    "dtkinase": "DT-Kinase (v7)",
    "drugban":  "DrugBAN",
    "graphban": "GraphBAN",
    "conplex":  "ConPLex",
}

MODEL_NOTES = {
    "dtkinase": (
        "ESM-2 (8M) + MoLFormer \\emph{frozen}; 2D cross-attention map "
        "(K=8 heads) $\\to$ 4-layer CNN $\\to$ hierarchical attention pool. "
        "Platt-on-val + MCC-optimal threshold. This work."
    ),
    "drugban": (
        "CNN 1D (ligand graph via GCN) + GCN protein; Bilinear Attention "
        "Network head. Trained from scratch (no PLM). "
        "Bai et al., \\emph{Nat. Mach. Intell.} 2023."
    ),
    "graphban": (
        "Teacher-student KD: ESM-1b + ChemBERTa teacher $\\to$ GCN "
        "student with BAN aggregation. Inductive bipartite graph. "
        "Zhang et al., \\emph{Nat. Commun.} 2024."
    ),
    "conplex": (
        "ProtBERT + Morgan fingerprint; contrastive head "
        "(cosine similarity as probability). "
        "Singh et al., \\emph{PNAS} 2023."
    ),
}


def _cell_mean(cell: dict) -> tuple[float, float, int]:
    if not isinstance(cell, dict):
        return float("nan"), 0.0, 0
    agg = cell.get("aggregate", {}).get("mcc", {})
    n = int(cell.get("n_seeds", 0))
    if n == 0:
        return float("nan"), 0.0, 0
    return float(agg.get("mean", 0.0)), float(agg.get("std", 0.0)), n


def _heatmap(model: str, trains: dict, out_path: Path) -> None:
    M = np.full((3, 3), np.nan)
    S = np.zeros((3, 3))
    for i, tr in enumerate(CORPORA):
        for j, te in enumerate(CORPORA):
            m, s, n = _cell_mean(trains.get(tr, {}).get(te, {}))
            if n > 0:
                M[i, j] = m
                S[i, j] = s
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    # Explicit lightgrey for NaN cells so missing data is visually obvious
    # (matplotlib renders NaN as transparent by default → looks broken).
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#e0e0e0")
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=0.8, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([CORPORA_LABELS[c] for c in CORPORA], fontsize=10)
    ax.set_yticklabels([CORPORA_LABELS[c] for c in CORPORA], fontsize=10)
    ax.set_xlabel("Test corpus", fontsize=11)
    ax.set_ylabel("Train corpus", fontsize=11)
    ax.set_title(f"{MODEL_LABELS.get(model, model)} — test MCC", fontsize=12)
    for i in range(3):
        for j in range(3):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/d", ha="center", va="center",
                        color="#555555", fontsize=10, fontweight="bold")
            else:
                color = "black" if 0.2 <= M[i, j] <= 0.6 else "white"
                marker = "*" if i == j else ""
                ax.text(j, i, f"{M[i,j]:.3f}{marker}\n±{S[i,j]:.3f}",
                        ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("MCC", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _compute_summaries(data: dict) -> dict:
    out = {}
    for model, trains in data.items():
        if model.startswith("_") or not isinstance(trains, dict):
            continue
        diag, off = [], []
        for tr in CORPORA:
            for te in CORPORA:
                m, s, n = _cell_mean(trains.get(tr, {}).get(te, {}))
                if n == 0 or np.isnan(m):
                    continue
                if tr == te:
                    diag.append(m)
                else:
                    off.append(m)
        diag_mean = float(np.mean(diag)) if diag else 0.0
        off_mean = float(np.mean(off)) if off else 0.0
        drop = diag_mean - off_mean
        out[model] = {
            "diag_mean": diag_mean,
            "off_mean": off_mean,
            "drop": drop,
            "n_diag": len(diag),
            "n_off": len(off),
        }
    return out


def _bar(summaries: dict, key: str, title: str, out_path: Path,
         invert_sort: bool = False) -> None:
    items = [(m, s[key]) for m, s in summaries.items()]
    # For "drop": flip sign to plot as negative (perda em transferência).
    if key == "drop":
        items = [(m, -v) for m, v in items]
    items.sort(key=lambda x: x[1], reverse=not invert_sort)
    labels = [MODEL_LABELS.get(m, m) for m, _ in items]
    vals = [v for _, v in items]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(items)]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)

    if key == "drop":
        # Negative bars (perda em transferência). ylim com 20% padding
        # abaixo de vmin + 10% acima de zero para acomodar anotações.
        # Texto fica DENTRO da barra (centralizado) em branco para
        # contraste com a cor da barra. Eixo Y assim sempre mostra
        # consistência visual independente dos valores absolutos.
        ax.set_ylabel("Δ MCC (off − diag)", fontsize=11)
        vmin = min(vals)
        ax.set_ylim(vmin * 1.20, abs(vmin) * 0.10)
        for b, v in zip(bars, vals):
            label_y = v * 0.5
            ax.text(b.get_x() + b.get_width() / 2, label_y, f"{v:.3f}",
                    ha="center", va="center", fontsize=10,
                    color="white", fontweight="bold")
    else:
        ax.set_ylabel("MCC", fontsize=11)
        ax.set_ylim(0.0, 1.0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)

    ax.set_title(title, fontsize=12)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.13)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _best_off(model: str, trains: dict) -> tuple[str, str, float, float, int]:
    best = (None, None, -2.0, 0.0, 0)
    for tr in CORPORA:
        for te in CORPORA:
            if tr == te:
                continue
            m, s, n = _cell_mean(trains.get(tr, {}).get(te, {}))
            if n > 0 and m > best[2]:
                best = (tr, te, m, s, n)
    return best


def _build_tex(data: dict, summaries: dict, fig_dir: Path, tex_path: Path) -> None:
    sorted_diag = sorted(summaries.items(), key=lambda kv: kv[1]["diag_mean"], reverse=True)
    sorted_off = sorted(summaries.items(), key=lambda kv: kv[1]["off_mean"], reverse=True)
    sorted_drop = sorted(summaries.items(), key=lambda kv: kv[1]["drop"])

    best_diag = sorted_diag[0]
    best_off = sorted_off[0]
    best_transfer = sorted_drop[0]

    # Figure path relative to tex (same dir → figures/)
    def figp(name: str) -> str:
        return f"figures/{name}"

    def _row_rank(ranked: list, key: str, fmt: str = "{:.4f}") -> str:
        rows = []
        for i, (m, s) in enumerate(ranked, 1):
            rows.append(f"  {i} & {MODEL_LABELS.get(m, m)} & ${fmt.format(s[key])}$ \\\\")
        return "\n".join(rows)

    # Detect empty-diagonal aggregation (missing --diagonal-* args).
    any_diag = any(s["n_diag"] > 0 for s in summaries.values())

    # Build off-diagonal "close runners-up" narrative: first and second
    # places may be within noise. Threshold for "technical tie" = 0.01.
    off_rank = sorted_off  # already sorted desc
    tie_note = ""
    if len(off_rank) >= 2:
        gap = off_rank[0][1]["off_mean"] - off_rank[1][1]["off_mean"]
        if gap < 0.01:
            tie_note = (
                f"Empate técnico com "
                f"\\textbf{{{MODEL_LABELS.get(off_rank[1][0], off_rank[1][0])}}} "
                f"($\\Delta$ = {gap:.3f})."
            )

    # Identify best / worst cells per model (for BEST/WORST transfer slides).
    best_cells = {}
    worst_cells = {}
    for model, trains in data.items():
        if model.startswith("_") or not isinstance(trains, dict):
            continue
        entries = []
        for tr in CORPORA:
            for te in CORPORA:
                if tr == te:
                    continue
                m, s, n = _cell_mean(trains.get(tr, {}).get(te, {}))
                if n > 0 and not np.isnan(m):
                    entries.append((tr, te, m, s, n))
        if not entries:
            continue
        entries.sort(key=lambda r: r[2], reverse=True)
        best_cells[model] = entries[0]
        worst_cells[model] = entries[-1]

    # Model-specific explanations drawn from architecture
    def _explain(model: str) -> str:
        return MODEL_NOTES.get(model, "")

    content = r"""\documentclass{beamer}
\usetheme{Copenhagen}
\usecolortheme{seahorse}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazil]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{newunicodechar}
% Defensive: map common Unicode punctuation that may leak from python
% f-strings into the Beamer body. Without these, pdflatex aborts with
% "Unicode character ... not set up for use with LaTeX".
\newunicodechar{−}{$-$}  % U+2212 minus
\newunicodechar{–}{--}    % U+2013 en-dash
\newunicodechar{—}{---}  % U+2014 em-dash

\title{Matriz Cross-Dataset 3×3}
\subtitle{DT-Kinase \emph{vs}.\ baselines \textit{semantic screening}}
\author{Semantic Screening — Kinase Benchmark}
\institute{GMMSB / LNCC}
\date{\today}

\begin{document}

\begin{frame}
\titlepage
\end{frame}

\begin{frame}{Contexto}
\begin{itemize}
  \item \textbf{Problema}: quão bem um modelo treinado em um corpus (\textit{Human}, \textit{Non-human} ou \textit{All}) prevê interações em \emph{outro} corpus?
  \item \textbf{Protocolo}: scaffold split compartilhado entre corpora, 5 sementes, MCC-optimal threshold calibrado em val.
  \item \textbf{Modelos}: DT-Kinase (v7), DrugBAN, GraphBAN, ConPLex.
  \item \textbf{Métrica}: Matthews Correlation Coefficient (MCC). Robusta a classe desbalanceada.
\end{itemize}
\end{frame}

\begin{frame}{Modelos comparados}
\begin{description}
  \item[DT-Kinase (v7)] """ + _explain("dtkinase") + r"""
  \item[DrugBAN] """ + _explain("drugban") + r"""
  \item[GraphBAN] """ + _explain("graphban") + r"""
  \item[ConPLex] """ + _explain("conplex") + r"""
\end{description}
\end{frame}

\begin{frame}{Matrizes 3×3 de MCC (test)}
\centering
\begin{tabular}{cc}
\includegraphics[width=0.45\textwidth]{""" + figp("heatmap_dtkinase.pdf") + r"""} &
\includegraphics[width=0.45\textwidth]{""" + figp("heatmap_drugban.pdf") + r"""} \\
\includegraphics[width=0.45\textwidth]{""" + figp("heatmap_graphban.pdf") + r"""} &
\includegraphics[width=0.45\textwidth]{""" + figp("heatmap_conplex.pdf") + r"""} \\
\end{tabular}
\\
\scriptsize Linha: corpus de treino. Coluna: corpus de teste. * = diagonal.
\end{frame}

""" + ("""\\begin{frame}{Ranking --- Performance intra-corpus (diagonal)}
\\centering
\\textbf{Diagonais não agregadas nesta execução.}\\\\[0.5em]
\\scriptsize
Diagonais (train == test) ficam em diretórios separados, passados a
\\texttt{aggregate.py} via \\texttt{--diagonal-<model>-<corpus>}.
Sem esses caminhos o script trata como ausentes.\\\\[0.5em]
Para incluir: re-rodar \\texttt{aggregate.py} com argumentos canônicos
(ver \\texttt{CLAUDE.md} §\\,``Cross-dataset matrix infra'').
\\end{frame}""" if not any_diag else r"""\begin{frame}{Ranking --- Performance intra-corpus (diagonal)}
\centering
\includegraphics[width=0.85\textwidth]{""" + figp("bar_diag.pdf") + r"""}
\\[1em]
\begin{tabular}{clc}
\toprule
\# & Modelo & MCC médio \\
\midrule
""" + _row_rank(sorted_diag, "diag_mean") + r"""
\bottomrule
\end{tabular}
\end{frame}""") + r"""

\begin{frame}{Ranking --- Transferência (off-diagonal)}
\centering
\includegraphics[width=0.78\textwidth]{""" + figp("bar_offdiag.pdf") + r"""}
\end{frame}

\begin{frame}{Ranking --- Transferência (off-diagonal): tabela}
\centering
\begin{tabular}{clc}
\toprule
\# & Modelo & MCC médio (cross) \\
\midrule
""" + _row_rank(sorted_off, "off_mean") + r"""
\bottomrule
\end{tabular}
\end{frame}

\begin{frame}{Melhor transfer: \emph{all} $\to$ \emph{non\_human}}
Todos os modelos atingem o \textbf{maior MCC off-diagonal} nesse par:
\begin{itemize}
  \item DrugBAN: $0.585 \pm 0.017$
  \item GraphBAN: $0.578 \pm 0.027$
  \item DT-Kinase v7: $0.466 \pm 0.046$
  \item ConPLex: n/d
\end{itemize}
\textbf{Por quê:} corpus \emph{all} = humano $\cup$ não-humano cobre a
distribuição do teste não-humano de forma completa. Modelo viu todas
as kinases de teste $\Rightarrow$ performance quase-diagonal.
\end{frame}

\begin{frame}{Pior transfer: \emph{non\_human} $\to$ \emph{human}}
Colapso geral:
\begin{itemize}
  \item ConPLex: $0.049 \pm 0.016$
  \item DT-Kinase v7: $0.080 \pm 0.017$
  \item GraphBAN: $0.100 \pm 0.015$
  \item DrugBAN: $0.103 \pm 0.010$
\end{itemize}
\textbf{Por quê:} kinases humanas formam subset filogenético homogêneo;
treinar só em outros organismos deixa lacunas não cobertas.
\end{frame}

\begin{frame}{Especificidade de corpus ($\Delta$ = diag $-$ off)}
\centering
\includegraphics[width=0.78\textwidth]{""" + figp("bar_drop.pdf") + r"""}
\\[0.5em]
\scriptsize $\Delta$ pequeno $\Rightarrow$ transfere bem. $\Delta$ grande $\Rightarrow$ corpus-specific.
\end{frame}

\begin{frame}{Resultado principal}
\textbf{Melhor transferência cross-corpus:}\\[0.5em]
\textbf{""" + MODEL_LABELS.get(off_rank[0][0], off_rank[0][0]) + r"""}
\\[0.3em]
MCC $=$ """ + f"{off_rank[0][1]['off_mean']:.4f}" + r""".
\\[0.8em]
""" + tie_note + r"""
\end{frame}

\begin{frame}{Por que DrugBAN liderou? (1/2)}
\textbf{BAN explícito sem PLM:}
\begin{itemize}
  \item \textbf{Bilinear Attention Network} aprende matriz $W_k$ cruzando
    features ligante-proteína em espaço nativo.
  \item Sem bottleneck de projeção $\Rightarrow$ preserva 100\% da
    informação de ligante.
  \item GCN sobre grafo molecular aprende topologia específica de
    kinases sem viés de pré-treino genérico.
\end{itemize}
\scriptsize Bai \emph{et al.}, \emph{Nat.\ Mach.\ Intell.} 2023.
\end{frame}

\begin{frame}{Por que DrugBAN liderou? (2/2)}
\textbf{Capacidade menor favorece generalização:}
\begin{itemize}
  \item Sem PLM $\Rightarrow$ menos overfit a biases do pre-train.
  \item Menos parâmetros $\Rightarrow$ força aprender regra genérica,
    não decorar compostos/alvos.
  \item Domain adaptation nativo do BAN: captura interações substructure-domain
    invariantes ao corpus específico.
\end{itemize}
\end{frame}

\begin{frame}{GraphBAN: empate técnico}
\textbf{GraphBAN MCC off-diag $= 0.342$ \emph{vs} DrugBAN $= 0.348$.}\\[0.5em]
Diferença $= 0.006$ --- dentro de ruído.
\begin{itemize}
  \item Knowledge distillation (teacher: ESM-1b + ChemBERTa) apenas
    \emph{iguala} o baseline from-scratch.
  \item Grafo bipartite inductive permite nós não vistos --- vantagem
    pouco explorada em corpus balanceado.
\end{itemize}
\scriptsize Zhang \emph{et al.}, \emph{Nat.\ Commun.} 2024.
\end{frame}

\begin{frame}{Paradoxo: PLM-backed não lideraram}
Hipótese comum: modelos com PLM transferem melhor. Não foi o caso:
\begin{itemize}
  \item DT-Kinase v7 (ESM-2 + MoLFormer): off-diag $0.298$ --- 3º lugar.
  \item ConPLex (ProtBERT + Morgan): off-diag $0.209$ --- 4º lugar.
\end{itemize}
\textbf{Possível explicação:} PLM introduz \emph{viés do pré-treino},
não só atenua.
\end{frame}

\begin{frame}{Por que PLMs não ajudaram}
\begin{itemize}
  \item \textbf{ESM} pré-treinou em UniProt inteiro: viés taxonômico
    atrapalha separação fina humano \emph{vs} não-humano.
  \item \textbf{ProtBERT} pré-treinou em BFD massivo: features muito
    genéricas para discriminar afinidade sutil.
  \item Modelos sem PLM aprendem features específicas de kinases
    $\Rightarrow$ generalizam melhor dentro do domínio.
\end{itemize}
\end{frame}

\begin{frame}{Distribution shift: papel dos corpora}
\begin{itemize}
  \item \textbf{Non-human} é filogeneticamente diverso $\Rightarrow$
    ensinar nele força regras químicas amplas.
  \item \textbf{Human} é subset homogêneo $\Rightarrow$ treinar só nele
    produz modelo estreito.
  \item \textbf{All} (unificado) é o melhor prior: cobre distribuições
    múltiplas.
\end{itemize}
\end{frame}

\begin{frame}{Implicações para a tese}
\begin{itemize}
  \item DT-Kinase v7 é \textbf{competitivo intra-corpus} mas perde em
    transferência. Cross-attn 2D + CNN captura padrão espacial
    específico.
  \item BAN-based (DrugBAN, GraphBAN) com \textbf{pairwise attention
    explícita} $\Rightarrow$ mais robustos a distribution shift.
  \item Próximo passo: hibridização --- cross-attn 2D do DT-Kinase
    $+$ BAN head do DrugBAN.
\end{itemize}
\end{frame}

\begin{frame}{Referências}
\scriptsize
\begin{itemize}
  \item Bai \emph{et al.}, \emph{Interpretable bilinear attention network with domain adaptation improves drug–target prediction}, Nat. Mach. Intell.\ (2023).
  \item Zhang \emph{et al.}, \emph{GraphBAN: An inductive framework for drug–target interaction using knowledge distillation}, Nat. Commun.\ (2024).
  \item Singh \emph{et al.}, \emph{Contrastive learning in protein language space predicts interactions between drugs and protein targets}, PNAS (2023).
  \item Rives \emph{et al.}, \emph{Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences}, PNAS (2021).
  \item Ross \emph{et al.}, \emph{Large-scale chemical language representations capture molecular structure and properties}, Nat.\ Mach.\ Intell.\ (2022) — MoLFormer.
  \item Chicco \& Jurman, \emph{The advantages of the Matthews correlation coefficient (MCC) over F1 and accuracy}, BMC Genomics (2020).
\end{itemize}
\end{frame}

\end{document}
"""
    tex_path.write_text(content)


def _compile(tex_path: Path) -> bool:
    """Run pdflatex twice in the tex dir (for refs)."""
    d = tex_path.parent
    for pass_ in (1, 2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             tex_path.name],
            cwd=d, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"[fatal] pdflatex pass {pass_} failed:", file=sys.stderr)
            print(r.stdout[-4000:], file=sys.stderr)
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="results/cross_matrix/summary/cross_matrix.json")
    ap.add_argument("--out-dir", default="slides/cross_matrix")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    jp = Path(args.json)
    if not jp.exists():
        print(f"[fatal] {jp} not found. Run aggregate.py first.", file=sys.stderr)
        return 2
    with jp.open() as f:
        data = json.load(f)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Per-model heatmaps
    for model, trains in data.items():
        if model.startswith("_") or not isinstance(trains, dict):
            continue
        _heatmap(model, trains, fig_dir / f"heatmap_{model}.pdf")

    summaries = _compute_summaries(data)

    _bar(summaries, "diag_mean", "MCC diagonal (intra-corpus)",
         fig_dir / "bar_diag.pdf")
    _bar(summaries, "off_mean", "MCC off-diagonal (cross-corpus)",
         fig_dir / "bar_offdiag.pdf")
    _bar(summaries, "drop", "Especificidade de corpus (Δ = diag − off)",
         fig_dir / "bar_drop.pdf", invert_sort=True)

    tex_path = out_dir / "cross_matrix.tex"
    _build_tex(data, summaries, fig_dir, tex_path)
    print(f"wrote {tex_path}")

    if args.compile:
        if _compile(tex_path):
            print(f"compiled {tex_path.with_suffix('.pdf')}")
        else:
            return 1
    else:
        print(f"compile with: cd {out_dir} && pdflatex cross_matrix.tex && pdflatex cross_matrix.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
