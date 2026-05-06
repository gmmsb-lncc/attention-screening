#!/usr/bin/env python3
"""Evaluate the 3-model human_kinome committee on the canonical test set.

For each corpus (human, non_human, all):
  - Load 5-seed averaged probs/thresholds for DT-Kinase + DrugBAN + ConPLex
    using the IN-DOMAIN checkpoint of each corpus.
  - Compute committee_prob = mean of 3 calibrated probs
  - Compute committee_thr  = mean of 3 thresholds (canonical τ̄)
  - Apply dedupe by (seq_id, chembl_id)
  - Compute MCC, AUROC, F1, accuracy, confusion matrix
  - Per-system + committee metrics

Outputs:
  results/inference/committee_human_kinome_test_eval/
    metrics.csv            per-corpus, per-system metrics
    confusion_matrices.png 3 panels (one per corpus)
    heatmap_mcc.png        MCC heatmap committee + 4 models × 3 corpora
    REPORT.md              tabular summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, f1_score,
                              accuracy_score, confusion_matrix)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "inference" / "experiments"))

from committee_vs_individual import (  # type: ignore
    load_5seed, dedupe_predictions, load_test_keys,
)

CORPORA = ["human", "non_human", "all"]
COMMITTEE_MEMBERS = ["dtkinase", "drugban", "conplex"]
ALL_MODELS = ["dtkinase", "drugban", "graphban", "conplex"]

OUT_DIR = REPO / "results" / "inference" / "committee_human_kinome_test_eval"


def system_metrics(y: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        mcc=float(matthews_corrcoef(y, pred)),
        auroc=float(roc_auc_score(y, prob)),
        f1=float(f1_score(y, pred, zero_division=0)),
        accuracy=float(accuracy_score(y, pred)),
        threshold=float(thr),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


def evaluate_corpus(corpus: str) -> dict:
    print(f"\n=== {corpus} ===")

    probs: dict[str, np.ndarray] = {}
    thrs: dict[str, float] = {}
    y_true: np.ndarray | None = None
    for m in ALL_MODELS:
        p, y, t = load_5seed(m, corpus)
        probs[m] = p
        thrs[m] = t
        y_true = y if y_true is None else y_true

    keys, _ = load_test_keys(corpus)
    n0 = len(y_true)

    probs_d: dict[str, np.ndarray] = {}
    y_d: np.ndarray | None = None
    for m in ALL_MODELS:
        p_d, yt_d, _ = dedupe_predictions(probs[m], y_true, keys)
        probs_d[m] = p_d
        if y_d is None:
            y_d = yt_d
    print(f"  dedupe: {n0} → {len(y_d)}")

    metrics_per_system: dict[str, dict] = {}

    # Individual models
    for m in ALL_MODELS:
        metrics_per_system[m] = system_metrics(y_d, probs_d[m], thrs[m])

    # 4-model legacy committee
    full_prob = np.mean([probs_d[m] for m in ALL_MODELS], axis=0)
    full_thr = np.mean([thrs[m] for m in ALL_MODELS])
    metrics_per_system["committee_4model"] = system_metrics(y_d, full_prob, full_thr)

    # 3-model human_kinome committee (canonical)
    com_prob = np.mean([probs_d[m] for m in COMMITTEE_MEMBERS], axis=0)
    com_thr = np.mean([thrs[m] for m in COMMITTEE_MEMBERS])
    metrics_per_system["committee_human_kinome"] = system_metrics(y_d, com_prob, com_thr)

    return dict(
        n=len(y_d),
        n_pos=int(y_d.sum()),
        n_neg=int((1 - y_d).sum()),
        metrics=metrics_per_system,
    )


def plot_confusion_matrices(results: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, corpus in zip(axes, CORPORA):
        m = results[corpus]["metrics"]["committee_human_kinome"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        # Normalize by row (true class)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["non-binder", "binder"])
        ax.set_yticklabels(["non-binder", "binder"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{corpus}\n(n={results[corpus]['n']}, "
                     f"MCC={m['mcc']:.3f})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                        fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Comitê human_kinome (DT-Kinase + DrugBAN + ConPLex) — "
                 "Matrizes de confusão normalizadas",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  → wrote {out_path}")


def plot_metric_heatmap(results: dict, metric: str, out_path: Path,
                         vmin: float = 0.4, vmax: float = 0.9) -> None:
    systems_order = (ALL_MODELS
                     + ["committee_4model", "committee_human_kinome"])
    M = np.zeros((len(systems_order), len(CORPORA)))
    for j, c in enumerate(CORPORA):
        for i, s in enumerate(systems_order):
            M[i, j] = results[c]["metrics"][s][metric]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CORPORA)))
    ax.set_xticklabels(CORPORA)
    ax.set_yticks(range(len(systems_order)))
    nice = {"dtkinase": "DT-Kinase", "drugban": "DrugBAN",
            "graphban": "GraphBAN", "conplex": "ConPLex",
            "committee_4model": "Comitê 4-model (legacy)",
            "committee_human_kinome": "Comitê human_kinome ★"}
    ax.set_yticklabels([nice[s] for s in systems_order])
    ax.set_xlabel("Test corpus"); ax.set_ylabel("System")
    ax.set_title(f"{metric.upper()} — comitê vs modelos individuais")
    for i in range(len(systems_order)):
        for j in range(len(CORPORA)):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}",
                    ha="center", va="center",
                    color="black" if v > (vmin + vmax) / 2 else "white",
                    fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label=metric.upper())
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  → wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for corpus in CORPORA:
        results[corpus] = evaluate_corpus(corpus)

    # CSV with all metrics
    rows: list[dict] = []
    for corpus, info in results.items():
        for sys_name, m in info["metrics"].items():
            rows.append({
                "corpus": corpus, "system": sys_name,
                "n": info["n"], "n_pos": info["n_pos"], "n_neg": info["n_neg"],
                **m,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metrics.csv", index=False)
    print(f"\n  → wrote {OUT_DIR / 'metrics.csv'}")

    # Plots
    plot_confusion_matrices(results, OUT_DIR / "confusion_matrices.png")
    plot_metric_heatmap(results, "mcc", OUT_DIR / "heatmap_mcc.png",
                        vmin=0.4, vmax=0.6)
    plot_metric_heatmap(results, "auroc", OUT_DIR / "heatmap_auroc.png",
                        vmin=0.75, vmax=0.90)
    plot_metric_heatmap(results, "f1", OUT_DIR / "heatmap_f1.png",
                        vmin=0.5, vmax=0.85)

    # REPORT
    lines: list[str] = []
    lines.append("# Comitê human_kinome — Avaliação no conjunto de teste\n")
    lines.append("**Composição do comitê**: DT-Kinase + DrugBAN + ConPLex.\n")
    lines.append("**Protocolo**: 5-seed averaged probabilities + thresholds, "
                 "in-domain checkpoint per corpus, dedupe por "
                 "`(seq_id, chembl_id)`, limiar canônico $\\overline{\\tau}$ "
                 "(média dos 3 limiares individuais).\n")
    for corpus in CORPORA:
        info = results[corpus]
        lines.append(f"## Corpus: {corpus}\n")
        lines.append(f"$n$ = {info['n']} pares únicos "
                     f"(positivos = {info['n_pos']}, negativos = {info['n_neg']}).\n")
        lines.append("| sistema | MCC | AUROC | F1 | Accuracy | TP | FP | TN | FN | thr |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for sys_name in (ALL_MODELS + ["committee_4model", "committee_human_kinome"]):
            m = info["metrics"][sys_name]
            mark = " ★" if sys_name == "committee_human_kinome" else ""
            lines.append(
                f"| {sys_name}{mark} | {m['mcc']:.4f} | {m['auroc']:.4f} | "
                f"{m['f1']:.4f} | {m['accuracy']:.4f} | "
                f"{m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | "
                f"{m['threshold']:.3f} |"
            )
        lines.append("")

    lines.append("## Figuras\n")
    lines.append("- `confusion_matrices.png` (matrizes de confusão normalizadas, "
                 "comitê human_kinome em cada corpus)")
    lines.append("- `heatmap_mcc.png` (MCC, comitê + 4 modelos × 3 corpora)")
    lines.append("- `heatmap_auroc.png` (AUROC)")
    lines.append("- `heatmap_f1.png` (F1)\n")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"  → wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
