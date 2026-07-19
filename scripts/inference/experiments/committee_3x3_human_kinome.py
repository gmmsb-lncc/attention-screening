#!/usr/bin/env python3
"""3x3 cross-dataset matrix for the human_kinome 3-model committee.

Builds the (train_corpus × test_corpus) matrix of committee MCC/AUROC/F1
for the 3-model committee `DT-Kinase + DrugBAN + ConPLex` (default
human_kinome panel; --models override available).

Inputs per cell (train, test):
  - DT-Kinase
      Diagonal:  results/{benchmark_*_8M_*}/test/level4_cnn_8M/{train}/seed_{s}/raw_predictions.npz
      Off-diag:  results/cross_matrix/dtkinase/{train}_to_{test}/seed_{s}/metrics.json
                 (legacy schema: y_prob + raw_labels lists)
  - DrugBAN
      Diagonal:  DrugBAN/results_universal/results_universal/{train}/seed_{s}/raw_predictions.npz
      Off-diag:  results/cross_matrix/drugban/{train}_to_{test}/seed_{s}/raw_predictions.npz
  - ConPLex
      Diagonal:  ConPLex/results_universal/{train}/seed_{s}/raw_predictions.npz
      Off-diag:  results/cross_matrix/conplex/{train}_to_{test}/seed_{s}/raw_predictions.npz

Aggregation per cell:
  1. 5-seed averaging (probs + thresholds) per model.
  2. Dedupe pairs by (seq_id, chembl_id) using universal_test.tsv filtered
     by test corpus.
  3. committee_prob = mean of N_model calibrated probs.
  4. committee_thr  = mean of N_model thresholds (canonical τ̄).
  5. Compute MCC, AUROC, F1, accuracy, confusion matrix.

Outputs (default --out-dir results/inference/committee_3x3_human_kinome/):
  matrix.csv               cell-by-cell metrics (committee + per-model)
  heatmap_mcc.{png,pdf}    3x3 MCC heatmap (committee)
  heatmap_auroc.{png,pdf}
  heatmap_f1.{png,pdf}
  per_model_heatmap.png    grid of 4 heatmaps (DT-K, DrugBAN, GraphBAN*, ConPLex)
  confusion_grid.png       3x3 grid of confusion matrices (committee)
  REPORT.md                tabular summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, f1_score,
                              accuracy_score, confusion_matrix)

REPO = Path(__file__).resolve().parents[3]

CORPORA = ["human", "non_human", "all"]
SEEDS = [42, 123, 456, 789, 1024]


def diagonal_dt_kinase(corpus: str) -> Path:
    table = {
        "human":     "benchmark_human_8M_13_05_2026/test/level4_cnn_8M/human",
        "non_human": "benchmark_non_human_8M_13_05_2026_v3/test/level4_cnn_8M/non_human",
        "all":       "all/benchmark_all_8M_13_04_2026/test/level4_cnn_8M/all",
    }
    return REPO / "results" / table[corpus]


def diagonal_drugban(corpus: str) -> Path:
    return REPO / "DrugBAN" / "results_universal" / "results_universal" / corpus


def diagonal_conplex(corpus: str) -> Path:
    return REPO / "ConPLex" / "results_universal" / corpus


def offdiag(model: str, train: str, test: str) -> Path:
    return REPO / "results" / "cross_matrix" / model / f"{train}_to_{test}"


def load_seed_probs_dt_kinase(path: Path, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (prob, y_true, threshold) for a single seed."""
    npz_path = path / f"seed_{seed}" / "raw_predictions.npz"
    json_path = path / f"seed_{seed}" / "metrics.json"
    if npz_path.exists():
        d = np.load(npz_path)
        prob = d["test_y_prob"] if "test_y_prob" in d.files else d["y_prob"]
        y = d["test_y_true"] if "test_y_true" in d.files else d["y_true"]
        result_path = path / f"seed_{seed}" / "level4_cnn_results.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())["Split by Scaffold"]["MLP"]
            thr = result["val_threshold"]
        else:
            cal_path = path / f"seed_{seed}" / "level4_cnn_calibration.json"
            thr = json.loads(cal_path.read_text())["threshold"] if cal_path.exists() else 0.5
        return np.asarray(prob, dtype=np.float64), np.asarray(y, dtype=np.int32), float(thr)
    if json_path.exists():
        m = json.loads(json_path.read_text())
        return (np.asarray(m["y_prob"], dtype=np.float64),
                np.asarray(m["raw_labels"], dtype=np.int32),
                float(m["threshold"]))
    raise FileNotFoundError(f"DT-Kinase predictions not found at {path}/seed_{seed}/")


def load_seed_probs_baseline(path: Path, seed: int, side_key: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (prob, y_true, threshold) for DrugBAN/GraphBAN/ConPLex."""
    npz_path = path / f"seed_{seed}" / "raw_predictions.npz"
    d = np.load(npz_path)
    prob = d["test_y_prob"] if "test_y_prob" in d.files else d["y_prob"]
    y = d["test_y_true"] if "test_y_true" in d.files else d["y_true"]
    side = path / f"seed_{seed}" / f"{side_key}_calibration.json"
    thr = json.loads(side.read_text())["threshold"] if side.exists() else 0.5
    return np.asarray(prob, dtype=np.float64), np.asarray(y, dtype=np.int32), float(thr)


def load_5seed(model: str, train: str, test: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (mean_prob, y_true, mean_threshold) across 5 seeds."""
    diagonal = (train == test)
    if model == "dtkinase":
        path = diagonal_dt_kinase(train) if diagonal else offdiag("dtkinase", train, test)
        loader = load_seed_probs_dt_kinase
        loader_args: tuple = ()
    elif model == "drugban":
        path = diagonal_drugban(train) if diagonal else offdiag("drugban", train, test)
        loader = load_seed_probs_baseline
        loader_args = ("drugban",)
    elif model == "graphban":
        path = (REPO / "GraphBAN" / "results_universal" / train) if diagonal \
               else offdiag("graphban", train, test)
        loader = load_seed_probs_baseline
        loader_args = ("graphban",)
    elif model == "conplex":
        path = diagonal_conplex(train) if diagonal else offdiag("conplex", train, test)
        loader = load_seed_probs_baseline
        loader_args = ("conplex",)
    else:
        raise ValueError(f"unknown model: {model}")

    probs_per_seed: list[np.ndarray] = []
    thrs_per_seed: list[float] = []
    y_true: np.ndarray | None = None
    for s in SEEDS:
        try:
            prob, y, thr = loader(path, s, *loader_args)
        except FileNotFoundError as e:
            print(f"  WARN missing {model} {train}->{test} seed_{s}: {e}")
            continue
        probs_per_seed.append(prob)
        thrs_per_seed.append(thr)
        if y_true is None:
            y_true = y
    if not probs_per_seed:
        raise RuntimeError(f"no seeds available for {model} {train}->{test}")
    mean_prob = np.mean(np.stack(probs_per_seed, axis=0), axis=0)
    mean_thr = float(np.mean(thrs_per_seed))
    return mean_prob, y_true, mean_thr


def load_test_keys(test_corpus: str) -> tuple[np.ndarray, np.ndarray]:
    """(pair_key, seq_id) for the test corpus subset of universal_test.tsv."""
    df = pd.read_csv(REPO / "scaffolds_splits" / "output" / "universal_test.tsv", sep="\t")
    if test_corpus != "all":
        df = df[df.get("dataset_source", df.get("organism")) == test_corpus]
    keys = (df["seq_id"].astype(str) + "__" + df["chembl_id"].astype(str)).to_numpy()
    seq_ids = df["seq_id"].astype(str).to_numpy()
    return keys, seq_ids


def dedupe_predictions(prob: np.ndarray, y: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse duplicate (seq_id, chembl_id) rows: mean prob, max label."""
    df = pd.DataFrame({"key": keys, "prob": prob, "y": y})
    g = df.groupby("key", as_index=False).agg({"prob": "mean", "y": "max"})
    g = g.sort_values("key").reset_index(drop=True)
    return g["prob"].to_numpy(), g["y"].to_numpy().astype(np.int32), g["key"].to_numpy()


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


def evaluate_cell(models: list[str], train: str, test: str) -> dict:
    print(f"\n--- cell {train} -> {test} ---")

    probs_per_model: dict[str, np.ndarray] = {}
    thrs_per_model: dict[str, float] = {}
    y_true: np.ndarray | None = None
    for m in models:
        prob, y, thr = load_5seed(m, train, test)
        probs_per_model[m] = prob
        thrs_per_model[m] = thr
        y_true = y if y_true is None else y_true

    # Sanity: align lengths (some baselines may have val+test concatenated;
    # we use only test_y_prob through load_5seed, so len should match
    # universal_test filtered by test corpus).
    n0 = len(y_true)
    keys, seq_ids = load_test_keys(test)
    if len(keys) != n0:
        print(f"  WARN length mismatch: y_true={n0}, keys={len(keys)} — "
              f"using length min")
    n0 = min(n0, len(keys))
    keys = keys[:n0]
    seq_ids = seq_ids[:n0]
    y_true = y_true[:n0]
    for m in models:
        probs_per_model[m] = probs_per_model[m][:n0]

    # Dedupe
    probs_d: dict[str, np.ndarray] = {}
    y_d: np.ndarray | None = None
    for m in models:
        p_d, yt_d, _ = dedupe_predictions(probs_per_model[m], y_true, keys)
        probs_d[m] = p_d
        if y_d is None:
            y_d = yt_d
    print(f"  dedupe: {n0} -> {len(y_d)}")

    # Per-model metrics
    per_model_m: dict[str, dict] = {}
    for m in models:
        per_model_m[m] = system_metrics(y_d, probs_d[m], thrs_per_model[m])

    # Committee
    com_prob = np.mean([probs_d[m] for m in models], axis=0)
    com_thr = float(np.mean([thrs_per_model[m] for m in models]))
    com_m = system_metrics(y_d, com_prob, com_thr)

    return dict(
        train=train, test=test, n=int(len(y_d)),
        committee=com_m, per_model=per_model_m,
    )


def plot_heatmap(M: np.ndarray, models_label: str, metric: str,
                 corpora: list[str], out_path: Path,
                 vmin: float, vmax: float, cmap: str = "RdYlGn") -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(corpora))); ax.set_yticks(range(len(corpora)))
    ax.set_xticklabels(corpora); ax.set_yticklabels(corpora)
    ax.set_xlabel("Test corpus"); ax.set_ylabel("Train corpus")
    ax.set_title(f"{metric.upper()} — {models_label}")
    for i in range(len(corpora)):
        for j in range(len(corpora)):
            v = M[i, j]
            txt = f"{v:.3f}" if not np.isnan(v) else "n/a"
            color = "black" if not np.isnan(v) and v > (vmin + vmax) / 2 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric.upper())
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_confusion_grid(results: dict, corpora: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(len(corpora), len(corpora),
                              figsize=(11, 11), sharex=True, sharey=True)
    for i, train in enumerate(corpora):
        for j, test in enumerate(corpora):
            ax = axes[i, j]
            cell = results.get((train, test))
            if cell is None:
                # Missing cell — render placeholder.
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(["non-bind", "bind"], fontsize=8)
                ax.set_yticklabels(["non-bind", "bind"], fontsize=8)
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=14, color="#888",
                        transform=ax.transAxes)
                ax.set_title(f"{train}→{test}\n(missing)", fontsize=9)
                continue
            m = cell["committee"]
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm / row_sums
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["non-bind", "bind"], fontsize=8)
            ax.set_yticklabels(["non-bind", "bind"], fontsize=8)
            ax.set_title(f"{train}→{test}\nMCC={m['mcc']:.3f}", fontsize=9)
            cell_labels = [["TN", "FP"], ["FN", "TP"]]
            for ii in range(2):
                for jj in range(2):
                    ax.text(jj, ii,
                            f"{cell_labels[ii][jj]}: {cm[ii, jj]}",
                            ha="center", va="center",
                            color="white" if cm_norm[ii, jj] > 0.5 else "black",
                            fontsize=9)
    fig.suptitle("Comitê — matrizes confusão por (train, test)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="dtkinase,drugban,conplex",
                    help="comma-separated committee members "
                         "(default: 3-model human_kinome panel)")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "results" / "inference" / "committee_3x3_human_kinome")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Committee = {models}")
    print(f"Output    = {args.out_dir}")

    results: dict[tuple[str, str], dict] = {}
    for train in CORPORA:
        for test in CORPORA:
            try:
                results[(train, test)] = evaluate_cell(models, train, test)
            except Exception as e:
                print(f"  FAILED {train}->{test}: {e}")
                results[(train, test)] = None

    # Build matrices
    rows: list[dict] = []
    M_mcc = np.full((3, 3), np.nan)
    M_auroc = np.full((3, 3), np.nan)
    M_f1 = np.full((3, 3), np.nan)
    per_model_M: dict[str, np.ndarray] = {m: np.full((3, 3), np.nan) for m in models}

    for i, train in enumerate(CORPORA):
        for j, test in enumerate(CORPORA):
            r = results.get((train, test))
            if r is None:
                continue
            cm = r["committee"]
            M_mcc[i, j] = cm["mcc"]
            M_auroc[i, j] = cm["auroc"]
            M_f1[i, j] = cm["f1"]
            rows.append({
                "train": train, "test": test, "n": r["n"],
                "system": "committee", **cm,
            })
            for m in models:
                per_model_M[m][i, j] = r["per_model"][m]["mcc"]
                rows.append({
                    "train": train, "test": test, "n": r["n"],
                    "system": m, **r["per_model"][m],
                })

    # CSV
    pd.DataFrame(rows).to_csv(args.out_dir / "matrix.csv", index=False)
    print(f"\n  wrote {args.out_dir / 'matrix.csv'}")

    # Heatmaps
    members = "+".join(models)
    plot_heatmap(M_mcc, f"comitê ({members})", "MCC", CORPORA,
                 args.out_dir / "heatmap_mcc.png", vmin=0.20, vmax=0.60)
    plot_heatmap(M_auroc, f"comitê ({members})", "AUROC", CORPORA,
                 args.out_dir / "heatmap_auroc.png", vmin=0.65, vmax=0.90)
    plot_heatmap(M_f1, f"comitê ({members})", "F1", CORPORA,
                 args.out_dir / "heatmap_f1.png", vmin=0.40, vmax=0.85)

    # Per-model heatmaps grid
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.2))
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        Mm = per_model_M[m]
        im = ax.imshow(Mm, cmap="RdYlGn", vmin=0.20, vmax=0.60, aspect="equal")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CORPORA); ax.set_yticklabels(CORPORA)
        ax.set_xlabel("Test"); ax.set_ylabel("Train")
        ax.set_title(m)
        for i in range(3):
            for j in range(3):
                v = Mm[i, j]
                txt = f"{v:.3f}" if not np.isnan(v) else "n/a"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="black" if not np.isnan(v) and v > 0.4 else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("MCC por modelo individual — 3×3 cross-corpus")
    plt.tight_layout()
    plt.savefig(args.out_dir / "per_model_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig(args.out_dir / "per_model_heatmap.pdf", bbox_inches="tight")
    plt.close()

    # Confusion grid
    plot_confusion_grid(results, CORPORA, args.out_dir / "confusion_grid.png")

    # REPORT
    lines: list[str] = []
    lines.append("# Matriz cross-corpus 3×3 — comitê 3-model human_kinome\n")
    lines.append(f"**Comitê**: {' + '.join(models)}.\n")
    lines.append(f"**Protocolo**: 5-seed averaging, dedupe (seq_id, chembl_id), "
                 f"limiar canônico $\\overline{{\\tau}}$.\n")
    lines.append("## Comitê — MCC (rows=train, cols=test)\n")
    lines.append("| train \\ test | " + " | ".join(CORPORA) + " |")
    lines.append("|---|" + "---|" * len(CORPORA))
    for i, train in enumerate(CORPORA):
        cells = [f"{M_mcc[i, j]:.4f}" if not np.isnan(M_mcc[i, j]) else "n/a"
                 for j in range(len(CORPORA))]
        lines.append(f"| **{train}** | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Comitê — AUROC\n")
    lines.append("| train \\ test | " + " | ".join(CORPORA) + " |")
    lines.append("|---|" + "---|" * len(CORPORA))
    for i, train in enumerate(CORPORA):
        cells = [f"{M_auroc[i, j]:.4f}" if not np.isnan(M_auroc[i, j]) else "n/a"
                 for j in range(len(CORPORA))]
        lines.append(f"| **{train}** | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Confusion matrix por cell (TP / FP / TN / FN)\n")
    lines.append("| train | test | n | TP | FP | TN | FN | MCC | AUROC | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for train in CORPORA:
        for test in CORPORA:
            r = results.get((train, test))
            if r is None:
                continue
            c = r["committee"]
            lines.append(
                f"| {train} | {test} | {r['n']} | {c['tp']} | {c['fp']} | "
                f"{c['tn']} | {c['fn']} | {c['mcc']:.4f} | "
                f"{c['auroc']:.4f} | {c['f1']:.4f} |"
            )
    lines.append("")

    lines.append("## Figuras\n")
    lines.append("- `heatmap_mcc.{png,pdf}` — MCC 3×3")
    lines.append("- `heatmap_auroc.{png,pdf}` — AUROC 3×3")
    lines.append("- `heatmap_f1.{png,pdf}` — F1 3×3")
    lines.append("- `per_model_heatmap.{png,pdf}` — heatmaps lado-a-lado dos modelos individuais")
    lines.append("- `confusion_grid.{png,pdf}` — 9 matrizes de confusão (3×3 grid)\n")

    (args.out_dir / "REPORT.md").write_text("\n".join(lines))
    print(f"  wrote {args.out_dir / 'REPORT.md'}")

    # Final summary of missing cells (helps user fix coverage gaps).
    missing = [(tr, te) for tr in CORPORA for te in CORPORA
               if results.get((tr, te)) is None]
    if missing:
        print("\n" + "=" * 60)
        print(" PARTIAL COVERAGE — missing cells:")
        for tr, te in missing:
            print(f"   {tr} -> {te}")
        print(" To fill the gaps, run:")
        print("   bash scripts/thesis_followups/cross_dataset_matrix/"
              "run_cross_matrix.sh")
        print(" Or copy the missing seed_*/raw_predictions.npz from a host")
        print(" that already has them (e.g. results/cross_matrix/{model}/")
        print(" {train}_to_{test}/seed_*/).")
        print("=" * 60)


if __name__ == "__main__":
    main()
