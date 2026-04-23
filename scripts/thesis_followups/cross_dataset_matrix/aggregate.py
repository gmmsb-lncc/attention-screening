#!/usr/bin/env python3
"""Aggregate the 3x3 cross-dataset evaluation matrix.

Inputs (off-diagonal cells):
    results/cross_matrix/{model}/{train}_to_{test}/seed_{s}/
        - metrics.json          (DT-Kinase v7 — threshold+calibration applied)
        - raw_predictions.npz   (baselines — val+test probs, threshold applied here)

Inputs (diagonal cells, imported):
    configured via --diagonal-<model>-<corpus> flags or env vars.
    Each points to a directory containing seed_{s}/{metrics.json | raw_predictions.npz}.

Outputs:
    {out_dir}/cross_matrix_{model}.csv   per-seed and aggregated metrics
    {out_dir}/cross_matrix.tex           LaTeX 3x3 table fragment
    {out_dir}/cross_matrix.json          full report

Protocol: baseline metrics apply the MCC-optimal threshold derived on
val_y_prob (from the training corpus val set) to test_y_prob. This matches
the "keep threshold from training corpus" decision in the plan.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODELS = ["dtkinase", "drugban", "graphban", "conplex"]
CORPORA = ["human", "non_human", "all"]
METRICS = ["mcc", "auroc", "auprc", "f1", "accuracy", "precision", "recall"]
MODEL_LABELS = {
    "dtkinase": "DT-Kinase v7",
    "drugban": "DrugBAN",
    "graphban": "GraphBAN",
    "conplex": "ConPLex",
}
CORPUS_LABELS = {"human": "H", "non_human": "NH", "all": "All"}


def _f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """F1-optimal threshold via sorted sweep. Matches recalibrate_baselines_f1val
    (DrugBAN + GraphBAN native criterion). Tie-break: closest to 0.5."""
    y = y_true.astype(np.int64)
    if y.size == 0 or len(np.unique(y)) < 2:
        return 0.5
    order = np.argsort(y_prob, kind="mergesort")[::-1]
    ps, ls = y_prob[order], y[order]
    total_pos = float((ls == 1).sum())
    tp_cum = np.cumsum(ls == 1, dtype=np.float64)
    fp_cum = np.cumsum(ls == 0, dtype=np.float64)
    last_idx = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]
    tp = tp_cum[last_idx]
    fp = fp_cum[last_idx]
    fn = total_pos - tp
    thr = ps[last_idx]
    sentinel = np.nextafter(float(ps.max()), np.inf)
    tp = np.r_[0.0, tp]
    fp = np.r_[0.0, fp]
    fn = np.r_[total_pos, fn]
    thr = np.r_[sentinel, thr]
    den = 2 * tp + fp + fn
    sc = np.where(den > 0, (2 * tp) / den, 0.0)
    best = float(np.nanmax(sc))
    ties = np.where(np.isclose(sc, best, rtol=1e-9, atol=1e-12))[0]
    bidx = int(ties[np.argmin(np.abs(thr[ties] - 0.5))])
    return float(thr[bidx])


# Per-model threshold criterion on the training-corpus validation set.
# Matches the thesis protocol declared in capitulo5 tables.
_MODEL_THRESHOLD_CRITERION = {
    "dtkinase": "mcc",
    "conplex":  "mcc",
    "drugban":  "f1",
    "graphban": "f1",
}


def _mcc_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Vectorised two-pass MCC-optimal threshold.

    Matches the semantics of benchmark/levels/level4_cnn._best_mcc_threshold
    (coarse grid + fine sweep) but avoids calling sklearn.matthews_corrcoef
    per threshold — crucial for large validation sets (~69k rows).
    """
    y = y_true.astype(np.int64)
    if y.size == 0 or len(np.unique(y)) < 2:
        return 0.5

    def _best(thresholds: np.ndarray) -> tuple[float, float]:
        P = float((y == 1).sum())
        N = float((y == 0).sum())
        best_thr, best_mcc = 0.5, -1.0
        for thr in thresholds:
            pred = y_prob >= thr
            tp = float((pred & (y == 1)).sum())
            fp = float((pred & (y == 0)).sum())
            fn = P - tp
            tn = N - fp
            denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if denom == 0:
                continue
            mcc = (tp * tn - fp * fn) / denom
            if mcc > best_mcc:
                best_mcc = mcc
                best_thr = float(thr)
        return best_thr, best_mcc

    coarse = np.linspace(0.01, 0.99, 100)
    best_thr, _ = _best(coarse)
    lo, hi = max(0.01, best_thr - 0.05), min(0.99, best_thr + 0.05)
    fine = np.linspace(lo, hi, 100)
    best_thr2, _ = _best(fine)
    return float(best_thr2)


def _metrics_from_preds(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    preds = (y_prob >= threshold).astype(int)
    two_class = len(np.unique(y_true)) == 2
    return {
        "mcc": float(matthews_corrcoef(y_true, preds)) if two_class else 0.0,
        "auroc": float(roc_auc_score(y_true, y_prob)) if two_class else 0.0,
        "auprc": float(average_precision_score(y_true, y_prob)) if two_class else 0.0,
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }


def _read_cell_dtkinase(seed_dir: Path) -> dict | None:
    """Read DT-Kinase v7 metrics. Supports two layouts:

    1. Cross-dataset eval (eval_checkpoint_on_dataset.py output): metrics.json
       carries full metrics + y_prob + raw_labels.
    2. Diagonal benchmark output: level4_cnn_results.json + raw_predictions.npz
       (npz has y_true + y_prob post-Platt; metrics live in the JSON).
    """
    # Layout 1: cross-dataset
    mpath = seed_dir / "metrics.json"
    if mpath.exists():
        with open(mpath) as fh:
            m = json.load(fh)
        auprc = 0.0
        if "y_prob" in m and "raw_labels" in m:
            y = np.asarray(m["raw_labels"])
            p = np.asarray(m["y_prob"])
            if len(np.unique(y)) == 2:
                auprc = float(average_precision_score(y, p))
        return {
            "mcc": float(m.get("mcc", 0.0)),
            "auroc": float(m.get("auroc", 0.0)),
            "auprc": auprc,
            "f1": float(m.get("f1", 0.0)),
            "accuracy": float(m.get("accuracy", 0.0)),
            "precision": float(m.get("precision", 0.0)),
            "recall": float(m.get("recall", 0.0)),
            "threshold": float(m.get("threshold", 0.5)),
            "n": int(m.get("n_samples", 0)),
        }

    # Layout 2: diagonal benchmark output
    jpath = seed_dir / "level4_cnn_results.json"
    npz_path = seed_dir / "raw_predictions.npz"
    if jpath.exists():
        with open(jpath) as fh:
            j = json.load(fh)
        # nested format: {"Split by Scaffold": {"MLP": {...}}}
        inner = next(iter(j.values())) if j else {}
        if isinstance(inner, dict):
            inner = next(iter(inner.values())) if inner else {}
        auprc = 0.0
        n = 0
        if npz_path.exists():
            d = np.load(npz_path, allow_pickle=True)
            if "y_true" in d and "y_prob" in d and len(np.unique(d["y_true"])) == 2:
                auprc = float(average_precision_score(d["y_true"], d["y_prob"]))
                n = int(len(d["y_true"]))
        return {
            "mcc": float(inner.get("mcc", 0.0)),
            "auroc": float(inner.get("auc", inner.get("auroc", 0.0))),
            "auprc": auprc,
            "f1": float(inner.get("f1", 0.0)),
            "accuracy": float(inner.get("accuracy", 0.0)),
            "precision": float(inner.get("precision", 0.0)),
            "recall": float(inner.get("recall", 0.0)),
            "threshold": float(inner.get("threshold", inner.get("val_threshold", 0.5))),
            "n": n,
        }
    return None


def _read_cell_baseline(seed_dir: Path, criterion: str = "mcc") -> dict | None:
    """Read a baseline .npz and compute test metrics at the val-derived threshold.

    criterion='mcc' → MCC-optimal on val (ConPLex native).
    criterion='f1'  → F1-optimal on val (DrugBAN + GraphBAN native).
    """
    npz = seed_dir / "raw_predictions.npz"
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    keys = set(d.keys())
    if not {"val_y_true", "val_y_prob", "test_y_true", "test_y_prob"} <= keys:
        return None
    val_y, val_p = d["val_y_true"], d["val_y_prob"]
    test_y, test_p = d["test_y_true"], d["test_y_prob"]
    if criterion == "f1":
        tau = _f1_threshold(val_y, val_p)
    else:
        tau = _mcc_threshold(val_y, val_p)
    out = _metrics_from_preds(test_y, test_p, tau)
    out["threshold_criterion"] = criterion
    return out


def _scan_cell(model: str, cell_dir: Path) -> dict:
    """Scan one (train, test) cell directory for all seed_* subdirs."""
    per_seed: dict[int, dict] = {}
    if not cell_dir.exists():
        return {"per_seed": per_seed, "n_seeds": 0, "aggregate": {}}
    for sd in sorted(cell_dir.glob("seed_*")):
        if not sd.is_dir():
            continue
        try:
            seed = int(sd.name.replace("seed_", ""))
        except ValueError:
            continue
        if model == "dtkinase":
            cell = _read_cell_dtkinase(sd)
        else:
            criterion = _MODEL_THRESHOLD_CRITERION.get(model, "mcc")
            cell = _read_cell_baseline(sd, criterion=criterion)
        if cell:
            per_seed[seed] = cell
    agg: dict[str, dict[str, float]] = {}
    for mname in METRICS:
        vals = [s[mname] for s in per_seed.values() if mname in s]
        if vals:
            agg[mname] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
                "values": vals,
            }
    return {"per_seed": per_seed, "n_seeds": len(per_seed), "aggregate": agg}


def _diagonal_path(args: argparse.Namespace, model: str, corpus: str) -> Path | None:
    key = f"diagonal_{model}_{corpus}"
    raw = getattr(args, key, None)
    if raw:
        return Path(raw)
    env_key = f"DIAG_{model.upper()}_{corpus.upper()}"
    env_val = os.environ.get(env_key)
    if env_val:
        return Path(env_val)
    return None


def _build_table(report: dict) -> tuple[str, str]:
    """Return (csv_text, latex_text) for the 3x3 table (MCC primary)."""
    # CSV: one row per (model, train, test)
    csv_lines = ["model,train,test,kind,mcc_mean,mcc_std,auroc_mean,auprc_mean,f1_mean,n_seeds"]
    for model in MODELS:
        for train in CORPORA:
            for test in CORPORA:
                cell = report.get(model, {}).get(train, {}).get(test, {})
                agg = cell.get("aggregate", {})
                kind = "diagonal" if train == test else "cross"
                mcc = agg.get("mcc", {})
                auroc = agg.get("auroc", {})
                auprc = agg.get("auprc", {})
                f1 = agg.get("f1", {})
                csv_lines.append(",".join([
                    model, train, test, kind,
                    f"{mcc.get('mean', float('nan')):.4f}",
                    f"{mcc.get('std', float('nan')):.4f}",
                    f"{auroc.get('mean', float('nan')):.4f}",
                    f"{auprc.get('mean', float('nan')):.4f}",
                    f"{f1.get('mean', float('nan')):.4f}",
                    str(cell.get("n_seeds", 0)),
                ]))

    # LaTeX 3x3 per-model table (MCC mean +/- std; italic for diagonal imports)
    tex = ["% Auto-generated by scripts/thesis_followups/cross_dataset_matrix/aggregate.py",
           "\\begin{tabular}{l|ccc}",
           "\\hline",
           "\\multicolumn{4}{c}{\\textbf{Cross-dataset MCC (mean $\\pm$ std over seeds)}} \\\\",
           "\\hline",
           " & " + " & ".join(f"Test: {CORPUS_LABELS[c]}" for c in CORPORA) + " \\\\",
           "\\hline"]
    for model in MODELS:
        tex.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{MODEL_LABELS[model]}}}}} \\\\")
        for train in CORPORA:
            row = [f"Train: {CORPUS_LABELS[train]}"]
            for test in CORPORA:
                cell = report.get(model, {}).get(train, {}).get(test, {})
                agg = cell.get("aggregate", {}).get("mcc", {})
                if agg:
                    mean = agg["mean"]
                    std = agg["std"]
                    formatted = f"{mean:.3f}$\\pm${std:.3f}"
                    if train == test:
                        formatted = f"\\textit{{{formatted}}}"
                    row.append(formatted)
                else:
                    row.append("--")
            tex.append(" & ".join(row) + " \\\\")
        tex.append("\\hline")
    tex.append("\\end{tabular}")
    return "\n".join(csv_lines), "\n".join(tex)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/cross_matrix",
                    help="Dir with {model}/{train}_to_{test}/seed_*/ layout")
    ap.add_argument("--out-dir", default="results/cross_matrix/summary")
    for model in MODELS:
        for corpus in CORPORA:
            ap.add_argument(f"--diagonal-{model}-{corpus}", default=None,
                            help=f"Diagonal dir for {model} train=test={corpus}. "
                                 f"Fallback env: DIAG_{model.upper()}_{corpus.upper()}")
    args = ap.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {}
    for model in MODELS:
        report[model] = {}
        for train in CORPORA:
            report[model][train] = {}
            for test in CORPORA:
                if train == test:
                    diag_path = _diagonal_path(args, model, train)
                    cell_dir = diag_path if diag_path is not None else None
                else:
                    cell_dir = root / model / f"{train}_to_{test}"
                if cell_dir is None:
                    report[model][train][test] = {"per_seed": {}, "n_seeds": 0, "aggregate": {}}
                    continue
                report[model][train][test] = _scan_cell(model, cell_dir)

    # Leakage reports
    leakage = {}
    filter_root = root / "filters"
    if filter_root.exists():
        for sub in sorted(filter_root.iterdir()):
            lr = sub / "leakage_report.json"
            if lr.exists():
                with open(lr) as fh:
                    leakage[sub.name] = json.load(fh)
    report["_leakage"] = leakage

    # Emit
    (out_dir / "cross_matrix.json").write_text(json.dumps(report, indent=2, default=str))
    csv_text, latex_text = _build_table(report)
    (out_dir / "cross_matrix.csv").write_text(csv_text + "\n")
    (out_dir / "cross_matrix.tex").write_text(latex_text + "\n")

    # Console summary
    print(f"Wrote {out_dir}/cross_matrix.{{csv,tex,json}}")
    for model in MODELS:
        for train in CORPORA:
            for test in CORPORA:
                agg = report[model][train][test].get("aggregate", {}).get("mcc", {})
                n = report[model][train][test].get("n_seeds", 0)
                if n:
                    kind = "diag" if train == test else "cross"
                    print(f"  {model:10s} {train:>9s}->{test:<9s} [{kind}] "
                          f"MCC={agg['mean']:.4f}±{agg['std']:.4f} (n={n})")


if __name__ == "__main__":
    main()
