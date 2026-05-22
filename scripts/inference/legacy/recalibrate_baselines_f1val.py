"""
Recalibração pós-hoc DrugBAN + GraphBAN: MCC-val → F1-val.

Carrega raw_predictions.npz salvos (contêm val + test predictions),
re-aplica sweep F1-val para obter novo τ*, recalcula métricas de teste.

Uso (em amazonita-01):
    cd ${PROJECT_ROOT}
    python recalibrate_baselines_f1val.py \
        --results-root results/ \
        --output recalibrated_f1val.json

Saída: JSON agregado por (modelo, corpus) com
{MCC, F1, Acc, P, R, AUROC} ± σ sob calibração F1-val, e
thresholds τ* por semente.

Hipóteses do script:
- raw_predictions.npz contém keys val_y_true, val_y_prob, test_y_true, test_y_prob
- nome do diretório segue padrão results/<model>/.../seed_<N>/raw_predictions.npz
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def optimize_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Sweep F1-optimal threshold sobre val. Replica DrugBAN/GraphBAN post-patch."""
    if len(y_true) == 0:
        return 0.5, 0.0
    order = np.argsort(y_prob, kind="mergesort")[::-1]
    ps, ls = y_prob[order], y_true[order]
    total_pos = float((ls == 1).sum())
    total_neg = float((ls == 0).sum())
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
    return float(thr[bidx]), best


def recalibrate_dir(directory: Path) -> dict:
    """Carrega raw_predictions.npz, aplica F1-val, devolve métricas de teste."""
    pred_file = directory / "raw_predictions.npz"
    if not pred_file.exists():
        return {}
    d = np.load(pred_file, allow_pickle=True)
    keys = set(d.keys())
    if not {"val_y_true", "val_y_prob", "test_y_true", "test_y_prob"} <= keys:
        return {}
    val_yt, val_yp = d["val_y_true"], d["val_y_prob"]
    test_yt, test_yp = d["test_y_true"], d["test_y_prob"]
    tau, val_f1 = optimize_threshold_f1(val_yt, val_yp)
    pred = (test_yp >= tau).astype(int)
    out = {
        "tau_f1val": tau,
        "val_f1_at_tau": val_f1,
        "test_mcc": float(matthews_corrcoef(test_yt, pred)),
        "test_f1": float(f1_score(test_yt, pred, zero_division=0)),
        "test_acc": float(accuracy_score(test_yt, pred)),
        "test_precision": float(precision_score(test_yt, pred, zero_division=0)),
        "test_recall": float(recall_score(test_yt, pred, zero_division=0)),
        "test_auroc": float(roc_auc_score(test_yt, test_yp))
        if len(set(test_yt)) == 2
        else 0.0,
        "n_val": int(len(val_yt)),
        "n_test": int(len(test_yt)),
    }
    return out


def find_seeds(model_dir: Path) -> dict:
    """Procura seed_*/raw_predictions.npz em model_dir."""
    seeds = {}
    for sd in sorted(model_dir.glob("seed_*")):
        if not sd.is_dir():
            continue
        try:
            seed_num = int(sd.name.replace("seed_", ""))
        except ValueError:
            continue
        rec = recalibrate_dir(sd)
        if rec:
            seeds[seed_num] = rec
    return seeds


def aggregate(seeds: dict) -> dict:
    keys = ["test_mcc", "test_f1", "test_acc", "test_precision", "test_recall", "test_auroc"]
    if not seeds:
        return {}
    out = {"n_seeds": len(seeds), "taus": [s["tau_f1val"] for s in seeds.values()]}
    for k in keys:
        vals = [s[k] for s in seeds.values()]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0)), "values": vals}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--output", default="recalibrated_f1val.json")
    ap.add_argument("--models", nargs="+", default=["drugban", "graphban"])
    args = ap.parse_args()

    root = Path(args.results_root)
    report = {}
    for model in args.models:
        for cand in [root / model, root / model / "results"]:
            if not cand.exists():
                continue
            for corpus_dir in sorted(cand.iterdir()):
                if not corpus_dir.is_dir() or corpus_dir.name == "results":
                    continue
                corpus = corpus_dir.name
                seeds = find_seeds(corpus_dir)
                if seeds:
                    agg = aggregate(seeds)
                    report.setdefault(model, {})[corpus] = {
                        "source_dir": str(corpus_dir),
                        "aggregate": agg,
                        "per_seed": seeds,
                    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved recalibrated metrics to {args.output}")
    for model, corpora in report.items():
        for corpus, data in corpora.items():
            agg = data["aggregate"]
            n = agg["n_seeds"]
            mcc = agg["test_mcc"]
            f1_ = agg["test_f1"]
            p = agg["test_precision"]
            r = agg["test_recall"]
            print(
                f"  {model}/{corpus}: n={n} | "
                f"MCC={mcc['mean']:.4f}±{mcc['std']:.4f} "
                f"F1={f1_['mean']:.4f}±{f1_['std']:.4f} "
                f"P={p['mean']:.4f} R={r['mean']:.4f}"
            )


if __name__ == "__main__":
    main()
