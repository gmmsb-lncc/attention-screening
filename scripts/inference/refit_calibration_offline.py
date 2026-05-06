"""Refit Platt calibration sidecars offline from saved val predictions.

The original `level4_cnn_calibration.json` files contain identity Platt
(a=1, b=0) because the training pipeline saved val predictions
post-sigmoid in `raw_predictions.npz` but never wrote a fitted
calibrator into the sidecar. At inference time `dtkinase_score.py`
re-applies the identity Platt, leaving probabilities uncalibrated.

This script reads each (corpus, seed)/raw_predictions.npz, recovers the
pre-Platt logits via inverse sigmoid, fits a logistic regression
`y_true ~ logit`, sweeps thresholds for MCC-optimal cutoff on the
post-Platt probabilities, and writes a corrected sidecar JSON.

No GPU, no data files, no checkpoint reload — pure offline fix.

Usage:
    python scripts/inference/refit_calibration_offline.py \
        --root results/benchmark_human_8M_13_05_2026/test/level4_cnn_8M/human \
        --corpus human
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef


def fit_platt(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Fit Platt on (logit, y) with sklearn LogisticRegression.

    Output (a, b) parametrise prob = sigmoid(a * logit + b).
    """
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(logits.reshape(-1, 1), y_true)
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    return a, b


def mcc_optimal_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """2-pass threshold sweep mirroring the training-time recipe.

    100 coarse points over (min, max) of probs, then 100 fine points
    +/- 0.05 around the coarse argmax.
    """
    pmin, pmax = float(probs.min()), float(probs.max())
    coarse = np.linspace(pmin, pmax, 100)
    best_t, best_mcc = 0.5, -1.0
    for t in coarse:
        mcc = matthews_corrcoef(y_true, (probs >= t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    fine = np.linspace(max(pmin, best_t - 0.05), min(pmax, best_t + 0.05), 100)
    for t in fine:
        mcc = matthews_corrcoef(y_true, (probs >= t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return float(best_t), float(best_mcc)


def refit_seed(seed_dir: Path, corpus: str, eps: float = 1e-6) -> dict:
    npz = np.load(seed_dir / "raw_predictions.npz")
    y_true = npz["y_true"].astype(int)
    y_prob = np.clip(npz["y_prob"].astype(np.float64), eps, 1.0 - eps)
    logits = np.log(y_prob / (1.0 - y_prob))
    a, b = fit_platt(logits, y_true)
    cal_probs = 1.0 / (1.0 + np.exp(-(a * logits + b)))
    thr, mcc = mcc_optimal_threshold(cal_probs, y_true)
    sidecar = {
        "platt_a":   a,
        "platt_b":   b,
        "threshold": thr,
        "calibration_metric": "mcc",
        "val_score": mcc,
        "n_val":     int(len(y_true)),
        "model":     "dtkinase",
        "corpus":    corpus,
        "seed":      int(seed_dir.name.split("_")[-1]),
        "source":    str(seed_dir / "raw_predictions.npz"),
        "note":      "platt fitted offline from raw_predictions.npz logits",
    }
    out = seed_dir / "level4_cnn_calibration.json"
    out.write_text(json.dumps(sidecar, indent=2))
    return sidecar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="dir containing seed_*/ subdirs with raw_predictions.npz")
    ap.add_argument("--corpus", choices=["human", "non_human", "all"], required=True)
    args = ap.parse_args()

    seed_dirs = sorted([d for d in args.root.iterdir()
                        if d.is_dir() and d.name.startswith("seed_")])
    if not seed_dirs:
        raise SystemExit(f"no seed_*/ subdirs under {args.root}")
    print(f"refitting {len(seed_dirs)} seeds under {args.root}")
    for sd in seed_dirs:
        s = refit_seed(sd, args.corpus)
        print(f"  {sd.name}: a={s['platt_a']:+.4f} b={s['platt_b']:+.4f} "
              f"thr={s['threshold']:.4f} val_mcc={s['val_score']:.4f}")


if __name__ == "__main__":
    main()
