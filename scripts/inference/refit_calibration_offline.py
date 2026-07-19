"""Run a test-only Platt diagnostic from saved test predictions.

The benchmark `raw_predictions.npz` files used here contain test predictions,
not validation predictions. Fitting Platt or selecting a threshold from them is
test leakage and must never produce the canonical calibration sidecar.

This script reads each (corpus, seed)/raw_predictions.npz, recovers the
pre-Platt logits via inverse sigmoid, fits a logistic regression
`y_true ~ logit`, sweeps thresholds for MCC-optimal cutoff on the
post-Platt probabilities, and writes a clearly marked test-only diagnostic.

No GPU, no data files, no checkpoint reload — pure offline fix.

Usage:
    python scripts/inference/refit_calibration_offline.py \
        --root results/benchmark_human_8M_13_05_2026/test/level4_cnn_8M/human \
        --corpus human --unsafe-test-only
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
        "test_score": mcc,
        "n_test":     int(len(y_true)),
        "model":     "dtkinase",
        "corpus":    corpus,
        "seed":      int(seed_dir.name.split("_")[-1]),
        "source":    str(seed_dir / "raw_predictions.npz"),
        "note":      "UNSAFE diagnostic: Platt and threshold fitted on test predictions",
    }
    out = seed_dir / "level4_cnn_calibration_TEST_ONLY.json"
    out.write_text(json.dumps(sidecar, indent=2))
    return sidecar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="dir containing seed_*/ subdirs with raw_predictions.npz")
    ap.add_argument("--corpus", choices=["human", "non_human", "all"], required=True)
    ap.add_argument(
        "--unsafe-test-only", action="store_true",
        help="acknowledge that this diagnostic fits and evaluates on test data",
    )
    args = ap.parse_args()
    if not args.unsafe_test_only:
        ap.error(
            "raw_predictions.npz contains test predictions; use "
            "refit_calibration_proper.py for valid calibration, or pass "
            "--unsafe-test-only to write a non-canonical diagnostic"
        )

    seed_dirs = sorted([d for d in args.root.iterdir()
                        if d.is_dir() and d.name.startswith("seed_")])
    if not seed_dirs:
        raise SystemExit(f"no seed_*/ subdirs under {args.root}")
    print(f"refitting {len(seed_dirs)} seeds under {args.root}")
    for sd in seed_dirs:
        s = refit_seed(sd, args.corpus)
        print(f"  {sd.name}: a={s['platt_a']:+.4f} b={s['platt_b']:+.4f} "
              f"thr={s['threshold']:.4f} test_mcc={s['test_score']:.4f}")


if __name__ == "__main__":
    main()
