#!/usr/bin/env python3
"""Choose the operating point on validation and evaluate held-out test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_mcc_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(-probability, kind="mergesort")
    prob_sorted = probability[order]
    y_sorted = y[order].astype(np.int64)
    last = np.r_[np.flatnonzero(prob_sorted[:-1] != prob_sorted[1:]), len(y) - 1]
    tp = np.cumsum(y_sorted)[last]
    fp = (last + 1) - tp
    positives = int(y_sorted.sum())
    negatives = len(y_sorted) - positives
    fn = positives - tp
    tn = negatives - fp
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )
    thresholds = prob_sorted[last]
    best_score = scores.max()
    tied = np.flatnonzero(np.isclose(scores, best_score, rtol=1e-9, atol=1e-12))
    return float(thresholds[tied[np.argmin(np.abs(thresholds[tied] - 0.5))]])


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = probability >= threshold
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "threshold": float(threshold),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
    }


def load_predictions(data_path: Path, prediction_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    prediction = pd.read_csv(prediction_path)
    if len(data) != len(prediction):
        raise ValueError(f"row mismatch: {data_path}={len(data)}, {prediction_path}={len(prediction)}")
    data = data.copy()
    data["probability"] = prediction["pred"].to_numpy(dtype=float)
    if not data["probability"].between(0, 1).all():
        raise ValueError(f"probabilities outside [0, 1] in {prediction_path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/chemglam/universal"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    val = load_predictions(args.data_root / "val.csv", args.val_predictions)
    test = load_predictions(args.data_root / "test.csv", args.test_predictions)
    threshold = best_mcc_threshold(val["label"].to_numpy(), val["probability"].to_numpy())

    result = {
        "protocol": "threshold selected on universal validation; frozen on test",
        "validation": metrics(val["label"].to_numpy(), val["probability"].to_numpy(), threshold),
        "test": {},
    }
    for corpus in ("all", "human", "non_human"):
        subset = test if corpus == "all" else test[test["dataset_source"] == corpus]
        result["test"][corpus] = metrics(
            subset["label"].to_numpy(), subset["probability"].to_numpy(), threshold
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chemglam_results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.output / "raw_predictions.npz",
        y_true=test["label"].to_numpy(dtype=np.int8),
        y_prob=test["probability"].to_numpy(dtype=np.float32),
        dataset_source=test["dataset_source"].to_numpy(dtype=str),
        threshold=np.asarray(threshold),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
