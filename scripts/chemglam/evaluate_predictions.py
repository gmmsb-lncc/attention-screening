#!/usr/bin/env python3
"""Choose the operating point on validation and evaluate held-out test.

The emitted files intentionally mirror the baseline benchmark contract:
raw predictions for every available split, a validation-derived calibration
sidecar, per-seed metrics, identifiers for pairwise comparison, and enough
provenance to reproduce the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


IDENTIFIER_COLUMNS = (
    "source_row", "target_id", "smiles", "target_sequence", "dataset_source"
)


def load_predictions(data_path: Path, prediction_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    prediction = pd.read_csv(prediction_path)
    if len(data) != len(prediction):
        raise ValueError(f"row mismatch: {data_path}={len(data)}, {prediction_path}={len(prediction)}")
    if "pred" not in prediction:
        raise ValueError(f"{prediction_path}: missing prediction column 'pred'")
    data = data.copy()
    data["probability"] = prediction["pred"].to_numpy(dtype=float)
    if not data["probability"].between(0, 1).all():
        raise ValueError(f"probabilities outside [0, 1] in {prediction_path}")
    return data


def _npz_fields(split: str, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return stable, split-prefixed arrays for pairwise auditing."""
    fields: dict[str, np.ndarray] = {
        f"{split}_y_true": frame["label"].to_numpy(dtype=np.int8),
        f"{split}_y_prob": frame["probability"].to_numpy(dtype=np.float32),
    }
    for column in IDENTIFIER_COLUMNS:
        if column in frame:
            fields[f"{split}_{column}"] = frame[column].fillna("").to_numpy(dtype=str)
    return fields


def _json_path(value: Path | None) -> str | None:
    return str(value) if value is not None else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path)
    parser.add_argument("--corpus", choices=("all", "human", "non_human"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--elapsed-seconds", type=float)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root or Path("data/chemglam") / args.corpus
    val = load_predictions(data_root / "val.csv", args.val_predictions)
    test = load_predictions(data_root / "test.csv", args.test_predictions)
    train = None
    if args.train_predictions is not None:
        train = load_predictions(data_root / "train.csv", args.train_predictions)
    threshold = best_mcc_threshold(val["label"].to_numpy(), val["probability"].to_numpy())

    val_metrics = metrics(val["label"].to_numpy(), val["probability"].to_numpy(), threshold)
    test_metrics = metrics(test["label"].to_numpy(), test["probability"].to_numpy(), threshold)
    result: dict[str, Any] = {
        "model": "ChemGLaM",
        "corpus": args.corpus,
        "seed": args.seed,
        "split": "universal_scaffold",
        "protocol": "MCC-optimal threshold selected on validation; frozen on held-out test",
        "model_selection": "minimum validation loss (upstream ChemGLaM criterion)",
        "threshold_optimization": "validation MCC-optimal (no test leakage)",
        "checkpoint": _json_path(args.checkpoint),
        "config": _json_path(args.config),
        "elapsed_seconds": args.elapsed_seconds,
        "validation": val_metrics,
        "test": test_metrics,
        "artifacts": {
            "raw_predictions": "raw_predictions.npz",
            "calibration": "chemglam_calibration.json",
        },
    }
    if train is not None:
        result["train"] = metrics(
            train["label"].to_numpy(), train["probability"].to_numpy(), threshold
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chemglam_results.json").write_text(json.dumps(result, indent=2) + "\n")
    raw_fields = {
        **_npz_fields("val", val),
        **_npz_fields("test", test),
        "threshold": np.asarray(threshold, dtype=np.float64),
        # Legacy aliases retained for consumers that previously interpreted
        # unprefixed arrays as held-out test predictions.
        "y_true": test["label"].to_numpy(dtype=np.int8),
        "y_prob": test["probability"].to_numpy(dtype=np.float32),
    }
    if train is not None:
        raw_fields.update(_npz_fields("train", train))
    np.savez_compressed(args.output / "raw_predictions.npz", **raw_fields)

    calibration = {
        "threshold": float(threshold),
        "calibration_metric": "mcc",
        "val_score": float(val_metrics["mcc"]),
        "n_val": int(val_metrics["n"]),
        "model": "chemglam",
        "corpus": args.corpus,
        "seed": args.seed,
        "source": str(args.output / "raw_predictions.npz"),
        "checkpoint": _json_path(args.checkpoint),
    }
    (args.output / "chemglam_calibration.json").write_text(
        json.dumps(calibration, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
