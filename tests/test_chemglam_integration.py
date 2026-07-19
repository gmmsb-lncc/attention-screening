"""Unit tests for ChemGLaM benchmark artifact parity."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluate = _load("chemglam_evaluate", "scripts/chemglam/evaluate_predictions.py")
aggregate_results = _load("chemglam_aggregate", "scripts/chemglam/aggregate_results.py")
make_configs = _load("chemglam_make_configs", "scripts/chemglam/make_run_configs.py")

pytestmark = pytest.mark.unit


def _data(labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "source_row": np.arange(len(labels)),
        "split": "fixture",
        "smiles": [f"C{i}" for i in range(len(labels))],
        "target_sequence": [f"SEQ{i}" for i in range(len(labels))],
        "target_id": [f"P{i}" for i in range(len(labels))],
        "label": labels,
        "dataset_source": "human",
    })


def test_evaluator_saves_all_splits_ids_calibration_and_provenance(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    output = tmp_path / "result"
    data_root.mkdir()
    frames = {
        "train": _data([0, 1, 0, 1]),
        "val": _data([0, 1, 0, 1]),
        "test": _data([1, 0, 1, 0]),
    }
    probabilities = {
        "train": [0.1, 0.9, 0.2, 0.8],
        "val": [0.2, 0.8, 0.3, 0.7],
        "test": [0.9, 0.1, 0.75, 0.25],
    }
    prediction_paths = {}
    for split, frame in frames.items():
        frame.to_csv(data_root / f"{split}.csv", index=False)
        path = tmp_path / f"{split}_prediction.csv"
        pd.DataFrame({"pred": probabilities[split]}).to_csv(path, index=False)
        prediction_paths[split] = path

    argv = [
        "evaluate_predictions.py", "--corpus", "human", "--seed", "42",
        "--data-root", str(data_root), "--output", str(output),
        "--train-predictions", str(prediction_paths["train"]),
        "--val-predictions", str(prediction_paths["val"]),
        "--test-predictions", str(prediction_paths["test"]),
        "--checkpoint", "logs/chemglam_human_seed42/best_checkpoint.ckpt",
        "--config", "results/chemglam/config.json",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert evaluate.main() == 0

    result = json.loads((output / "chemglam_results.json").read_text())
    calibration = json.loads((output / "chemglam_calibration.json").read_text())
    raw = np.load(output / "raw_predictions.npz")
    assert result["seed"] == 42
    assert set(("train", "validation", "test")).issubset(result)
    assert calibration["calibration_metric"] == "mcc"
    assert calibration["threshold"] == result["validation"]["threshold"]
    for field in (
        "train_y_true", "train_y_prob", "val_y_true", "val_y_prob",
        "test_y_true", "test_y_prob", "val_target_id", "test_smiles",
    ):
        assert field in raw.files


def test_multiseed_aggregation_uses_population_std():
    runs = []
    for seed, mcc in ((42, 0.4), (123, 0.6)):
        metric_row = {
            "threshold": 0.5, "mcc": mcc, "auroc": 0.7, "auprc": 0.65,
            "f1": 0.6, "precision": 0.61, "recall": 0.59, "accuracy": 0.7,
        }
        runs.append({
            "model": "ChemGLaM", "corpus": "human", "seed": seed,
            "validation": dict(metric_row), "test": dict(metric_row),
        })
    result = aggregate_results.aggregate_runs(runs)
    assert result["seeds"] == [42, 123]
    assert result["aggregate"]["test"]["mcc"]["mean"] == pytest.approx(0.5)
    assert result["aggregate"]["test"]["mcc"]["std"] == pytest.approx(0.1)


def test_run_configs_isolate_corpus_cache_and_train_evaluation(tmp_path, monkeypatch):
    output = tmp_path / "configs"
    monkeypatch.setattr(sys, "argv", [
        "make_run_configs.py", "--base", str(REPO / "configs/chemglam_universal.json"),
        "--corpus", "non_human", "--seed", "123", "--output", str(output),
    ])
    assert make_configs.main() == 0
    train = json.loads((output / "train.json").read_text())
    train_eval = json.loads((output / "train_eval.json").read_text())
    test = json.loads((output / "test.json").read_text())
    assert train["dataset_csv_path"].endswith("non_human/train_valid.csv")
    assert train_eval["dataset_csv_path"].endswith("non_human/train.csv")
    assert train_eval["cache_dir"] == "chemglam_non_human_train"
    assert test["cache_dir"] == "chemglam_non_human_test"
