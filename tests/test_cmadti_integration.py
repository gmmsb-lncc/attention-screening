"""Pure contract tests for the canonical CMA-DTI integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


protocol = load_module("cmadti_protocol", "scripts/cmadti/protocol.py")
prepare = load_module("cmadti_prepare", "scripts/cmadti/prepare_universal.py")
aggregate = load_module("cmadti_aggregate", "scripts/cmadti/aggregate_results.py")

pytestmark = pytest.mark.unit


def test_publication_fidelity_and_leakage_free_threshold_contract():
    config = yaml.safe_load((REPO / "configs/cmadti_universal.yaml").read_text())
    assert config["model"]["attention_heads"] == 2
    assert config["training"]["epochs"] == 100
    assert config["protocol"]["model_selection"] == "validation_auroc"
    assert config["protocol"]["threshold_selection"] == "validation_mcc"
    assert config["publication"]["doi"] == "10.3389/fbinf.2026.1861685"


def test_threshold_is_selected_on_validation_and_metrics_use_frozen_value():
    val_y = np.array([0, 0, 1, 1])
    val_prob = np.array([0.1, 0.3, 0.6, 0.9])
    threshold = protocol.best_mcc_threshold(val_y, val_prob)
    assert threshold == pytest.approx(0.6)
    test = protocol.metrics(np.array([0, 1]), np.array([0.55, 0.65]), threshold)
    assert test["threshold"] == pytest.approx(threshold)
    assert test["mcc"] == pytest.approx(1.0)


def test_converter_preserves_identifiers_and_rows(tmp_path):
    source = tmp_path / "human_train.tsv"
    pd.DataFrame({
        "canonical_smiles": ["CCO", "CCN"], "seq": ["AAAA", "BBBB"],
        "seq_id": ["P1", "P2"], "chembl_id": ["C1", "C2"],
        "label": [0, 1], "dataset_source": ["human", "human"],
    }).to_csv(source, sep="\t", index=False)
    result = prepare.convert(source, "train", "human")
    assert list(result.columns) == [
        "source_row", "split", "SMILES", "Protein", "target_id",
        "chembl_id", "Y", "dataset_source",
    ]
    assert len(result) == 2
    assert result["source_row"].tolist() == [0, 1]


def test_multiseed_aggregate_population_std():
    runs = []
    for seed, mcc in ((42, 0.4), (123, 0.6)):
        row = {"mcc": mcc, "auroc": 0.7, "auprc": 0.6, "f1": 0.5,
               "precision": 0.5, "recall": 0.5, "accuracy": 0.7,
               "loss": 0.4, "threshold": 0.5}
        runs.append({"corpus": "human", "seed": seed, "train": dict(row),
                     "validation": dict(row), "test": dict(row)})
    result = aggregate.aggregate_runs(runs)
    assert result["aggregate"]["test"]["mcc"]["mean"] == pytest.approx(0.5)
    assert result["aggregate"]["test"]["mcc"]["std"] == pytest.approx(0.1)
