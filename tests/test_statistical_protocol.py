"""Sanity tests for scripts/statistical_analysis/.

These tests exercise the statistical protocol implementations on
synthetic inputs (no GPU, no network, no real data required) plus a
single alignment check against the on-disk benchmark artifacts.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.statistical_analysis import (
    HEDGES_NU_PAIRED, HEDGES_NU_UNPAIRED, hedges_J,
)
from scripts.statistical_analysis import data_loader, effect_size
from scripts.statistical_analysis import null_model, upper_limit


def test_hedges_J_paired_value():
    """J(4) = 0.8 exactly under Lakens 2013 / Borenstein approximation."""
    assert abs(hedges_J(HEDGES_NU_PAIRED) - 0.8000) < 1e-9


def test_hedges_J_unpaired_value():
    """J(8) approximately 0.9032 (2 (n - 1) with n = 5 each side)."""
    assert abs(hedges_J(HEDGES_NU_UNPAIRED) - 0.9032) < 1e-3


def test_hedges_J_invalid():
    with pytest.raises(ValueError):
        hedges_J(0)
    with pytest.raises(ValueError):
        hedges_J(-2)


def test_effect_size_paired_zero():
    """Identical per-seed metrics give g_paired = 0 with no variance."""
    a = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    out = effect_size._hedges_paired(a, a)
    assert out["g_paired"] == 0.0
    assert out["d_z"] == 0.0
    assert out["mean_delta"] == 0.0
    assert out["sd_delta"] == 0.0


def test_effect_size_paired_J4_factor():
    """g_paired = J(4) * d_z when SDs are well-defined."""
    a = np.array([0.55, 0.52, 0.58, 0.51, 0.54])
    b = np.array([0.50, 0.50, 0.50, 0.50, 0.50])
    out = effect_size._hedges_paired(a, b)
    expected_J = hedges_J(HEDGES_NU_PAIRED)
    assert abs(out["J_nu"] - expected_J) < 1e-9
    assert out["nu"] == HEDGES_NU_PAIRED
    if out["sd_delta"] > 0:
        expected_g = expected_J * (out["mean_delta"] / out["sd_delta"])
        assert abs(out["g_paired"] - expected_g) < 1e-9


def test_null_model_metrics_constant_yields_mcc_zero():
    """MCC of a constant predictor is 0 by construction."""
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    m = null_model._metrics_constant(y, majority=1)
    assert m["mcc"] == 0.0
    assert m["auroc"] == 0.5


def test_upper_limit_below_one_synthetic():
    """Adding 2-fold noise to pChEMBL must not yield perfect agreement."""
    rng = np.random.default_rng(0)
    p_real = rng.uniform(4, 9, size=500)
    y0 = (p_real >= 6.0).astype(int)
    samples = []
    for _ in range(500):
        p_noisy = p_real + rng.normal(0, 0.301, size=500)
        y_noisy = (p_noisy >= 6.0).astype(int)
        m = upper_limit._scoring_metrics(y0, y_noisy, p_noisy)
        samples.append(m["mcc"])
    arr = np.asarray(samples)
    median = float(np.median(arr))
    assert 0.5 < median < 1.0


def test_paired_delta_zero_when_inputs_equal():
    """Importing bootstrap_ci.paired_delta - identical seeds yield ~zero delta."""
    from scripts.thesis_followups.bootstrap_ci import paired_delta

    rng_data = np.random.default_rng(42)
    seeds = []
    for _ in range(5):
        labels = rng_data.integers(0, 2, size=200)
        logits = rng_data.normal(0, 1, size=200)
        seeds.append({"logits": logits, "labels": labels, "threshold": 0.0})
    rng = np.random.default_rng(0)
    cmp = paired_delta(seeds, seeds, n_boot=200, rng=rng)
    assert abs(cmp["mcc"]["median_delta"]) < 1e-9
    # Wilcoxon must not crash on identical inputs (degenerate path).
    assert cmp["mcc"]["wilcoxon_p"] >= 0.0


def test_data_loader_alignment_non_human():
    """Real-data alignment guard for the on-disk benchmark."""
    try:
        panel = data_loader.load_panel("non_human", seeds=(42,))
    except FileNotFoundError:
        pytest.skip("non_human seed_42 raw_predictions.npz not on disk")
    refs = panel["dtkinase"][0]["y_true"]
    for m, lst in panel.items():
        assert np.array_equal(lst[0]["y_true"], refs), \
            f"Misaligned y_true between dtkinase and {m}"


def test_resolve_bands_marks_primary():
    """0.05 absolute band must be marked as primary."""
    from scripts.statistical_analysis.tost_sensitivity import _resolve_bands

    bands = _resolve_bands(["0.03", "0.05", "0.07", "0.5sigma"], sigma_pooled=0.02)
    primary = [b for b in bands if b["primary"]]
    assert len(primary) == 1
    assert abs(primary[0]["value"] - 0.05) < 1e-9
    cohen_band = [b for b in bands if b["kind"] == "cohen_d"][0]
    assert abs(cohen_band["value"] - 0.5 * 0.02) < 1e-9
