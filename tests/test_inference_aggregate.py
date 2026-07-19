"""Unit tests for scripts/inference/aggregate.py.

Covers:
  - dedupe by (uniprot, chembl_id) before merge (no cartesian explosion)
  - soft mean across N models
  - agreement_count + tier mapping
  - rank fusion (Borda count, lower = better)
  - confidence = 1 - prob_std
  - partial committee (subset of models)
  - tie-breaking ordering (prob_mean DESC, agreement DESC, confidence DESC)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "inference"))

import aggregate  # noqa: E402


pytestmark = pytest.mark.unit


# ======================================================================
# Helpers
# ======================================================================

def _write_scores(tmp_path, model: str, rows: list[dict]) -> Path:
    """Write a scores_<model>.csv under tmp_path with the canonical schema."""
    df = pd.DataFrame(rows)
    p = tmp_path / f"scores_{model}.csv"
    df.to_csv(p, index=False)
    return p


def _make_pair(uniprot: str, chembl_id: str, prob: float, thr: float = 0.5) -> dict:
    return {
        "uniprot":   uniprot,
        "chembl_id": chembl_id,
        "prob":      prob,
        "pred":      int(prob >= thr),
        "threshold": thr,
    }


# ======================================================================
# Dedupe behavior
# ======================================================================

def test_dedupe_collapses_duplicate_keys(tmp_path):
    """50 rows with 27 unique (uniprot, chembl_id) keys → 27 output rows."""
    pairs = []
    n_unique = 27
    for i in range(n_unique):
        for _ in range(2 if i % 2 == 0 else 1):  # some duplicated, some not
            pairs.append(_make_pair(f"P{i:03d}", f"L{i:03d}", 0.6))

    for m in ("dtkinase", "drugban", "graphban", "conplex"):
        _write_scores(tmp_path, m, pairs)

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    df = pd.read_csv(out)
    assert len(df) == n_unique, f"expected {n_unique} rows post-dedupe, got {len(df)}"
    assert df["pair_id"].nunique() == n_unique


def test_dedupe_averages_duplicate_probs(tmp_path):
    """Two rows with same key but different probs → output prob = mean."""
    rows = [
        _make_pair("P1", "L1", 0.4),
        _make_pair("P1", "L1", 0.8),
    ]
    _write_scores(tmp_path, "dtkinase", rows)
    _write_scores(tmp_path, "drugban", rows)

    model_dfs = aggregate.load_model_scores(tmp_path)
    df_dtk = model_dfs["dtkinase"]
    assert len(df_dtk) == 1
    np.testing.assert_allclose(df_dtk["prob_dtkinase"].iloc[0], 0.6)


def test_dedupe_pred_uses_max(tmp_path):
    """Duplicate keys w/ different preds → max (OR-fold)."""
    rows = [
        {**_make_pair("P1", "L1", 0.3), "pred": 0},
        {**_make_pair("P1", "L1", 0.6), "pred": 1},
    ]
    _write_scores(tmp_path, "dtkinase", rows)
    _write_scores(tmp_path, "drugban", rows)

    model_dfs = aggregate.load_model_scores(tmp_path)
    assert int(model_dfs["dtkinase"]["pred_dtkinase"].iloc[0]) == 1


def test_rejects_prediction_on_a_different_scale(tmp_path):
    """CSV contract requires pred == (prob >= threshold) row by row."""
    inconsistent = {
        "uniprot": "P1", "chembl_id": "L1", "prob": 0.75,
        "pred": 0, "threshold": 0.50,
    }
    _write_scores(tmp_path, "dtkinase", [inconsistent])
    _write_scores(tmp_path, "drugban", [_make_pair("P1", "L1", 0.75)])
    with pytest.raises(ValueError, match="inconsistent with prob >= threshold"):
        aggregate.load_model_scores(tmp_path)


# ======================================================================
# Aggregation: soft mean, agreement, tier
# ======================================================================

def test_soft_mean_and_tier_strong(tmp_path):
    """All 4 models predict binder with prob ~0.8 → STRONG, PoE on prob_mean,
    arithmetic mean on prob_soft_mean."""
    pair = ("P1", "L1")
    probs = {"dtkinase": 0.84, "drugban": 0.79, "graphban": 0.82, "conplex": 0.71}
    for m, p in probs.items():
        _write_scores(tmp_path, m, [_make_pair(*pair, p)])

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    df = pd.read_csv(out)
    assert len(df) == 1
    # Canonical (PoE = geometric mean) on prob_mean / prob_committee.
    expected_poe = float(np.exp(np.mean(np.log(list(probs.values())))))
    np.testing.assert_allclose(df["prob_mean"].iloc[0], expected_poe)
    np.testing.assert_allclose(df["prob_committee"].iloc[0], expected_poe)
    # Soft-mean preserved on prob_soft_mean for backward compatibility.
    np.testing.assert_allclose(
        df["prob_soft_mean"].iloc[0], np.mean(list(probs.values())))
    assert int(df["agreement_count"].iloc[0]) == 4
    assert df["tier"].iloc[0] == "STRONG"


def test_poe_applies_the_same_rule_to_scores_and_thresholds(tmp_path):
    pair = ("P1", "L1")
    probs = {"dtkinase": 0.70, "drugban": 0.80,
             "graphban": 0.60, "conplex": 0.55}
    thresholds = {"dtkinase": 0.45, "drugban": 0.50,
                  "graphban": 0.40, "conplex": 0.52}
    for model, prob in probs.items():
        _write_scores(tmp_path, model, [
            _make_pair(*pair, prob, thr=thresholds[model])
        ])

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    row = pd.read_csv(out).iloc[0]
    expected_score = float(np.exp(np.mean(np.log(list(probs.values())))))
    expected_threshold = float(
        np.exp(np.mean(np.log(list(thresholds.values()))))
    )
    np.testing.assert_allclose(row["prob_committee"], expected_score)
    np.testing.assert_allclose(row["thr_committee"], expected_threshold)
    assert int(row["pred_committee"]) == int(expected_score >= expected_threshold)


def test_tier_likely(tmp_path):
    """3/4 models agree → LIKELY."""
    pair = ("P1", "L1")
    probs = {"dtkinase": 0.65, "drugban": 0.72, "graphban": 0.55, "conplex": 0.31}
    for m, p in probs.items():
        _write_scores(tmp_path, m, [_make_pair(*pair, p)])

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    df = pd.read_csv(out)
    assert int(df["agreement_count"].iloc[0]) == 3
    assert df["tier"].iloc[0] == "LIKELY"


def test_tier_uncertain(tmp_path):
    pair = ("P1", "L1")
    probs = {"dtkinase": 0.55, "drugban": 0.60, "graphban": 0.40, "conplex": 0.30}
    for m, p in probs.items():
        _write_scores(tmp_path, m, [_make_pair(*pair, p)])

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    df = pd.read_csv(out)
    assert int(df["agreement_count"].iloc[0]) == 2
    assert df["tier"].iloc[0] == "UNCERTAIN"


def test_tier_unlikely(tmp_path):
    pair = ("P1", "L1")
    probs = {"dtkinase": 0.30, "drugban": 0.20, "graphban": 0.40, "conplex": 0.55}
    for m, p in probs.items():
        _write_scores(tmp_path, m, [_make_pair(*pair, p)])

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    df = pd.read_csv(out)
    assert int(df["agreement_count"].iloc[0]) <= 1
    assert df["tier"].iloc[0] == "UNLIKELY"


def test_confidence_inverse_of_std(tmp_path):
    """confidence = 1 - prob_std."""
    pair = ("P1", "L1")
    probs = [0.10, 0.50, 0.90]                 # large spread
    for i, p in enumerate(probs):
        m = ["dtkinase", "drugban", "graphban"][i]
        _write_scores(tmp_path, m, [_make_pair(*pair, p)])

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    df = pd.read_csv(out)
    expected_std = float(np.std(probs))
    np.testing.assert_allclose(df["prob_std"].iloc[0], expected_std, rtol=1e-5)
    np.testing.assert_allclose(df["confidence"].iloc[0], 1 - expected_std, rtol=1e-5)


# ======================================================================
# Rank fusion (Borda)
# ======================================================================

def test_rank_fusion_lower_is_better(tmp_path):
    """High-prob pairs across all models → low (better) rank_fusion."""
    pairs = [
        _make_pair("P1", "L1", 0.95),  # winner everywhere
        _make_pair("P2", "L2", 0.50),
        _make_pair("P3", "L3", 0.10),  # loser everywhere
    ]
    for m in ("dtkinase", "drugban", "graphban"):
        _write_scores(tmp_path, m, pairs)

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    df = pd.read_csv(out)
    df = df.set_index("pair_id")

    # P1 has rank 1 in each model → rank_fusion = 3
    assert df.loc["P1__L1", "rank_fusion"] == 3.0
    # P2 has rank 2 in each → rank_fusion = 6
    assert df.loc["P2__L2", "rank_fusion"] == 6.0
    # P3 has rank 3 in each → rank_fusion = 9
    assert df.loc["P3__L3", "rank_fusion"] == 9.0


# ======================================================================
# Partial committee (subset of models)
# ======================================================================

def test_partial_committee_three_models(tmp_path):
    """ConPLex absent → committee operates with 3 models. PoE applied over
    the 3 present probs; arithmetic mean preserved on prob_soft_mean."""
    pair = ("P1", "L1")
    probs = {"dtkinase": 0.8, "drugban": 0.7, "graphban": 0.6}
    for m, p in probs.items():
        _write_scores(tmp_path, m, [_make_pair(*pair, p)])

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    df = pd.read_csv(out)
    assert len(df) == 1
    expected_poe = float(np.exp(np.mean(np.log(list(probs.values())))))
    np.testing.assert_allclose(df["prob_mean"].iloc[0], expected_poe)
    np.testing.assert_allclose(
        df["prob_soft_mean"].iloc[0], np.mean(list(probs.values())))
    assert int(df["agreement_count"].iloc[0]) == 3
    assert df["tier"].iloc[0] == "STRONG"   # 3/3 in 3-model regime


def test_partial_committee_two_models_minimum(tmp_path):
    pair = ("P1", "L1")
    _write_scores(tmp_path, "dtkinase", [_make_pair(*pair, 0.8)])
    _write_scores(tmp_path, "drugban",  [_make_pair(*pair, 0.7)])

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    df = pd.read_csv(out)
    assert len(df) == 1


def test_one_model_raises(tmp_path):
    """Single model is not a committee → must raise."""
    _write_scores(tmp_path, "dtkinase", [_make_pair("P1", "L1", 0.8)])
    with pytest.raises(RuntimeError, match="at least 2"):
        aggregate.load_model_scores(tmp_path)


# ======================================================================
# Tier rescale (Anexo B Tabela B.6)
# ======================================================================

@pytest.mark.parametrize("agreement,expected", [
    (4, "STRONG"), (3, "LIKELY"), (2, "UNCERTAIN"), (1, "UNLIKELY"), (0, "UNLIKELY"),
])
def test_assign_tier_n4(agreement, expected):
    assert aggregate.assign_tier(agreement, 4) == expected


@pytest.mark.parametrize("agreement,expected", [
    (5, "STRONG"), (4, "LIKELY"), (3, "UNCERTAIN"),
    (2, "UNCERTAIN"), (1, "UNLIKELY"), (0, "UNLIKELY"),
])
def test_assign_tier_n5(agreement, expected):
    assert aggregate.assign_tier(agreement, 5) == expected


def test_five_model_committee_includes_chemglam(tmp_path):
    pair = ("P1", "L1")
    probs = {
        "dtkinase": 0.8, "drugban": 0.7, "graphban": 0.6,
        "conplex": 0.65, "chemglam": 0.75,
    }
    for model, probability in probs.items():
        _write_scores(tmp_path, model, [_make_pair(*pair, probability)])

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    row = pd.read_csv(out).iloc[0]
    assert row["tier"] == "STRONG"
    assert int(row["agreement_count"]) == 5
    assert "prob_chemglam" in row.index


@pytest.mark.parametrize("agreement,expected", [
    (3, "STRONG"), (2, "LIKELY"), (1, "UNCERTAIN"), (0, "UNLIKELY"),
])
def test_assign_tier_n3(agreement, expected):
    assert aggregate.assign_tier(agreement, 3) == expected


@pytest.mark.parametrize("agreement,expected", [
    (2, "STRONG"), (1, "LIKELY"), (0, "UNLIKELY"),
])
def test_assign_tier_n2(agreement, expected):
    assert aggregate.assign_tier(agreement, 2) == expected


# ======================================================================
# Sorting
# ======================================================================

def test_output_sorted_by_prob_mean_desc(tmp_path):
    pairs = [
        _make_pair("P1", "L1", 0.20),
        _make_pair("P2", "L2", 0.90),
        _make_pair("P3", "L3", 0.50),
    ]
    for m in ("dtkinase", "drugban"):
        _write_scores(tmp_path, m, pairs)

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out)])
    df = pd.read_csv(out)
    assert list(df["pair_id"]) == ["P2__L2", "P3__L3", "P1__L1"]


def test_top_k_subset_emitted(tmp_path):
    pairs = [_make_pair(f"P{i}", f"L{i}", 0.5 + 0.05*i) for i in range(10)]
    for m in ("dtkinase", "drugban"):
        _write_scores(tmp_path, m, pairs)

    out = tmp_path / "consensus.csv"
    aggregate_cli(["--scores-dir", str(tmp_path), "--out", str(out), "--top-k", "3"])
    top = pd.read_csv(tmp_path / "consensus.top.csv")
    assert len(top) == 3
    # highest prob first
    assert top["prob_mean"].iloc[0] > top["prob_mean"].iloc[-1]


# ======================================================================
# Helpers
# ======================================================================

def aggregate_cli(argv: list[str]) -> None:
    """Invoke aggregate.main() with patched argv."""
    old = sys.argv
    sys.argv = ["aggregate.py"] + argv
    try:
        aggregate.main()
    finally:
        sys.argv = old
