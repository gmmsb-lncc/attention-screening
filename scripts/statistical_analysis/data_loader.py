"""Shared I/O for the statistical-analysis toolkit.

Resolves the per-model, per-corpus, per-seed path conventions and
normalizes the differing NPZ key schemas into a single in-memory dict
schema:

    {
        "model": str,
        "corpus": str,
        "seed": int,
        "split": str,              # "test" or "val"
        "y_true": np.ndarray[int], # binary labels
        "y_prob": np.ndarray[float], # in [0, 1]
        "logits": np.ndarray[float], # logit(y_prob), epsilon-clamped
        "threshold": float,        # MCC- or F1-optimal on val
        "platt_a": float | None,   # only for DT-Kinase
        "platt_b": float | None,
    }

Cross-model alignment (np.array_equal on y_true) is asserted on first
multi-model load per (corpus, seed); failures raise immediately.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = REPO_ROOT / "scaffolds_splits" / "output"

# Path templates per (model, corpus). Date globs (`*`) are resolved at load.
_DT_KINASE_PATTERNS = {
    "human":     "results/benchmark_human_8M_*/test/level4_cnn_8M/human/seed_{seed}",
    "non_human": "results/benchmark_non_human_8M_*/test/level4_cnn_8M/non_human/seed_{seed}",
    "all":       "results/all/benchmark_all_8M_*/test/level4_cnn_8M/all/seed_{seed}",
}
_BASELINE_PATTERNS = {
    "drugban":  "DrugBAN/results_universal/results_universal/{corpus}/seed_{seed}",
    "graphban": "GraphBAN/results_universal/{corpus}/seed_{seed}",
    "conplex":  "ConPLex/results_universal/{corpus}/seed_{seed}",
}
_BASELINE_CAL_NAMES = {
    "drugban":  "drugban_calibration.json",
    "graphban": "graphban_calibration.json",
    "conplex":  "conplex_calibration.json",
}
_TEST_TSV_NAMES = {
    "human":     "human_test.tsv",
    "non_human": "non_human_test.tsv",
    "all":       "universal_test.tsv",
}


def _resolve_dtkinase_dir(corpus: str, seed: int) -> Path:
    pattern = _DT_KINASE_PATTERNS[corpus].format(seed=seed)
    matches = sorted(glob.glob(str(REPO_ROOT / pattern)))
    if not matches:
        raise FileNotFoundError(
            f"No DT-Kinase result dir matched: {pattern}")
    if len(matches) > 1:
        # Prefer the most recently dated match (lexicographic sort puts
        # YYYY_MM_DD newest last).
        return Path(matches[-1])
    return Path(matches[0])


def _resolve_baseline_dir(model: str, corpus: str, seed: int) -> Path:
    pattern = _BASELINE_PATTERNS[model].format(corpus=corpus, seed=seed)
    return REPO_ROOT / pattern


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def load_predictions(model: str, corpus: str, seed: int,
                    split: str = "test") -> dict:
    """Load a single (model, corpus, seed, split) prediction artifact."""
    if split not in ("test", "val"):
        raise ValueError(f"split must be 'test' or 'val', got {split!r}")

    if model == "dtkinase":
        result_dir = _resolve_dtkinase_dir(corpus, seed)
        npz_path = result_dir / "raw_predictions.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path)
        if split != "test":
            raise ValueError(
                "DT-Kinase NPZ does not store val predictions; pass split='test'.")
        y_true = data["y_true"].astype(np.int64)
        y_prob = data["y_prob"].astype(np.float64)
        cal_path = result_dir / "level4_cnn_results.json"
    elif model in _BASELINE_PATTERNS:
        result_dir = _resolve_baseline_dir(model, corpus, seed)
        npz_path = result_dir / "raw_predictions.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path)
        prefix = "test_" if split == "test" else "val_"
        if f"{prefix}y_true" not in data.files:
            raise KeyError(
                f"{model} NPZ missing {prefix}y_true (keys: {list(data.files)})")
        y_true = data[f"{prefix}y_true"].astype(np.int64)
        y_prob = data[f"{prefix}y_prob"].astype(np.float64)
        cal_path = result_dir / _BASELINE_CAL_NAMES[model]
    else:
        raise ValueError(f"Unknown model {model!r}")

    threshold = 0.5
    platt_a = None
    platt_b = None
    if cal_path.exists():
        with cal_path.open() as fh:
            cal = json.load(fh)
        if model == "dtkinase":
            result = cal["Split by Scaffold"]["MLP"]
            threshold = float(result["val_threshold"])
        else:
            threshold = float(cal.get("threshold", 0.5))
            platt_a = cal.get("platt_a")
            platt_b = cal.get("platt_b")
        if platt_a is not None:
            platt_a = float(platt_a)
        if platt_b is not None:
            platt_b = float(platt_b)

    return {
        "model": model,
        "corpus": corpus,
        "seed": int(seed),
        "split": split,
        "y_true": y_true,
        "y_prob": y_prob,
        "logits": _logit(y_prob),
        "threshold": threshold,
        "platt_a": platt_a,
        "platt_b": platt_b,
    }


def load_all_seeds(model: str, corpus: str, seeds=None,
                  split: str = "test") -> list[dict]:
    """Load predictions for one (model, corpus) across all seeds."""
    from . import SEEDS  # local import to avoid circular at module load

    if seeds is None:
        seeds = SEEDS
    return [load_predictions(model, corpus, s, split) for s in seeds]


def load_test_pchembl(corpus: str) -> np.ndarray:
    """Load per-sample pChEMBL values from the test TSV.

    Returns an array aligned with y_true in the corresponding NPZ files.
    Column 7 (1-indexed) is `pchembl_value` in all three TSVs.
    """
    tsv_path = SPLITS_DIR / _TEST_TSV_NAMES[corpus]
    if not tsv_path.exists():
        raise FileNotFoundError(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t")
    if "pchembl_value" not in df.columns:
        raise KeyError(
            f"pchembl_value column missing from {tsv_path}; cols: {list(df.columns)}")
    return df["pchembl_value"].to_numpy(dtype=np.float64)


def assert_aligned(seeds_data: list[list[dict]]) -> None:
    """Verify that y_true matches across models for the same (corpus, seed).

    Argument: a list (one per model) of lists (one per seed) of pred dicts.
    Asserts np.array_equal on every (seed_index, model_a, model_b) pair.
    """
    if not seeds_data:
        return
    n_seeds = len(seeds_data[0])
    for s_idx in range(n_seeds):
        ref = seeds_data[0][s_idx]["y_true"]
        for m_idx in range(1, len(seeds_data)):
            other = seeds_data[m_idx][s_idx]["y_true"]
            if not np.array_equal(ref, other):
                raise AssertionError(
                    f"Test-set misalignment at seed_index={s_idx}: "
                    f"{seeds_data[0][s_idx]['model']} vs "
                    f"{seeds_data[m_idx][s_idx]['model']} "
                    f"(corpus={seeds_data[0][s_idx]['corpus']})")


def load_panel(corpus: str, models=None, seeds=None,
               split: str = "test") -> dict[str, list[dict]]:
    """Load the full {model: [per-seed dicts]} panel for a corpus."""
    from . import MODELS

    if models is None:
        models = MODELS
    panel = {m: load_all_seeds(m, corpus, seeds, split) for m in models}
    assert_aligned(list(panel.values()))
    return panel
