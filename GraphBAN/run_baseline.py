"""Train GraphBAN baseline on thesis kinase datasets with multi-seed evaluation.

This script wraps GraphBAN's training pipeline to:
1. Extract ESM-1b and ChemBERTa features (once for all seeds)
2. Generate teacher GAE embeddings per seed (bipartite graph KD)
3. Train GraphBAN (inductive mode with domain adaptation) per seed
4. Model selection by validation AUROC (GraphBAN published criterion)
5. Optimize decision threshold on VALIDATION set maximizing MCC (no test leakage)
6. Apply val-optimized threshold to TEST set for final metrics
7. Save aggregated results as JSON with full methodology provenance

Methodology alignment with DT-Kinase:
- Same scaffold splits (Bemis-Murcko 80/10/10)
- Same 5 canonical seeds {42, 123, 456, 789, 1024}
- Threshold calibrated on validation set (MCC-optimal), applied to test
- Model selection: GraphBAN uses val AUROC (its published criterion)
  DT-Kinase uses val MCC — each uses its own published protocol

IMPORTANT: GraphBAN's original inductive mode uses test set for validation
(val_generator = test_dataset), causing test leakage. This wrapper corrects
that by using the actual validation split for model selection.

Usage:
    python run_baseline.py --dataset non_human
    python run_baseline.py --dataset human --seeds 42 123 456
    python run_baseline.py --dataset all --max-epoch 30  # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Apply DGL graphbolt compatibility shim before any DGL import
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import dgl_compat  # noqa: F401, E402 — must be before dgl

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

GRAPHBAN_SRC = SCRIPT_DIR / "src"
GRAPHBAN_INDUCTIVE = GRAPHBAN_SRC / "inductive_mode"
CANONICAL_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_TEACHER_EPOCHS = 10


# ---------------------------------------------------------------------------
# GraphBAN module imports
# ---------------------------------------------------------------------------

def setup_graphban_imports():
    """Import GraphBAN modules after path setup."""
    if GRAPHBAN_INDUCTIVE.exists():
        sys.path.insert(0, str(GRAPHBAN_INDUCTIVE))
        # Also add src root (teacher_gae expects it)
        sys.path.insert(0, str(GRAPHBAN_SRC))
    try:
        from configs import get_cfg_defaults
        from dataloader import DTIDataset, DTIDataset2, MultiDataLoader
        from models import GraphBAN, binary_cross_entropy, cross_entropy_logits
        from trainer import Trainer
        from utils import graph_collate_func, graph_collate_func2, mkdir, set_seed
        from domain_adaptator import Discriminator

        return {
            "get_cfg_defaults": get_cfg_defaults,
            "DTIDataset": DTIDataset,
            "DTIDataset2": DTIDataset2,
            "MultiDataLoader": MultiDataLoader,
            "GraphBAN": GraphBAN,
            "binary_cross_entropy": binary_cross_entropy,
            "cross_entropy_logits": cross_entropy_logits,
            "Trainer": Trainer,
            "graph_collate_func": graph_collate_func,
            "graph_collate_func2": graph_collate_func2,
            "mkdir": mkdir,
            "set_seed": set_seed,
            "Discriminator": Discriminator,
        }
    except ImportError as e:
        print(f"ERROR: Cannot import GraphBAN modules: {e}")
        print(f"Make sure GraphBAN is cloned at: {GRAPHBAN_SRC}")
        print("Run: bash setup_env.sh")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Feature extraction (replicates run_model.py logic, runs once for all seeds)
# ---------------------------------------------------------------------------

def extract_esm_features(df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Extract ESM-1b protein embeddings (1280-d mean pool).

    Replicates GraphBAN's run_model.py Get_Protein_Feature function.
    Processes unique proteins only and merges back.
    """
    import esm

    print("\n  Loading ESM-1b model (esm1b_t33_650M_UR50S)...")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)

    # Truncate proteins to 1022 (ESM-1b limit)
    df = df.copy()
    df["Protein"] = df["Protein"].apply(lambda x: x[:1022] if len(x) > 1022 else x)

    pro_list = df["Protein"].unique()
    print(f"  Extracting ESM features for {len(pro_list)} unique proteins...")

    dictionary = {}
    data_tmp = [(f"protein{i}", p) for i, p in enumerate(pro_list)]

    batch_size = 5
    for i in tqdm(range(0, len(data_tmp), batch_size), desc="  ESM"):
        batch = data_tmp[i : i + batch_size]
        if not batch:
            continue
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_reps = results["representations"][33]
        for j, (_, seq) in enumerate(batch):
            emb = token_reps[j, 1 : len(seq) + 1].mean(0).cpu().numpy()
            dictionary[seq] = emb

    esm_df = pd.DataFrame(list(dictionary.items()), columns=["Protein", "esm"])
    df = pd.merge(df, esm_df, on="Protein", how="left")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("  ESM feature extraction complete.")
    return df


def extract_chemberta_features(df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Extract ChemBERTa-77M-MTR drug embeddings (384-d CLS token).

    Replicates GraphBAN's run_model.py get_embeddings function.
    Processes unique SMILES only and merges back.
    """
    from transformers import AutoTokenizer, RobertaModel

    print("\n  Loading ChemBERTa-77M-MTR model...")
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    model = RobertaModel.from_pretrained(model_name, add_pooling_layer=False, use_safetensors=True)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df_unique = df.drop_duplicates(subset="SMILES").copy()
    print(f"  Extracting ChemBERTa features for {len(df_unique)} unique SMILES...")

    emblist = []
    for _, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="  ChemBERTa"):
        encodings = tokenizer(
            row["SMILES"],
            return_tensors="pt",
            padding="max_length",
            max_length=290,
            truncation=True,
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            output = model(**encodings)
            emb = output.last_hidden_state[0, 0, :].cpu().numpy().astype(np.float64)
            emblist.append(emb)

    df_unique["fcfp"] = emblist
    df = pd.merge(df, df_unique[["SMILES", "fcfp"]], on="SMILES", how="left")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("  ChemBERTa feature extraction complete.")
    return df


def extract_features_cached(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract ESM + ChemBERTa features with caching (once for all seeds)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "features_extracted.pkl"

    if cache_file.exists():
        print("\n  Loading cached features...")
        cached = pd.read_pickle(cache_file)
        return cached["train"], cached["val"], cached["test"]

    print("\n  Extracting features (this will be cached for future runs)...")

    # Combine all splits for joint feature extraction (ensures same embeddings)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["_split"] = "train"
    val_df["_split"] = "val"
    test_df["_split"] = "test"
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Extract features on combined data
    combined = extract_esm_features(combined, device)
    combined = extract_chemberta_features(combined, device)

    # Split back
    train_out = combined[combined["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    val_out = combined[combined["_split"] == "val"].drop(columns=["_split"]).reset_index(drop=True)
    test_out = combined[combined["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)

    # Cache for reuse
    pd.to_pickle({"train": train_out, "val": val_out, "test": test_out}, cache_file)
    print(f"  Features cached at {cache_file}")

    return train_out, val_out, test_out


# ---------------------------------------------------------------------------
# Teacher GAE embeddings
# ---------------------------------------------------------------------------

def generate_teacher_embeddings(
    train_csv: Path, seed: int, output_parquet: Path, epochs: int = DEFAULT_TEACHER_EPOCHS
) -> None:
    """Run teacher_gae.py as subprocess to generate teacher embeddings.

    The teacher GAE operates on the bipartite CPI graph of the training set,
    producing per-interaction embeddings (256-d) saved as a parquet file.
    """
    teacher_script = GRAPHBAN_INDUCTIVE / "teacher_gae.py"
    if not teacher_script.exists():
        raise FileNotFoundError(f"teacher_gae.py not found: {teacher_script}")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(teacher_script),
        "--train_path", str(train_csv),
        "--seed", str(seed),
        "--teacher_path", str(output_parquet),
        "--epoch", str(epochs),
    ]

    print(f"    Running teacher_gae.py (seed={seed}, epochs={epochs})...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(GRAPHBAN_SRC),
        timeout=3600,
    )

    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"teacher_gae.py failed (exit code {result.returncode})")

    if not output_parquet.exists():
        raise RuntimeError(f"Teacher embeddings not produced: {output_parquet}")

    print(f"    Teacher embeddings saved: {output_parquet}")


# ---------------------------------------------------------------------------
# Fair evaluation protocol (same as DrugBAN wrapper)
# ---------------------------------------------------------------------------

def _collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (y_true, y_prob) from a data loader using the given model.

    Replicates GraphBAN's Trainer.test() prediction collection logic.
    """
    from models import binary_cross_entropy, cross_entropy_logits

    y_true, y_prob = [], []
    with torch.no_grad():
        model.eval()
        for batch in data_loader:
            v_d, sm, v_p, esm_feat, labels = batch
            sm = torch.tensor(sm, dtype=torch.float32)
            sm = torch.reshape(sm, (sm.shape[0], 1, 384))
            esm_feat = torch.tensor(esm_feat, dtype=torch.float32)
            esm_feat = torch.reshape(esm_feat, (sm.shape[0], 1, 1280))
            v_d = v_d.to(device)
            sm = sm.to(device)
            v_p = v_p.to(device)
            esm_feat = esm_feat.to(device)
            labels = labels.float().to(device)
            _, _, _, score = model(v_d, sm, v_p, esm_feat, device)

            if n_class == 1:
                n, _ = binary_cross_entropy(score, labels)
            else:
                n, _ = cross_entropy_logits(score, labels)

            y_prob.extend(n.to("cpu").tolist())
            y_true.extend(labels.to("cpu").tolist())

    return np.array(y_true), np.array(y_prob)


def optimize_threshold_on_validation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "mcc",
) -> tuple[float, float]:
    """Optimize decision threshold on validation predictions (no test leakage).

    Mirrors DT-Kinase methodology (crossattention_split_analysis/training/evaluator.py):
    sweep all unique predicted probabilities as candidate thresholds, select the
    one maximizing MCC. Ties broken by threshold closest to 0.5.

    Uses vectorized cumulative sums for O(N log N) performance.
    """
    if len(y_true) == 0:
        return 0.5, 0.0

    order = np.argsort(y_prob, kind="mergesort")[::-1]
    probs_sorted = y_prob[order]
    labels_sorted = y_true[order]

    total_pos = float((labels_sorted == 1).sum())
    total_neg = float((labels_sorted == 0).sum())

    tp_cum = np.cumsum(labels_sorted == 1, dtype=np.float64)
    fp_cum = np.cumsum(labels_sorted == 0, dtype=np.float64)

    last_indices = np.r_[np.where(np.diff(probs_sorted) != 0)[0], len(probs_sorted) - 1]
    tp = tp_cum[last_indices]
    fp = fp_cum[last_indices]
    fn = total_pos - tp
    tn = total_neg - fp
    thresholds = probs_sorted[last_indices]

    sentinel = np.nextafter(float(np.max(probs_sorted)), np.inf)
    tp = np.concatenate(([0.0], tp))
    fp = np.concatenate(([0.0], fp))
    fn = np.concatenate(([total_pos], fn))
    tn = np.concatenate(([total_neg], tn))
    thresholds = np.concatenate(([sentinel], thresholds))

    if metric == "mcc":
        denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        valid = denom_sq > 0
        scores = np.zeros_like(denom_sq)
        scores[valid] = (tp[valid] * tn[valid] - fp[valid] * fn[valid]) / np.sqrt(denom_sq[valid])
    elif metric == "f1":
        denom = (2 * tp) + fp + fn
        scores = np.where(denom > 0, (2 * tp) / denom, 0.0)
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    best_score = float(np.nanmax(scores))
    tie_idx = np.where(np.isclose(scores, best_score, rtol=1e-9, atol=1e-12))[0]
    best_idx = int(tie_idx[np.argmin(np.abs(thresholds[tie_idx] - 0.5))])

    return float(thresholds[best_idx]), best_score


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Compute all metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------

def train_single_seed(
    cfg,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    teacher_parquet: Path,
    seed: int,
    output_dir: Path,
    device: torch.device,
    modules: dict,
) -> dict:
    """Train GraphBAN for a single seed and return metrics.

    Protocol:
      1. Load teacher embeddings and attach to training data
      2. Create datasets/dataloaders with PROPER val split (fixes original leak)
      3. Train with GraphBAN's Trainer (model selected by val AUROC)
      4. Collect predictions on VALIDATION set with best model
      5. Optimize threshold on validation maximizing MCC (no test leakage)
      6. Collect predictions on TEST set with best model
      7. Apply val-optimized threshold to test predictions → final metrics
    """
    print(f"\n{'─'*50}")
    print(f"  Seed: {seed}")
    print(f"{'─'*50}")

    modules["set_seed"](seed)

    seed_output = output_dir / f"seed_{seed}"
    modules["mkdir"](str(seed_output))

    # Update config for this seed
    cfg.defrost()
    cfg.SOLVER.SEED = seed
    cfg.RESULT.OUTPUT_DIR = str(seed_output)
    cfg.freeze()

    # Load teacher embeddings and attach to training data
    train_emb = pd.read_parquet(teacher_parquet)
    train_emb["Array"] = train_emb.apply(lambda row: np.array(row), axis=1)
    train_emb.drop(train_emb.columns.difference(["Array"]), axis=1, inplace=True)

    train_df_seed = train_df.copy()
    train_df_seed["teacher_emb"] = train_emb["Array"].values

    print(f"  Data: train={len(train_df_seed)}, val={len(val_df)}, test={len(test_df)}")

    # Create datasets
    train_dataset = modules["DTIDataset2"](train_df_seed.index.values, train_df_seed)
    val_dataset = modules["DTIDataset"](val_df.index.values, val_df)
    test_dataset = modules["DTIDataset"](test_df.index.values, test_df)

    # Create dataloaders
    params_train = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": True,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": True,
        "collate_fn": modules["graph_collate_func2"],
    }
    params_eval = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": False,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": False,
        "collate_fn": modules["graph_collate_func"],
    }

    training_generator = torch.utils.data.DataLoader(train_dataset, **params_train)

    # FAIR SPLIT: Use actual validation set (not test set like original inductive mode)
    val_generator = torch.utils.data.DataLoader(val_dataset, **params_eval)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params_eval)

    # Build model
    model = modules["GraphBAN"](**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Setup domain adaptation
    if cfg.DA.USE:
        source_generator = torch.utils.data.DataLoader(train_dataset, **params_train)
        target_generator = torch.utils.data.DataLoader(val_dataset, **{**params_eval, "shuffle": True, "drop_last": True})
        n_batches = max(len(source_generator), len(target_generator))
        multi_generator = modules["MultiDataLoader"](
            dataloaders=[source_generator, target_generator],
            n_batches=n_batches,
        )
        domain_dmm = modules["Discriminator"](
            input_size=cfg.DA.RANDOM_DIM,
            n_class=cfg.DECODER.BINARY,
        ).to(device)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        multi_generator = None
        domain_dmm = None
        opt_da = None

    # Create trainer
    train_loader = multi_generator if cfg.DA.USE else training_generator
    trainer = modules["Trainer"](
        model, opt, device,
        train_loader,
        val_generator,
        test_generator,
        opt_da=opt_da,
        discriminator=domain_dmm,
        experiment=None,
        **cfg,
    )

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    # Handle return value (Trainer.train returns either dict or tuple)
    if isinstance(result, tuple):
        result_metrics = result[0] if isinstance(result[0], dict) else {}
    elif isinstance(result, dict):
        result_metrics = result
    else:
        result_metrics = {}

    # --- Fair evaluation protocol (mirrors DT-Kinase methodology) ---
    n_class = cfg.DECODER.BINARY

    # Step 1: Collect predictions on VALIDATION set with best model
    print("  Collecting validation predictions for fair threshold optimization...")
    val_y_true, val_y_prob = _collect_predictions(
        trainer.best_model, val_generator, device, n_class,
    )

    # Step 2: Optimize threshold on validation maximizing MCC (no test leakage)
    val_threshold, val_best_mcc = optimize_threshold_on_validation(
        val_y_true, val_y_prob, metric="mcc",
    )
    print(f"  Val-optimized threshold={val_threshold:.4f} (val MCC={val_best_mcc:.4f})")

    # Step 3: Collect predictions on TEST set with best model
    print("  Collecting test predictions...")
    test_y_true, test_y_prob = _collect_predictions(
        trainer.best_model, test_generator, device, n_class,
    )

    # Step 4: Apply val-optimized threshold to test → primary (fair) metrics
    metrics = compute_metrics_at_threshold(test_y_true, test_y_prob, val_threshold)
    metrics["threshold_source"] = "validation_mcc"
    metrics["val_threshold"] = val_threshold
    metrics["val_best_mcc"] = val_best_mcc
    metrics["training_time_s"] = round(elapsed, 1)
    metrics["best_epoch"] = result_metrics.get("best_epoch", -1)
    metrics["model_selection"] = "val_auroc"

    # Also record GraphBAN's native metrics for transparency
    native_threshold = result_metrics.get("thred_optim", 0.5)
    native_test_metrics = compute_metrics_at_threshold(
        test_y_true, test_y_prob, native_threshold,
    )
    metrics["graphban_native"] = {
        "threshold": native_threshold,
        "threshold_source": "test_f1_optimal (GraphBAN original — NOT used for comparison)",
        "mcc": native_test_metrics["mcc"],
        "f1": native_test_metrics["f1"],
        "accuracy": native_test_metrics["accuracy"],
        "auroc": result_metrics.get("auroc", native_test_metrics["auroc"]),
        "auprc": result_metrics.get("auprc", None),
        "sensitivity": result_metrics.get("sensitivity", None),
        "specificity": result_metrics.get("specificity", None),
        "note": (
            "GraphBAN original protocol uses test set for both validation and "
            "threshold optimization. These metrics are recorded for transparency "
            "but NOT used for comparison with DT-Kinase."
        ),
    }

    print(f"  Results (seed={seed}, fair protocol):")
    print(f"    MCC={metrics['mcc']:.4f}  AUROC={metrics['auroc']:.4f}  "
          f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")
    print(f"    Threshold={val_threshold:.4f} (from validation MCC)")
    print(f"    [GraphBAN native: MCC={native_test_metrics['mcc']:.4f} "
          f"threshold={native_threshold:.4f} (test-set F1, not used)]")
    print(f"    Time: {elapsed:.1f}s  Best epoch: {metrics['best_epoch']}")

    # Save raw predictions for reproducibility
    np.savez(
        seed_output / "raw_predictions.npz",
        val_y_true=val_y_true,
        val_y_prob=val_y_prob,
        test_y_true=test_y_true,
        test_y_prob=test_y_prob,
    )

    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(all_metrics: list[dict]) -> dict:
    """Compute mean +/- std across seeds."""
    metric_names = ["accuracy", "f1", "precision", "recall", "mcc", "auroc"]
    agg = {}
    for m in metric_names:
        values = [r[m] for r in all_metrics]
        agg[m] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
    agg["val_threshold"] = {
        "mean": float(np.mean([r["val_threshold"] for r in all_metrics])),
        "std": float(np.std([r["val_threshold"] for r in all_metrics])),
        "values": [r["val_threshold"] for r in all_metrics],
    }
    agg["training_time_s"] = {
        "mean": float(np.mean([r["training_time_s"] for r in all_metrics])),
    }
    # Aggregate GraphBAN native metrics for comparison
    native_mcc_vals = [
        r["graphban_native"]["mcc"] for r in all_metrics if "graphban_native" in r
    ]
    if native_mcc_vals:
        agg["graphban_native_mcc"] = {
            "mean": float(np.mean(native_mcc_vals)),
            "std": float(np.std(native_mcc_vals)),
            "values": native_mcc_vals,
        }
    return agg


def print_summary_table(agg: dict, dataset: str, seeds: list[int]) -> None:
    """Print a formatted summary table."""
    print(f"\n{'='*60}")
    print(f"  GraphBAN Baseline — {dataset} ({len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"  {'Metric':<12} {'Mean':>8} {'± Std':>8}")
    print(f"  {'─'*28}")
    for m in ["mcc", "auroc", "f1", "accuracy", "precision", "recall"]:
        vals = agg[m]
        print(f"  {m.upper():<12} {vals['mean']:>8.4f} {vals['std']:>8.4f}")
    print(f"  {'─'*28}")
    if "graphban_native_mcc" in agg:
        native = agg["graphban_native_mcc"]
        print(f"  {'MCC (native)':<12} {native['mean']:>8.4f} {native['std']:>8.4f}  "
              f"(test-set threshold, not used)")
    print(f"  Avg training time: {agg['training_time_s']['mean']:.1f}s per seed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GraphBAN baseline training")
    parser.add_argument(
        "--dataset",
        choices=["non_human", "human", "all"],
        default="non_human",
        help="Dataset to train on",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=CANONICAL_SEEDS,
        help="Random seeds (default: 42 123 456 789 1024)",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Override max epochs (for quick testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--teacher-epochs",
        type=int,
        default=DEFAULT_TEACHER_EPOCHS,
        help="Number of teacher GAE epochs (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: GraphBAN/results/<dataset>)",
    )
    parser.add_argument(
        "--no-da",
        action="store_true",
        help="Disable domain adaptation (uses simpler training)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0, increase on GPU servers)",
    )
    args = parser.parse_args()

    # Validate dataset exists — auto-prepare if missing
    dataset_path = SCRIPT_DIR / "datasets" / "kinase" / args.dataset / "scaffold"
    if not dataset_path.exists() or not (dataset_path / "train.csv").exists():
        print(f"Dataset not found at {dataset_path}")
        print(f"Auto-preparing data for '{args.dataset}'...")
        from prepare_data import prepare_dataset
        prepare_dataset(args.dataset, SCRIPT_DIR)
        if not (dataset_path / "train.csv").exists():
            print(f"ERROR: Data preparation failed. Check scaffold splits at:")
            print(f"  {SCRIPT_DIR.parent / 'scaffolds_splits' / 'output'}")
            sys.exit(1)
        print("Data prepared successfully.\n")

    # Setup
    modules = setup_graphban_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config (GraphBAN defaults + inductive DA settings)
    cfg = modules["get_cfg_defaults"]()
    config_path = GRAPHBAN_INDUCTIVE / "GraphBAN_DA.yaml"
    if config_path.exists():
        cfg.merge_from_file(str(config_path))

    if args.max_epoch is not None:
        cfg.defrost()
        cfg.SOLVER.MAX_EPOCH = args.max_epoch
        cfg.freeze()
    if args.batch_size is not None:
        cfg.defrost()
        cfg.SOLVER.BATCH_SIZE = args.batch_size
        cfg.freeze()
    if args.num_workers > 0:
        cfg.defrost()
        cfg.SOLVER.NUM_WORKERS = args.num_workers
        cfg.freeze()
    if args.no_da:
        cfg.defrost()
        cfg.DA.USE = False
        cfg.DA.TASK = False
        cfg.freeze()

    # Determine output directory
    output_base = args.output_dir if args.output_dir else (SCRIPT_DIR / "results" / args.dataset)
    output_base = Path(output_base).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset} (scaffold split)")
    print(f"Seeds: {args.seeds}")
    print(f"Max epochs: {cfg.SOLVER.MAX_EPOCH}")
    print(f"Batch size: {cfg.SOLVER.BATCH_SIZE}")
    print(f"Workers: {cfg.SOLVER.NUM_WORKERS}")
    print(f"Domain adaptation: {'enabled' if cfg.DA.USE else 'disabled'}")
    print(f"Teacher GAE epochs: {args.teacher_epochs}")
    print(f"Output: {output_base}")
    print(f"Device: {device}")

    # Load data
    train_csv = dataset_path / "train.csv"
    val_csv = dataset_path / "val.csv"
    test_csv = dataset_path / "test.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"\nData loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Extract features (once for all seeds, with caching)
    cache_dir = output_base / "feature_cache"
    train_df, val_df, test_df = extract_features_cached(
        train_df, val_df, test_df, device, cache_dir,
    )

    # Training loop over seeds
    all_metrics = []
    for seed in args.seeds:
        # Step 1: Generate teacher embeddings for this seed
        teacher_dir = output_base / f"seed_{seed}" / "teacher"
        teacher_dir.mkdir(parents=True, exist_ok=True)
        teacher_parquet = teacher_dir / "teacher_embeddings.parquet"

        if not teacher_parquet.exists():
            generate_teacher_embeddings(
                train_csv, seed, teacher_parquet, epochs=args.teacher_epochs,
            )
        else:
            print(f"\n    Using cached teacher embeddings: {teacher_parquet}")

        # Step 2: Train GraphBAN with fair evaluation
        metrics = train_single_seed(
            cfg=cfg.clone(),
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            teacher_parquet=teacher_parquet,
            seed=seed,
            output_dir=output_base,
            device=device,
            modules=modules,
        )
        all_metrics.append({"seed": seed, **metrics})

    # Aggregate
    agg = aggregate_results(all_metrics)
    print_summary_table(agg, args.dataset, args.seeds)

    # Save results
    results_file = output_base / "graphban_results.json"
    output = {
        "model": "GraphBAN",
        "dataset": args.dataset,
        "split": "scaffold",
        "seeds": args.seeds,
        "methodology": {
            "model_selection": "validation AUROC (GraphBAN published criterion)",
            "threshold_optimization": "validation MCC-optimal (no test leakage)",
            "threshold_metric": "mcc",
            "fair_split_fix": (
                "GraphBAN's original inductive mode uses test set as validation "
                "(val_generator = test_dataset). This wrapper corrects that by "
                "using the actual validation split for model selection, matching "
                "the DT-Kinase fair evaluation protocol."
            ),
            "features": {
                "protein": "ESM-1b (esm1b_t33_650M_UR50S, 1280-d mean pool)",
                "drug": "ChemBERTa-77M-MTR (384-d CLS token)",
                "teacher": f"GAE on bipartite CPI graph (256-d, {args.teacher_epochs} epochs)",
            },
            "note": (
                "Threshold is calibrated on validation set predictions by sweeping "
                "all unique probability values and selecting the one that maximizes "
                "MCC. This mirrors the DT-Kinase protocol in "
                "crossattention_split_analysis/training/evaluator.py. "
                "GraphBAN's native test-set F1-optimal threshold is also recorded "
                "under per_seed[].graphban_native for transparency but is NOT used "
                "for comparison."
            ),
        },
        "config": {
            "max_epoch": cfg.SOLVER.MAX_EPOCH,
            "batch_size": cfg.SOLVER.BATCH_SIZE,
            "lr": cfg.SOLVER.LR,
            "da_lr": cfg.SOLVER.DA_LR,
            "domain_adaptation": cfg.DA.USE,
            "da_method": cfg.DA.METHOD if cfg.DA.USE else None,
            "decoder_binary": cfg.DECODER.BINARY,
            "teacher_gae_epochs": args.teacher_epochs,
        },
        "aggregate": agg,
        "per_seed": all_metrics,
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_file}")
    print("Done!")


if __name__ == "__main__":
    main()
