"""Train DrugBAN baseline on thesis kinase datasets with multi-seed evaluation.

This script wraps DrugBAN's training pipeline to:
1. Build molecular graphs (GCN) and protein features (CNN) via DrugBAN's own encoders
2. Train DrugBAN (bilinear attention network + optional domain adaptation) per seed
3. Model selection by validation AUROC (DrugBAN published criterion)
4. Optimize decision threshold on VALIDATION set maximizing MCC (no test leakage)
5. Apply val-optimized threshold to TEST set for final metrics
6. Save aggregated results as JSON with full methodology provenance

Methodology alignment with DT-Kinase:
- Same scaffold splits (Bemis-Murcko 80/10/10)
- Same 5 canonical seeds {42, 123, 456, 789, 1024}
- Threshold calibrated on validation set (MCC-optimal), applied to test
- Model selection: DrugBAN uses val AUROC (its published criterion)
  DT-Kinase uses val MCC — each uses its own published protocol

Usage:
    python run_baseline.py --dataset non_human
    python run_baseline.py --dataset human --seeds 42 123 456
    python run_baseline.py --dataset all --max-epoch 30  # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Suppress torch.tensor(sourceTensor) deprecation warnings from DrugBAN's
# trainer.py (upstream code, not modified here).
warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor.*use sourceTensor.clone",
    category=UserWarning,
)

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

DRUGBAN_SRC = SCRIPT_DIR / "src"
CANONICAL_SEEDS = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# DrugBAN module imports
# ---------------------------------------------------------------------------

def setup_drugban_imports() -> dict:
    """Import DrugBAN modules after path setup.

    NOTE: Our DrugBAN/configs/ directory (YAML files) shadows the upstream
    src/configs/ Python package. We load config.py by absolute path to avoid
    the name collision.
    """
    if DRUGBAN_SRC.exists():
        if str(DRUGBAN_SRC) in sys.path:
            sys.path.remove(str(DRUGBAN_SRC))
        sys.path.insert(0, str(DRUGBAN_SRC))

    try:
        # Load config from absolute path to avoid configs/ name collision
        import importlib.util
        config_path = DRUGBAN_SRC / "configs.py"
        spec = importlib.util.spec_from_file_location("drugban_configs", str(config_path))
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        get_cfg_defaults = cfg_mod.get_cfg_defaults

        from dataloader import DTIDataset, MultiDataLoader
        from models import DrugBAN, binary_cross_entropy, cross_entropy_logits
        from trainer import Trainer
        from utils import graph_collate_func, mkdir, set_seed
        from domain_adaptator import Discriminator

        return {
            "get_cfg_defaults": get_cfg_defaults,
            "DTIDataset": DTIDataset,
            "MultiDataLoader": MultiDataLoader,
            "DrugBAN": DrugBAN,
            "binary_cross_entropy": binary_cross_entropy,
            "cross_entropy_logits": cross_entropy_logits,
            "Trainer": Trainer,
            "graph_collate_func": graph_collate_func,
            "mkdir": mkdir,
            "set_seed": set_seed,
            "Discriminator": Discriminator,
        }
    except ImportError as e:
        print(f"ERROR: Cannot import DrugBAN modules: {e}")
        print(f"Make sure DrugBAN is cloned at: {DRUGBAN_SRC}")
        print("Run: bash setup_env.sh")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Fair evaluation protocol (same logic as GraphBAN wrapper)
# ---------------------------------------------------------------------------

def _collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (y_true, y_prob) from a data loader using the given model.

    DrugBAN batch format from graph_collate_func: (batched_graph, protein_tensor, labels)
    DrugBAN model.forward(bg_d, v_p, mode="eval") returns (v_d, v_p, score, att)
    """
    from models import binary_cross_entropy, cross_entropy_logits

    y_true, y_prob = [], []
    with torch.no_grad():
        model.eval()
        for batch in data_loader:
            v_d, v_p, labels = batch
            v_d = v_d.to(device)
            v_p = v_p.to(device)
            labels = labels.float().to(device)
            # DrugBAN forward in eval mode: returns (v_d, v_p, score, att)
            _, _, score, _ = model(v_d, v_p, mode="eval")

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

    Vectorized sweep of all unique predicted probabilities as candidate
    thresholds; selects one maximizing MCC. Ties broken by threshold
    closest to 0.5.
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
    seed: int,
    output_dir: Path,
    device: torch.device,
    modules: dict,
) -> dict:
    """Train DrugBAN for a single seed and return metrics.

    Protocol:
      1. Create datasets/dataloaders
      2. Train with DrugBAN's Trainer (model selected by val AUROC)
      3. Collect predictions on VALIDATION set with best model
      4. Optimize threshold on validation maximizing MCC (no test leakage)
      5. Collect predictions on TRAIN and TEST sets with best model
      6. Apply val-optimized threshold to all splits → fair metrics
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

    print(f"  Data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Create datasets
    train_dataset = modules["DTIDataset"](train_df.index.values, train_df)
    val_dataset = modules["DTIDataset"](val_df.index.values, val_df)
    test_dataset = modules["DTIDataset"](test_df.index.values, test_df)

    # Create dataloaders
    params_train = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": True,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": True,
        "collate_fn": modules["graph_collate_func"],
    }
    params_eval = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": False,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": False,
        "collate_fn": modules["graph_collate_func"],
    }

    training_generator = torch.utils.data.DataLoader(train_dataset, **params_train)
    val_generator = torch.utils.data.DataLoader(val_dataset, **params_eval)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params_eval)

    # Build model
    model = modules["DrugBAN"](**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Setup domain adaptation (if enabled)
    if cfg.DA.USE:
        source_generator = torch.utils.data.DataLoader(train_dataset, **params_train)
        target_generator = torch.utils.data.DataLoader(
            val_dataset, **{**params_eval, "shuffle": True, "drop_last": True}
        )
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
    n_class = cfg.DECODER.BINARY
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

    # Step 3: Collect predictions on TRAIN set with best model (for overfit diagnosis)
    print("  Collecting train predictions...")
    train_eval_generator = torch.utils.data.DataLoader(train_dataset, **params_eval)
    train_y_true, train_y_prob = _collect_predictions(
        trainer.best_model, train_eval_generator, device, n_class,
    )

    # Step 4: Collect predictions on TEST set with best model
    print("  Collecting test predictions...")
    test_y_true, test_y_prob = _collect_predictions(
        trainer.best_model, test_generator, device, n_class,
    )

    # Step 5: Apply val-optimized threshold to all splits → primary (fair) metrics
    train_metrics = compute_metrics_at_threshold(train_y_true, train_y_prob, val_threshold)
    val_metrics = compute_metrics_at_threshold(val_y_true, val_y_prob, val_threshold)
    test_metrics = compute_metrics_at_threshold(test_y_true, test_y_prob, val_threshold)

    # Also record DrugBAN's native metrics for transparency
    native_threshold = result_metrics.get("thred_optim", 0.5)
    native_test_metrics = compute_metrics_at_threshold(
        test_y_true, test_y_prob, native_threshold,
    )

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "val_threshold": val_threshold,
        "threshold_source": "validation_mcc",
        "training_time_s": round(elapsed, 1),
        "best_epoch": result_metrics.get("best_epoch", -1),
        "model_selection": "val_auroc",
        "drugban_native": {
            "threshold": native_threshold,
            "threshold_source": "test_f1_optimal (DrugBAN original — NOT used for comparison)",
            "mcc": native_test_metrics["mcc"],
            "f1": native_test_metrics["f1"],
            "accuracy": native_test_metrics["accuracy"],
            "auroc": result_metrics.get("auroc", native_test_metrics["auroc"]),
            "auprc": result_metrics.get("auprc", None),
            "sensitivity": result_metrics.get("sensitivity", None),
            "specificity": result_metrics.get("specificity", None),
            "note": (
                "DrugBAN original protocol uses test set for threshold optimization. "
                "These metrics are recorded for transparency but NOT used for "
                "comparison with DT-Kinase."
            ),
        },
    }

    print(f"  Results (seed={seed}, fair protocol, threshold={val_threshold:.4f}):")
    print(f"    {'Split':<6}  {'MCC':>7}  {'AUROC':>7}  {'F1':>7}  {'Acc':>7}")
    print(f"    {'─'*38}")
    for split_name, split_m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"    {split_name:<6}  {split_m['mcc']:>7.4f}  {split_m['auroc']:>7.4f}  "
              f"{split_m['f1']:>7.4f}  {split_m['accuracy']:>7.4f}")
    print(f"    [DrugBAN native: MCC={native_test_metrics['mcc']:.4f} "
          f"threshold={native_threshold:.4f} (test-set F1, not used)]")
    print(f"    Time: {elapsed:.1f}s  Best epoch: {metrics['best_epoch']}")

    # Save raw predictions for reproducibility
    np.savez(
        seed_output / "raw_predictions.npz",
        train_y_true=train_y_true,
        train_y_prob=train_y_prob,
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
    """Compute mean +/- std across seeds for train/val/test splits."""
    metric_names = ["accuracy", "f1", "precision", "recall", "mcc", "auroc"]
    agg: dict = {}

    # Aggregate per split
    for split in ["train", "val", "test"]:
        agg[split] = {}
        for m in metric_names:
            values = [r[split][m] for r in all_metrics]
            agg[split][m] = {
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
    # Aggregate DrugBAN native metrics for comparison
    native_mcc_vals = [
        r["drugban_native"]["mcc"] for r in all_metrics if "drugban_native" in r
    ]
    if native_mcc_vals:
        agg["drugban_native_mcc"] = {
            "mean": float(np.mean(native_mcc_vals)),
            "std": float(np.std(native_mcc_vals)),
            "values": native_mcc_vals,
        }
    return agg


def print_summary_table(agg: dict, dataset: str, seeds: list[int]) -> None:
    """Print a formatted summary table showing train/val/test metrics."""
    print(f"\n{'='*72}")
    print(f"  DrugBAN Baseline — {dataset} ({len(seeds)} seeds)")
    print(f"{'='*72}")
    print(f"  {'Metric':<12} {'Train Mean':>11} {'± Std':>7}  {'Val Mean':>9} {'± Std':>7}  {'Test Mean':>10} {'± Std':>7}")
    print(f"  {'─'*68}")
    for m in ["mcc", "auroc", "f1", "accuracy", "precision", "recall"]:
        tr = agg["train"][m]
        vl = agg["val"][m]
        te = agg["test"][m]
        print(f"  {m.upper():<12} {tr['mean']:>11.4f} {tr['std']:>7.4f}  "
              f"{vl['mean']:>9.4f} {vl['std']:>7.4f}  "
              f"{te['mean']:>10.4f} {te['std']:>7.4f}")
    print(f"  {'─'*68}")
    if "drugban_native_mcc" in agg:
        native = agg["drugban_native_mcc"]
        print(f"  {'MCC (native)':<12} {'N/A':>11} {'':>7}  {'N/A':>9} {'':>7}  "
              f"{native['mean']:>10.4f} {native['std']:>7.4f}  (test-set threshold, not used)")
    print(f"  Avg training time: {agg['training_time_s']['mean']:.1f}s per seed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DrugBAN baseline training")
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
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: DrugBAN/results/<dataset>)",
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
    modules = setup_drugban_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config (DrugBAN defaults)
    cfg = modules["get_cfg_defaults"]()
    config_path = SCRIPT_DIR / "configs" / "kinase.yaml"
    if config_path.exists():
        cfg.merge_from_file(str(config_path))
    else:
        # Fallback: use upstream default config if our kinase config doesn't exist
        upstream_config = DRUGBAN_SRC / "configs" / "DrugBAN.yaml"
        if upstream_config.exists():
            cfg.merge_from_file(str(upstream_config))

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

    # Training loop over seeds
    all_metrics = []
    total_start = time.time()
    for seed in args.seeds:
        metrics = train_single_seed(
            cfg=cfg.clone(),
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            seed=seed,
            output_dir=output_base,
            device=device,
            modules=modules,
        )
        all_metrics.append({"seed": seed, **metrics})

    total_elapsed = time.time() - total_start

    # Aggregate
    agg = aggregate_results(all_metrics)
    print_summary_table(agg, args.dataset, args.seeds)

    # Save results
    results_file = output_base / "drugban_results.json"
    output = {
        "model": "DrugBAN",
        "dataset": args.dataset,
        "split": "scaffold",
        "seeds": args.seeds,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "methodology": {
            "model_selection": "validation AUROC (DrugBAN published criterion)",
            "threshold_optimization": "validation MCC-optimal (no test leakage)",
            "threshold_metric": "mcc",
            "features": {
                "drug": "Molecular graph (GCN, atom features 75-d → 128-d hidden layers)",
                "protein": "Sequence (CNN with kernels [3,6,9], 128 filters each)",
            },
            "note": (
                "Threshold is calibrated on validation set predictions by sweeping "
                "all unique probability values and selecting the one that maximizes "
                "MCC. This mirrors the DT-Kinase protocol. "
                "DrugBAN's native test-set F1-optimal threshold is also recorded "
                "under per_seed[].drugban_native for transparency but is NOT used "
                "for comparison."
            ),
        },
        "config": {
            "max_epoch": cfg.SOLVER.MAX_EPOCH,
            "batch_size": cfg.SOLVER.BATCH_SIZE,
            "lr": cfg.SOLVER.LR,
            "domain_adaptation": cfg.DA.USE,
            "da_method": cfg.DA.METHOD if cfg.DA.USE else None,
            "decoder_binary": cfg.DECODER.BINARY,
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
