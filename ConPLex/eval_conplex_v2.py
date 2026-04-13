#!/usr/bin/env python3
"""
eval_conplex_v2.py — Evaluate ConPLex with MCC-optimal threshold from validation set.

Protocol (identical to DT-Kinase):
  1. Run inference on the VALIDATION set → collect (score, label) pairs
  2. Sweep thresholds on validation → select τ* = argmax_τ MCC(val, τ)
  3. Run inference on the TEST set → apply τ* → compute final metrics
  4. No test data is used for threshold selection (zero leakage)

Outputs per experiment:
  - val_predictions.csv     : raw validation scores
  - test_predictions.csv    : raw test scores  
  - results_v2.json         : all metrics with both τ=0.5 and τ* thresholds
  - threshold_sweep.csv     : full sweep data for analysis

Usage:
    python eval_conplex_v2.py \
        --checkpoint best_models/trained_non_human_rep0/trained_non_human_rep0_best_model.pt \
        --data-dir dataset/kinase_non_human \
        --exp-id conplex_v2_non_human_rep0 \
        --device 0

Batch mode (all reps):
    for dataset in non_human human; do
        for rep in 0 1 2; do
            python eval_conplex_v2.py \
                --checkpoint best_models/trained_${dataset}_rep${rep}/trained_${dataset}_rep${rep}_best_model.pt \
                --data-dir dataset/kinase_${dataset} \
                --exp-id conplex_v2_${dataset}_rep${rep} \
                --device 0
        done
    done
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from time import time
from tqdm import tqdm

# Add ConPLex src to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(SCRIPT_DIR)

from src import architectures as model_types
from src.data import DTIDataModule, drug_target_collate_fn
from src.utils import set_random_seed, get_featurizer, get_logger

logg = get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Threshold calibration utilities
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Matthews Correlation Coefficient from binary arrays."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return float(tp * tn - fp * fn) / denom


def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray,
                           n_thresholds: int = 1000) -> tuple:
    """
    Sweep thresholds to find MCC-optimal operating point.
    
    Returns:
        (best_threshold, best_mcc, sweep_df)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    results = []
    
    for tau in thresholds:
        preds = (scores >= tau).astype(int)
        mcc = compute_mcc(labels, preds)
        
        tp = np.sum((labels == 1) & (preds == 1))
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(labels)
        
        results.append({
            'threshold': tau, 'mcc': mcc, 'f1': f1,
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        })
    
    sweep_df = pd.DataFrame(results)
    best_idx = sweep_df['mcc'].idxmax()
    best_threshold = sweep_df.loc[best_idx, 'threshold']
    best_mcc = sweep_df.loc[best_idx, 'mcc']
    
    return best_threshold, best_mcc, sweep_df


def compute_metrics_at_threshold(scores: np.ndarray, labels: np.ndarray,
                                  threshold: float) -> dict:
    """Compute all classification metrics at a given threshold (pure numpy)."""
    preds = (scores >= threshold).astype(int)
    
    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    
    n = len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n
    mcc = compute_mcc(labels, preds)
    
    # AUROC via trapezoidal rule (threshold-invariant)
    from sklearn.metrics import roc_auc_score
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5  # degenerate case
    
    return {
        'MCC': float(mcc),
        'AUROC': float(auroc),
        'F1': float(f1),
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'threshold': float(threshold),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(model, dataloader, device, use_cuda: bool, desc: str = "Inference"):
    """Run model inference, return (scores, labels) as numpy arrays."""
    all_scores = []
    all_labels = []
    
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
        for batch in tqdm(dataloader, desc=desc):
            drug, target, label = batch
            drug = drug.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).int()
            
            pred = model(drug, target)
            pred_sigmoid = torch.sigmoid(pred)
            
            all_scores.append(pred_sigmoid.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    
    if use_cuda:
        torch.cuda.synchronize()
    
    return np.concatenate(all_scores), np.concatenate(all_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = ArgumentParser(description="ConPLex v2: MCC-optimal threshold from validation")
    parser.add_argument("--checkpoint", required=True, help="Path to state_dict .pt")
    parser.add_argument("--data-dir", required=True, help="Directory with train/val/test.csv")
    parser.add_argument("--exp-id", required=True, help="Experiment name for output")
    parser.add_argument("--drug-featurizer", default="MorganFeaturizer")
    parser.add_argument("--target-featurizer", default="ProtBertFeaturizer")
    parser.add_argument("--model-architecture", default="SimpleCoembedding")
    parser.add_argument("--latent-dimension", type=int, default=1024)
    parser.add_argument("--latent-distance", default="Cosine")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=int, default=0, help="CUDA device (-1 for CPU)")
    parser.add_argument("--output-dir", default="./results_v2", help="Where to save results")
    parser.add_argument("--n-thresholds", type=int, default=2000,
                        help="Number of threshold candidates for sweep")
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(42)

    # ── Auto-tune num_workers ──────────────────────────────────────────────
    if args.num_workers is None:
        n_cpus = os.cpu_count() or 4
        args.num_workers = min(n_cpus // 2, 8)
        args.num_workers = max(args.num_workers, 2)

    # ── Device & CUDA optimizations ────────────────────────────────────────
    use_cuda = torch.cuda.is_available() and args.device >= 0
    device = torch.device(f"cuda:{args.device}" if use_cuda else "cpu")
    logg.info(f"Device: {device}")

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(args.device)
        logg.info(f"GPU: {gpu_name}")

    # ── Data ───────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir).resolve()
    logg.info(f"Data directory: {data_dir}")

    drug_featurizer = get_featurizer(args.drug_featurizer, save_dir=data_dir)
    target_featurizer = get_featurizer(args.target_featurizer, save_dir=data_dir)

    datamodule = DTIDataModule(
        data_dir,
        drug_featurizer,
        target_featurizer,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    datamodule.prepare_data()
    datamodule.setup(stage=None)  # Load BOTH val and test (stage='test' skips val)

    # Build dataloaders for BOTH val and test
    def make_loader(dataset, desc):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=drug_target_collate_fn,
            pin_memory=use_cuda,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

    val_loader = make_loader(datamodule.data_val, "val")
    test_loader = make_loader(datamodule.data_test, "test")
    
    logg.info(f"Val set:  {len(datamodule.data_val)} samples, {len(val_loader)} batches")
    logg.info(f"Test set: {len(datamodule.data_test)} samples, {len(test_loader)} batches")

    # ── Model ──────────────────────────────────────────────────────────────
    drug_shape = drug_featurizer.shape
    target_shape = target_featurizer.shape

    arch_name = args.model_architecture

    model = getattr(model_types, arch_name)(
        drug_shape=drug_shape,
        target_shape=target_shape,
        latent_dimension=args.latent_dimension,
        latent_distance=args.latent_distance,
        classify=True,
    )

    ckpt_path = Path(args.checkpoint).resolve()
    logg.info(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Inference on VALIDATION set → threshold calibration
    # ══════════════════════════════════════════════════════════════════════
    logg.info("=" * 60)
    logg.info("PHASE 1: Validation inference + threshold calibration")
    logg.info("=" * 60)
    
    t0 = time()
    val_scores, val_labels = run_inference(model, val_loader, device, use_cuda, "Val")
    t_val = time() - t0
    logg.info(f"Val inference: {t_val:.2f}s ({len(val_labels)/t_val:.0f} samples/s)")
    
    # Find MCC-optimal threshold on validation
    best_tau, best_val_mcc, sweep_df = find_optimal_threshold(
        val_scores, val_labels, n_thresholds=args.n_thresholds
    )
    
    # Also compute val metrics at fixed τ=0.5 for comparison
    val_metrics_fixed = compute_metrics_at_threshold(val_scores, val_labels, 0.5)
    val_metrics_optimal = compute_metrics_at_threshold(val_scores, val_labels, best_tau)
    
    logg.info(f"Val MCC @ τ=0.500: {val_metrics_fixed['MCC']:.4f}")
    logg.info(f"Val MCC @ τ*={best_tau:.3f}: {val_metrics_optimal['MCC']:.4f}")
    logg.info(f"Optimal threshold τ* = {best_tau:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Inference on TEST set → apply τ* from validation
    # ══════════════════════════════════════════════════════════════════════
    logg.info("=" * 60)
    logg.info("PHASE 2: Test inference with calibrated threshold")
    logg.info("=" * 60)
    
    t0 = time()
    test_scores, test_labels = run_inference(model, test_loader, device, use_cuda, "Test")
    t_test = time() - t0
    logg.info(f"Test inference: {t_test:.2f}s ({len(test_labels)/t_test:.0f} samples/s)")
    
    # Compute test metrics with BOTH thresholds
    test_metrics_fixed = compute_metrics_at_threshold(test_scores, test_labels, 0.5)
    test_metrics_calibrated = compute_metrics_at_threshold(test_scores, test_labels, best_tau)
    
    logg.info(f"Test MCC @ τ=0.500 (v1): {test_metrics_fixed['MCC']:.4f}")
    logg.info(f"Test MCC @ τ*={best_tau:.3f} (v2): {test_metrics_calibrated['MCC']:.4f}")
    logg.info(f"  ΔmCC = {test_metrics_calibrated['MCC'] - test_metrics_fixed['MCC']:+.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Save results
    # ══════════════════════════════════════════════════════════════════════
    output_dir = Path(args.output_dir) / args.exp_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pd.DataFrame({'prediction': val_scores, 'label': val_labels})\
        .to_csv(output_dir / 'val_predictions.csv', index=False)
    pd.DataFrame({'prediction': test_scores, 'label': test_labels})\
        .to_csv(output_dir / 'test_predictions.csv', index=False)
    
    # Save threshold sweep
    sweep_df.to_csv(output_dir / 'threshold_sweep.csv', index=False)

    # Save comprehensive results
    results = {
        'protocol': 'MCC-optimal threshold from validation (zero test leakage)',
        'optimal_threshold': round(best_tau, 4),
        'n_threshold_candidates': args.n_thresholds,
        
        'val': {
            'n_samples': int(len(val_labels)),
            'n_pos': int(val_labels.sum()),
            'n_neg': int(len(val_labels) - val_labels.sum()),
            'pos_ratio': round(float(val_labels.mean()), 4),
            'score_mean': round(float(val_scores.mean()), 4),
            'score_std': round(float(val_scores.std()), 4),
            'at_fixed_0.5': {k: round(v, 6) if isinstance(v, float) else v 
                            for k, v in val_metrics_fixed.items()},
            'at_optimal': {k: round(v, 6) if isinstance(v, float) else v 
                          for k, v in val_metrics_optimal.items()},
        },
        
        'test': {
            'n_samples': int(len(test_labels)),
            'n_pos': int(test_labels.sum()),
            'n_neg': int(len(test_labels) - test_labels.sum()),
            'pos_ratio': round(float(test_labels.mean()), 4),
            'score_mean': round(float(test_scores.mean()), 4),
            'score_std': round(float(test_scores.std()), 4),
            'at_fixed_0.5': {k: round(v, 6) if isinstance(v, float) else v 
                            for k, v in test_metrics_fixed.items()},
            'at_calibrated': {k: round(v, 6) if isinstance(v, float) else v 
                             for k, v in test_metrics_calibrated.items()},
        },
        
        'improvement': {
            'MCC': round(test_metrics_calibrated['MCC'] - test_metrics_fixed['MCC'], 6),
            'F1': round(test_metrics_calibrated['F1'] - test_metrics_fixed['F1'], 6),
            'Accuracy': round(test_metrics_calibrated['Accuracy'] - test_metrics_fixed['Accuracy'], 6),
        },
        
        'checkpoint': str(ckpt_path),
        'data_dir': str(data_dir),
        'eval_time_val_sec': round(t_val, 2),
        'eval_time_test_sec': round(t_test, 2),
        'device': str(device),
        'gpu': torch.cuda.get_device_name(args.device) if use_cuda else 'CPU',
    }

    with open(output_dir / 'results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f" ConPLex v2 — {args.exp_id}")
    print(f" Protocol: val-calibrated threshold (no test leakage)")
    print(f"{'='*65}")
    print(f"  Optimal τ* (from val): {best_tau:.4f}")
    print(f"")
    print(f"  {'Metric':<12} {'τ=0.5 (v1)':>12} {'τ*={:.3f} (v2)':>15} {'Δ':>10}".format(best_tau))
    print(f"  {'-'*12} {'-'*12} {'-'*15} {'-'*10}")
    for metric in ['MCC', 'AUROC', 'F1', 'Accuracy', 'Precision', 'Recall']:
        v1 = test_metrics_fixed[metric]
        v2 = test_metrics_calibrated[metric]
        delta = v2 - v1
        print(f"  {metric:<12} {v1:>12.4f} {v2:>15.4f} {delta:>+10.4f}")
    print(f"{'='*65}\n")
    
    logg.info(f"Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    main()
