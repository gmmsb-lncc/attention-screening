#!/usr/bin/env python3
"""
eval_crossdomain.py — Cross-domain evaluation of ConPLex pretrained on DAVIS,
                      evaluated zero-shot on universal kinase test sets.

Analogous to the GraphBAN cross-domain evaluation (BioSNAP, BindingDB, KIBA → kinase test).
Here: DAVIS → kinase test (Non-Human, Human, All).

Protocol:
  1. Load pretrained ConPLex weights (trained on DAVIS by original authors)
  2. For each kinase dataset (non_human, human, all):
     a. Run inference on the VALIDATION set → find MCC-optimal τ*
     b. Run inference on the TEST set → apply τ* → compute metrics
     c. Also report metrics at fixed τ=0.5 for reference
  3. Save comprehensive results JSON and summary table

Note: Unlike GraphBAN's 3 source datasets × 5 seeds, ConPLex has a single
      pretrained checkpoint (DAVIS, ProtBERT). Results have no seed variance.

Usage:
    python eval_crossdomain.py --device 0
    python eval_crossdomain.py --device -1          # CPU
    python eval_crossdomain.py --checkpoint models/esm_epoch5_state_dict.pt  # ESM variant
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
from datetime import datetime
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
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mcc(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return float(tp * tn - fp * fn) / denom


def find_optimal_threshold(scores, labels, n_thresholds=2000):
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_tau, best_mcc = 0.5, -1.0
    for tau in thresholds:
        preds = (scores >= tau).astype(int)
        mcc = compute_mcc(labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_tau = tau
    return best_tau, best_mcc


def compute_metrics(scores, labels, threshold):
    from sklearn.metrics import roc_auc_score, average_precision_score
    preds = (scores >= threshold).astype(int)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))

    n = len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n
    mcc = compute_mcc(labels, preds)

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5
    try:
        auprc = average_precision_score(labels, scores)
    except ValueError:
        auprc = 0.0

    return {
        'MCC': round(float(mcc), 4),
        'AUROC': round(float(auroc), 4),
        'AUPRC': round(float(auprc), 4),
        'F1': round(float(f1), 4),
        'Accuracy': round(float(accuracy), 4),
        'Sensitivity': round(float(recall), 4),
        'Specificity': round(float(specificity), 4),
        'Precision': round(float(precision), 4),
        'Recall': round(float(recall), 4),
        'threshold': round(float(threshold), 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(model, dataloader, device, use_cuda, desc="Inference"):
    all_scores, all_labels = [], []
    model.eval()
    with torch.no_grad():
        if use_cuda:
            ctx = torch.cuda.amp.autocast()
        else:
            import contextlib
            ctx = contextlib.nullcontext()
        with ctx:
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
    parser = ArgumentParser(description="ConPLex Cross-Domain: DAVIS → Kinase test sets")
    parser.add_argument("--checkpoint", default="models/protbert_epoch3_state_dict.pt",
                        help="Path to pretrained state_dict .pt (default: DAVIS/ProtBERT)")
    parser.add_argument("--drug-featurizer", default="MorganFeaturizer")
    parser.add_argument("--target-featurizer", default="ProtBertFeaturizer")
    parser.add_argument("--model-architecture", default="SimpleCoembedding")
    parser.add_argument("--latent-dimension", type=int, default=1024)
    parser.add_argument("--latent-distance", default="Cosine")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=int, default=0, help="CUDA device (-1 for CPU)")
    parser.add_argument("--output-dir", default="./results_crossdomain",
                        help="Where to save results")
    parser.add_argument("--datasets", nargs="+",
                        default=["kinase_non_human", "kinase_human", "kinase_all"],
                        help="Kinase test sets to evaluate on")
    return parser.parse_args()


def evaluate_single_dataset(model, data_dir, device, use_cuda, args):
    """Evaluate pretrained model on a single kinase dataset (val + test)."""
    logg.info(f"  Loading data from {data_dir}")

    drug_featurizer = get_featurizer(args.drug_featurizer, save_dir=data_dir)
    target_featurizer = get_featurizer(args.target_featurizer, save_dir=data_dir)

    datamodule = DTIDataModule(
        data_dir,
        drug_featurizer,
        target_featurizer,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers or min(os.cpu_count() // 2, 8),
    )
    datamodule.prepare_data()
    datamodule.setup(stage=None)

    def make_loader(dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers or min(os.cpu_count() // 2, 8),
            collate_fn=drug_target_collate_fn,
            pin_memory=use_cuda,
        )

    val_loader = make_loader(datamodule.data_val)
    test_loader = make_loader(datamodule.data_test)

    n_val = len(datamodule.data_val)
    n_test = len(datamodule.data_test)
    logg.info(f"  Val: {n_val} samples | Test: {n_test} samples")

    # Phase 1: Validation → threshold calibration
    t0 = time()
    val_scores, val_labels = run_inference(model, val_loader, device, use_cuda, "Val")
    t_val = time() - t0
    best_tau, best_val_mcc = find_optimal_threshold(val_scores, val_labels)
    logg.info(f"  Val MCC-optimal τ* = {best_tau:.4f} (MCC = {best_val_mcc:.4f})")

    # Phase 2: Test → apply τ* and also τ=0.5
    t0 = time()
    test_scores, test_labels = run_inference(model, test_loader, device, use_cuda, "Test")
    t_test = time() - t0

    metrics_fixed = compute_metrics(test_scores, test_labels, 0.5)
    metrics_calibrated = compute_metrics(test_scores, test_labels, best_tau)

    logg.info(f"  Test MCC @ τ=0.5: {metrics_fixed['MCC']:.4f}")
    logg.info(f"  Test MCC @ τ*={best_tau:.3f}: {metrics_calibrated['MCC']:.4f}")

    return {
        'n_val': n_val,
        'n_test': n_test,
        'n_pos_test': int(test_labels.sum()),
        'pos_ratio_test': round(float(test_labels.mean()), 4),
        'optimal_threshold': round(best_tau, 4),
        'val_mcc_at_optimal': round(best_val_mcc, 4),
        'test_at_fixed_0.5': metrics_fixed,
        'test_at_calibrated': metrics_calibrated,
        'eval_time_val_sec': round(t_val, 2),
        'eval_time_test_sec': round(t_test, 2),
    }


def main():
    args = parse_args()
    set_random_seed(42)

    if args.num_workers is None:
        args.num_workers = min((os.cpu_count() or 4) // 2, 8)
        args.num_workers = max(args.num_workers, 2)

    use_cuda = torch.cuda.is_available() and args.device >= 0
    device = torch.device(f"cuda:{args.device}" if use_cuda else "cpu")
    logg.info(f"Device: {device}")

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Load pretrained model ─────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint).resolve()
    logg.info(f"Loading pretrained checkpoint: {ckpt_path}")

    # Determine shapes from featurizers
    dummy_dir = Path("dataset/DAVIS")
    drug_featurizer = get_featurizer(args.drug_featurizer, save_dir=dummy_dir)
    target_featurizer = get_featurizer(args.target_featurizer, save_dir=dummy_dir)

    model = getattr(model_types, args.model_architecture)(
        drug_shape=drug_featurizer.shape,
        target_shape=target_featurizer.shape,
        latent_dimension=args.latent_dimension,
        latent_distance=args.latent_distance,
        classify=True,
    )
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logg.info(f"Model loaded: {n_params:,} parameters")

    # ── Evaluate on each kinase dataset ───────────────────────────────────
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(ckpt_path),
            'source_dataset': 'DAVIS',
            'protocol': 'Zero-shot cross-domain (pretrained on DAVIS → kinase test sets)',
            'threshold_calibration': 'MCC-optimal on kinase validation set',
            'device': str(device),
            'gpu': torch.cuda.get_device_name(args.device) if use_cuda else 'CPU',
            'model_params': n_params,
        },
        'datasets': {},
    }

    for ds_name in args.datasets:
        logg.info(f"\n{'='*60}")
        logg.info(f"Evaluating: {ds_name}")
        logg.info(f"{'='*60}")

        data_dir = Path(f"dataset/{ds_name}")
        if not data_dir.exists():
            logg.warning(f"  Dataset {ds_name} not found at {data_dir}, skipping")
            continue

        ds_results = evaluate_single_dataset(model, data_dir, device, use_cuda, args)
        results['datasets'][ds_name] = ds_results

    # ── Save results ──────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / "conplex_crossdomain_davis.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    logg.info(f"\nResults saved to {out_file}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  ConPLex Cross-Domain: DAVIS → Kinase Test Sets (zero-shot)")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"{'='*80}")

    header = f"  {'Test Set':<15} {'AUROC':>8} {'AUPRC':>8} {'MCC':>8} {'F1':>8} {'Acc.':>8} {'Sens.':>8} {'Spec.':>8} {'τ*':>6}"
    print(header)
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for ds_name, ds_res in results['datasets'].items():
        m = ds_res['test_at_calibrated']
        short = ds_name.replace('kinase_', '').replace('_', '-').title()
        tau = ds_res['optimal_threshold']
        print(f"  {short:<15} {m['AUROC']:>8.3f} {m['AUPRC']:>8.3f} {m['MCC']:>8.3f} "
              f"{m['F1']:>8.3f} {m['Accuracy']:>8.3f} {m['Sensitivity']:>8.3f} "
              f"{m['Specificity']:>8.3f} {tau:>6.3f}")

    # Reference: DT-Kinase in-domain
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'DT-Kinase (NH)':<15} {'0.814':>8} {'---':>8} {'0.512':>8} "
          f"{'0.788':>8} {'0.752':>8} {'0.867':>8} {'~0.65':>8} {'---':>6}")
    print(f"  {'DT-Kinase (H)':<15} {'0.807':>8} {'---':>8} {'0.457':>8} "
          f"{'0.649':>8} {'0.751':>8} {'0.650':>8} {'~0.75':>8} {'---':>6}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    main()
