#!/usr/bin/env python3
"""
eval_conplex.py — Evaluate a pretrained ConPLex model on a given dataset.

Optimized for RTX 4090 + multi-core CPU:
  - pin_memory=True for fast CPU→GPU transfer
  - Configurable num_workers for parallel data loading
  - TF32 and cuDNN autotuner enabled
  - torch.compile for compiled model (PyTorch 2.x)

Usage:
    python eval_conplex.py \
        --checkpoint models/protbert_epoch3_state_dict.pt \
        --data-dir dataset/kinase/non_human \
        --exp-id eval_pretrained_non_human \
        --device 0
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
import torchmetrics

# Add ConPLex src to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(SCRIPT_DIR)

from src import architectures as model_types
from src.data import DTIDataModule, drug_target_collate_fn
from src.utils import set_random_seed, get_featurizer, get_logger

logg = get_logger()


def parse_args():
    parser = ArgumentParser(description="Evaluate ConPLex on a dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to state_dict .pt")
    parser.add_argument("--data-dir", required=True, help="Directory with train/val/test.csv")
    parser.add_argument("--exp-id", required=True, help="Experiment name for output")
    parser.add_argument("--drug-featurizer", default="MorganFeaturizer")
    parser.add_argument("--target-featurizer", default="ProtBertFeaturizer")
    parser.add_argument("--model-architecture", default="SimpleCoembeddingNoSigmoid")
    parser.add_argument("--latent-dimension", type=int, default=1024)
    parser.add_argument("--latent-distance", default="Cosine")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: auto = CPU_count/2, max 8)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device (-1 for CPU)")
    parser.add_argument("--output-dir", default="./results", help="Where to save results")
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
        vram = torch.cuda.get_device_properties(args.device).total_mem // (1024**2)
        logg.info(f"GPU: {torch.cuda.get_device_name(args.device)} ({vram} MB)")
        logg.info(f"cuDNN benchmark: ON, TF32: ON")

    logg.info(f"DataLoader workers: {args.num_workers}, batch_size: {args.batch_size}")

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
    datamodule.setup(stage="test")

    # Enable pin_memory for fast CPU→GPU transfer
    test_loader = torch.utils.data.DataLoader(
        datamodule.data_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=drug_target_collate_fn,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    logg.info(f"Test set: {len(datamodule.data_test)} samples, "
              f"{len(test_loader)} batches")

    # ── Model ──────────────────────────────────────────────────────────────
    drug_shape = drug_featurizer.shape
    target_shape = target_featurizer.shape
    logg.info(f"Model: {args.model_architecture} "
              f"(drug={drug_shape}, target={target_shape}, latent={args.latent_dimension})")

    # Handle the NoSigmoid alias
    arch_name = args.model_architecture
    if arch_name == "SimpleCoembeddingNoSigmoid":
        arch_name = "SimpleCoembedding"

    model = getattr(model_types, arch_name)(
        drug_shape=drug_shape,
        target_shape=target_shape,
        latent_dimension=args.latent_dimension,
        latent_distance=args.latent_distance,
        classify=True,
    )

    # Load checkpoint
    ckpt_path = Path(args.checkpoint).resolve()
    logg.info(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Try torch.compile for PyTorch 2.x speedup
    if hasattr(torch, 'compile') and use_cuda:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logg.info("torch.compile: enabled (reduce-overhead)")
        except Exception as e:
            logg.warning(f"torch.compile failed, using eager mode: {e}")

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics = {
        "AUROC": torchmetrics.AUROC(task="binary").to(device),
        "AUPRC": torchmetrics.AveragePrecision(task="binary").to(device),
    }

    # ── Inference ──────────────────────────────────────────────────────────
    all_preds = []
    all_labels = []
    logg.info("Running inference...")
    t0 = time()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
        for batch in tqdm(test_loader, desc="Eval"):
            drug, target, label = batch
            drug = drug.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).int()

            pred = model(drug, target)
            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())

            for met in metrics.values():
                met(pred, label)

    if use_cuda:
        torch.cuda.synchronize()
    t1 = time()

    # ── Compute final metrics ──────────────────────────────────────────────
    results = {}
    for name, met in metrics.items():
        results[name] = met.compute().item()
        logg.info(f"{name}: {results[name]:.4f}")

    results["eval_time_sec"] = round(t1 - t0, 2)
    results["throughput_samples_per_sec"] = round(len(datamodule.data_test) / (t1 - t0), 1)
    results["n_test_samples"] = len(datamodule.data_test)
    results["checkpoint"] = str(ckpt_path)
    results["data_dir"] = str(data_dir)
    results["batch_size"] = args.batch_size
    results["num_workers"] = args.num_workers
    results["device"] = str(device)
    if use_cuda:
        results["gpu"] = torch.cuda.get_device_name(args.device)

    # ── Save ───────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) / args.exp_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logg.info(f"Results saved to {output_dir / 'results.json'}")

    preds_cat = torch.cat(all_preds).numpy()
    labels_cat = torch.cat(all_labels).numpy()
    pred_df = pd.DataFrame({"prediction": preds_cat, "label": labels_cat})
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    logg.info(f"Predictions saved to {output_dir / 'predictions.csv'}")

    print(f"\n{'='*55}")
    print(f" ConPLex Evaluation: {args.exp_id}")
    print(f"{'='*55}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"{'='*55}\n")

    return results


if __name__ == "__main__":
    main()
