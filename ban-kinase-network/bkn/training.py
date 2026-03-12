"""Single-seed training loop for BKN."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluation import (
    collect_predictions,
    compute_metrics_at_threshold,
    optimize_threshold_on_validation,
)
from .trainer_patch import patch_trainer_for_mcc_logging


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
    """Train BKN for a single seed and return per-split metrics.

    Protocol identical to GraphBAN/run_baseline.py:
      1. Attach teacher embeddings to training data
      2. Build datasets / data loaders for train, val, test
      3. Train model (best checkpoint selected by val AUROC)
      4. Collect val predictions → MCC-optimal threshold (no test leakage)
      5. Collect train + test predictions → apply val threshold
      6. Return ``{train, val, test}`` metric dicts
    """
    print(f"\n{'─'*50}")
    print(f"  Seed: {seed}")
    print(f"{'─'*50}")

    modules["set_seed"](seed)

    seed_output = output_dir / f"seed_{seed}"
    modules["mkdir"](str(seed_output))

    cfg.defrost()
    cfg.SOLVER.SEED = seed
    cfg.RESULT.OUTPUT_DIR = str(seed_output)
    cfg.freeze()

    # Attach teacher embeddings to training split
    train_emb = pd.read_parquet(teacher_parquet)
    train_emb["Array"] = train_emb.apply(lambda row: np.array(row), axis=1)
    train_emb.drop(train_emb.columns.difference(["Array"]), axis=1, inplace=True)

    train_df_seed = train_df.copy()
    train_df_seed["teacher_emb"] = train_emb["Array"].values

    print(f"  Data: train={len(train_df_seed)}, val={len(val_df)}, test={len(test_df)}")

    train_dataset = modules["DTIDataset2"](train_df_seed.index.values, train_df_seed)
    val_dataset = modules["DTIDataset"](val_df.index.values, val_df)
    test_dataset = modules["DTIDataset"](test_df.index.values, test_df)

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
    val_generator = torch.utils.data.DataLoader(val_dataset, **params_eval)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params_eval)

    model = modules["GraphBAN"](**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

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

    patch_trainer_for_mcc_logging(trainer, val_generator, device, n_class)

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    if isinstance(result, tuple):
        result_metrics = result[0] if isinstance(result[0], dict) else {}
    elif isinstance(result, dict):
        result_metrics = result
    else:
        result_metrics = {}

    # --- Fair evaluation protocol ---
    print("  Collecting validation predictions for threshold optimization...")
    val_y_true, val_y_prob = collect_predictions(
        trainer.best_model, val_generator, device, n_class,
    )

    val_threshold, val_best_mcc = optimize_threshold_on_validation(
        val_y_true, val_y_prob, metric="mcc",
    )
    print(f"  Val-optimized threshold={val_threshold:.4f} (val MCC={val_best_mcc:.4f})")

    print("  Collecting train predictions...")
    # Use DTIDataset (5-tuple) not DTIDataset2 (6-tuple with teacher) so that
    # graph_collate_func can unpack correctly during evaluation.
    train_eval_dataset = modules["DTIDataset"](train_df_seed.index.values, train_df_seed)
    train_eval_gen = torch.utils.data.DataLoader(train_eval_dataset, **params_eval)
    train_y_true, train_y_prob = collect_predictions(
        trainer.best_model, train_eval_gen, device, n_class,
    )

    print("  Collecting test predictions...")
    test_y_true, test_y_prob = collect_predictions(
        trainer.best_model, test_generator, device, n_class,
    )

    train_metrics = compute_metrics_at_threshold(train_y_true, train_y_prob, val_threshold)
    val_metrics = compute_metrics_at_threshold(val_y_true, val_y_prob, val_threshold)
    test_metrics = compute_metrics_at_threshold(test_y_true, test_y_prob, val_threshold)

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
        "graphban_native": {
            "threshold": native_threshold,
            "threshold_source": (
                "test_f1_optimal (GraphBAN original — NOT used for comparison)"
            ),
            "mcc": native_test_metrics["mcc"],
            "auroc": result_metrics.get("auroc", native_test_metrics["auroc"]),
            "auprc": result_metrics.get("auprc", None),
        },
    }

    print(f"  Results (seed={seed}, threshold={val_threshold:.4f}):")
    print(f"    {'Split':<6}  {'MCC':>7}  {'AUROC':>7}  {'F1':>7}  {'Acc':>7}")
    print(f"    {'─'*38}")
    for split_name, split_m in [
        ("Train", train_metrics),
        ("Val", val_metrics),
        ("Test", test_metrics),
    ]:
        print(
            f"    {split_name:<6}  {split_m['mcc']:>7.4f}  {split_m['auroc']:>7.4f}  "
            f"{split_m['f1']:>7.4f}  {split_m['accuracy']:>7.4f}"
        )
    print(f"    Time: {elapsed:.1f}s  Best epoch: {metrics['best_epoch']}")

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
