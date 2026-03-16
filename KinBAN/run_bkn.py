"""BAN-Kinase-Network (BKN) — CLI entry point.

Self-contained variant of GraphBAN with three encoder substitutions:
  1. Protein: ESM-2 650M (esm2_t33_650M_UR50D) instead of ESM-1b
  2. Drug:    MoLFormer-XL (ibm/MoLFormer-XL-both-10pct, 768-d) instead of ChemBERTa
  3. Pooling: CLS-guided attention pooling for both encoders

All logic lives in the ``bkn/`` package.  This file is the thin CLI entry point.

Usage:
    python run_bkn.py --dataset non_human
    python run_bkn.py --dataset human --seeds 42 123 456
    python run_bkn.py --dataset all --max-epoch 10  # quick test
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# DGL graphbooth compatibility shim — must load before any DGL import.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import dgl_compat  # noqa: F401, E402

warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor.*use sourceTensor.clone",
    category=UserWarning,
)

import pandas as pd
import torch

from bkn import (
    CANONICAL_SEEDS,
    DEFAULT_TEACHER_EPOCHS,
    DRUG_EMB_DIM,
    SCRIPT_DIR,
    aggregate_results,
    extract_features_cached,
    generate_teacher_embeddings,
    print_summary_table,
    setup_bkn_imports,
    train_single_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BKN: GraphBAN with ESM-2 + MoLFormer + Attention Pooling"
    )
    parser.add_argument(
        "--dataset",
        choices=["non_human", "human", "all"],
        default="non_human",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=CANONICAL_SEEDS)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--teacher-epochs", type=int, default=DEFAULT_TEACHER_EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-da", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--molformer-path",
        type=str,
        default=None,
        help="Local path to MoLFormer-XL weights (skips HuggingFace download). "
             "Example: /tmp/molformer_dl",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    dataset_dir = SCRIPT_DIR / "datasets" / "kinase" / args.dataset / "scaffold"
    if not dataset_dir.exists() or not (dataset_dir / "train.csv").exists():
        print(f"Dataset not found at {dataset_dir}")
        print(f"Auto-preparing data for '{args.dataset}'...")
        sys.path.insert(0, str(SCRIPT_DIR))
        from prepare_data import prepare_dataset
        prepare_dataset(args.dataset, SCRIPT_DIR)
        if not (dataset_dir / "train.csv").exists():
            print("ERROR: Data preparation failed.")
            sys.exit(1)
        print("Data prepared successfully.\n")

    modules = setup_bkn_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = modules["get_cfg_defaults"]()
    config_path = SCRIPT_DIR / "configs" / "kinase_bkn.yaml"
    if config_path.exists():
        cfg.merge_from_file(str(config_path))

    if args.max_epoch is not None:
        cfg.defrost(); cfg.SOLVER.MAX_EPOCH = args.max_epoch; cfg.freeze()
    if args.batch_size is not None:
        cfg.defrost(); cfg.SOLVER.BATCH_SIZE = args.batch_size; cfg.freeze()
    if args.num_workers > 0:
        cfg.defrost(); cfg.SOLVER.NUM_WORKERS = args.num_workers; cfg.freeze()
    if args.no_da:
        cfg.defrost(); cfg.DA.USE = False; cfg.DA.TASK = False; cfg.freeze()

    # Override MoLFormer model path if a local directory is provided
    if args.molformer_path:
        import bkn.constants as _bkn_const
        _bkn_const.MOLFORMER_MODEL_NAME = str(args.molformer_path)

    output_base = Path(args.output_dir or (SCRIPT_DIR / "results" / args.dataset)).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    print("Model:    BKN (ESM-2 650M + MoLFormer-XL 768-d + AttentionPool)")
    print(f"Dataset:  {args.dataset} (scaffold split)")
    print(f"Seeds:    {args.seeds}")
    print(f"Epochs:   {cfg.SOLVER.MAX_EPOCH}  |  Batch: {cfg.SOLVER.BATCH_SIZE}")
    print(f"DA:       {'enabled' if cfg.DA.USE else 'disabled'}")
    print(f"Output:   {output_base}")
    print(f"Device:   {device}")

    train_df = pd.read_csv(dataset_dir / "train.csv")
    val_df   = pd.read_csv(dataset_dir / "val.csv")
    test_df  = pd.read_csv(dataset_dir / "test.csv")
    print(f"\nData: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    cache_dir = output_base / "feature_cache"
    train_df, val_df, test_df = extract_features_cached(
        train_df, val_df, test_df, device, cache_dir,
    )

    all_metrics = []
    for seed in args.seeds:
        teacher_dir = output_base / f"seed_{seed}" / "teacher"
        teacher_dir.mkdir(parents=True, exist_ok=True)
        teacher_parquet = teacher_dir / "teacher_embeddings.parquet"

        if not teacher_parquet.exists():
            generate_teacher_embeddings(
                dataset_dir / "train.csv",
                seed,
                teacher_parquet,
                epochs=args.teacher_epochs,
            )
        else:
            print(f"\n    Using cached teacher embeddings: {teacher_parquet}")

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

    agg = aggregate_results(all_metrics)
    print_summary_table(agg, args.dataset, args.seeds)

    results_file = output_base / "bkn_results.json"
    output_data = {
        "model": "BKN",
        "variant": "ESM-2-650M + MoLFormer-XL-768d + CLS-Attention-Pooling",
        "dataset": args.dataset,
        "split": "scaffold",
        "seeds": args.seeds,
        "methodology": {
            "protein_encoder": "ESM-2 esm2_t33_650M_UR50D (1280-d, frozen)",
            "drug_encoder": "MoLFormer-XL ibm/MoLFormer-XL-both-10pct (768-d, frozen)",
            "pooling": "CLS-guided attention pooling for both encoders",
            "model_selection": "validation AUROC",
            "threshold_optimization": "validation MCC-optimal (no test leakage)",
            "teacher": f"GAE on bipartite CPI graph (256-d, {args.teacher_epochs} epochs)",
            "trainer_info": (
                f"trainer.py in src/ statically patched for "
                f"(batch, 1, {DRUG_EMB_DIM}) drug embedding reshape."
            ),
        },
        "config": {
            "max_epoch": cfg.SOLVER.MAX_EPOCH,
            "batch_size": cfg.SOLVER.BATCH_SIZE,
            "lr": cfg.SOLVER.LR,
            "domain_adaptation": cfg.DA.USE,
            "drug_emb_dim": DRUG_EMB_DIM,
        },
        "aggregate": agg,
        "per_seed": all_metrics,
    }

    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {results_file}")
    print("Done!")


if __name__ == "__main__":
    main()
