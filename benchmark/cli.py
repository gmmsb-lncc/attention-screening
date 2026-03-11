"""Command-line interface for the benchmark pipeline.

Builds an ``argparse.ArgumentParser`` and converts parsed arguments
into a ``BenchmarkConfig`` instance.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import List

from benchmark.config import (
    DEFAULT_SCAFFOLD_SPLIT_DIR,
    LEVEL_0_EXPANSION,
    VALID_LEVELS,
    BenchmarkConfig,
)


def build_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified benchmark: scaffold split → models → comparative report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python semantic_screening_models.py --dataset human --embedding 8M --levels 1a 2 3 4\n"
            "  python semantic_screening_models.py --dataset human --embedding 8M --levels 1a 2 3 --finetune\n"
        ),
    )

    # --- required ---
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["human", "non_human", "all"],
        help="Dataset to benchmark. All options use the universal scaffold-disjoint split; "
             "'human'/'non_human' filter by dataset_source, 'all' uses all rows.",
    )
    parser.add_argument(
        "--embedding",
        default="8M",
        choices=["8M", "150M", "650M"],
        help="ESM-2 model shorthand (default: 8M)",
    )

    # --- level selection ---
    parser.add_argument(
        "--levels",
        default="1a,1b,1c,2,3,4,5a,5b,6a,6b",
        nargs="*",
        help=(
            "Levels to run: "
            "0=ClassicalML(1a+1b+1c+3), "
            "1a=FP, 1b=LigMeanPool, 1c=LigAttnPool, "
            "2=MeanPool, 3=AttnPool, 4=CrossAttn+AttnPool, "
            "5a=CrossAttn+AttnPool+GRL, 5b=AttnPool+GRL, "
            "6a=CrossAttn+BAN+GRL, 6b=AttnPool+BAN+GRL "
            "(default: 1a,1b,1c,2,3,4,5a,5b,6a,6b). "
            "Examples: --levels 0 OR --levels 1a 1b 1c 2 3 4 5a 5b 6a 6b"
        ),
    )

    # --- output ---
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Root results dir (default: ./results/benchmark_{dataset}_{embedding})",
    )
    parser.add_argument(
        "--scaffold_split_dir",
        default=DEFAULT_SCAFFOLD_SPLIT_DIR,
        help=f"Scaffold split dir (default: {DEFAULT_SCAFFOLD_SPLIT_DIR})",
    )

    # --- reproducibility ---
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds for multi-seed runs (default: from config.DEFAULT_SEEDS)",
    )

    # --- flags ---
    parser.add_argument("--force", action="store_true", help="Force recalculation of all levels")
    parser.add_argument("--force_split", action="store_true", help="Force regeneration of scaffold splits")
    parser.add_argument("--debug", action="store_true", help="Debug mode (verbose output)")

    # --- deep-learning hyper-parameters ---
    parser.add_argument("--epochs", type=int, default=500, help="Max epochs for Level 4 (default: 500)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for Level 4 (default: 32)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5, 0=disable)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Level 4 (default: 1e-4)")

    # --- fine-tuning ---
    parser.add_argument("--finetune", action="store_true", help="Enable ESM-2 + MolFormer fine-tuning before levels")
    parser.add_argument("--use_finetuned", action="store_true", help="Use pre-existing fine-tuned embeddings")
    parser.add_argument("--finetune_epochs", type=int, default=100, help="Fine-tuning epochs (default: 100)")
    parser.add_argument("--finetune_lr", type=float, default=1e-5, help="Fine-tuning learning rate (default: 1e-5)")
    parser.add_argument("--finetune_batch_size", type=int, default=8, help="Fine-tuning batch size (default: 8)")

    return parser


def parse_levels(levels_arg: object) -> List[str]:
    """Parse levels from various formats: ``['1a', '2']``, ``'1a,2,3'``, etc.

    Returns a sorted, deduplicated list of valid level strings.

    Raises:
        SystemExit: When an invalid level is provided.
    """
    try:
        if isinstance(levels_arg, list):
            if not levels_arg:
                return sorted(VALID_LEVELS)
            parts: list[str] = []
            for item in levels_arg:
                parts.extend(re.split(r"[,\s]+", str(item).strip()))
        else:
            parts = re.split(r"[,\s]+", str(levels_arg).strip())

        levels = sorted({x.strip().lower() for x in parts if x.strip()})

        for level in levels:
            if level not in VALID_LEVELS:
                msg = f"Invalid level: {level}. Valid: {sorted(VALID_LEVELS)}"
                raise ValueError(msg)

        # Expand level 0 shortcut → 1a, 1b, 1c, 3
        if "0" in levels:
            levels = sorted({lv for lv in levels if lv != "0"} | set(LEVEL_0_EXPANSION))

        return levels if levels else sorted(VALID_LEVELS)

    except ValueError as exc:
        print(f"ERROR: Invalid --levels value: {exc}")
        sys.exit(1)


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Convert parsed CLI arguments into a ``BenchmarkConfig``."""
    return BenchmarkConfig(
        dataset=args.dataset,
        embedding=args.embedding,
        levels=parse_levels(args.levels),
        output_dir=args.output_dir,
        scaffold_split_dir=args.scaffold_split_dir,
        seeds=args.seeds,
        force=args.force,
        force_split=args.force_split,
        debug=args.debug,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        finetune=args.finetune,
        use_finetuned=args.use_finetuned,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_batch_size=args.finetune_batch_size,
    )
