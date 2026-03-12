"""BAN-Kinase-Network (BKN) package — public API."""
from __future__ import annotations

from .constants import (
    BKN_SRC,
    CANONICAL_SEEDS,
    DEFAULT_TEACHER_EPOCHS,
    DRUG_EMB_DIM,
    ESM2_EMB_DIM,
    GRAPHBAN_DIR,
    GRAPHBAN_INDUCTIVE,
    REPO_ROOT,
    SCRIPT_DIR,
)
from .evaluation import (
    collect_predictions,
    compute_metrics_at_threshold,
    optimize_threshold_on_validation,
)
from .features import extract_features_cached
from .loader import setup_bkn_imports
from .reporting import aggregate_results, print_summary_table
from .teacher import generate_teacher_embeddings
from .training import train_single_seed

__all__ = [
    # Constants
    "BKN_SRC",
    "CANONICAL_SEEDS",
    "DEFAULT_TEACHER_EPOCHS",
    "DRUG_EMB_DIM",
    "ESM2_EMB_DIM",
    "GRAPHBAN_DIR",
    "GRAPHBAN_INDUCTIVE",
    "REPO_ROOT",
    "SCRIPT_DIR",
    # Loader
    "setup_bkn_imports",
    # Feature extraction
    "extract_features_cached",
    # Teacher GAE
    "generate_teacher_embeddings",
    # Evaluation
    "collect_predictions",
    "compute_metrics_at_threshold",
    "optimize_threshold_on_validation",
    # Training
    "train_single_seed",
    # Reporting
    "aggregate_results",
    "print_summary_table",
]
