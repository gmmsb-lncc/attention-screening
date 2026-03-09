"""Centralized configuration for the benchmark pipeline.

Single source of truth for constants, paths, embedding mappings,
metric definitions, and the ``BenchmarkConfig`` dataclass that
carries runtime parameters across all modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Embedding model registry
# ---------------------------------------------------------------------------

SUPPORTED_EMBEDDINGS: Dict[str, str] = {
    "8M": "esm2_t6_8M_UR50D",
    "150M": "esm2_t30_150M_UR50D",
    "650M": "esm2_t33_650M_UR50D",
}

# ---------------------------------------------------------------------------
# Embedding dimensions
# ---------------------------------------------------------------------------

PROTEIN_DIMS: Dict[str, int] = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
}

MOLFORMER_DIM: int = 768  # MoLFormer per-token embeddings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EMBEDDING_BASE_PATH = "./results/protein_model_benchmark_{dataset_type}_v2"
DEFAULT_SCAFFOLD_SPLIT_DIR = "scaffolds_splits/output"

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRICS_ORDER: List[str] = ["accuracy", "mcc", "f1", "precision", "recall", "auc"]

# ---------------------------------------------------------------------------
# Level display labels (used in tables and plots)
# ---------------------------------------------------------------------------

LEVEL_LABELS: Dict[str, str] = {
    "level1a_fp_knn": "L1a (FP+KNN)",
    "level1a_fp_mlp": "L1a (FP+MLP)",
    "level1b_ligmean_knn": "L1b (LigMean+KNN)",
    "level1b_ligmean_mlp": "L1b (LigMean+MLP)",
    "level1c_ligattn_knn": "L1c (LigAttn+KNN)",
    "level1c_ligattn_mlp": "L1c (LigAttn+MLP)",
    "level2_meanpool_knn": "L2 (MeanPool+KNN)",
    "level2_meanpool_mlp": "L2 (MeanPool+MLP)",
    "level3_attnpool_knn": "L3 (AttnPool+KNN)",
    "level3_attnpool_mlp": "L3 (AttnPool+MLP)",
    "level4_crossatt_knn": "L4 (CrossAtt+KNN)",
    "level4_crossatt_mlp": "L4 (CrossAtt+MLP)",
    "level5_da_knn": "L5 (DA+KNN)",
    "level5_da_mlp": "L5 (DA+MLP)",
    "level5b_da_knn": "L5b (AttnPool+DA+KNN)",
    "level5b_da_mlp": "L5b (AttnPool+DA+MLP)",
    "level6a_ban_knn": "L6a (BAN+CrossAttn+KNN)",
    "level6a_ban_mlp": "L6a (BAN+CrossAttn+MLP)",
    "level6b_ban_knn": "L6b (BAN+KNN)",
    "level6b_ban_mlp": "L6b (BAN+MLP)",
}

# ---------------------------------------------------------------------------
# Plotting palette (colorblind-friendly)
# ---------------------------------------------------------------------------

LEVEL_COLORS: Dict[str, str] = {
    "level1a_fp_knn": "#1b9e77",
    "level1a_fp_mlp": "#66c2a5",
    "level1b_ligmean_knn": "#4daf4a",
    "level1b_ligmean_mlp": "#a6d854",
    "level1c_ligattn_knn": "#377eb8",
    "level1c_ligattn_mlp": "#7eb8da",
    "level2_meanpool_knn": "#7570b3",
    "level2_meanpool_mlp": "#a6a3d9",
    "level3_attnpool_knn": "#d95f02",
    "level3_attnpool_mlp": "#e78e3f",
    "level4_crossatt_knn": "#e7298a",
    "level4_crossatt_mlp": "#f06ab6",
    "level5_da_knn": "#e41a1c",
    "level5_da_mlp": "#fb6a4a",
    "level5b_da_knn": "#ff7f00",
    "level5b_da_mlp": "#fdbf6f",
    "level6a_ban_knn": "#984ea3",
    "level6a_ban_mlp": "#cab2d6",
    "level6b_ban_knn": "#a65628",
    "level6b_ban_mlp": "#d2a679",
}

# ---------------------------------------------------------------------------
# Valid levels
# ---------------------------------------------------------------------------

VALID_LEVELS = frozenset({"1a", "1b", "1c", "2", "3", "4", "5", "5b", "6a", "6b"})

# ---------------------------------------------------------------------------
# Activity threshold
# ---------------------------------------------------------------------------

PCHEMBL_ACTIVITY_THRESHOLD = 6.0  # IC50 <= 1000 nM


# ---------------------------------------------------------------------------
# BenchmarkConfig – immutable runtime configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable configuration object for a benchmark run.

    Constructed once from CLI arguments (or programmatic call) and
    threaded through every module.  The ``frozen=True`` prevents accidental
    mutation mid-pipeline.
    """

    # --- required ---
    dataset: str
    embedding: str  # shorthand: "8M", "150M", "650M"

    # --- level selection ---
    levels: List[str] = field(default_factory=lambda: ["1a", "1b", "1c", "2", "3", "4", "5", "5b", "6a", "6b"])

    # --- output ---
    output_dir: Optional[str] = None
    scaffold_split_dir: str = DEFAULT_SCAFFOLD_SPLIT_DIR

    # --- reproducibility ---
    seeds: Optional[List[int]] = None

    # --- flags ---
    force: bool = False
    force_split: bool = False
    debug: bool = False

    # --- deep-learning hyper-parameters (Level 4) ---
    epochs: int = 500
    batch_size: int = 32
    patience: int = 5
    learning_rate: float = 1e-4

    # --- fine-tuning ---
    finetune: bool = False
    use_finetuned: bool = False
    finetune_epochs: int = 100
    finetune_lr: float = 1e-5
    finetune_batch_size: int = 8

    # --- derived (computed in __post_init__) ---

    @property
    def embedding_name(self) -> str:
        """Full ESM-2 model name from shorthand."""
        return SUPPORTED_EMBEDDINGS[self.embedding]

    @property
    def resolved_output_dir(self) -> str:
        """Compute actual output directory."""
        if self.output_dir:
            return self.output_dir
        return f"./results/benchmark_{self.dataset}_{self.embedding}"

    @property
    def resolved_patience(self) -> Optional[int]:
        """Return ``None`` when patience is disabled (0)."""
        return self.patience if self.patience > 0 else None

    @property
    def resolved_seeds(self) -> List[int]:
        """Seeds to use, falling back to project defaults."""
        if self.seeds:
            return self.seeds
        try:
            from crossattention_split_analysis.config import DEFAULT_SEEDS
            return DEFAULT_SEEDS
        except ImportError:
            return [42, 123, 456, 789, 1024]

    def build_dir(self, dataset_type: str) -> str:
        """Return the embedding build directory for a given dataset type."""
        return os.path.join(
            EMBEDDING_BASE_PATH.format(dataset_type=dataset_type),
            self.embedding_name,
            "build",
        )

    def datasets_to_process(self) -> List[str]:
        """List of concrete datasets (expands 'all' → ['human', 'non_human'])."""
        if self.dataset == "all":
            return ["human", "non_human"]
        return [self.dataset]

    def __post_init__(self) -> None:
        """Validate configuration at construction time."""
        if self.embedding not in SUPPORTED_EMBEDDINGS:
            msg = (
                f"Unsupported embedding '{self.embedding}'. "
                f"Choose from: {list(SUPPORTED_EMBEDDINGS.keys())}"
            )
            raise ValueError(msg)
        if self.dataset not in {"human", "non_human", "all"}:
            msg = f"Invalid dataset '{self.dataset}'. Choose from: human, non_human, all"
            raise ValueError(msg)
        for level in self.levels:
            if level not in VALID_LEVELS:
                msg = f"Invalid level '{level}'. Valid levels: {sorted(VALID_LEVELS)}"
                raise ValueError(msg)
