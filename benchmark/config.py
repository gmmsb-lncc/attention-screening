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

MOLFORMER_DIM: int = 768  # MoLFormer per-token embeddings (kept for backward compat)
CHEMBERTA_DIM: int = 384  # ChemBERTa-77M-MTR per-token embeddings

LIGAND_MODEL_DIMS: Dict[str, int] = {
    "molformer": 768,
    "smited": 768,
    "chemberta": 384,
}

LIGAND_MATRIX_DIRS: Dict[str, List[str]] = {
    "molformer": ["ligand_matrices", "molformer_matrix"],
    "smited": ["ligand_matrices"],
    "chemberta": ["chemberta_matrix"],
}

VALID_LIGAND_MODELS = frozenset({"molformer", "smited", "chemberta"})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EMBEDDING_BASE_PATH = "./results/protein_model_benchmark_{dataset_type}_v2"
DEFAULT_SCAFFOLD_SPLIT_DIR = "scaffolds_splits/output"
DEFAULT_SE3_FEATURES_SUBDIR = "se3_features"

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
    "level1b_ligmean_knn": "L1b (LigMeanPool+KNN)",
    "level1b_ligmean_mlp": "L1b (LigMeanPool+MLP)",
    "level1c_ligattn_knn": "L1c (LigAttnPool+KNN)",
    "level1c_ligattn_mlp": "L1c (LigAttnPool+MLP)",
    "level2_meanpool_knn": "L2 (MeanPool+KNN)",
    "level2_meanpool_mlp": "L2 (MeanPool+MLP)",
    "level3_attnpool_knn": "L3 (AttnPool+KNN)",
    "level3_attnpool_mlp": "L3 (AttnPool+MLP)",
    "level4_crossatt_knn": "L4 (CrossAttn+AttnPool+KNN)",
    "level4_crossatt_mlp": "L4 (CrossAttn+AttnPool+MLP)",
    "level5_da_knn": "L5a (CrossAttn+AttnPool+GRL+KNN)",
    "level5_da_mlp": "L5a (CrossAttn+AttnPool+GRL+MLP)",
    "level5b_da_knn": "L5b (AttnPool+GRL+KNN)",
    "level5b_da_mlp": "L5b (AttnPool+GRL+MLP)",
    "level6a_ban_knn": "L6a (CrossAttn+BAN+GRL+KNN)",
    "level6a_ban_mlp": "L6a (CrossAttn+BAN+GRL+MLP)",
    "level6b_ban_knn": "L6b (AttnPool+BAN+GRL+KNN)",
    "level6b_ban_mlp": "L6b (AttnPool+BAN+GRL+MLP)",
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

VALID_LEVELS = frozenset({"0", "1a", "1b", "1c", "2", "3"})
OBSOLETE_LEVELS = frozenset({"4", "5a", "5b", "6a", "6b"})

# Level 0 is a shortcut for the classical ML baseline subset
LEVEL_0_EXPANSION = ["1a", "1b", "1c", "3"]

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

    # --- ligand model ---
    ligand_model: str = "molformer"  # "molformer", "smited", or "chemberta"
    use_se3_ligand: bool = False
    se3_features_dir: Optional[str] = None

    # --- level selection ---
    levels: List[str] = field(default_factory=lambda: ["1a", "1b", "1c", "2", "3"])

    # --- output ---
    output_dir: Optional[str] = None
    scaffold_split_dir: str = DEFAULT_SCAFFOLD_SPLIT_DIR

    # --- reproducibility ---
    seeds: Optional[List[int]] = None

    # --- mode ---
    mode: str = "train"  # "train" = fit on train, eval on val; "test" = fit on val, eval on test

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
    def ligand_dim(self) -> int:
        """Ligand embedding dimension for the selected ligand model."""
        return LIGAND_MODEL_DIMS.get(self.ligand_model, 768)

    @property
    def resolved_output_dir(self) -> str:
        """Compute actual output directory.

        Always appends ``/{mode}`` so that train and test phases
        write to separate directories even when the user provides
        the same ``--output_dir``.
        """
        base = self.output_dir if self.output_dir else f"./results/benchmark_{self.dataset}_{self.embedding}"
        return os.path.join(base, self.mode)

    @property
    def resolved_se3_feature_dirs(self) -> List[str]:
        """Default SE3 feature directories following embedding build layout."""
        dirs: List[str] = []
        for ds in self.datasets_to_process():
            dirs.append(
                os.path.join(self.build_dir(ds), DEFAULT_SE3_FEATURES_SUBDIR)
            )
        return dirs

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
        """List of concrete datasets for embedding directory iteration.

        Expands ``'all'`` → ``['human', 'non_human']`` because embedding
        files are stored per-corpus.  Split loading always uses the
        universal scaffold files (see ``dataset_source_filter``).
        """
        if self.dataset == "all":
            return ["human", "non_human"]
        return [self.dataset]

    @property
    def dataset_source_filter(self) -> Optional[str]:
        """Filter value for the ``dataset_source`` column in universal splits.

        Returns ``None`` for ``'all'`` (no filtering — use every row),
        or the corpus name (``'human'`` / ``'non_human'``) otherwise.
        """
        if self.dataset == "all":
            return None
        return self.dataset

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
        if self.mode not in {"train", "test"}:
            msg = f"Invalid mode '{self.mode}'. Choose from: train, test"
            raise ValueError(msg)
        if self.ligand_model not in VALID_LIGAND_MODELS:
            msg = (
                f"Invalid ligand_model '{self.ligand_model}'. "
                f"Choose from: {sorted(VALID_LIGAND_MODELS)}"
            )
            raise ValueError(msg)
        if self.se3_features_dir is not None and not self.se3_features_dir.strip():
            msg = "--se3-features-dir cannot be empty when provided"
            raise ValueError(msg)
        for level in self.levels:
            if level in OBSOLETE_LEVELS:
                msg = (
                    f"Level '{level}' is obsolete and no longer supported. "
                    "Use levels up to 3: 0, 1a, 1b, 1c, 2, 3."
                )
                raise ValueError(msg)
            if level not in VALID_LEVELS:
                msg = f"Invalid level '{level}'. Valid levels: {sorted(VALID_LEVELS)}"
                raise ValueError(msg)
