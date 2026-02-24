"""Configuration and constants for CrossAttention split analysis."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal

# =============================================================================
# EMBEDDINGS
# =============================================================================

SUPPORTED_EMBEDDINGS = {
    '8M': 'esm2_t6_8M_UR50D',
    '150M': 'esm2_t30_150M_UR50D',
    '650M': 'esm2_t33_650M_UR50D'
}

PROTEIN_DIMS = {
    'esm2_t6_8M_UR50D': 320,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t33_650M_UR50D': 1280
}

# Ligand embedding dimensions
LIGAND_DIM = 768  # SMI-TED per-token embeddings
MOLFORMER_DIM = 768  # MoLFormer per-token embeddings

# Ligand matrix directory names
LIGAND_MATRIX_DIRS = {
    'smited': 'ligand_matrices',      # Original SMI-TED matrices
    'molformer': 'molformer_matrix',  # MoLFormer matrices
}

# Max sequence length for attention matrices
MAX_SEQ_LEN = 1024


# =============================================================================
# PATHS
# =============================================================================

DATASET_PATHS = {
    'human': './tests/datasets/kinase_human_compounds.tsv',
    'non_human': './tests/datasets/kinase_non_human_compounds.tsv',
    'all': './tests/datasets/kinase_all_compounds.tsv'
}

EMBEDDING_BASE_PATH = './results/protein_model_benchmark_{dataset_type}_v2'

# For dataset 'all', we need to search in both human and non_human directories
# since the matrices are stored separately
EMBEDDING_BASE_PATHS_ALL = [
    './results/protein_model_benchmark_human_v2',
    './results/protein_model_benchmark_non_human_v2'
]


# =============================================================================
# AFFINITY THRESHOLD
# =============================================================================

@dataclass
class AffinityThresholdConfig:
    """
    Configuration for converting continuous affinity to binary labels.

    Uses pChEMBL values (logarithmic scale) instead of direct nM values.
    The threshold of 6.0 corresponds to 1000 nM (1 μM), which is standard in kinase drug discovery.

    Conversion: pChEMBL = -log10(IC50/Kd in M)
    - 1000 nM = 1e-6 M → pChEMBL = 6.0
    - Active: pChEMBL >= 6.0 (higher pChEMBL = lower concentration = higher affinity)

    See encoder_compare/config.py for detailed scientific justification.
    """
    threshold_pchembl: float = 6.0  # Equivalent to 1000 nM
    column_name: str = 'pchembl_value'
    label_column: str = 'label'

    def to_dict(self) -> Dict:
        # Calculate equivalent nM value: nM = 10^(9 - pChEMBL)
        equivalent_nm = 10 ** (9 - self.threshold_pchembl)
        return {
            'threshold_pchembl': self.threshold_pchembl,
            'threshold_nm_equivalent': equivalent_nm,
            'threshold_uM_equivalent': equivalent_nm / 1000,
            'column_name': self.column_name,
            'interpretation': f'active if {self.column_name} >= {self.threshold_pchembl} (equivalent to <= {equivalent_nm:.0f} nM)'
        }


DEFAULT_AFFINITY_THRESHOLD = AffinityThresholdConfig()


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Training configuration for CrossAttention model.

    Attributes:
        protein_dim: Protein embedding dimension (or MAX_SEQ_LEN for attention)
        ligand_dim: Ligand embedding dimension (768 for SMI-TED)
        hidden_dim: Hidden dimension for all layers
        model_variant: Model variant ('cnn_crossattn', 'cross_attention_lite', or 'diffusion')
        num_cnn_layers: Number of CNN encoder layers
        num_cross_attn_layers: Number of cross-attention layers
        num_heads: Number of attention heads
        ff_dim: Feed-forward layer dimension
        dropout: Dropout rate
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: AdamW weight decay
        num_epochs: Maximum training epochs
        patience: Early stopping patience (epochs without improvement)
        classification_weight: Weight for BCE classification loss
        regression_weight: Weight for MSE regression loss
        optimize_threshold: If True, calibrate decision threshold on validation
        threshold_metric: Metric optimized for threshold selection
        fixed_threshold: Fixed threshold used when optimize_threshold=False
        affinity_threshold: Configuration for label generation
    """
    # Model architecture
    protein_dim: int = 640
    ligand_dim: int = 768
    hidden_dim: int = 256
    model_variant: Literal['cnn_crossattn', 'cross_attention_lite', 'diffusion'] = 'cnn_crossattn'
    num_cnn_layers: int = 3
    num_cross_attn_layers: int = 2
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1

    # Diffusion-specific (used only when model_variant='diffusion')
    diffusion_steps: int = 200
    diffusion_beta_start: float = 1e-4
    diffusion_beta_end: float = 0.02
    diffusion_layers: int = 4
    diffusion_loss_weight: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 500
    patience: Optional[int] = 30
    max_grad_norm: float = 1.0
    classification_weight: float = 1.0
    regression_weight: float = 0.5
    optimize_threshold: bool = True
    threshold_metric: str = 'mcc'
    fixed_threshold: float = 0.5

    # Reproducibility
    affinity_threshold: AffinityThresholdConfig = field(
        default_factory=lambda: DEFAULT_AFFINITY_THRESHOLD
    )

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'protein_dim': self.protein_dim,
            'ligand_dim': self.ligand_dim,
            'hidden_dim': self.hidden_dim,
            'model_variant': self.model_variant,
            'num_cnn_layers': self.num_cnn_layers,
            'num_cross_attn_layers': self.num_cross_attn_layers,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'diffusion_steps': self.diffusion_steps,
            'diffusion_beta_start': self.diffusion_beta_start,
            'diffusion_beta_end': self.diffusion_beta_end,
            'diffusion_layers': self.diffusion_layers,
            'diffusion_loss_weight': self.diffusion_loss_weight,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'max_grad_norm': self.max_grad_norm,
            'classification_weight': self.classification_weight,
            'regression_weight': self.regression_weight,
            'optimize_threshold': self.optimize_threshold,
            'threshold_metric': self.threshold_metric,
            'fixed_threshold': self.fixed_threshold,
            'affinity_threshold': self.affinity_threshold.to_dict()
        }


# =============================================================================
# STATISTICAL CONFIGURATION
# =============================================================================

MIN_SEEDS_FOR_STATISTICS = 3
RECOMMENDED_SEEDS_FOR_PUBLICATION = 5
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds for statistical rigor


# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================

AVAILABLE_SCENARIOS = {
    'scaffold': 'Split by Scaffold'
}

DEFAULT_SCENARIOS = ['scaffold']
