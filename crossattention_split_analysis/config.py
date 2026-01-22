"""Configuration and constants for CrossAttention split analysis."""

from dataclasses import dataclass, field
from typing import Dict

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

# Ligand embedding dimension (SMI-TED)
LIGAND_DIM = 768

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


# =============================================================================
# AFFINITY THRESHOLD
# =============================================================================

@dataclass
class AffinityThresholdConfig:
    """
    Configuration for converting continuous affinity to binary labels.

    The 1000 nM (1 μM) threshold is standard in kinase drug discovery.
    See encoder_compare/config.py for detailed scientific justification.
    """
    threshold_nm: float = 1000.0
    column_name: str = 'standard_value'
    label_column: str = 'label'

    def to_dict(self) -> Dict:
        return {
            'threshold_nm': self.threshold_nm,
            'threshold_uM': self.threshold_nm / 1000,
            'column_name': self.column_name,
            'interpretation': f'active if {self.column_name} <= {self.threshold_nm} nM'
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
        affinity_threshold: Configuration for label generation
    """
    # Model architecture
    protein_dim: int = 640
    ligand_dim: int = 768
    hidden_dim: int = 256
    num_cnn_layers: int = 3
    num_cross_attn_layers: int = 2
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 500
    patience: int = 30
    max_grad_norm: float = 1.0

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
            'num_cnn_layers': self.num_cnn_layers,
            'num_cross_attn_layers': self.num_cross_attn_layers,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'max_grad_norm': self.max_grad_norm,
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
    'new_compound_new_kinase': 'New Compound + New Kinase',
    'compound': 'Split by Compound',
    'random': 'Random Split'
}

DEFAULT_SCENARIOS = ['new_compound_new_kinase', 'compound', 'random']
