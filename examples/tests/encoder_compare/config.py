"""Configuration and constants for encoder comparison experiments."""

from dataclasses import dataclass, field
from typing import Dict, Optional

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

LIGAND_DIM = 768


# =============================================================================
# PATHS
# =============================================================================

DATASET_PATHS = {
    'human': '/media/storage/leon/attention-screening/tests/datasets/kinase_human_compounds.tsv',
    'non_human': '/media/storage/leon/attention-screening/tests/datasets/kinase_non_human_compounds.tsv',
}

EMBEDDING_BASE_PATH = '/media/storage/leon/attention-screening/results/protein_model_benchmark_{dataset_type}_v2'


# =============================================================================
# ENCODER TYPES
# =============================================================================

ENCODER_TYPES = ['linear', 'cnn', 'cnn_attention']


# =============================================================================
# AFFINITY THRESHOLD CONFIGURATION
# =============================================================================

@dataclass
class AffinityThresholdConfig:
    """
    Configuration for converting continuous affinity values to binary labels.

    The threshold defines what constitutes an "active" compound:
    - standard_value <= threshold → active (label=1)
    - standard_value > threshold → inactive (label=0)

    Attributes:
        threshold_nm: Threshold in nanomolar (nM). Default: 1000 nM (1 μM).
        column_name: Name of the column containing affinity values.
        label_column: Name of the output label column.

    Scientific Justification:
        The 1000 nM (1 μM) threshold is commonly used in drug discovery as it
        represents a reasonable cutoff for "active" compounds in biochemical
        assays. This threshold is consistent with:
        - ChEMBL activity classification
        - FDA guidance for lead compound potency
        - Standard kinase inhibitor screening practices

        However, this threshold should be validated for your specific:
        - Assay type (Ki, Kd, IC50, EC50)
        - Target class (kinases vs other protein families)
        - Downstream application (screening vs optimization)

    References:
        - Merck Molecular Activity Challenge (Kaggle)
        - ChEMBL database documentation
        - Hughes et al. (2011) "Principles of early drug discovery"
    """
    threshold_nm: float = 1000.0  # 1 μM
    column_name: str = 'standard_value'
    label_column: str = 'label'

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'threshold_nm': self.threshold_nm,
            'threshold_uM': self.threshold_nm / 1000,
            'column_name': self.column_name,
            'label_column': self.label_column,
            'interpretation': f'active if {self.column_name} <= {self.threshold_nm} nM'
        }


# Default threshold configuration
DEFAULT_AFFINITY_THRESHOLD = AffinityThresholdConfig()


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Training hyperparameters.

    Attributes:
        protein_dim: Protein embedding dimension (depends on ESM model)
        ligand_dim: Ligand embedding dimension (768 for ChemBERTa)
        hidden_dim: Hidden dimension for all layers
        num_heads: Number of attention heads
        num_cross_attn_layers: Number of cross-attention layers
        ff_dim: Feed-forward layer dimension
        dropout: Dropout rate
        num_epochs: Maximum number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: AdamW weight decay
        max_grad_norm: Gradient clipping threshold
        affinity_threshold: Configuration for label generation
    """
    protein_dim: int = 640
    ligand_dim: int = 768
    hidden_dim: int = 256
    num_heads: int = 8
    num_cross_attn_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.1
    num_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 5e-5  # Reduced from 1e-4 for better stability
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient clipping to prevent exploding gradients
    affinity_threshold: AffinityThresholdConfig = field(
        default_factory=lambda: DEFAULT_AFFINITY_THRESHOLD
    )

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'protein_dim': self.protein_dim,
            'ligand_dim': self.ligand_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_cross_attn_layers': self.num_cross_attn_layers,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            'affinity_threshold': self.affinity_threshold.to_dict()
        }
