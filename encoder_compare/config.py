"""Configuration and constants for encoder comparison experiments."""

from dataclasses import dataclass
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

LIGAND_DIM = 768


# =============================================================================
# PATHS
# =============================================================================

DATASET_PATHS = {
    'human': '/media/storage/leon/semantic-screening/tests/datasets/kinase_human_compounds.tsv',
    'non_human': '/media/storage/leon/semantic-screening/tests/datasets/kinase_non_human_compounds.tsv',
}

EMBEDDING_BASE_PATH = '/media/storage/leon/semantic-screening/results/protein_model_benchmark_{dataset_type}_v2'


# =============================================================================
# ENCODER TYPES
# =============================================================================

ENCODER_TYPES = ['linear', 'cnn', 'cnn_attention']


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    protein_dim: int = 640
    ligand_dim: int = 768
    hidden_dim: int = 256
    num_heads: int = 8
    num_cross_attn_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.1
    num_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient clipping to prevent exploding gradients
