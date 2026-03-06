"""Level 5-Lite: Cross-Attention with Pre-calculated Embeddings.

This module implements a simplified version of Level 5 that:
- Uses pre-calculated MoLFormer + ESM-2 embeddings
- Applies Transformer encoders + Bidirectional Cross-Attention
- Eliminates complex SMILES-to-atoms alignment

Target: MCC 0.48-0.54 (vs Level 1 baseline: 0.428)
"""

from .model import Level5LiteModel
from .encoders import ProteinEncoder, LigandEncoder
from .attention import BidirectionalCrossAttention, AttentionPooling
from .classifier import ClassifierHead
from .dataset import Level5LiteDataset, collate_level5_lite, create_level5_lite_dataloaders
from .trainer import (
    Level5LiteConfig,
    Level5LiteTrainer,
    train_level5_lite,
    compute_metrics,
    optimize_threshold,
)

__all__ = [
    # Model
    "Level5LiteModel",
    "ProteinEncoder",
    "LigandEncoder",
    "BidirectionalCrossAttention",
    "AttentionPooling",
    "ClassifierHead",
    # Dataset
    "Level5LiteDataset",
    "collate_level5_lite",
    "create_level5_lite_dataloaders",
    # Training
    "Level5LiteConfig",
    "Level5LiteTrainer",
    "train_level5_lite",
    "compute_metrics",
    "optimize_threshold",
]
