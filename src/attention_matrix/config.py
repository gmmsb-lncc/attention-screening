"""
Configuration for Attention Matrix Module.

Single Responsibility: Configuration management only.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class AttentionMatrixConfig:
    """
    Configuration for the Attention Matrix pipeline.
    
    Attributes:
        protein_dim: Dimension of protein embeddings (ESM2: 320, 480, 640, 1280, etc.)
        ligand_dim: Dimension of ligand embeddings (SMI-TED: 768)
        
        max_protein_len: Maximum protein sequence length (truncate/pad)
        max_ligand_len: Maximum ligand SMILES length (truncate/pad)
        
        hidden_dim: Hidden dimension for projections and attention
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        dropout: Dropout rate
        
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        epochs: Maximum training epochs
        early_stopping_patience: Early stopping patience
        
        activity_threshold: pChEMBL threshold for binary classification (default: 7.0 = 100nM)
        device: Compute device ('auto', 'cpu', 'cuda', 'mps')
        random_state: Random seed for reproducibility
    """
    
    # Embedding dimensions
    protein_dim: int = 320
    ligand_dim: int = 768
    
    # Sequence lengths
    max_protein_len: int = 256
    max_ligand_len: int = 64
    
    # Model architecture
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.2
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    early_stopping_patience: int = 10
    
    # Task configuration
    activity_threshold: float = 7.0  # pChEMBL threshold
    regression_weight: float = 0.7
    classification_weight: float = 0.3
    
    # Device and reproducibility
    device: str = "auto"
    random_state: int = 42
    
    # Data splitting
    test_size: float = 0.1
    val_size: float = 0.1
    n_protein_clusters: Optional[int] = None  # Auto if None
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate dimensions
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'AttentionMatrixConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def get_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device != 'auto':
            return self.device
        
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
