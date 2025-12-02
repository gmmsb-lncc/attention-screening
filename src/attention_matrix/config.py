"""
Configuration for Attention Matrix Module.

Single Responsibility: Configuration management only.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class AttentionMatrixConfig:
    """
    Configuration for the Attention Matrix pipeline.
    
    Attributes:
        protein_model: Name of ESM model (determines protein_dim and max_protein_len)
        protein_dim: Dimension of protein embeddings (auto-detected from protein_model)
        ligand_dim: Dimension of ligand embeddings (SMI-TED: 768)
        
        max_protein_len: Maximum protein sequence length (auto-detected from protein_model)
        max_ligand_len: Maximum ligand SMILES length (truncate/pad)
        
        hidden_dim: Hidden dimension for projections and attention
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        dropout: Dropout rate
        
        positional_encoding_type: Type of positional encoding ('sinusoidal' or 'rope')
        
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        epochs: Maximum training epochs
        early_stopping_patience: Early stopping patience
        
        activity_threshold: pChEMBL threshold for binary classification (default: 7.0 = 100nM)
        device: Compute device ('auto', 'cpu', 'cuda', 'mps')
        random_state: Random seed for reproducibility
    """
    
    # ESM Model (determines protein_dim and max_protein_len automatically)
    protein_model: str = 'esm2_t6_8M_UR50D'
    
    # Embedding dimensions (auto-detected if protein_model is set)
    protein_dim: Optional[int] = None  # Will be set from protein_model
    ligand_dim: int = 768
    
    # Sequence lengths (auto-detected if protein_model is set)
    max_protein_len: Optional[int] = None  # Will be set from protein_model
    max_ligand_len: int = 512  # SMI-TED max tokens
    
    # Model architecture
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.2
    
    # Positional encoding
    positional_encoding_type: str = 'rope'  # 'sinusoidal' or 'rope' (rope recommended)
    
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
        """Validate and auto-configure from protein_model."""
        # Auto-detect dimensions from protein_model
        self._configure_from_protein_model()
        
        # Validate dimensions
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        
        # Validate positional encoding type
        valid_pe_types = ['sinusoidal', 'rope']
        if self.positional_encoding_type not in valid_pe_types:
            raise ValueError(
                f"positional_encoding_type must be one of {valid_pe_types}, "
                f"got '{self.positional_encoding_type}'"
            )
    
    def _configure_from_protein_model(self):
        """Auto-configure protein_dim and max_protein_len from protein_model."""
        try:
            from src.build.core.constants import get_esm_model_info
            model_info = get_esm_model_info(self.protein_model)
            
            # Only set if not explicitly provided
            if self.protein_dim is None:
                self.protein_dim = model_info['dim']
            
            if self.max_protein_len is None:
                self.max_protein_len = model_info['max_len']
                
        except (ImportError, ValueError):
            # Fallback to defaults if constants not available
            if self.protein_dim is None:
                self.protein_dim = 320
            if self.max_protein_len is None:
                self.max_protein_len = 1024
    
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
