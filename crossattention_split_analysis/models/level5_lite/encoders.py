"""Encoder modules for Level 5-Lite.

SIMPLIFIED VERSION: Removed redundant Transformer encoders.

ESM-2 and MoLFormer are already pre-trained Transformers with self-attention.
Adding another Transformer encoder on top is redundant and hurts performance.

This module now contains only simple linear projections to align dimensions.
The actual learning happens in the cross-attention layer.
"""

import torch
import torch.nn as nn


class ProteinEncoder(nn.Module):
    """Simple projection for ESM-2 per-residue embeddings.
    
    FIXED: Removed redundant Transformer encoder.
    
    Why this is correct:
    - ESM-2 already contains 33 self-attention layers (for 650M model)
    - ESM-2 was trained on 250M+ sequences
    - Adding more self-attention is redundant and computationally wasteful
    - We only need to project to the correct dimension for cross-attention
    
    This reduces parameters from ~6M to ~164K per encoder.
    """
    
    def __init__(
        self,
        input_dim: int = 320,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """Initialize ProteinEncoder.
        
        Args:
            input_dim: ESM-2 embedding dimension (320 for 8M, 640 for 150M, 1280 for 650M)
            hidden_dim: Hidden dimension for cross-attention
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple projection with normalization and dropout
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [batch, seq_len, input_dim] ESM-2 embeddings
            mask: [batch, seq_len] padding mask (unused, kept for API compatibility)
        
        Returns:
            [batch, seq_len, hidden_dim] projected representations
        """
        return self.proj(x)


class LigandEncoder(nn.Module):
    """Simple projection for MoLFormer per-token embeddings.
    
    FIXED: Removed redundant Transformer encoder.
    
    Why this is correct:
    - MoLFormer is a 1.1B parameter Transformer (RoBERTa architecture)
    - MoLFormer has 12 self-attention layers
    - Adding more self-attention is redundant
    - We only need to project to the correct dimension for cross-attention
    
    This reduces parameters from ~6M to ~393K per encoder.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """Initialize LigandEncoder.
        
        Args:
            input_dim: MoLFormer embedding dimension (768)
            hidden_dim: Hidden dimension for cross-attention
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple projection with normalization and dropout
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [batch, n_tokens, input_dim] MoLFormer embeddings
            mask: [batch, n_tokens] padding mask (unused, kept for API compatibility)
        
        Returns:
            [batch, n_tokens, hidden_dim] projected representations
        """
        return self.proj(x)
