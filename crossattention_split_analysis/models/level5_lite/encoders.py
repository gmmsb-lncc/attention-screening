"""Encoder modules for Level 5-Lite.

Contains:
- ProteinEncoder: Transforms ESM-2 per-residue embeddings
- LigandEncoder: Transforms MoLFormer per-token embeddings
"""

import torch
import torch.nn as nn


class ProteinEncoder(nn.Module):
    """Encoder for ESM-2 per-residue embeddings.
    
    ESM-2 already encodes evolutionary and structural information.
    The Transformer encoder refines representations for binding prediction.
    
    Scientific justification:
    - ESM-2 was trained on 250M+ sequences (general knowledge)
    - Transformer encoder specializes for binding (light fine-tuning)
    - Self-attention captures long-range dependencies in the sequence
    """
    
    def __init__(
        self,
        input_dim: int = 320,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize ProteinEncoder.
        
        Args:
            input_dim: ESM-2 embedding dimension (320 for 8M, 640 for 150M, 1280 for 650M)
            hidden_dim: Hidden dimension for transformer layers
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear projection to uniform dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [batch, seq_len, input_dim] ESM-2 embeddings
            mask: [batch, seq_len] padding mask (True = pad token)
        
        Returns:
            [batch, seq_len, hidden_dim] encoded representations
        """
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return x


class LigandEncoder(nn.Module):
    """Encoder for MoLFormer per-token embeddings.
    
    MoLFormer already encodes molecular chemistry (1.1B parameters).
    The Transformer encoder refines representations for binding prediction.
    
    Scientific justification:
    - MoLFormer was trained on 2M+ molecules (chemical knowledge)
    - Self-attention between SMILES tokens captures local dependencies
    - No need for GNN because MoLFormer already understands structure
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize LigandEncoder.
        
        Args:
            input_dim: MoLFormer embedding dimension (768)
            hidden_dim: Hidden dimension for transformer layers
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [batch, n_tokens, input_dim] MoLFormer embeddings
            mask: [batch, n_tokens] padding mask (True = pad token)
        
        Returns:
            [batch, n_tokens, hidden_dim] encoded representations
        """
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return x
