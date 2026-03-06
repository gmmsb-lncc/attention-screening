"""Attention Pooling Layer for aggregating token embeddings into fixed-size vectors.

Instead of simple mean pooling, this uses learned attention weights to create
context-aware pooled representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Learned attention pooling to aggregate sequence embeddings.
    
    Args:
        input_dim: Dimension of input embeddings
        hidden_dim: Dimension of attention hidden layer (default: input_dim // 2)
        dropout: Dropout rate (default: 0.1)
    
    Shape:
        Input: (batch_size, seq_len, input_dim)
        Mask: (batch_size, seq_len) - 1 for valid tokens, 0 for padding
        Output: (batch_size, input_dim)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim // 2
        
        # Two-layer attention network
        self.attention = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, seq_len, dim)
            mask: (batch, seq_len) - 1 for valid, 0 for padding
            
        Returns:
            pooled: (batch, dim) - weighted average of embeddings
        """
        # Compute attention scores: (batch, seq_len, 1)
        attn_scores = self.attention(embeddings)
        
        # Apply mask if provided (set padding to -inf before softmax)
        if mask is not None:
            # Expand mask: (batch, seq_len) -> (batch, seq_len, 1)
            mask = mask.unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax over sequence length: (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted sum: (batch, dim)
        pooled = (embeddings * attn_weights).sum(dim=1)
        
        return pooled
