"""
Pooling strategies for converting per-token embeddings into fixed-size vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns to weight tokens by importance.
    
    This is more sophisticated than mean pooling because it learns which
    parts of the sequence are most relevant for the downstream task.
    
    Architecture:
        1. Query vector (learnable parameter)
        2. Compute attention scores between query and all tokens
        3. Softmax to get weights
        4. Weighted sum of token embeddings
    """
    
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Dimension of the input embeddings
        """
        super().__init__()
        # Learnable query vector
        self.attention_query = nn.Parameter(torch.randn(hidden_dim))
        
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            embeddings: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len] - 1 for valid tokens, 0 for padding
            
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        # Compute attention scores: [batch_size, seq_len]
        attention_scores = torch.matmul(embeddings, self.attention_query)
        
        # Apply mask if provided (set padding to -inf before softmax)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask.bool(), float('-inf'))
        
        # Normalize to weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Weighted sum: [batch_size, hidden_dim]
        pooled = torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=1)
        
        return pooled


def attention_pooling_numpy(matrix: torch.Tensor, query: torch.Tensor = None) -> torch.Tensor:
    """
    Simple attention pooling for numpy arrays (inference mode).
    
    Args:
        matrix: [seq_len, hidden_dim] - per-token embeddings
        query: [hidden_dim] - attention query (optional, defaults to mean)
        
    Returns:
        vector: [hidden_dim] - pooled representation
    """
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.from_numpy(matrix)
    
    if len(matrix.shape) == 2:
        matrix = matrix.unsqueeze(0)  # [1, seq_len, hidden_dim]
    
    hidden_dim = matrix.shape[-1]
    
    # Use mean as query if not provided
    if query is None:
        query = matrix.mean(dim=1).squeeze(0)  # [hidden_dim]
    
    # Compute attention scores
    attention_scores = torch.matmul(matrix, query)  # [1, seq_len]
    attention_weights = F.softmax(attention_scores, dim=1)  # [1, seq_len]
    
    # Weighted sum
    pooled = torch.sum(matrix * attention_weights.unsqueeze(-1), dim=1)  # [1, hidden_dim]
    
    return pooled.squeeze(0)  # [hidden_dim]
