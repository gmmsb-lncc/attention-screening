"""
Positional Encoding implementations for sequence models.

This module provides different positional encoding strategies:
1. Sinusoidal (Vaswani et al., 2017) - Fixed, additive encoding
2. RoPE (Su et al., 2021) - Rotary Position Embedding, unlimited sequence length

RoPE is the recommended choice for modern architectures as it:
- Supports sequences of any length without pre-allocation
- Preserves relative position information in attention
- Is used by state-of-the-art models (LLaMA, ESM-2, etc.)

References:
    - Vaswani et al. (2017). "Attention Is All You Need". NeurIPS.
    - Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding". arXiv:2104.09864.

Author: DockTKinase Team
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
    
    Adds fixed positional information to embeddings using sine/cosine functions.
    
    Limitations:
        - Requires pre-defined max_len
        - Positions beyond max_len cannot be encoded
        - Additive encoding can interfere with learned representations
    
    Args:
        d_model: Dimension of the model embeddings
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            [batch, seq_len, d_model] with positional encoding added
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}. "
                f"Use RoPE for unlimited sequence lengths."
            )
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - Su et al., 2021.
    
    RoPE encodes position information through rotation of query/key vectors,
    which preserves relative position information in the attention mechanism.
    
    Key advantages over sinusoidal encoding:
        1. No sequence length limit - computed on-the-fly
        2. Relative position awareness - improves attention patterns
        3. Better extrapolation - works on longer sequences than trained
        4. Multiplicative instead of additive - preserves embedding geometry
    
    Mathematical formulation:
        For position m and dimension 2i, 2i+1:
        RoPE(x, m) = [x_{2i} * cos(mθ_i) - x_{2i+1} * sin(mθ_i),
                      x_{2i} * sin(mθ_i) + x_{2i+1} * cos(mθ_i)]
        
        where θ_i = 10000^(-2i/d)
    
    Reference:
        Su et al. (2021). "RoFormer: Enhanced Transformer with 
        Rotary Position Embedding". arXiv:2104.09864.
    
    Args:
        dim: Dimension of the embeddings (must be even)
        base: Base for the frequency computation (default: 10000)
        max_seq_len_cached: Maximum sequence length to cache (default: 2048)
    """
    
    def __init__(
        self, 
        dim: int, 
        base: int = 10000,
        max_seq_len_cached: int = 2048
    ):
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")
        
        self.dim = dim
        self.base = base
        self.max_seq_len_cached = max_seq_len_cached
        
        # Compute inverse frequencies: θ_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        
        # Pre-compute for initial max length
        self._update_cos_sin_cache(max_seq_len_cached)
        
        logger.info(
            f"RoPE initialized: dim={dim}, base={base}, "
            f"max_cached={max_seq_len_cached}"
        )
    
    def _update_cos_sin_cache(self, seq_len: int) -> None:
        """Update the cached cos/sin values if sequence is longer than cache."""
        if seq_len <= self._seq_len_cached:
            return
        
        self._seq_len_cached = seq_len
        
        # Compute position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies: outer product of positions and inverse frequencies
        # freqs: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate for pairs: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Cache cos and sin: [1, seq_len, dim]
        self._cos_cached = emb.cos().unsqueeze(0)
        self._sin_cached = emb.sin().unsqueeze(0)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half of the dimensions.
        
        Splits x into two halves and rotates: [x1, x2] -> [-x2, x1]
        
        Args:
            x: [..., dim] tensor
            
        Returns:
            [..., dim] tensor with rotated pairs
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply rotary positional embedding.
        
        Args:
            x: [batch, seq_len, dim] or [batch, heads, seq_len, dim]
            seq_len: Optional explicit sequence length
            
        Returns:
            Tensor with same shape as input, with RoPE applied
        """
        if seq_len is None:
            # Infer seq_len from input shape
            if x.dim() == 3:
                seq_len = x.size(1)
            elif x.dim() == 4:
                seq_len = x.size(2)
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")
        
        # Update cache if needed
        self._update_cos_sin_cache(seq_len)
        
        # Get cached values
        cos = self._cos_cached[:, :seq_len, :].to(x.dtype)
        sin = self._sin_cached[:, :seq_len, :].to(x.dtype)
        
        # Handle different input shapes
        if x.dim() == 4:
            # [batch, heads, seq_len, dim] -> add head dimension to cos/sin
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        return (x * cos) + (self._rotate_half(x) * sin)
    
    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors for attention.
        
        This is the typical usage pattern in attention mechanisms.
        
        Args:
            q: Query tensor [batch, heads, seq_len, dim]
            k: Key tensor [batch, heads, seq_len, dim]
            seq_len: Optional explicit sequence length
            
        Returns:
            Tuple of (q_rotated, k_rotated) with same shapes
        """
        q_rotated = self.forward(q, seq_len)
        k_rotated = self.forward(k, seq_len)
        return q_rotated, k_rotated


class RoPEMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Position Embedding.
    
    Drop-in replacement for standard multi-head attention that uses RoPE
    instead of additive positional encoding.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear projections
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # RoPE - applied to head dimension
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with RoPE.
        
        Args:
            query: [batch, seq_q, d_model]
            key: [batch, seq_k, d_model]
            value: [batch, seq_k, d_model]
            mask: Optional attention mask
            
        Returns:
            [batch, seq_q, d_model]
        """
        batch_size = query.size(0)
        seq_q = query.size(1)
        seq_k = key.size(1)
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q, k = self.rope.apply_rotary_pos_emb(q, k)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back: [batch, seq_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_q, self.d_model
        )
        
        return self.out_proj(attn_output)


def create_positional_encoding(
    d_model: int,
    encoding_type: str = "rope",
    **kwargs
) -> nn.Module:
    """
    Factory function to create positional encoding.
    
    Args:
        d_model: Model dimension
        encoding_type: "rope" (recommended) or "sinusoidal"
        **kwargs: Additional arguments for the encoding
        
    Returns:
        Positional encoding module
    """
    if encoding_type == "rope":
        return RotaryPositionalEmbedding(dim=d_model, **kwargs)
    elif encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model=d_model, **kwargs)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


# Aliases for backward compatibility
PositionalEncoding = SinusoidalPositionalEncoding
RoPE = RotaryPositionalEmbedding
