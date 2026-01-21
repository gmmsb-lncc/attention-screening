"""
Encoder Variants for Protein-Ligand Affinity Prediction.

This module provides different encoder architectures to process per-token
embeddings before the cross-attention mechanism.

Variants:
    1. LinearEncoder: Simple projection (baseline)
    2. CNNEncoder: Convolutional encoder (current default)
    3. CNNAttentionEncoder: CNN + Local Self-Attention (hybrid)

Author: DockTKinase Team
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# VARIANT 1: LINEAR ENCODER (Baseline)
# =============================================================================

class LinearEncoder(nn.Module):
    """
    Simple linear projection encoder.

    This is a baseline that just projects the input embeddings to hidden_dim.
    Lets the cross-attention do most of the work.

    Architecture:
        Input [batch, seq_len, input_dim]
        → Linear(input_dim, hidden_dim)
        → LayerNorm
        → Dropout
        → Output [batch, seq_len, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        **kwargs  # Ignore extra args for compatibility
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = "linear"

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: [batch, seq_len] padding mask (1=valid, 0=padding)
        Returns:
            [batch, seq_len, hidden_dim]
        """
        out = self.projection(x)

        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return out


# =============================================================================
# VARIANT 2: CNN ENCODER (Current Default)
# =============================================================================

class Conv1DBlock(nn.Module):
    """1D Convolutional block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_residual = (in_channels == out_channels)
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x)

        if self.use_residual:
            out = out + residual

        return self.dropout(self.activation(out))


class CNNEncoder(nn.Module):
    """
    CNN encoder for extracting local patterns.

    Architecture:
        Input [batch, seq_len, input_dim]
        → Linear projection to hidden_dim
        → Multi-scale Conv1D blocks (kernel sizes 3, 5, 7)
        → LayerNorm
        → Output [batch, seq_len, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = "cnn"

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-scale CNN layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            kernel_size = kernel_sizes[i % len(kernel_sizes)]
            self.conv_layers.append(
                Conv1DBlock(hidden_dim, hidden_dim, kernel_size, dropout)
            )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]

        # Transpose for conv1d: [batch, hidden_dim, seq_len]
        x = x.transpose(1, 2)

        # Apply CNN layers
        for conv in self.conv_layers:
            x = conv(x)

        # Transpose back: [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)

        # Apply mask
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return self.layer_norm(x)


# =============================================================================
# VARIANT 3: CNN + LOCAL ATTENTION ENCODER (Hybrid)
# =============================================================================

class LocalSelfAttention(nn.Module):
    """
    Local self-attention with window-based attention pattern.

    More efficient than full self-attention, focuses on local context.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        window_size: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len]
        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply local window mask (optional optimization for very long sequences)
        if seq_len > self.window_size * 2:
            # Create local attention mask
            local_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            for i in range(seq_len):
                start = max(0, i - self.window_size)
                end = min(seq_len, i + self.window_size + 1)
                local_mask[i, start:end] = False
            attn_scores = attn_scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply padding mask
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = torch.matmul(attn_probs, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)

        return out


class CNNAttentionEncoder(nn.Module):
    """
    Hybrid CNN + Local Self-Attention encoder.

    Combines CNN for local pattern extraction with self-attention
    for capturing longer-range dependencies.

    Architecture:
        Input [batch, seq_len, input_dim]
        → Linear projection
        → CNN blocks (local patterns)
        → Local Self-Attention (longer-range)
        → LayerNorm
        → Output [batch, seq_len, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_cnn_layers: int = 2,
        num_attention_layers: int = 1,
        num_heads: int = 4,
        kernel_sizes: Tuple[int, ...] = (3, 5),
        window_size: int = 64,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = "cnn_attention"

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # CNN layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_cnn_layers):
            kernel_size = kernel_sizes[i % len(kernel_sizes)]
            self.conv_layers.append(
                Conv1DBlock(hidden_dim, hidden_dim, kernel_size, dropout)
            )

        # Self-attention layers
        self.attention_layers = nn.ModuleList()
        self.attention_norms = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.ff_norms = nn.ModuleList()

        for _ in range(num_attention_layers):
            self.attention_layers.append(
                LocalSelfAttention(hidden_dim, num_heads, window_size, dropout)
            )
            self.attention_norms.append(nn.LayerNorm(hidden_dim))

            # Feed-forward
            self.ff_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ))
            self.ff_norms.append(nn.LayerNorm(hidden_dim))

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project input
        x = self.input_proj(x)

        # CNN layers
        x = x.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        # Self-attention layers with residual connections
        for attn, attn_norm, ff, ff_norm in zip(
            self.attention_layers, self.attention_norms,
            self.ff_layers, self.ff_norms
        ):
            # Self-attention with residual
            residual = x
            x = attn_norm(x)
            x = residual + attn(x, mask)

            # Feed-forward with residual
            residual = x
            x = ff_norm(x)
            x = residual + ff(x)

        x = self.final_norm(x)

        # Apply mask
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

ENCODER_REGISTRY = {
    'linear': LinearEncoder,
    'cnn': CNNEncoder,
    'cnn_attention': CNNAttentionEncoder
}


def create_encoder(
    encoder_type: Literal['linear', 'cnn', 'cnn_attention'],
    input_dim: int,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create encoder by type.

    Args:
        encoder_type: One of 'linear', 'cnn', 'cnn_attention'
        input_dim: Input embedding dimension
        hidden_dim: Output hidden dimension
        dropout: Dropout rate
        **kwargs: Additional encoder-specific arguments

    Returns:
        Encoder module
    """
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available: {list(ENCODER_REGISTRY.keys())}")

    encoder_class = ENCODER_REGISTRY[encoder_type]
    return encoder_class(input_dim, hidden_dim, dropout=dropout, **kwargs)
