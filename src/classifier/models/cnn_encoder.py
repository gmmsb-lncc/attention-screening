"""
CNN Encoders for Protein and Ligand Embedding Processing.

This module contains CNN encoder architectures for processing per-token
embeddings from protein language models (ESM-2) and ligand encoders (SMI-TED).

Encoders:
    - CNNEncoder: Standard encoder with multi-scale kernels
    - OptimizedCNNEncoder: Efficient encoder with dilated depthwise separable convs

Design Rationale:
    The CNN encoder extracts LOCAL patterns (motifs) from sequence embeddings.
    ESM-2 already captures GLOBAL context via self-attention, so the CNN
    provides complementary local feature extraction.

Scientific References:
    - Dilated Convolutions: Yu & Koltun (2016). ICLR.
    - Depthwise Separable: Chollet (2017). CVPR.
    - Squeeze-and-Excitation: Hu et al. (2018). CVPR.
    - Pre-LayerNorm: Xiong et al. (2020). ICML.
    - Residual Learning: He et al. (2016). CVPR.

Author: DockTKinase Team
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

from .cnn_blocks import Conv1DBlock, OptimizedConv1DBlock

logger = logging.getLogger(__name__)


class CNNEncoder(nn.Module):
    """
    Standard CNN encoder for extracting local patterns from embedding matrices.
    
    Architecture:
        Input [batch, seq_len, embed_dim]
            │
            ▼
        Linear Projection (embed_dim → hidden_dim)
            │
            ▼
        Conv1DBlock (kernel=3) + Residual
            │
            ▼
        Conv1DBlock (kernel=5) + Residual
            │
            ▼
        Conv1DBlock (kernel=7) + Residual
            │
            ▼
        LayerNorm
            │
            ▼
        Output [batch, seq_len, hidden_dim]
    
    Receptive Field:
        With kernels (3, 5, 7) and 2 convs per block:
        RF = 1 + 2×(3-1) + 2×(5-1) + 2×(7-1) = 25 positions
    
    Args:
        input_dim: Input embedding dimension (e.g., 2560 for ESM-2 3B)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of conv blocks (default: 3)
        kernel_sizes: Tuple of kernel sizes for each layer (default: (3, 5, 7))
        dropout: Dropout probability (default: 0.1)
    
    Example:
        >>> encoder = CNNEncoder(input_dim=2560, hidden_dim=256)
        >>> protein_emb = torch.randn(32, 500, 2560)  # [batch, seq_len, esm_dim]
        >>> features = encoder(protein_emb)  # [32, 500, 256]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-scale CNN layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            kernel_size = kernel_sizes[i % len(kernel_sizes)]
            self.conv_layers.append(
                Conv1DBlock(hidden_dim, hidden_dim, kernel_size, dropout)
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Calculate receptive field
        self._receptive_field = self._calculate_receptive_field(kernel_sizes)
        
        logger.debug(
            f"CNNEncoder: input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"layers={num_layers}, receptive_field={self._receptive_field}"
        )
    
    def _calculate_receptive_field(self, kernel_sizes: Tuple[int, ...]) -> int:
        """Calculate total receptive field of the encoder."""
        rf = 1
        for i in range(self.num_layers):
            k = kernel_sizes[i % len(kernel_sizes)]
            rf += 2 * (k - 1)  # 2 convs per block
        return rf
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode sequence embeddings.
        
        Args:
            x: Input embeddings [batch, seq_len, input_dim]
            mask: Padding mask [batch, seq_len], 1=valid, 0=padding
            
        Returns:
            Encoded features [batch, seq_len, hidden_dim]
        """
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # Transpose for conv1d: [batch, hidden_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Transpose back: [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        return self.layer_norm(x)
    
    @property
    def receptive_field(self) -> int:
        """Get the receptive field size."""
        return self._receptive_field
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class OptimizedCNNEncoder(nn.Module):
    """
    Optimized CNN encoder with dilated depthwise separable convolutions.
    
    Key improvements over CNNEncoder:
        1. Dilated convolutions: Exponentially growing receptive field
        2. Depthwise separable: ~3x fewer parameters per layer
        3. Squeeze-and-Excitation: Adaptive channel recalibration
        4. Pre-LayerNorm: More stable training dynamics
    
    Architecture:
        Input [batch, seq_len, embed_dim]
            │
            ▼
        Linear + LayerNorm + GELU (embed_dim → hidden_dim)
            │
            ▼
        OptimizedConv1DBlock (dilation=1) ← RF grows by 2×(k-1)×1
            │
            ▼
        OptimizedConv1DBlock (dilation=2) ← RF grows by 2×(k-1)×2
            │
            ▼
        OptimizedConv1DBlock (dilation=4) ← RF grows by 2×(k-1)×4
            │
            ▼
        LayerNorm
            │
            ▼
        Output [batch, seq_len, hidden_dim]
    
    Receptive Field Comparison (kernel=3):
        - Standard (no dilation):    1 + 4 + 4 + 4 = 13
        - Dilated (d=1,2,4):         1 + 4 + 8 + 16 = 29
        - Dilated (d=1,2,4,8):       1 + 4 + 8 + 16 + 32 = 61
    
    Parameter Comparison (hidden_dim=256, kernel=3):
        - Standard Conv: 256 × 256 × 3 × 2 = 393,216 per block
        - Depthwise Sep: (256 × 3 + 256 × 256) × 2 = 132,608 per block
        - Reduction: ~66%
    
    Args:
        input_dim: Input embedding dimension
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of conv blocks (default: 3)
        kernel_size: Convolution kernel size (default: 3)
        dilations: Dilation factors per layer (default: (1, 2, 4))
        dropout: Dropout probability (default: 0.1)
        use_se: Use Squeeze-and-Excitation (default: True)
    
    Example:
        >>> encoder = OptimizedCNNEncoder(input_dim=2560, hidden_dim=256)
        >>> protein_emb = torch.randn(32, 500, 2560)
        >>> features = encoder(protein_emb)  # [32, 500, 256]
        >>> print(f"Parameters: {encoder.count_parameters():,}")
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        use_se: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.use_se = use_se
        
        # Input projection with normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Dilated convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilations[i % len(dilations)]
            self.conv_layers.append(
                OptimizedConv1DBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_se=use_se
                )
            )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # Calculate and store receptive field
        self._receptive_field = self._calculate_receptive_field(kernel_size, dilations)
        self._dilations = dilations
        
        logger.info(
            f"OptimizedCNNEncoder: input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"layers={num_layers}, dilations={dilations}, "
            f"receptive_field={self._receptive_field}, use_se={use_se}"
        )
    
    def _calculate_receptive_field(
        self, 
        kernel_size: int, 
        dilations: Tuple[int, ...]
    ) -> int:
        """
        Calculate total receptive field with dilations.
        
        For each block with 2 convolutions:
            RF_increase = 2 × (kernel_size - 1) × dilation
        """
        rf = 1
        for i in range(self.num_layers):
            d = dilations[i % len(dilations)]
            rf += 2 * (kernel_size - 1) * d
        return rf
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode sequence embeddings with optimized convolutions.
        
        Args:
            x: Input embeddings [batch, seq_len, input_dim]
            mask: Padding mask [batch, seq_len], 1=valid, 0=padding
            
        Returns:
            Encoded features [batch, seq_len, hidden_dim]
        """
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # Transpose for conv1d: [batch, hidden_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply dilated conv layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Transpose back: [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        return self.output_norm(x)
    
    @property
    def receptive_field(self) -> int:
        """Get the receptive field size."""
        return self._receptive_field
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> dict:
        """Get encoder configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'kernel_size': self.kernel_size,
            'dilations': self._dilations,
            'use_se': self.use_se,
            'receptive_field': self._receptive_field,
            'parameters': self.count_parameters()
        }


def create_encoder(
    input_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.1,
    optimized: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create CNN encoder.
    
    Args:
        input_dim: Input embedding dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout probability
        optimized: Whether to use optimized encoder
        **kwargs: Additional arguments passed to encoder
        
    Returns:
        CNNEncoder or OptimizedCNNEncoder instance
    
    Example:
        >>> # Standard encoder
        >>> encoder = create_encoder(2560, optimized=False)
        
        >>> # Optimized encoder
        >>> encoder = create_encoder(2560, optimized=True, use_se=True)
    """
    if optimized:
        return OptimizedCNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
    else:
        return CNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
