"""
Optimized CNN Building Blocks for Sequence Embedding Processing.

This module contains modular CNN components designed for processing
protein and ligand embeddings from language models (ESM-2, SMI-TED).

Components:
    - SqueezeExcitation: Channel recalibration (Hu et al., 2018)
    - DepthwiseSeparableConv1d: Efficient convolution (Chollet, 2017)
    - OptimizedConv1DBlock: Pre-LayerNorm + SE + Depthwise Separable

Design Principles:
    - Single Responsibility: Each class has one purpose
    - Open/Closed: Extensible without modification
    - Dependency Inversion: Depend on abstractions (nn.Module)

References:
    - Hu et al. (2018). "Squeeze-and-Excitation Networks". CVPR.
    - Chollet (2017). "Xception: Deep Learning with Depthwise Separable Convolutions". CVPR.
    - Xiong et al. (2020). "On Layer Normalization in the Transformer Architecture". ICML.
    - He et al. (2016). "Identity Mappings in Deep Residual Networks". ECCV.

Author: DockTKinase Team
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for adaptive channel recalibration.
    
    Learns to emphasize informative channels and suppress less useful ones
    through a global pooling → FC → FC → sigmoid pathway.
    
    Reference:
        Hu et al. (2018). "Squeeze-and-Excitation Networks". CVPR.
        "SE blocks adaptively recalibrate channel-wise feature responses
        by explicitly modelling interdependencies between channels."
    
    Why use for embeddings:
        ESM-2 embeddings have 2560 dimensions encoding different properties
        (structure, hydrophobicity, conservation). SE learns which dimensions
        are important for each sample adaptively.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 4)
    
    Example:
        >>> se = SqueezeExcitation(256, reduction=4)
        >>> x = torch.randn(32, 256, 100)  # [batch, channels, seq_len]
        >>> out = se(x)  # Same shape, recalibrated channels
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        
        # Bottleneck dimension (minimum 8 to avoid too aggressive reduction)
        reduced_channels = max(channels // reduction, 8)
        
        # Squeeze: Global Average Pooling
        # Excitation: FC → GELU → FC → Sigmoid
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.GELU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
        self._channels = channels
        self._reduction = reduction
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply squeeze-and-excitation.
        
        Args:
            x: Input tensor [batch, channels, seq_len]
            
        Returns:
            Recalibrated tensor [batch, channels, seq_len]
        """
        batch_size = x.size(0)
        
        # Squeeze: [batch, channels, seq_len] → [batch, channels, 1] → [batch, channels]
        squeezed = self.squeeze(x).view(batch_size, -1)
        
        # Excitation: [batch, channels] → [batch, channels]
        scale = self.excitation(squeezed)
        
        # Scale: [batch, channels, 1] * [batch, channels, seq_len]
        return x * scale.unsqueeze(-1)
    
    def extra_repr(self) -> str:
        return f'channels={self._channels}, reduction={self._reduction}'


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable 1D Convolution for efficient feature extraction.
    
    Factorizes standard convolution into:
    1. Depthwise: Spatial convolution per channel (groups=in_channels)
    2. Pointwise: 1x1 convolution to mix channels
    
    Parameter reduction:
        Standard Conv: in_channels × out_channels × kernel_size
        Depthwise Sep: in_channels × kernel_size + in_channels × out_channels
        Ratio: ~kernel_size times fewer parameters
    
    Reference:
        Chollet (2017). "Xception: Deep Learning with Depthwise Separable Convolutions". CVPR.
        "Depthwise separable convolutions [...] lead to both a better accuracy and
        a lower parameter count than Inception V3."
    
    Why use for embeddings:
        - ESM-2 dimensions encode orthogonal properties → process spatially first
        - Reduces overfitting on small drug discovery datasets
        - ~3x fewer parameters per layer
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        dilation: Dilation factor (default: 1)
        padding: Padding size (auto-calculated if None)
        bias: Whether to use bias (default: False for efficiency)
    
    Example:
        >>> conv = DepthwiseSeparableConv1d(256, 256, kernel_size=3, dilation=2)
        >>> x = torch.randn(32, 256, 100)
        >>> out = conv(x)  # [32, 256, 100]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        padding: Optional[int] = None,
        bias: bool = False
    ):
        super().__init__()
        
        # Auto-calculate padding for 'same' output size
        if padding is None:
            padding = (kernel_size - 1) * dilation // 2
        
        # Depthwise: each channel convolved separately
        self.depthwise = nn.Conv1d(
            in_channels, 
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Key: makes it depthwise
            bias=bias
        )
        
        # Pointwise: 1x1 convolution to mix channels
        self.pointwise = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=1,
            bias=bias
        )
        
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._dilation = dilation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply depthwise separable convolution.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            
        Returns:
            Output tensor [batch, out_channels, seq_len]
        """
        return self.pointwise(self.depthwise(x))
    
    def extra_repr(self) -> str:
        return (f'in_channels={self._in_channels}, out_channels={self._out_channels}, '
                f'kernel_size={self._kernel_size}, dilation={self._dilation}')


class OptimizedConv1DBlock(nn.Module):
    """
    Optimized 1D Convolutional block combining best practices:
    
    1. Pre-LayerNorm: More stable gradients (Xiong et al., 2020)
    2. Depthwise Separable: Fewer parameters (Chollet, 2017)
    3. Squeeze-and-Excitation: Adaptive channels (Hu et al., 2018)
    4. Residual Connection: Better gradient flow (He et al., 2016)
    
    Architecture:
        Input ─────────────────────────────────────┐
          │                                        │ (residual)
          ▼                                        │
        GroupNorm ──► DepthwiseSepConv ──► GELU    │
          │                                        │
          ▼                                        │
        GroupNorm ──► DepthwiseSepConv             │
          │                                        │
          ▼                                        │
        SqueezeExcitation (optional)               │
          │                                        │
          └────────────────── + ◄──────────────────┘
          │
          ▼
        GELU ──► Dropout ──► Output
    
    Note: Uses GroupNorm(1, channels) instead of LayerNorm to avoid transposes.
          GroupNorm with num_groups=1 is equivalent to LayerNorm.
    
    Args:
        channels: Number of channels (in = out for residual)
        kernel_size: Convolution kernel size (default: 3)
        dilation: Dilation factor for larger receptive field (default: 1)
        dropout: Dropout probability (default: 0.1)
        use_se: Whether to use Squeeze-and-Excitation (default: True)
        se_reduction: SE reduction ratio (default: 4)
    
    Example:
        >>> block = OptimizedConv1DBlock(256, kernel_size=3, dilation=2)
        >>> x = torch.randn(32, 256, 100)
        >>> out = block(x)  # [32, 256, 100]
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        use_se: bool = True,
        se_reduction: int = 4
    ):
        super().__init__()
        
        self.use_se = use_se
        
        # Pre-LayerNorm using GroupNorm(1, C) ≡ LayerNorm
        # Avoids expensive transpose operations
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        
        # Depthwise separable convolutions
        self.conv1 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size, dilation=dilation
        )
        self.conv2 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size, dilation=dilation
        )
        
        # Activations and regularization
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Squeeze-and-Excitation
        if use_se:
            self.se = SqueezeExcitation(channels, reduction=se_reduction)
        
        self._channels = channels
        self._kernel_size = kernel_size
        self._dilation = dilation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply optimized convolution block with residual connection.
        
        Args:
            x: Input tensor [batch, channels, seq_len]
            
        Returns:
            Output tensor [batch, channels, seq_len]
        """
        residual = x
        
        # First conv: Pre-norm → Conv → GELU → Dropout
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second conv: Pre-norm → Conv
        x = self.norm2(x)
        x = self.conv2(x)
        
        # Squeeze-and-Excitation
        if self.use_se:
            x = self.se(x)
        
        # Residual connection + activation
        x = x + residual
        x = self.activation(x)
        
        return self.dropout(x)
    
    def extra_repr(self) -> str:
        return (f'channels={self._channels}, kernel_size={self._kernel_size}, '
                f'dilation={self._dilation}, use_se={self.use_se}')


class Conv1DBlock(nn.Module):
    """
    Standard 1D Convolutional block with residual connection.
    
    This is the original (non-optimized) implementation for comparison.
    Use OptimizedConv1DBlock for better efficiency.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 3)
        dropout: Dropout probability (default: 0.1)
        use_residual: Whether to use residual connection (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        
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
        
        # Projection for residual if dimensions don't match
        if use_residual and in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, 1)
            self.use_residual = True
        else:
            self.proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            [batch, out_channels, seq_len]
        """
        residual = x
        out = self.conv(x)
        
        if self.use_residual:
            if self.proj is not None:
                residual = self.proj(residual)
            out = out + residual
        
        return self.dropout(self.activation(out))
