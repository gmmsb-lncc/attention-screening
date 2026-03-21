"""
Tests for Optimized CNN Encoder Components.

This module validates the CNN blocks and encoders for correctness,
parameter counts, and receptive field calculations.

Run with: pytest tests/test_cnn_encoder.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.models.cnn_blocks import (
    SqueezeExcitation,
    DepthwiseSeparableConv1d,
    OptimizedConv1DBlock,
    Conv1DBlock
)
from src.classifier.models.cnn_encoder import (
    CNNEncoder,
    OptimizedCNNEncoder,
    create_encoder
)


class TestSqueezeExcitation:
    """Tests for Squeeze-and-Excitation block."""
    
    def test_output_shape(self):
        """Output shape should match input shape."""
        se = SqueezeExcitation(channels=256, reduction=4)
        x = torch.randn(32, 256, 100)  # [batch, channels, seq_len]
        out = se(x)
        assert out.shape == x.shape
    
    def test_scale_range(self):
        """SE scales should be in [0, 1] due to sigmoid."""
        se = SqueezeExcitation(channels=64, reduction=4)
        x = torch.randn(8, 64, 50)
        
        # Access internal computation
        squeezed = se.squeeze(x).view(8, -1)
        scale = se.excitation(squeezed)
        
        assert scale.min() >= 0.0
        assert scale.max() <= 1.0
    
    def test_parameter_count(self):
        """Check parameter reduction ratio."""
        channels = 256
        reduction = 4
        se = SqueezeExcitation(channels, reduction)
        
        reduced = max(channels // reduction, 8)
        expected_params = channels * reduced + reduced * channels  # Two linear layers, no bias
        actual_params = sum(p.numel() for p in se.parameters())
        
        assert actual_params == expected_params


class TestDepthwiseSeparableConv1d:
    """Tests for Depthwise Separable Convolution."""
    
    def test_output_shape(self):
        """Output shape should have correct channels and same length."""
        conv = DepthwiseSeparableConv1d(in_channels=128, out_channels=256, kernel_size=3)
        x = torch.randn(16, 128, 100)
        out = conv(x)
        
        assert out.shape == (16, 256, 100)
    
    def test_parameter_reduction(self):
        """Depthwise separable should have fewer parameters than standard conv."""
        in_ch, out_ch, k = 256, 256, 3
        
        # Standard conv
        standard = nn.Conv1d(in_ch, out_ch, k, padding=k//2)
        standard_params = sum(p.numel() for p in standard.parameters())
        
        # Depthwise separable
        dw_sep = DepthwiseSeparableConv1d(in_ch, out_ch, k)
        dw_sep_params = sum(p.numel() for p in dw_sep.parameters())
        
        # Depthwise separable should have ~k times fewer parameters
        reduction_ratio = standard_params / dw_sep_params
        assert reduction_ratio > 2.5  # Should be ~3x reduction
    
    def test_dilation(self):
        """Dilation should increase receptive field without changing output size."""
        for dilation in [1, 2, 4]:
            conv = DepthwiseSeparableConv1d(64, 64, kernel_size=3, dilation=dilation)
            x = torch.randn(8, 64, 100)
            out = conv(x)
            assert out.shape == x.shape, f"Failed for dilation={dilation}"


class TestOptimizedConv1DBlock:
    """Tests for Optimized Conv1D Block."""
    
    def test_output_shape(self):
        """Output shape should match input shape (residual block)."""
        block = OptimizedConv1DBlock(channels=256, kernel_size=3, dilation=1)
        x = torch.randn(16, 256, 100)
        out = block(x)
        assert out.shape == x.shape
    
    def test_with_se(self):
        """Block with SE should have slightly more parameters."""
        block_no_se = OptimizedConv1DBlock(channels=128, use_se=False)
        block_with_se = OptimizedConv1DBlock(channels=128, use_se=True)
        
        params_no_se = sum(p.numel() for p in block_no_se.parameters())
        params_with_se = sum(p.numel() for p in block_with_se.parameters())
        
        # SE adds extra parameters for channel attention
        assert params_with_se > params_no_se
        overhead = (params_with_se - params_no_se) / params_no_se
        # SE overhead is ~25% for small channels, acceptable for the benefits
        assert overhead < 0.30  # Less than 30% overhead
    
    def test_gradient_flow(self):
        """Gradients should flow through residual connection."""
        block = OptimizedConv1DBlock(channels=64)
        x = torch.randn(4, 64, 50, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestCNNEncoder:
    """Tests for standard CNN Encoder."""
    
    def test_output_shape(self):
        """Output should have hidden_dim channels."""
        encoder = CNNEncoder(input_dim=2560, hidden_dim=256, num_layers=3)
        x = torch.randn(8, 500, 2560)  # [batch, seq_len, esm_dim]
        out = encoder(x)
        
        assert out.shape == (8, 500, 256)
    
    def test_receptive_field(self):
        """Check receptive field calculation."""
        encoder = CNNEncoder(input_dim=768, hidden_dim=128, num_layers=3)
        # With kernels (3, 5, 7): RF = 1 + 2×2 + 2×4 + 2×6 = 25
        assert encoder.receptive_field == 25
    
    def test_masking(self):
        """Masked positions should be zeroed."""
        encoder = CNNEncoder(input_dim=512, hidden_dim=64, num_layers=2)
        x = torch.randn(4, 100, 512)
        mask = torch.ones(4, 100)
        mask[:, 50:] = 0  # Mask second half
        
        out = encoder(x, mask=mask)
        
        # Masked positions should be zero
        assert (out[:, 50:, :] == 0).all()


class TestOptimizedCNNEncoder:
    """Tests for Optimized CNN Encoder."""
    
    def test_output_shape(self):
        """Output should have hidden_dim channels."""
        encoder = OptimizedCNNEncoder(
            input_dim=2560, hidden_dim=256, num_layers=3
        )
        x = torch.randn(8, 500, 2560)
        out = encoder(x)
        
        assert out.shape == (8, 500, 256)
    
    def test_receptive_field_with_dilation(self):
        """Dilations should increase receptive field."""
        encoder = OptimizedCNNEncoder(
            input_dim=768, 
            hidden_dim=128, 
            num_layers=3,
            kernel_size=3,
            dilations=(1, 2, 4)
        )
        # RF = 1 + 2×2×1 + 2×2×2 + 2×2×4 = 1 + 4 + 8 + 16 = 29
        assert encoder.receptive_field == 29
    
    def test_parameter_reduction(self):
        """Optimized encoder should have fewer parameters."""
        standard = CNNEncoder(input_dim=2560, hidden_dim=256, num_layers=3)
        optimized = OptimizedCNNEncoder(input_dim=2560, hidden_dim=256, num_layers=3)
        
        std_params = standard.count_parameters()
        opt_params = optimized.count_parameters()
        
        print(f"\nParameter comparison:")
        print(f"  Standard:  {std_params:,}")
        print(f"  Optimized: {opt_params:,}")
        print(f"  Reduction: {(1 - opt_params/std_params)*100:.1f}%")
        
        # Optimized should have at least 30% fewer parameters
        assert opt_params < std_params * 0.7
    
    def test_get_config(self):
        """Config should contain all relevant info."""
        encoder = OptimizedCNNEncoder(
            input_dim=2560,
            hidden_dim=256,
            num_layers=4,
            dilations=(1, 2, 4, 8)
        )
        config = encoder.get_config()
        
        assert 'input_dim' in config
        assert 'receptive_field' in config
        assert 'parameters' in config
        assert config['num_layers'] == 4


class TestCreateEncoder:
    """Tests for encoder factory function."""
    
    def test_create_standard(self):
        """Factory should create standard encoder when optimized=False."""
        encoder = create_encoder(input_dim=768, optimized=False)
        assert isinstance(encoder, CNNEncoder)
    
    def test_create_optimized(self):
        """Factory should create optimized encoder when optimized=True."""
        encoder = create_encoder(input_dim=768, optimized=True)
        assert isinstance(encoder, OptimizedCNNEncoder)
    
    def test_kwargs_passed(self):
        """Additional kwargs should be passed to encoder."""
        encoder = create_encoder(
            input_dim=768, 
            optimized=True, 
            use_se=False,
            dilations=(1, 1, 1)
        )
        assert encoder.use_se == False


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_protein_ligand_encoding(self):
        """Simulate encoding protein and ligand embeddings."""
        # Create encoders
        protein_encoder = create_encoder(input_dim=2560, hidden_dim=256, optimized=True)
        ligand_encoder = create_encoder(input_dim=768, hidden_dim=256, optimized=True)
        
        # Simulate batch
        batch_size = 16
        protein_emb = torch.randn(batch_size, 500, 2560)  # ESM-2 3B
        ligand_emb = torch.randn(batch_size, 50, 768)     # SMI-TED
        
        # Encode
        protein_features = protein_encoder(protein_emb)
        ligand_features = ligand_encoder(ligand_emb)
        
        assert protein_features.shape == (batch_size, 500, 256)
        assert ligand_features.shape == (batch_size, 50, 256)
    
    def test_training_step(self):
        """Simulate a training step with gradients."""
        encoder = create_encoder(input_dim=768, hidden_dim=128, optimized=True)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
        
        # Forward pass
        x = torch.randn(8, 100, 768)
        out = encoder(x)
        
        # Backward pass
        loss = out.mean()
        loss.backward()
        
        # Check gradients exist
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()


def print_comparison_table():
    """Print a comparison table for documentation."""
    print("\n" + "="*70)
    print("CNN ENCODER COMPARISON")
    print("="*70)
    
    configs = [
        (2560, 256, 3, "ESM-2 3B"),
        (1280, 256, 3, "ESM-2 650M"),
        (768, 256, 3, "SMI-TED"),
    ]
    
    print(f"\n{'Input Dim':<12} {'Model':<12} {'Hidden':<8} {'Layers':<8} {'Params':<12} {'RF':<6}")
    print("-"*70)
    
    for input_dim, hidden_dim, num_layers, name in configs:
        standard = CNNEncoder(input_dim, hidden_dim, num_layers)
        optimized = OptimizedCNNEncoder(input_dim, hidden_dim, num_layers)
        
        std_params = standard.count_parameters()
        opt_params = optimized.count_parameters()
        
        print(f"{name:<12} {'Standard':<12} {hidden_dim:<8} {num_layers:<8} {std_params:>10,} {standard.receptive_field:<6}")
        print(f"{'':<12} {'Optimized':<12} {hidden_dim:<8} {num_layers:<8} {opt_params:>10,} {optimized.receptive_field:<6}")
        reduction = (1 - opt_params/std_params) * 100
        print(f"{'':<12} {'Reduction':<12} {'':<8} {'':<8} {reduction:>9.1f}%")
        print()
    
    print("="*70)


if __name__ == "__main__":
    # Run quick validation
    print("Running quick validation...")
    
    # Test basic functionality
    encoder = OptimizedCNNEncoder(input_dim=2560, hidden_dim=256)
    x = torch.randn(4, 100, 2560)
    out = encoder(x)
    
    print(f"✓ Input shape:  {tuple(x.shape)}")
    print(f"✓ Output shape: {tuple(out.shape)}")
    print(f"✓ Parameters:   {encoder.count_parameters():,}")
    print(f"✓ Receptive field: {encoder.receptive_field}")
    
    # Print comparison
    print_comparison_table()
    
    print("\n✅ All quick validations passed!")
