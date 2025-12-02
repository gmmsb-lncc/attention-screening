"""
Tests for Positional Encoding implementations.

Tests RoPE (Rotary Position Embedding) and Sinusoidal encoding.

Author: DockTKinase Team
Date: December 2025
"""

import pytest
import torch
import torch.nn as nn
import math


class TestRotaryPositionalEmbedding:
    """Tests for RoPE implementation."""
    
    def test_basic_forward(self):
        """Basic forward pass works."""
        from src.classifier.models.positional_encoding import RotaryPositionalEmbedding
        
        rope = RotaryPositionalEmbedding(dim=64)
        x = torch.randn(2, 100, 64)  # [batch, seq, dim]
        
        out = rope(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_variable_sequence_length(self):
        """RoPE handles variable sequence lengths."""
        from src.classifier.models.positional_encoding import RotaryPositionalEmbedding
        
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len_cached=128)
        
        # Test different sequence lengths
        for seq_len in [50, 100, 200, 500, 1000]:
            x = torch.randn(2, seq_len, 64)
            out = rope(x)
            assert out.shape == (2, seq_len, 64), f"Failed for seq_len={seq_len}"
    
    def test_beyond_initial_cache(self):
        """RoPE extends cache for sequences beyond initial max."""
        from src.classifier.models.positional_encoding import RotaryPositionalEmbedding
        
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len_cached=100)
        
        # Sequence longer than initial cache
        x = torch.randn(2, 500, 64)
        out = rope(x)
        
        assert out.shape == x.shape
        assert rope._seq_len_cached >= 500
    
    def test_4d_input(self):
        """RoPE works with 4D input [batch, heads, seq, dim]."""
        from src.classifier.models.positional_encoding import RotaryPositionalEmbedding
        
        rope = RotaryPositionalEmbedding(dim=32)
        x = torch.randn(2, 8, 100, 32)  # [batch, heads, seq, head_dim]
        
        out = rope(x)
        
        assert out.shape == x.shape
    
    def test_apply_to_qk(self):
        """apply_rotary_pos_emb works for Q and K."""
        from src.classifier.models.positional_encoding import RotaryPositionalEmbedding
        
        rope = RotaryPositionalEmbedding(dim=32)
        
        q = torch.randn(2, 8, 100, 32)
        k = torch.randn(2, 8, 100, 32)
        
        q_rot, k_rot = rope.apply_rotary_pos_emb(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_relative_position_preserved(self):
        """RoPE preserves relative position information in attention."""
        from src.classifier.models.positional_encoding import RotaryPositionalEmbedding
        
        rope = RotaryPositionalEmbedding(dim=64)
        
        # Create identical tokens at different positions
        x = torch.ones(1, 10, 64)
        x_rotated = rope(x)
        
        # Different positions should have different encodings
        assert not torch.allclose(x_rotated[0, 0], x_rotated[0, 5], atol=1e-5)
    
    def test_dimension_must_be_even(self):
        """RoPE requires even dimension."""
        from src.classifier.models.positional_encoding import RotaryPositionalEmbedding
        
        with pytest.raises(ValueError, match="must be even"):
            RotaryPositionalEmbedding(dim=63)
    
    def test_deterministic(self):
        """Same input produces same output."""
        from src.classifier.models.positional_encoding import RotaryPositionalEmbedding
        
        rope = RotaryPositionalEmbedding(dim=64)
        x = torch.randn(2, 100, 64)
        
        out1 = rope(x)
        out2 = rope(x)
        
        assert torch.allclose(out1, out2)


class TestSinusoidalPositionalEncoding:
    """Tests for Sinusoidal encoding."""
    
    def test_basic_forward(self):
        """Basic forward pass works."""
        from src.classifier.models.positional_encoding import SinusoidalPositionalEncoding
        
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000)
        x = torch.randn(2, 100, 256)
        
        out = pe(x)
        
        assert out.shape == x.shape
    
    def test_max_length_limit(self):
        """Sinusoidal encoding fails beyond max_len."""
        from src.classifier.models.positional_encoding import SinusoidalPositionalEncoding
        
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=100)
        x = torch.randn(2, 200, 256)  # Beyond max_len
        
        with pytest.raises(ValueError, match="exceeds max_len"):
            pe(x)
    
    def test_additive_encoding(self):
        """Sinusoidal encoding is additive."""
        from src.classifier.models.positional_encoding import SinusoidalPositionalEncoding
        
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(1, 50, 64)
        
        out = pe(x)
        
        # Output should be the positional encoding itself
        assert not torch.allclose(out, x)


class TestRoPEMultiHeadAttention:
    """Tests for RoPE-based Multi-Head Attention."""
    
    def test_self_attention(self):
        """Self-attention works."""
        from src.classifier.models.positional_encoding import RoPEMultiHeadAttention
        
        attn = RoPEMultiHeadAttention(d_model=256, num_heads=8)
        x = torch.randn(2, 100, 256)
        
        out = attn(x, x, x)
        
        assert out.shape == x.shape
    
    def test_cross_attention(self):
        """Cross-attention works with different Q and K lengths."""
        from src.classifier.models.positional_encoding import RoPEMultiHeadAttention
        
        attn = RoPEMultiHeadAttention(d_model=256, num_heads=8)
        
        query = torch.randn(2, 100, 256)
        key = torch.randn(2, 200, 256)
        value = torch.randn(2, 200, 256)
        
        out = attn(query, key, value)
        
        assert out.shape == query.shape
    
    def test_variable_lengths(self):
        """Attention works with very long sequences."""
        from src.classifier.models.positional_encoding import RoPEMultiHeadAttention
        
        attn = RoPEMultiHeadAttention(d_model=128, num_heads=4)
        
        # Long sequence - would fail with sinusoidal if max_len is small
        x = torch.randn(1, 2000, 128)
        out = attn(x, x, x)
        
        assert out.shape == x.shape
    
    def test_gradient_flow(self):
        """Gradients flow through attention."""
        from src.classifier.models.positional_encoding import RoPEMultiHeadAttention
        
        attn = RoPEMultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 50, 64, requires_grad=True)
        
        out = attn(x, x, x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestCreatePositionalEncoding:
    """Tests for factory function."""
    
    def test_create_rope(self):
        """Factory creates RoPE."""
        from src.classifier.models.positional_encoding import create_positional_encoding
        
        pe = create_positional_encoding(d_model=64, encoding_type="rope")
        
        from src.classifier.models.positional_encoding import RotaryPositionalEmbedding
        assert isinstance(pe, RotaryPositionalEmbedding)
    
    def test_create_sinusoidal(self):
        """Factory creates Sinusoidal."""
        from src.classifier.models.positional_encoding import create_positional_encoding
        
        pe = create_positional_encoding(d_model=64, encoding_type="sinusoidal")
        
        from src.classifier.models.positional_encoding import SinusoidalPositionalEncoding
        assert isinstance(pe, SinusoidalPositionalEncoding)
    
    def test_invalid_type(self):
        """Factory raises error for unknown type."""
        from src.classifier.models.positional_encoding import create_positional_encoding
        
        with pytest.raises(ValueError, match="Unknown encoding type"):
            create_positional_encoding(d_model=64, encoding_type="unknown")


class TestRoPEVsSinusoidal:
    """Comparison tests between RoPE and Sinusoidal."""
    
    def test_rope_handles_longer_sequences(self):
        """RoPE handles sequences that would fail with sinusoidal."""
        from src.classifier.models.positional_encoding import (
            RotaryPositionalEmbedding, 
            SinusoidalPositionalEncoding
        )
        
        dim = 64
        short_max = 100
        long_seq = 500
        
        # Sinusoidal fails
        sin_pe = SinusoidalPositionalEncoding(d_model=dim, max_len=short_max)
        x_long = torch.randn(1, long_seq, dim)
        
        with pytest.raises(ValueError):
            sin_pe(x_long)
        
        # RoPE succeeds
        rope = RotaryPositionalEmbedding(dim=dim, max_seq_len_cached=short_max)
        out = rope(x_long)
        assert out.shape == x_long.shape
    
    def test_output_dimensions_match(self):
        """Both produce same output dimensions."""
        from src.classifier.models.positional_encoding import (
            RotaryPositionalEmbedding, 
            SinusoidalPositionalEncoding
        )
        
        dim = 64
        seq_len = 100
        
        rope = RotaryPositionalEmbedding(dim=dim)
        sin_pe = SinusoidalPositionalEncoding(d_model=dim, max_len=1000, dropout=0.0)
        
        x = torch.randn(2, seq_len, dim)
        
        rope_out = rope(x)
        sin_out = sin_pe(x)
        
        assert rope_out.shape == sin_out.shape == x.shape


class TestCrossAttentionModelWithRoPE:
    """Tests for CrossAttentionAffinityModel with RoPE support."""
    
    def test_model_with_sinusoidal(self):
        """Model works with sinusoidal encoding."""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        model = CrossAttentionAffinityModel(
            protein_dim=128,
            ligand_dim=64,
            hidden_dim=32,
            num_cnn_layers=2,
            num_cross_attn_layers=1,
            num_heads=4,
            positional_encoding_type='sinusoidal'
        )
        
        protein = torch.randn(2, 100, 128)
        ligand = torch.randn(2, 50, 64)
        
        output = model(protein, ligand)
        
        assert 'classification' in output
        assert 'regression' in output
        assert output['classification'].shape == (2, 1)
        assert output['regression'].shape == (2, 1)
    
    def test_model_with_rope(self):
        """Model works with RoPE encoding."""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        model = CrossAttentionAffinityModel(
            protein_dim=128,
            ligand_dim=64,
            hidden_dim=32,
            num_cnn_layers=2,
            num_cross_attn_layers=1,
            num_heads=4,
            positional_encoding_type='rope'
        )
        
        protein = torch.randn(2, 100, 128)
        ligand = torch.randn(2, 50, 64)
        
        output = model(protein, ligand)
        
        assert 'classification' in output
        assert 'regression' in output
        assert output['classification'].shape == (2, 1)
    
    def test_rope_handles_long_sequences(self):
        """RoPE model handles sequences beyond max_len."""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        model = CrossAttentionAffinityModel(
            protein_dim=64,
            ligand_dim=32,
            hidden_dim=32,
            num_cnn_layers=1,
            num_cross_attn_layers=1,
            num_heads=4,
            max_protein_len=100,  # Initial cache size
            max_ligand_len=50,
            positional_encoding_type='rope'
        )
        
        # Sequences much longer than initial max_len
        protein = torch.randn(1, 500, 64)
        ligand = torch.randn(1, 200, 32)
        
        output = model(protein, ligand)
        
        assert output['classification'].shape == (1, 1)
    
    def test_config_includes_encoding_type(self):
        """Model config includes positional_encoding_type."""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        model = CrossAttentionAffinityModel(
            protein_dim=64,
            ligand_dim=32,
            hidden_dim=32,
            positional_encoding_type='rope'
        )
        
        info = model.get_architecture_info()
        
        assert info['config']['positional_encoding_type'] == 'rope'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
