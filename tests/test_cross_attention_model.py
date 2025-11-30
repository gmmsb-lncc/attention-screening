"""
Tests for CrossAttentionAffinityModel.

Phase 2 - Multi-Matrix Branch
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCrossAttentionModel:
    """Tests for CrossAttentionAffinityModel"""
    
    def test_model_import(self):
        """Test that model can be imported"""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        assert CrossAttentionAffinityModel is not None
    
    def test_model_creation_default(self):
        """Test model creation with default parameters"""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        model = CrossAttentionAffinityModel()
        assert model is not None
        assert model.count_parameters() > 0
    
    def test_model_creation_custom(self):
        """Test model creation with custom parameters"""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        model = CrossAttentionAffinityModel(
            protein_dim=320,  # ESM-2 8M
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1,
            num_heads=4
        )
        
        assert model is not None
        assert model.protein_dim == 320
        assert model.ligand_dim == 768
        assert model.hidden_dim == 64
    
    def test_forward_pass(self):
        """Test forward pass with random data"""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        model = CrossAttentionAffinityModel(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1,
            num_heads=4
        )
        
        batch_size = 2
        protein_len = 50
        ligand_len = 30
        
        protein_matrix = torch.randn(batch_size, protein_len, 320)
        ligand_matrix = torch.randn(batch_size, ligand_len, 768)
        
        output = model(protein_matrix, ligand_matrix)
        
        assert 'classification' in output
        assert 'regression' in output
        assert output['classification'].shape == (batch_size, 1)
        assert output['regression'].shape == (batch_size, 1)
    
    def test_forward_with_masks(self):
        """Test forward pass with padding masks"""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        model = CrossAttentionAffinityModel(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1,
            num_heads=4
        )
        
        batch_size = 2
        protein_len = 50
        ligand_len = 30
        
        protein_matrix = torch.randn(batch_size, protein_len, 320)
        ligand_matrix = torch.randn(batch_size, ligand_len, 768)
        
        # Create masks (1 = valid, 0 = padding)
        protein_mask = torch.ones(batch_size, protein_len)
        protein_mask[0, 40:] = 0  # First sample has 40 valid tokens
        
        ligand_mask = torch.ones(batch_size, ligand_len)
        ligand_mask[1, 20:] = 0  # Second sample has 20 valid tokens
        
        output = model(
            protein_matrix, ligand_matrix,
            protein_mask=protein_mask, ligand_mask=ligand_mask
        )
        
        assert output['classification'].shape == (batch_size, 1)
        assert output['regression'].shape == (batch_size, 1)
    
    def test_factory_function(self):
        """Test create_cross_attention_model factory"""
        from src.classifier.models.cross_attention_model import create_cross_attention_model
        
        # Test different sizes
        model_small = create_cross_attention_model(size='small')
        model_base = create_cross_attention_model(size='base')
        model_large = create_cross_attention_model(size='large')
        
        # Larger models should have more parameters
        assert model_small.count_parameters() < model_base.count_parameters()
        assert model_base.count_parameters() < model_large.count_parameters()
    
    def test_factory_with_model_names(self):
        """Test factory with different model names"""
        from src.classifier.models.cross_attention_model import create_cross_attention_model
        
        model = create_cross_attention_model(
            protein_model='esm2_t6_8M_UR50D',
            ligand_model='SMI-TED',
            size='small'
        )
        
        assert model.protein_dim == 320  # ESM-2 8M dimension
        assert model.ligand_dim == 768
    
    def test_architecture_info(self):
        """Test get_architecture_info method"""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        model = CrossAttentionAffinityModel(hidden_dim=64)
        info = model.get_architecture_info()
        
        assert 'model_type' in info
        assert info['model_type'] == 'CrossAttentionAffinityModel'
        assert 'config' in info
        assert 'total_parameters' in info
        assert 'encoder_params' in info
    
    def test_gradient_flow(self):
        """Test that gradients flow properly"""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        model = CrossAttentionAffinityModel(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1
        )
        
        protein_matrix = torch.randn(2, 50, 320, requires_grad=True)
        ligand_matrix = torch.randn(2, 30, 768, requires_grad=True)
        
        output = model(protein_matrix, ligand_matrix)
        loss = output['classification'].sum() + output['regression'].sum()
        loss.backward()
        
        # Check gradients exist
        assert protein_matrix.grad is not None
        assert ligand_matrix.grad is not None
        
        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestMultiTaskLoss:
    """Tests for MultiTaskLoss"""
    
    def test_loss_import(self):
        """Test that loss can be imported"""
        from src.classifier.models.cross_attention_model import MultiTaskLoss
        assert MultiTaskLoss is not None
    
    def test_loss_computation(self):
        """Test loss computation"""
        from src.classifier.models.cross_attention_model import MultiTaskLoss
        
        loss_fn = MultiTaskLoss()
        
        cls_logits = torch.randn(4, 1)
        reg_output = torch.randn(4, 1)
        cls_target = torch.randint(0, 2, (4, 1)).float()
        reg_target = torch.randn(4, 1)
        
        losses = loss_fn(cls_logits, reg_output, cls_target, reg_target)
        
        assert 'total' in losses
        assert 'classification' in losses
        assert 'regression' in losses
        assert losses['total'].item() >= 0
    
    def test_loss_with_mask(self):
        """Test loss with regression mask"""
        from src.classifier.models.cross_attention_model import MultiTaskLoss
        
        loss_fn = MultiTaskLoss()
        
        cls_logits = torch.randn(4, 1)
        reg_output = torch.randn(4, 1)
        cls_target = torch.randint(0, 2, (4, 1)).float()
        reg_target = torch.randn(4, 1)
        reg_mask = torch.tensor([[1], [1], [0], [0]])  # Only first 2 have regression labels
        
        losses = loss_fn(cls_logits, reg_output, cls_target, reg_target, reg_mask)
        
        assert 'total' in losses
        assert losses['total'].item() >= 0
    
    def test_loss_weighting(self):
        """Test custom loss weights"""
        from src.classifier.models.cross_attention_model import MultiTaskLoss
        
        loss_fn = MultiTaskLoss(classification_weight=2.0, regression_weight=0.5)
        
        cls_logits = torch.randn(4, 1)
        reg_output = torch.randn(4, 1)
        cls_target = torch.randint(0, 2, (4, 1)).float()
        reg_target = torch.randn(4, 1)
        
        losses = loss_fn(cls_logits, reg_output, cls_target, reg_target)
        
        # Total should be weighted sum
        expected = 2.0 * losses['classification'] + 0.5 * losses['regression']
        assert torch.isclose(losses['total'], expected, atol=1e-5)


class TestCNNEncoder:
    """Tests for CNNEncoder"""
    
    def test_encoder_forward(self):
        """Test CNN encoder forward pass"""
        from src.classifier.models.cross_attention_model import CNNEncoder
        
        encoder = CNNEncoder(
            input_dim=320,
            hidden_dim=64,
            num_layers=2
        )
        
        x = torch.randn(2, 50, 320)
        output = encoder(x)
        
        assert output.shape == (2, 50, 64)
    
    def test_encoder_with_mask(self):
        """Test CNN encoder with mask"""
        from src.classifier.models.cross_attention_model import CNNEncoder
        
        encoder = CNNEncoder(input_dim=320, hidden_dim=64)
        
        x = torch.randn(2, 50, 320)
        mask = torch.ones(2, 50)
        mask[0, 30:] = 0
        
        output = encoder(x, mask)
        
        # Check masked positions are zeroed
        assert output.shape == (2, 50, 64)
        assert torch.allclose(output[0, 30:], torch.zeros(20, 64), atol=1e-5)


class TestCrossAttention:
    """Tests for CrossAttention"""
    
    def test_cross_attention_forward(self):
        """Test cross-attention forward pass"""
        from src.classifier.models.cross_attention_model import CrossAttention
        
        attn = CrossAttention(hidden_dim=64, num_heads=4)
        
        query = torch.randn(2, 50, 64)  # protein
        key = torch.randn(2, 30, 64)    # ligand
        value = torch.randn(2, 30, 64)
        
        output = attn(query, key, value)
        
        assert output.shape == (2, 50, 64)
    
    def test_cross_attention_with_mask(self):
        """Test cross-attention with key mask"""
        from src.classifier.models.cross_attention_model import CrossAttention
        
        attn = CrossAttention(hidden_dim=64, num_heads=4)
        
        query = torch.randn(2, 50, 64)
        key = torch.randn(2, 30, 64)
        value = torch.randn(2, 30, 64)
        key_mask = torch.ones(2, 30)
        key_mask[0, 20:] = 0
        
        output = attn(query, key, value, key_mask)
        
        assert output.shape == (2, 50, 64)


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_training_step(self):
        """Test complete training step"""
        from src.classifier.models.cross_attention_model import (
            CrossAttentionAffinityModel, MultiTaskLoss
        )
        
        # Create model
        model = CrossAttentionAffinityModel(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create loss function
        loss_fn = MultiTaskLoss()
        
        # Create batch
        batch_size = 4
        protein_matrix = torch.randn(batch_size, 50, 320)
        ligand_matrix = torch.randn(batch_size, 30, 768)
        cls_target = torch.randint(0, 2, (batch_size, 1)).float()
        reg_target = torch.randn(batch_size, 1) * 2 + 6
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        output = model(protein_matrix, ligand_matrix)
        losses = loss_fn(
            output['classification'],
            output['regression'],
            cls_target,
            reg_target
        )
        
        losses['total'].backward()
        optimizer.step()
        
        # Check loss decreased after step would require multiple steps
        # Here we just verify the training step completes
        assert True
    
    def test_model_save_load(self):
        """Test model save and load"""
        import tempfile
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        model = CrossAttentionAffinityModel(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64
        )
        
        # Get initial prediction
        x_protein = torch.randn(1, 50, 320)
        x_ligand = torch.randn(1, 30, 768)
        
        model.eval()
        with torch.no_grad():
            initial_output = model(x_protein, x_ligand)
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pt') as f:
            torch.save(model.state_dict(), f.name)
            
            # Create new model and load weights
            new_model = CrossAttentionAffinityModel(
                protein_dim=320,
                ligand_dim=768,
                hidden_dim=64
            )
            new_model.load_state_dict(torch.load(f.name))
        
        # Check predictions match
        new_model.eval()
        with torch.no_grad():
            loaded_output = new_model(x_protein, x_ligand)
        
        assert torch.allclose(
            initial_output['classification'],
            loaded_output['classification'],
            atol=1e-5
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
