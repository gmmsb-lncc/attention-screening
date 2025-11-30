"""
Tests for Matrix Pipeline Components.

Tests for:
- MatrixEmbeddingDataset
- CrossAttentionTrainer
- MatrixAffinityPipeline

Author: DockTKinase Team
Date: November 2025
"""

import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path


class TestMatrixDataloader:
    """Tests for matrix dataloader components."""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample data for testing."""
        # Create directories
        prot_dir = tmp_path / 'protein_matrix_embeddings'
        lig_dir = tmp_path / 'ligand_matrix_embeddings'
        prot_dir.mkdir()
        lig_dir.mkdir()
        
        # Create dummy matrices
        np.save(prot_dir / 'P1_matrix.npy', np.random.randn(50, 320))
        np.save(prot_dir / 'P2_matrix.npy', np.random.randn(80, 320))
        np.save(lig_dir / 'C1_matrix.npy', np.random.randn(20, 768))
        np.save(lig_dir / 'C2_matrix.npy', np.random.randn(15, 768))
        
        # Create DataFrame
        df = pd.DataFrame({
            'seq_id': ['P1', 'P1', 'P2', 'P2'],
            'chembl_id': ['C1', 'C2', 'C1', 'C2'],
            'label': [1, 0, 0, 1],
            'pchembl_value': [7.5, 6.2, 5.8, 8.1]
        })
        
        return df, str(prot_dir), str(lig_dir)
    
    def test_dataset_creation(self, sample_data):
        """Test MatrixEmbeddingDataset creation."""
        from src.classifier.utils.matrix_dataloader import MatrixEmbeddingDataset
        
        df, prot_dir, lig_dir = sample_data
        dataset = MatrixEmbeddingDataset(df, prot_dir, lig_dir)
        
        assert len(dataset) == 4
    
    def test_dataset_getitem(self, sample_data):
        """Test dataset __getitem__."""
        from src.classifier.utils.matrix_dataloader import MatrixEmbeddingDataset
        
        df, prot_dir, lig_dir = sample_data
        dataset = MatrixEmbeddingDataset(df, prot_dir, lig_dir)
        
        sample = dataset[0]
        
        assert 'protein_matrix' in sample
        assert 'ligand_matrix' in sample
        assert 'label' in sample
        assert 'regression_target' in sample
        assert isinstance(sample['protein_matrix'], torch.Tensor)
        assert isinstance(sample['ligand_matrix'], torch.Tensor)
    
    def test_collate_batch(self, sample_data):
        """Test collate function with padding."""
        from src.classifier.utils.matrix_dataloader import (
            MatrixEmbeddingDataset, collate_matrix_batch
        )
        
        df, prot_dir, lig_dir = sample_data
        dataset = MatrixEmbeddingDataset(df, prot_dir, lig_dir)
        
        batch = [dataset[i] for i in range(2)]
        collated = collate_matrix_batch(batch)
        
        assert collated['protein_matrix'].shape[0] == 2  # batch size
        assert 'protein_mask' in collated
        assert 'ligand_mask' in collated
        # Check padding worked
        assert collated['protein_matrix'].shape[1] == max(50, 50)  # P1 is used for both
    
    def test_create_dataloader(self, sample_data):
        """Test dataloader creation."""
        from src.classifier.utils.matrix_dataloader import create_matrix_dataloader
        
        df, prot_dir, lig_dir = sample_data
        loader = create_matrix_dataloader(df, prot_dir, lig_dir, batch_size=2)
        
        batch = next(iter(loader))
        assert batch['protein_matrix'].shape[0] == 2


class TestCrossAttentionTrainer:
    """Tests for CrossAttentionTrainer."""
    
    @pytest.fixture
    def trainer_setup(self, tmp_path):
        """Create trainer setup."""
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        from src.classifier.utils.matrix_trainer import CrossAttentionTrainer, TrainingConfig
        from src.classifier.utils.matrix_dataloader import MatrixEmbeddingDataset, collate_matrix_batch
        from torch.utils.data import DataLoader
        
        # Create directories
        prot_dir = tmp_path / 'protein_matrix_embeddings'
        lig_dir = tmp_path / 'ligand_matrix_embeddings'
        prot_dir.mkdir()
        lig_dir.mkdir()
        
        # Create dummy matrices (small for test)
        np.save(prot_dir / 'P1_matrix.npy', np.random.randn(20, 320))
        np.save(prot_dir / 'P2_matrix.npy', np.random.randn(25, 320))
        np.save(lig_dir / 'C1_matrix.npy', np.random.randn(10, 768))
        np.save(lig_dir / 'C2_matrix.npy', np.random.randn(8, 768))
        
        # Create DataFrame
        df = pd.DataFrame({
            'seq_id': ['P1'] * 4 + ['P2'] * 4,
            'chembl_id': ['C1', 'C2'] * 4,
            'label': [1, 0] * 4,
            'pchembl_value': [7.5, 6.2] * 4
        })
        
        # Create dataset and loader
        dataset = MatrixEmbeddingDataset(df, str(prot_dir), str(lig_dir))
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_matrix_batch)
        
        # Create model (small for test)
        model = CrossAttentionAffinityModel(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1,
            num_heads=4
        )
        
        # Create config
        config = TrainingConfig(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1,
            num_epochs=2,
            batch_size=4,
            save_dir=str(tmp_path / 'checkpoints')
        )
        
        return model, config, loader
    
    def test_trainer_creation(self, trainer_setup):
        """Test trainer creation."""
        from src.classifier.utils.matrix_trainer import CrossAttentionTrainer
        
        model, config, loader = trainer_setup
        trainer = CrossAttentionTrainer(model, config, loader, loader)
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
    
    def test_train_epoch(self, trainer_setup):
        """Test single training epoch."""
        from src.classifier.utils.matrix_trainer import CrossAttentionTrainer
        
        model, config, loader = trainer_setup
        trainer = CrossAttentionTrainer(model, config, loader)
        
        trainer.current_epoch = 1
        metrics = trainer.train_epoch()
        
        assert 'loss' in metrics
        assert 'cls_loss' in metrics
        assert 'reg_loss' in metrics
        assert metrics['loss'] > 0
    
    def test_validate(self, trainer_setup):
        """Test validation."""
        from src.classifier.utils.matrix_trainer import CrossAttentionTrainer
        
        model, config, loader = trainer_setup
        trainer = CrossAttentionTrainer(model, config, loader, loader)
        
        metrics = trainer.validate()
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'auc' in metrics
    
    def test_save_load_checkpoint(self, trainer_setup):
        """Test checkpoint save/load."""
        from src.classifier.utils.matrix_trainer import CrossAttentionTrainer
        
        model, config, loader = trainer_setup
        trainer = CrossAttentionTrainer(model, config, loader)
        
        # Train a bit
        trainer.current_epoch = 1
        trainer.train_epoch()
        
        # Save checkpoint
        trainer.save_checkpoint('test_checkpoint.pt')
        
        # Check file exists
        checkpoint_path = Path(config.save_dir) / 'test_checkpoint.pt'
        assert checkpoint_path.exists()
        
        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))
        assert trainer.current_epoch == 1


class TestMetricTracker:
    """Tests for MetricTracker."""
    
    def test_update_and_get(self):
        """Test metric tracking."""
        from src.classifier.utils.matrix_trainer import MetricTracker
        
        tracker = MetricTracker()
        tracker.update('loss', 1.0)
        tracker.update('loss', 0.8)
        tracker.update('loss', 0.6)
        
        assert tracker.get_last('loss') == 0.6
        assert tracker.get_best('loss', mode='min') == 0.6
        assert tracker.get_best('loss', mode='max') == 1.0
    
    def test_to_dict(self):
        """Test conversion to dict."""
        from src.classifier.utils.matrix_trainer import MetricTracker
        
        tracker = MetricTracker()
        tracker.update('loss', 1.0)
        tracker.update('accuracy', 0.9)
        
        d = tracker.to_dict()
        assert 'loss' in d
        assert 'accuracy' in d


class TestEarlyStopping:
    """Tests for EarlyStopping."""
    
    def test_improvement_resets_counter(self):
        """Test that improvement resets patience counter."""
        from src.classifier.utils.matrix_trainer import EarlyStopping
        
        stopper = EarlyStopping(patience=3, min_delta=0.01)
        
        assert stopper(1.0) == False
        assert stopper(0.9) == False  # Improved
        assert stopper.counter == 0
    
    def test_no_improvement_increments_counter(self):
        """Test that no improvement increments counter."""
        from src.classifier.utils.matrix_trainer import EarlyStopping
        
        stopper = EarlyStopping(patience=3, min_delta=0.01)
        
        stopper(1.0)
        stopper(1.0)  # No improvement
        assert stopper.counter == 1
        stopper(1.0)  # No improvement
        assert stopper.counter == 2
    
    def test_triggers_stop(self):
        """Test that early stopping triggers after patience."""
        from src.classifier.utils.matrix_trainer import EarlyStopping
        
        stopper = EarlyStopping(patience=2, min_delta=0.01)
        
        stopper(1.0)
        stopper(1.0)  # +1
        assert stopper(1.0) == True  # +2 = patience


class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        from src.classifier.utils.matrix_trainer import TrainingConfig
        
        config = TrainingConfig()
        
        assert config.protein_dim == 2560
        assert config.ligand_dim == 768
        assert config.batch_size == 32
        assert config.num_epochs == 100
    
    def test_custom_values(self):
        """Test custom configuration."""
        from src.classifier.utils.matrix_trainer import TrainingConfig
        
        config = TrainingConfig(
            protein_dim=320,
            batch_size=16,
            num_epochs=50
        )
        
        assert config.protein_dim == 320
        assert config.batch_size == 16
        assert config.num_epochs == 50
    
    def test_to_dict_from_dict(self):
        """Test serialization."""
        from src.classifier.utils.matrix_trainer import TrainingConfig
        
        config = TrainingConfig(protein_dim=320)
        d = config.to_dict()
        
        config2 = TrainingConfig.from_dict(d)
        assert config2.protein_dim == 320


class TestMatrixPipelineConfig:
    """Tests for MatrixPipelineConfig."""
    
    def test_get_embedding_dims(self):
        """Test embedding dimension lookup."""
        from src.classifier.utils.matrix_pipeline import MatrixPipelineConfig
        
        config = MatrixPipelineConfig(
            protein_model='esm2_t33',
            ligand_model='smited'
        )
        
        prot_dim, lig_dim = config.get_embedding_dims()
        assert prot_dim == 1280
        assert lig_dim == 768
    
    def test_different_models(self):
        """Test different model configurations."""
        from src.classifier.utils.matrix_pipeline import MatrixPipelineConfig
        
        configs = [
            ('esm2_t6', 320),
            ('esm2_t33', 1280),
            ('esm2_t36', 2560),
            ('boltz', 384),
            ('esmc', 1152),
        ]
        
        for model, expected_dim in configs:
            config = MatrixPipelineConfig(protein_model=model)
            prot_dim, _ = config.get_embedding_dims()
            assert prot_dim == expected_dim, f"Failed for {model}"


class TestModuleExports:
    """Test that all expected classes are exported."""
    
    def test_dataloader_exports(self):
        """Test dataloader module exports."""
        from src.classifier.utils import (
            MatrixEmbeddingDataset,
            VectorEmbeddingDataset,
            collate_matrix_batch,
            create_matrix_dataloader
        )
        
        assert MatrixEmbeddingDataset is not None
        assert collate_matrix_batch is not None
    
    def test_trainer_exports(self):
        """Test trainer module exports."""
        from src.classifier.utils import (
            TrainingConfig,
            MetricTracker,
            EarlyStopping,
            CrossAttentionTrainer,
            train_cross_attention_model
        )
        
        assert TrainingConfig is not None
        assert CrossAttentionTrainer is not None
    
    def test_pipeline_exports(self):
        """Test pipeline module exports."""
        from src.classifier.utils import (
            MatrixPipelineConfig,
            MatrixAffinityPipeline,
            run_matrix_pipeline
        )
        
        assert MatrixPipelineConfig is not None
        assert MatrixAffinityPipeline is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
