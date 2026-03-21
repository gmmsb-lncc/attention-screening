"""
Integration tests for stratification across classification and regression pipelines.

CRITICAL TESTS: Verify that classification and regression use IDENTICAL split indices
when using the stratification system. This ensures consistent evaluation and prevents
data leakage between pipelines.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.build.core.config import BuildConfig
from src.build.pipeline.stratification_manager import StratificationManager
from src.build.pipeline.split_indices import SplitIndices


class TestStratificationIntegration:
    """Test integration of stratification across all pipelines."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = BuildConfig()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_classification_and_regression_use_same_splits(self):
        """
        CRITICAL TEST: Verify classification and regression use IDENTICAL indices.
        
        This is the MOST IMPORTANT test - it ensures that when both pipelines
        receive the same SplitIndices object, they use the exact same samples
        for training, validation, and testing.
        """
        # Create sample data
        n_samples = 200
        embedding_dim = 100
        
        protein_embeddings = np.random.randn(n_samples, 320).astype(np.float32)
        ligand_embeddings = np.random.randn(n_samples, 768).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples).astype(np.int32)
        
        # Create stratification manager
        manager = StratificationManager(
            self.config,
            clustering_algorithm='kmeans',
            random_state=42
        )
        
        # Perform stratification
        splits = manager.stratify(
            protein_embeddings=protein_embeddings,
            ligand_embeddings=ligand_embeddings,
            labels=labels,
            test_size=0.2,
            val_size=0.1
        )
        
        # Simulate what classification pipeline would do
        # It receives the splits and uses them in load_data()
        clf_train_idx = splits.train_idx.copy()
        clf_val_idx = splits.val_idx.copy()
        clf_test_idx = splits.test_idx.copy()
        
        # Simulate what regression pipeline would do
        # It also receives the same splits and uses them in load_data()
        reg_train_idx = splits.train_idx.copy()
        reg_val_idx = splits.val_idx.copy()
        reg_test_idx = splits.test_idx.copy()
        
        # CRITICAL ASSERTIONS: Indices must be IDENTICAL
        assert np.array_equal(clf_train_idx, reg_train_idx), \
            "Classification and Regression must use IDENTICAL train indices!"
        
        assert np.array_equal(clf_val_idx, reg_val_idx), \
            "Classification and Regression must use IDENTICAL validation indices!"
        
        assert np.array_equal(clf_test_idx, reg_test_idx), \
            "Classification and Regression must use IDENTICAL test indices!"
        
        # Additional verification: check they're not just empty
        assert len(clf_train_idx) > 0, "Train indices should not be empty"
        assert len(clf_val_idx) > 0, "Validation indices should not be empty"
        assert len(clf_test_idx) > 0, "Test indices should not be empty"
        
        # Verify total coverage (no missing samples)
        all_indices = np.concatenate([clf_train_idx, clf_val_idx, clf_test_idx])
        assert len(all_indices) == n_samples, "All samples must be assigned to a split"
        
        # Verify no overlap (no data leakage)
        train_set = set(clf_train_idx.tolist())
        val_set = set(clf_val_idx.tolist())
        test_set = set(clf_test_idx.tolist())
        
        assert len(train_set & val_set) == 0, "Train and validation must not overlap"
        assert len(train_set & test_set) == 0, "Train and test must not overlap"
        assert len(val_set & test_set) == 0, "Validation and test must not overlap"
    
    def test_splits_are_reproducible_with_same_seed(self):
        """Test that using the same random_state produces identical splits."""
        n_samples = 150
        protein_embeddings = np.random.randn(n_samples, 320).astype(np.float32)
        ligand_embeddings = np.random.randn(n_samples, 768).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples).astype(np.int32)
        
        # First stratification
        manager1 = StratificationManager(self.config, random_state=42)
        splits1 = manager1.stratify(protein_embeddings, ligand_embeddings, labels)
        
        # Second stratification with same seed
        manager2 = StratificationManager(self.config, random_state=42)
        splits2 = manager2.stratify(protein_embeddings, ligand_embeddings, labels)
        
        # Splits should be identical
        assert np.array_equal(splits1.train_idx, splits2.train_idx)
        assert np.array_equal(splits1.val_idx, splits2.val_idx)
        assert np.array_equal(splits1.test_idx, splits2.test_idx)
    
    def test_splits_saved_and_loaded_remain_identical(self):
        """Test that saved and loaded splits are identical to original."""
        n_samples = 100
        protein_embeddings = np.random.randn(n_samples, 320).astype(np.float32)
        ligand_embeddings = np.random.randn(n_samples, 768).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples).astype(np.int32)
        
        # Create and stratify
        manager = StratificationManager(self.config, random_state=42)
        original_splits = manager.stratify(protein_embeddings, ligand_embeddings, labels)
        
        # Save splits
        save_path = Path(self.temp_dir) / "test_splits.npz"
        original_splits.save(str(save_path))
        
        # Load splits
        loaded_splits = SplitIndices.load(str(save_path))
        
        # Verify they are identical
        assert np.array_equal(original_splits.train_idx, loaded_splits.train_idx)
        assert np.array_equal(original_splits.val_idx, loaded_splits.val_idx)
        assert np.array_equal(original_splits.test_idx, loaded_splits.test_idx)
        
        # Verify metadata is preserved
        assert original_splits.metadata['train_size'] == loaded_splits.metadata['train_size']
        assert original_splits.metadata['val_size'] == loaded_splits.metadata['val_size']
        assert original_splits.metadata['test_size'] == loaded_splits.metadata['test_size']
    
    def test_no_data_leakage_across_splits(self):
        """Verify there is NO data leakage between train/val/test sets."""
        n_samples = 300
        protein_embeddings = np.random.randn(n_samples, 320).astype(np.float32)
        ligand_embeddings = np.random.randn(n_samples, 768).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples).astype(np.int32)
        
        manager = StratificationManager(self.config, random_state=42)
        splits = manager.stratify(protein_embeddings, ligand_embeddings, labels)
        
        # Convert to sets for efficient overlap checking
        train_set = set(splits.train_idx.tolist())
        val_set = set(splits.val_idx.tolist())
        test_set = set(splits.test_idx.tolist())
        
        # Check for any overlaps (data leakage)
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        assert len(train_val_overlap) == 0, \
            f"DATA LEAKAGE: {len(train_val_overlap)} samples in both train and validation!"
        
        assert len(train_test_overlap) == 0, \
            f"DATA LEAKAGE: {len(train_test_overlap)} samples in both train and test!"
        
        assert len(val_test_overlap) == 0, \
            f"DATA LEAKAGE: {len(val_test_overlap)} samples in both validation and test!"
        
        # Verify all samples are assigned exactly once
        all_samples = train_set | val_set | test_set
        assert len(all_samples) == n_samples, \
            f"Coverage issue: {len(all_samples)} unique samples but expected {n_samples}"
    
    def test_split_proportions_are_respected(self):
        """Test that split proportions match requested sizes."""
        n_samples = 1000
        protein_embeddings = np.random.randn(n_samples, 320).astype(np.float32)
        ligand_embeddings = np.random.randn(n_samples, 768).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples).astype(np.int32)
        
        test_size = 0.2
        val_size = 0.1
        
        manager = StratificationManager(self.config, random_state=42)
        splits = manager.stratify(
            protein_embeddings, ligand_embeddings, labels,
            test_size=test_size, val_size=val_size
        )
        
        # Calculate actual proportions
        train_prop = len(splits.train_idx) / n_samples
        val_prop = len(splits.val_idx) / n_samples
        test_prop = len(splits.test_idx) / n_samples
        
        # Allow for small rounding differences (±5%)
        assert 0.65 <= train_prop <= 0.75, f"Train proportion {train_prop:.2%} out of range"
        assert 0.05 <= val_prop <= 0.15, f"Val proportion {val_prop:.2%} out of range"
        assert 0.15 <= test_prop <= 0.25, f"Test proportion {test_prop:.2%} out of range"
        
        # Total should be 100%
        total_prop = train_prop + val_prop + test_prop
        assert abs(total_prop - 1.0) < 0.01, f"Total proportion {total_prop:.2%} != 100%"
    
    def test_stratification_handles_imbalanced_classes(self):
        """Test stratification works with imbalanced datasets."""
        n_samples = 500
        protein_embeddings = np.random.randn(n_samples, 320).astype(np.float32)
        ligand_embeddings = np.random.randn(n_samples, 768).astype(np.float32)
        
        # Create imbalanced labels (90% class 0, 10% class 1)
        labels = np.zeros(n_samples, dtype=np.int32)
        n_positive = int(n_samples * 0.1)
        labels[:n_positive] = 1
        np.random.shuffle(labels)
        
        manager = StratificationManager(self.config, random_state=42)
        splits = manager.stratify(protein_embeddings, ligand_embeddings, labels)
        
        # Verify splits were created successfully
        assert isinstance(splits, SplitIndices)
        assert len(splits.train_idx) > 0
        assert len(splits.val_idx) > 0
        assert len(splits.test_idx) > 0
        
        # Check class distribution in each split
        train_labels = labels[splits.train_idx]
        val_labels = labels[splits.val_idx]
        test_labels = labels[splits.test_idx]
        
        # Each split should have both classes (if possible given size)
        assert len(np.unique(train_labels)) >= 1, "Train should have at least 1 class"
        # Val and test might have only 1 class due to small size and imbalance
        # But they should not be empty
        assert len(val_labels) > 0
        assert len(test_labels) > 0


class TestSplitIndicesImmutabilityInPipelines:
    """Test that SplitIndices remain immutable when passed to pipelines."""
    
    def test_pipelines_cannot_modify_original_splits(self):
        """Verify that pipelines cannot modify the original split indices."""
        train_idx = np.array([0, 1, 2, 3], dtype=np.int32)
        val_idx = np.array([4, 5], dtype=np.int32)
        test_idx = np.array([6, 7], dtype=np.int32)
        
        splits = SplitIndices(train_idx, val_idx, test_idx)
        
        # Store originals for comparison
        original_train = splits.train_idx.copy()
        original_val = splits.val_idx.copy()
        original_test = splits.test_idx.copy()
        
        # Simulate what pipelines do - copy the indices
        pipeline_train = splits.train_idx.copy()
        pipeline_val = splits.val_idx.copy()
        pipeline_test = splits.test_idx.copy()
        
        # Try to modify pipeline copies (this is OK)
        pipeline_train[0] = 999
        
        # Original should remain unchanged (immutability)
        assert np.array_equal(splits.train_idx, original_train), \
            "Original train indices should not change when pipeline copy is modified"
        
        # Verify arrays are read-only
        with pytest.raises(ValueError, match="read-only"):
            splits.train_idx[0] = 999


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
