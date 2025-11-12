"""
Unit tests for SplitIndices data class.

Tests validation, immutability, and save/load functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.build.pipeline.split_indices import SplitIndices


class TestSplitIndicesValidation:
    """Test validation logic for SplitIndices."""
    
    def test_valid_splits(self):
        """Test creating valid split indices."""
        train_idx = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        val_idx = np.array([5, 6], dtype=np.int32)
        test_idx = np.array([7, 8, 9], dtype=np.int32)
        
        splits = SplitIndices(train_idx, val_idx, test_idx)
        
        assert np.array_equal(splits.train_idx, train_idx)
        assert np.array_equal(splits.val_idx, val_idx)
        assert np.array_equal(splits.test_idx, test_idx)
    
    def test_auto_convert_to_int32(self):
        """Test automatic conversion to int32."""
        train_idx = np.array([0, 1, 2], dtype=np.int64)
        val_idx = np.array([3, 4], dtype=np.float64)
        test_idx = np.array([5, 6], dtype=np.int16)
        
        splits = SplitIndices(train_idx, val_idx, test_idx)
        
        assert splits.train_idx.dtype == np.int32
        assert splits.val_idx.dtype == np.int32
        assert splits.test_idx.dtype == np.int32
    
    def test_reject_overlap_train_val(self):
        """Test rejection of overlapping train and val indices."""
        train_idx = np.array([0, 1, 2], dtype=np.int32)
        val_idx = np.array([2, 3], dtype=np.int32)  # 2 overlaps!
        test_idx = np.array([4, 5], dtype=np.int32)
        
        with pytest.raises(ValueError, match="overlap"):
            SplitIndices(train_idx, val_idx, test_idx)
    
    def test_reject_overlap_train_test(self):
        """Test rejection of overlapping train and test indices."""
        train_idx = np.array([0, 1, 2], dtype=np.int32)
        val_idx = np.array([3, 4], dtype=np.int32)
        test_idx = np.array([2, 5], dtype=np.int32)  # 2 overlaps!
        
        with pytest.raises(ValueError, match="overlap"):
            SplitIndices(train_idx, val_idx, test_idx)
    
    def test_reject_overlap_val_test(self):
        """Test rejection of overlapping val and test indices."""
        train_idx = np.array([0, 1, 2], dtype=np.int32)
        val_idx = np.array([3, 4], dtype=np.int32)
        test_idx = np.array([4, 5], dtype=np.int32)  # 4 overlaps!
        
        with pytest.raises(ValueError, match="overlap"):
            SplitIndices(train_idx, val_idx, test_idx)
    
    def test_reject_duplicate_within_train(self):
        """Test rejection of duplicate indices within train."""
        train_idx = np.array([0, 1, 1, 2], dtype=np.int32)  # 1 duplicated!
        val_idx = np.array([3, 4], dtype=np.int32)
        test_idx = np.array([5, 6], dtype=np.int32)
        
        with pytest.raises(ValueError, match="unique"):
            SplitIndices(train_idx, val_idx, test_idx)
    
    def test_reject_empty_train(self):
        """Test rejection of empty train set."""
        train_idx = np.array([], dtype=np.int32)
        val_idx = np.array([0, 1], dtype=np.int32)
        test_idx = np.array([2, 3], dtype=np.int32)
        
        with pytest.raises(ValueError, match="empty"):
            SplitIndices(train_idx, val_idx, test_idx)
    
    def test_reject_empty_test(self):
        """Test rejection of empty test set."""
        train_idx = np.array([0, 1], dtype=np.int32)
        val_idx = np.array([2, 3], dtype=np.int32)
        test_idx = np.array([], dtype=np.int32)
        
        with pytest.raises(ValueError, match="empty"):
            SplitIndices(train_idx, val_idx, test_idx)
    
    def test_allow_empty_val(self):
        """Test that empty validation set is allowed."""
        train_idx = np.array([0, 1, 2], dtype=np.int32)
        val_idx = np.array([], dtype=np.int32)
        test_idx = np.array([3, 4], dtype=np.int32)
        
        splits = SplitIndices(train_idx, val_idx, test_idx)
        assert len(splits.val_idx) == 0


class TestSplitIndicesMetadata:
    """Test metadata functionality."""
    
    def test_default_metadata(self):
        """Test default metadata creation."""
        train_idx = np.array([0, 1, 2], dtype=np.int32)
        val_idx = np.array([3, 4], dtype=np.int32)
        test_idx = np.array([5, 6], dtype=np.int32)
        
        splits = SplitIndices(train_idx, val_idx, test_idx)
        
        assert 'train_size' in splits.metadata
        assert 'val_size' in splits.metadata
        assert 'test_size' in splits.metadata
        assert 'total_size' in splits.metadata
        assert 'created_at' in splits.metadata
        
        assert splits.metadata['train_size'] == 3
        assert splits.metadata['val_size'] == 2
        assert splits.metadata['test_size'] == 2
        assert splits.metadata['total_size'] == 7
    
    def test_custom_metadata(self):
        """Test custom metadata."""
        train_idx = np.array([0, 1], dtype=np.int32)
        val_idx = np.array([2], dtype=np.int32)
        test_idx = np.array([3], dtype=np.int32)
        
        custom_meta = {
            'algorithm': 'kmeans',
            'random_state': 42
        }
        
        splits = SplitIndices(train_idx, val_idx, test_idx, metadata=custom_meta)
        
        assert splits.metadata['algorithm'] == 'kmeans'
        assert splits.metadata['random_state'] == 42


class TestSplitIndicesSaveLoad:
    """Test save/load functionality."""
    
    def setup_method(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load(self):
        """Test saving and loading splits."""
        train_idx = np.array([0, 1, 2, 3], dtype=np.int32)
        val_idx = np.array([4, 5], dtype=np.int32)
        test_idx = np.array([6, 7, 8], dtype=np.int32)
        
        metadata = {'algorithm': 'kmeans', 'random_state': 42}
        splits = SplitIndices(train_idx, val_idx, test_idx, metadata=metadata)
        
        # Save
        filepath = Path(self.temp_dir) / "splits.npz"
        splits.save(str(filepath))
        
        assert filepath.exists()
        
        # Load
        loaded_splits = SplitIndices.load(str(filepath))
        
        # Verify indices
        assert np.array_equal(loaded_splits.train_idx, train_idx)
        assert np.array_equal(loaded_splits.val_idx, val_idx)
        assert np.array_equal(loaded_splits.test_idx, test_idx)
        
        # Verify metadata
        assert loaded_splits.metadata['algorithm'] == 'kmeans'
        assert loaded_splits.metadata['random_state'] == 42
    
    def test_save_with_empty_val(self):
        """Test saving splits with empty validation set."""
        train_idx = np.array([0, 1, 2], dtype=np.int32)
        val_idx = np.array([], dtype=np.int32)
        test_idx = np.array([3, 4], dtype=np.int32)
        
        splits = SplitIndices(train_idx, val_idx, test_idx)
        
        filepath = Path(self.temp_dir) / "splits_no_val.npz"
        splits.save(str(filepath))
        
        loaded_splits = SplitIndices.load(str(filepath))
        
        assert len(loaded_splits.val_idx) == 0
        assert np.array_equal(loaded_splits.train_idx, train_idx)
        assert np.array_equal(loaded_splits.test_idx, test_idx)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        filepath = Path(self.temp_dir) / "nonexistent.npz"
        
        with pytest.raises(FileNotFoundError):
            SplitIndices.load(str(filepath))


class TestSplitIndicesConversion:
    """Test conversion to/from dict."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        train_idx = np.array([0, 1, 2], dtype=np.int32)
        val_idx = np.array([3, 4], dtype=np.int32)
        test_idx = np.array([5, 6], dtype=np.int32)
        
        splits = SplitIndices(train_idx, val_idx, test_idx)
        splits_dict = splits.to_dict()
        
        assert 'train_idx' in splits_dict
        assert 'val_idx' in splits_dict
        assert 'test_idx' in splits_dict
        assert 'metadata' in splits_dict
        
        assert np.array_equal(splits_dict['train_idx'], train_idx)
        assert np.array_equal(splits_dict['val_idx'], val_idx)
        assert np.array_equal(splits_dict['test_idx'], test_idx)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        splits_dict = {
            'train_idx': np.array([0, 1, 2], dtype=np.int32),
            'val_idx': np.array([3, 4], dtype=np.int32),
            'test_idx': np.array([5, 6], dtype=np.int32),
            'metadata': {'algorithm': 'kmeans'}
        }
        
        splits = SplitIndices.from_dict(splits_dict)
        
        assert np.array_equal(splits.train_idx, splits_dict['train_idx'])
        assert np.array_equal(splits.val_idx, splits_dict['val_idx'])
        assert np.array_equal(splits.test_idx, splits_dict['test_idx'])
        assert splits.metadata['algorithm'] == 'kmeans'
    
    def test_roundtrip_conversion(self):
        """Test to_dict -> from_dict roundtrip."""
        train_idx = np.array([0, 1, 2], dtype=np.int32)
        val_idx = np.array([3, 4], dtype=np.int32)
        test_idx = np.array([5, 6], dtype=np.int32)
        
        original = SplitIndices(train_idx, val_idx, test_idx)
        reconstructed = SplitIndices.from_dict(original.to_dict())
        
        assert np.array_equal(reconstructed.train_idx, original.train_idx)
        assert np.array_equal(reconstructed.val_idx, original.val_idx)
        assert np.array_equal(reconstructed.test_idx, original.test_idx)


class TestSplitIndicesImmutability:
    """Test immutability of SplitIndices."""
    
    def test_cannot_modify_train_idx(self):
        """Test that train_idx cannot be modified after creation."""
        train_idx = np.array([0, 1, 2], dtype=np.int32)
        val_idx = np.array([3, 4], dtype=np.int32)
        test_idx = np.array([5, 6], dtype=np.int32)
        
        splits = SplitIndices(train_idx, val_idx, test_idx)
        
        # Should not be able to modify
        with pytest.raises(ValueError, match="read-only"):
            splits.train_idx[0] = 999
    
    def test_cannot_modify_val_idx(self):
        """Test that val_idx cannot be modified after creation."""
        train_idx = np.array([0, 1, 2], dtype=np.int32)
        val_idx = np.array([3, 4], dtype=np.int32)
        test_idx = np.array([5, 6], dtype=np.int32)
        
        splits = SplitIndices(train_idx, val_idx, test_idx)
        
        with pytest.raises(ValueError, match="read-only"):
            splits.val_idx[0] = 999
    
    def test_cannot_modify_test_idx(self):
        """Test that test_idx cannot be modified after creation."""
        train_idx = np.array([0, 1, 2], dtype=np.int32)
        val_idx = np.array([3, 4], dtype=np.int32)
        test_idx = np.array([5, 6], dtype=np.int32)
        
        splits = SplitIndices(train_idx, val_idx, test_idx)
        
        with pytest.raises(ValueError, match="read-only"):
            splits.test_idx[0] = 999


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
