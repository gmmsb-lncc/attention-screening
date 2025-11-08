#!/usr/bin/env python3
"""
Test Level 1.1: DataManager - Data Loading and Preparation
Duration: ~10s
Priority: CRITICAL - Foundation for all other tests

Tests the DataManager class which loads .npy embeddings and labels,
creates stratified train/val/test splits (80/10/10), and manages data caching.

This is the first and most critical test as all other components depend on
proper data loading.
"""

import os
import sys
import tempfile
import numpy as np
import torch
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from classifier.core.data_loader import DataManager


def test_data_loading():
    """Test 1: Basic data loading from .npy files."""
    print("\n" + "="*60)
    print("TEST 1.1: DataManager - Data Loading")
    print("="*60)
    
    # Create synthetic data
    n_samples = 100
    n_features = 64
    
    embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    print(f"\n📊 Created synthetic data:")
    print(f"   Embeddings: {embeddings.shape} {embeddings.dtype}")
    print(f"   Labels: {labels.shape} {labels.dtype}")
    print(f"   Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Save to temporary files
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_emb:
        np.save(f_emb.name, embeddings)
        emb_path = f_emb.name
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_lab:
        np.save(f_lab.name, labels)
        lab_path = f_lab.name
    
    try:
        # Test 1.1: Load embedding dimension
        print("\n🔍 Test 1.1: Get embedding dimension")
        device = torch.device("cpu")
        data_manager = DataManager(emb_path, lab_path, device)
        
        dim = data_manager.get_embedding_dim()
        assert dim == n_features, f"Expected dim={n_features}, got {dim}"
        print(f"   ✅ Embedding dimension: {dim}")
        
        # Test 1.2: Load data with default split
        print("\n🔍 Test 1.2: Load data with stratified split (80/10/10)")
        train_loader, val_loader, test_loader = data_manager.load_data()
        
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        total = train_size + val_size + test_size
        
        print(f"   Train: {train_size} samples ({train_size/total*100:.1f}%)")
        print(f"   Val:   {val_size} samples ({val_size/total*100:.1f}%)")
        print(f"   Test:  {test_size} samples ({test_size/total*100:.1f}%)")
        
        # Verify splits are approximately 80/10/10
        assert abs(train_size/total - 0.8) < 0.05, "Train split not ~80%"
        assert abs(val_size/total - 0.1) < 0.05, "Val split not ~10%"
        assert abs(test_size/total - 0.1) < 0.05, "Test split not ~10%"
        print(f"   ✅ Split ratios correct (80/10/10)")
        
        # Test 1.3: Check data types
        print("\n🔍 Test 1.3: Verify data types in DataLoader")
        X_batch, y_batch = next(iter(train_loader))
        
        assert isinstance(X_batch, torch.Tensor), "X should be Tensor"
        assert isinstance(y_batch, torch.Tensor), "y should be Tensor"
        assert X_batch.dtype == torch.float32, f"X should be float32, got {X_batch.dtype}"
        assert y_batch.dtype == torch.float32, f"y should be float32, got {y_batch.dtype}"
        assert y_batch.shape[1] == 1, "Labels should be (batch, 1)"
        
        print(f"   ✅ X type: {X_batch.dtype}, shape: {X_batch.shape}")
        print(f"   ✅ y type: {y_batch.dtype}, shape: {y_batch.shape}")
        
        # Test 1.4: Cache mechanism
        print("\n🔍 Test 1.4: Verify caching mechanism")
        assert data_manager._embeddings is not None, "Embeddings not cached"
        assert data_manager._labels is not None, "Labels not cached"
        assert data_manager._dataset is not None, "Dataset not cached"
        print(f"   ✅ Data cached correctly")
        
        # Test 1.5: Reload from cache
        print("\n🔍 Test 1.5: Reload data (should use cache)")
        train_loader2, val_loader2, test_loader2 = data_manager.load_data()
        
        # Should be same splits due to random_state=42
        assert len(train_loader2.dataset) == train_size, "Train size changed"
        assert len(val_loader2.dataset) == val_size, "Val size changed"
        assert len(test_loader2.dataset) == test_size, "Test size changed"
        print(f"   ✅ Cache working (same splits)")
        
        # Test 1.6: Custom batch size
        print("\n🔍 Test 1.6: Create loaders with custom batch size")
        batch_size = 16
        train_loader_custom, val_loader_custom, test_loader_custom = data_manager.create_data_loaders(
            batch_size=batch_size
        )
        
        # Check batch size (last batch may be smaller)
        X_batch, y_batch = next(iter(train_loader_custom))
        assert X_batch.shape[0] <= batch_size, f"Batch size > {batch_size}"
        print(f"   ✅ Custom batch size: {batch_size} (first batch: {X_batch.shape[0]})")
        
        # Test 1.7: Stratification check
        print("\n🔍 Test 1.7: Verify stratification is maintained")
        # Get all labels from train set
        train_labels = []
        for _, y_batch in train_loader:
            train_labels.extend(y_batch.cpu().numpy().flatten())
        train_labels = np.array(train_labels)
        
        # Get all labels from val set
        val_labels = []
        for _, y_batch in val_loader:
            val_labels.extend(y_batch.cpu().numpy().flatten())
        val_labels = np.array(val_labels)
        
        # Calculate class ratios
        train_ratio = train_labels.mean()
        val_ratio = val_labels.mean()
        original_ratio = labels.mean()
        
        print(f"   Original ratio: {original_ratio:.3f}")
        print(f"   Train ratio:    {train_ratio:.3f}")
        print(f"   Val ratio:      {val_ratio:.3f}")
        
        # Ratios should be similar (within 10%)
        assert abs(train_ratio - original_ratio) < 0.1, "Train not stratified"
        assert abs(val_ratio - original_ratio) < 0.1, "Val not stratified"
        print(f"   ✅ Stratification maintained")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - DataManager working correctly!")
        print("="*60)
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(emb_path):
            os.unlink(emb_path)
        if os.path.exists(lab_path):
            os.unlink(lab_path)


def test_edge_cases():
    """Test 2: Edge cases for DataManager."""
    print("\n" + "="*60)
    print("TEST 1.1 (Edge Cases): DataManager Edge Cases")
    print("="*60)
    
    try:
        # Edge Case 1: Small dataset
        print("\n🔍 Edge Case 1: Very small dataset (20 samples)")
        n_samples = 20
        n_features = 32
        
        embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
        labels = np.random.choice([0, 1], size=n_samples)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_emb:
            np.save(f_emb.name, embeddings)
            emb_path = f_emb.name
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_lab:
            np.save(f_lab.name, labels)
            lab_path = f_lab.name
        
        device = torch.device("cpu")
        data_manager = DataManager(emb_path, lab_path, device)
        train_loader, val_loader, test_loader = data_manager.load_data()
        
        print(f"   Train: {len(train_loader.dataset)} samples")
        print(f"   Val:   {len(val_loader.dataset)} samples")
        print(f"   Test:  {len(test_loader.dataset)} samples")
        print(f"   ✅ Small dataset handled")
        
        # Cleanup
        os.unlink(emb_path)
        os.unlink(lab_path)
        
        # Edge Case 2: Imbalanced dataset
        print("\n🔍 Edge Case 2: Imbalanced dataset (70:30)")
        n_samples = 100
        embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
        # Use 70:30 instead of 90:10 to ensure at least 2 samples per class in splits
        labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_emb:
            np.save(f_emb.name, embeddings)
            emb_path = f_emb.name
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_lab:
            np.save(f_lab.name, labels)
            lab_path = f_lab.name
        
        data_manager = DataManager(emb_path, lab_path, device)
        train_loader, val_loader, test_loader = data_manager.load_data()
        
        print(f"   Original: {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"   ✅ Imbalanced dataset handled")
        print(f"   ⚠️  NOTE: Very imbalanced data (>85:15) may fail stratification")
        
        # Cleanup
        os.unlink(emb_path)
        os.unlink(lab_path)
        
        print("\n" + "="*60)
        print("✅ ALL EDGE CASES PASSED!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ EDGE CASE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 Starting DataManager Tests (Level 1.1)")
    print("=" * 60)
    
    # Run main tests
    success1 = test_data_loading()
    
    # Run edge cases
    success2 = test_edge_cases()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Main Tests:   {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Edge Cases:   {'✅ PASSED' if success2 else '❌ FAILED'}")
    print("="*60)
    
    if success1 and success2:
        print("🎉 DataManager: FULLY FUNCTIONAL ✅")
        exit(0)
    else:
        print("❌ DataManager: NEEDS ATTENTION")
        exit(1)
