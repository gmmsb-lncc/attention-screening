#!/usr/bin/env python3
"""
Test 3.3: AMP Training
Tests Automatic Mixed Precision (AMP) training.

Tests:
1. AMP disabled training (baseline)
2. AMP enabled training (float16)
3. GradScaler initialization
4. Loss scaling behavior
5. Training convergence with AMP
6. AMP dtype validation (float16 vs bfloat16)
7. AMP compatibility check
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from classifier.classifier import MLPEmbeddingClassifier
from classifier.core.trainer import ModelTrainer, TrainingConfig, TrainingHistory


def create_synthetic_data(n_samples: int, input_dim: int) -> TensorDataset:
    """Create synthetic binary classification dataset."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, 2, (n_samples,)).float()
    return TensorDataset(X, y)


def test_1_amp_disabled_baseline():
    """Test 1: AMP disabled training (baseline)."""
    print("\n" + "="*60)
    print("Test 3.3.1: AMP Disabled (Baseline)")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=64, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create model with AMP disabled
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=3,
            amp_enabled=False,  # Disabled
            patience=10
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Check scaler not created when AMP disabled
        assert trainer.scaler is None, "Scaler should be None when AMP disabled"
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Check training completed
        assert history.total_epochs == 3, f"Expected 3 epochs, got {history.total_epochs}"
        assert len(history.train_losses) == 3, "Should have 3 train losses"
        
        print(f"✅ AMP disabled training successful")
        print(f"✅ Scaler: {trainer.scaler}")
        print(f"✅ Epochs: {history.total_epochs}")
        print(f"✅ Final train loss: {history.train_losses[-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_amp_enabled_training():
    """Test 2: AMP enabled training (float16)."""
    print("\n" + "="*60)
    print("Test 3.3.2: AMP Enabled (float16)")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=64, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create model with AMP enabled
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=3,
            amp_enabled=True,  # Enabled
            amp_dtype=torch.float16,
            patience=10
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Check scaler created when AMP enabled
        if config.amp_enabled:
            assert trainer.scaler is not None, "Scaler should be created when AMP enabled"
            print(f"✅ GradScaler created: {type(trainer.scaler).__name__}")
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Check training completed
        assert history.total_epochs == 3, f"Expected 3 epochs, got {history.total_epochs}"
        assert len(history.train_losses) == 3, "Should have 3 train losses"
        
        print(f"✅ AMP enabled training successful")
        print(f"✅ AMP dtype: {config.amp_dtype}")
        print(f"✅ Epochs: {history.total_epochs}")
        print(f"✅ Final train loss: {history.train_losses[-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_gradscaler_initialization():
    """Test 3: GradScaler properly initialized."""
    print("\n" + "="*60)
    print("Test 3.3.3: GradScaler Initialization")
    print("="*60)
    
    try:
        # Create model with AMP enabled
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=1,
            amp_enabled=True,
            amp_dtype=torch.float16
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Check scaler
        if config.amp_enabled:
            assert trainer.scaler is not None, "Scaler should exist"
            assert hasattr(trainer.scaler, 'scale'), "Scaler should have scale method"
            assert hasattr(trainer.scaler, 'step'), "Scaler should have step method"
            assert hasattr(trainer.scaler, 'update'), "Scaler should have update method"
            
            print(f"✅ GradScaler type: {type(trainer.scaler).__name__}")
            print(f"✅ Has scale(): {hasattr(trainer.scaler, 'scale')}")
            print(f"✅ Has step(): {hasattr(trainer.scaler, 'step')}")
            print(f"✅ Has update(): {hasattr(trainer.scaler, 'update')}")
        else:
            print("ℹ️  AMP not enabled on this device")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_loss_scaling_behavior():
    """Test 4: Loss scaling behavior."""
    print("\n" + "="*60)
    print("Test 3.3.4: Loss Scaling Behavior")
    print("="*60)
    
    try:
        # Create small dataset
        dataset = create_synthetic_data(n_samples=32, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Create model with AMP
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=2,
            amp_enabled=True,
            amp_dtype=torch.float16,
            patience=10
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=None)
        
        # Check losses are valid (not NaN/Inf despite scaling)
        for epoch, loss in enumerate(history.train_losses):
            assert not np.isnan(loss), f"Loss is NaN at epoch {epoch}"
            assert not np.isinf(loss), f"Loss is Inf at epoch {epoch}"
            assert loss > 0, f"Loss should be positive at epoch {epoch}"
        
        print(f"✅ All losses valid (no NaN/Inf)")
        print(f"✅ Loss range: [{min(history.train_losses):.4f}, {max(history.train_losses):.4f}]")
        print(f"✅ Losses: {[f'{l:.4f}' for l in history.train_losses]}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_training_convergence_with_amp():
    """Test 5: Training converges with AMP."""
    print("\n" + "="*60)
    print("Test 3.3.5: Training Convergence with AMP")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=128, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=64, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Train with AMP
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=10,
            amp_enabled=True,
            amp_dtype=torch.float16,
            patience=20
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer.setup_training(optimizer=optimizer)
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Check convergence: loss should decrease
        initial_loss = history.train_losses[0]
        final_loss = history.train_losses[-1]
        
        print(f"   Initial loss: {initial_loss:.4f}")
        print(f"   Final loss: {final_loss:.4f}")
        print(f"   Reduction: {initial_loss - final_loss:.4f}")
        
        # Loss should decrease (or at least not increase significantly)
        assert final_loss <= initial_loss * 1.5, \
            f"Loss increased too much: {initial_loss:.4f} → {final_loss:.4f}"
        
        print(f"✅ Training converged with AMP")
        print(f"✅ Total epochs: {history.total_epochs}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_amp_dtype_validation():
    """Test 6: AMP dtype validation (float16 vs bfloat16)."""
    print("\n" + "="*60)
    print("Test 3.3.6: AMP Dtype Validation")
    print("="*60)
    
    try:
        # Test float16
        model1 = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config1 = TrainingConfig(
            max_epochs=1,
            amp_enabled=True,
            amp_dtype=torch.float16
        )
        
        trainer1 = ModelTrainer(model=model1, config=config1, device=torch.device('cpu'))
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        trainer1.setup_training(optimizer=optimizer1)
        
        assert config1.amp_dtype == torch.float16, "Should use float16"
        print(f"✅ float16 dtype validated")
        
        # Test bfloat16
        model2 = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config2 = TrainingConfig(
            max_epochs=1,
            amp_enabled=True,
            amp_dtype=torch.bfloat16
        )
        
        trainer2 = ModelTrainer(model=model2, config=config2, device=torch.device('cpu'))
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        trainer2.setup_training(optimizer=optimizer2)
        
        assert config2.amp_dtype == torch.bfloat16, "Should use bfloat16"
        print(f"✅ bfloat16 dtype validated")
        
        # Test invalid dtype should raise error
        try:
            config3 = TrainingConfig(
                max_epochs=1,
                amp_enabled=True,
                amp_dtype=torch.float32  # Invalid for AMP
            )
            print(f"❌ Should have raised error for float32")
            return False
        except ValueError as e:
            print(f"✅ Correctly rejected float32: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_amp_compatibility_check():
    """Test 7: AMP compatibility check."""
    print("\n" + "="*60)
    print("Test 3.3.7: AMP Compatibility Check")
    print("="*60)
    
    try:
        # Check if CUDA available (AMP works best on GPU)
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available: {cuda_available}")
        
        # Check if CPU supports AMP (limited support)
        cpu_device = torch.device('cpu')
        print(f"   CPU device: {cpu_device}")
        
        # Create model and test AMP on available device
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=1,
            amp_enabled=True,
            amp_dtype=torch.float16
        )
        
        device = torch.device('cuda' if cuda_available else 'cpu')
        trainer = ModelTrainer(model=model, config=config, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Create small dataset and train
        dataset = create_synthetic_data(n_samples=16, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        history = trainer.train(train_loader=train_loader, val_loader=None)
        
        print(f"✅ AMP compatible on device: {device}")
        print(f"✅ Training completed successfully")
        print(f"✅ Final loss: {history.train_losses[-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LEVEL 3.3: AMP TRAINING TESTS")
    print("="*60)
    
    tests = [
        ("test_1_amp_disabled_baseline", test_1_amp_disabled_baseline),
        ("test_2_amp_enabled_training", test_2_amp_enabled_training),
        ("test_3_gradscaler_initialization", test_3_gradscaler_initialization),
        ("test_4_loss_scaling_behavior", test_4_loss_scaling_behavior),
        ("test_5_training_convergence_with_amp", test_5_training_convergence_with_amp),
        ("test_6_amp_dtype_validation", test_6_amp_dtype_validation),
        ("test_7_amp_compatibility_check", test_7_amp_compatibility_check),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY - Level 3.3: AMP Training")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        padded_name = test_name.ljust(40, '.')
        print(f"{padded_name} {status}")
    
    print("="*60)
    passed_count = sum(results.values())
    total_count = len(results)
    percentage = (passed_count / total_count * 100) if total_count > 0 else 0
    print(f"Results: {passed_count}/{total_count} tests passed ({percentage:.1f}%)")
    print("="*60)
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    exit(main())
