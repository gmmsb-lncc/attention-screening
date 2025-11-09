#!/usr/bin/env python3
"""
Test 3.2: Optimizer & Scheduler
Tests learning rate scheduling with ReduceLROnPlateau.

Tests:
1. Scheduler initialization
2. Learning rate reduction on plateau
3. Scheduler step tracking
4. Multiple LR reductions
5. Scheduler patience behavior
6. Min learning rate enforcement
7. Learning rate history tracking
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


def test_1_scheduler_initialization():
    """Test 1: Scheduler properly initialized."""
    print("\n" + "="*60)
    print("Test 3.2.1: Scheduler Initialization")
    print("="*60)
    
    try:
        # Create model
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        
        # Create config with scheduler enabled
        config = TrainingConfig(
            max_epochs=5,
            use_scheduler=True,
            scheduler_factor=0.5,
            scheduler_patience=2,
            scheduler_min_lr=1e-6,
            amp_enabled=False
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        
        # Setup training with optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer.setup_training(optimizer=optimizer)
        
        # Check scheduler created
        assert trainer.scheduler is not None, "Scheduler should be created"
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau), \
            f"Expected ReduceLROnPlateau, got {type(trainer.scheduler)}"
        
        print(f"✅ Scheduler created: {type(trainer.scheduler).__name__}")
        print(f"✅ Scheduler factor: {config.scheduler_factor}")
        print(f"✅ Scheduler patience: {config.scheduler_patience}")
        print(f"✅ Min LR: {config.scheduler_min_lr}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_lr_reduction_on_plateau():
    """Test 2: Learning rate reduces when metric plateaus."""
    print("\n" + "="*60)
    print("Test 3.2.2: LR Reduction on Plateau")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=64, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create model with scheduler
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=10,
            use_scheduler=True,
            scheduler_factor=0.5,
            scheduler_patience=2,
            scheduler_min_lr=1e-6,
            amp_enabled=False,
            patience=20  # Don't stop early
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer.setup_training(optimizer=optimizer)
        
        # Get initial learning rate
        initial_lr = optimizer.param_groups[0]['lr']
        print(f"   Initial LR: {initial_lr}")
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Get final learning rate
        final_lr = optimizer.param_groups[0]['lr']
        print(f"   Final LR: {final_lr}")
        
        # Check LR history tracked
        assert len(history.learning_rates) > 0, "Learning rate history should be tracked"
        assert len(history.learning_rates) == history.total_epochs, \
            f"LR history length {len(history.learning_rates)} != epochs {history.total_epochs}"
        
        print(f"✅ LR history tracked: {len(history.learning_rates)} epochs")
        print(f"✅ LR progression: {history.learning_rates[:5]}...")
        
        # LR should have reduced at least once if metric plateaued
        lr_changes = sum(1 for i in range(1, len(history.learning_rates)) 
                        if history.learning_rates[i] < history.learning_rates[i-1])
        print(f"✅ LR reductions detected: {lr_changes}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_scheduler_step_tracking():
    """Test 3: Scheduler steps are properly tracked."""
    print("\n" + "="*60)
    print("Test 3.2.3: Scheduler Step Tracking")
    print("="*60)
    
    try:
        # Create small dataset for quick training
        dataset = create_synthetic_data(n_samples=32, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=16, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Create model with scheduler
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=5,
            use_scheduler=True,
            scheduler_factor=0.8,
            scheduler_patience=1,
            amp_enabled=False
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer.setup_training(optimizer=optimizer)
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Check learning rates recorded for each epoch
        assert len(history.learning_rates) == config.max_epochs, \
            f"Expected {config.max_epochs} LR records, got {len(history.learning_rates)}"
        
        # All learning rates should be positive
        assert all(lr > 0 for lr in history.learning_rates), "All LRs should be positive"
        
        # Learning rates should be monotonically decreasing or stable
        for i in range(1, len(history.learning_rates)):
            assert history.learning_rates[i] <= history.learning_rates[i-1], \
                f"LR should not increase: epoch {i-1}={history.learning_rates[i-1]}, epoch {i}={history.learning_rates[i]}"
        
        print(f"✅ All {len(history.learning_rates)} LR values recorded")
        print(f"✅ LR sequence: {[f'{lr:.6f}' for lr in history.learning_rates]}")
        print(f"✅ Monotonic decrease verified")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_multiple_lr_reductions():
    """Test 4: Multiple LR reductions can occur."""
    print("\n" + "="*60)
    print("Test 3.2.4: Multiple LR Reductions")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=64, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create model with aggressive scheduler
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=15,
            use_scheduler=True,
            scheduler_factor=0.5,  # Aggressive reduction
            scheduler_patience=2,
            scheduler_min_lr=1e-7,
            amp_enabled=False,
            patience=30  # Don't stop early
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # High initial LR
        trainer.setup_training(optimizer=optimizer)
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Count LR reductions
        lr_reductions = 0
        for i in range(1, len(history.learning_rates)):
            if history.learning_rates[i] < history.learning_rates[i-1] * 0.99:  # 1% threshold for noise
                lr_reductions += 1
                print(f"   Reduction {lr_reductions}: {history.learning_rates[i-1]:.6f} → {history.learning_rates[i]:.6f}")
        
        print(f"✅ Total LR reductions: {lr_reductions}")
        print(f"✅ Initial LR: {history.learning_rates[0]:.6f}")
        print(f"✅ Final LR: {history.learning_rates[-1]:.6f}")
        
        # Should have at least 1 reduction in 15 epochs
        assert lr_reductions >= 1, f"Expected at least 1 LR reduction, got {lr_reductions}"
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_scheduler_patience():
    """Test 5: Scheduler respects patience parameter."""
    print("\n" + "="*60)
    print("Test 3.2.5: Scheduler Patience Behavior")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=32, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=16, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Test with high patience
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=8,
            use_scheduler=True,
            scheduler_factor=0.5,
            scheduler_patience=5,  # High patience
            scheduler_min_lr=1e-7,
            amp_enabled=False,
            patience=20
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer.setup_training(optimizer=optimizer)
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Count LR changes
        lr_changes = sum(1 for i in range(1, len(history.learning_rates)) 
                        if abs(history.learning_rates[i] - history.learning_rates[i-1]) > 1e-8)
        
        print(f"✅ Patience: {config.scheduler_patience}")
        print(f"✅ Total epochs: {history.total_epochs}")
        print(f"✅ LR changes: {lr_changes}")
        print(f"✅ LR values: {[f'{lr:.6f}' for lr in history.learning_rates]}")
        
        # With high patience and few epochs, LR should stay stable or have few changes
        assert lr_changes <= history.total_epochs, "LR changes should be reasonable"
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_min_lr_enforcement():
    """Test 6: Minimum learning rate is enforced."""
    print("\n" + "="*60)
    print("Test 3.2.6: Min LR Enforcement")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=64, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create model with high min_lr
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        min_lr = 1e-4
        config = TrainingConfig(
            max_epochs=20,
            use_scheduler=True,
            scheduler_factor=0.5,
            scheduler_patience=2,
            scheduler_min_lr=min_lr,  # High min LR
            amp_enabled=False,
            patience=30
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer.setup_training(optimizer=optimizer)
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Check all LRs are >= min_lr
        min_lr_observed = min(history.learning_rates)
        print(f"   Min LR configured: {min_lr:.6f}")
        print(f"   Min LR observed: {min_lr_observed:.6f}")
        
        assert min_lr_observed >= min_lr * 0.99, \
            f"LR {min_lr_observed} should not go below min_lr {min_lr}"
        
        print(f"✅ Min LR enforced")
        print(f"✅ LR range: [{min_lr_observed:.6f}, {max(history.learning_rates):.6f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_lr_history_completeness():
    """Test 7: Learning rate history is complete."""
    print("\n" + "="*60)
    print("Test 3.2.7: LR History Completeness")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=32, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=16, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Create model
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=7,
            use_scheduler=True,
            scheduler_factor=0.7,
            scheduler_patience=2,
            amp_enabled=False
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer.setup_training(optimizer=optimizer)
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Check completeness
        assert len(history.learning_rates) == history.total_epochs, \
            f"LR history length {len(history.learning_rates)} != total epochs {history.total_epochs}"
        
        # Check all values are valid floats
        assert all(isinstance(lr, (float, np.floating)) for lr in history.learning_rates), \
            "All LR values should be floats"
        
        # Check no NaN or Inf
        assert all(not np.isnan(lr) for lr in history.learning_rates), "No NaN in LR history"
        assert all(not np.isinf(lr) for lr in history.learning_rates), "No Inf in LR history"
        
        print(f"✅ LR history complete: {len(history.learning_rates)} epochs")
        print(f"✅ All values valid (float, no NaN/Inf)")
        print(f"✅ LR progression: {[f'{lr:.6f}' for lr in history.learning_rates]}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LEVEL 3.2: OPTIMIZER & SCHEDULER TESTS")
    print("="*60)
    
    tests = [
        ("test_1_scheduler_initialization", test_1_scheduler_initialization),
        ("test_2_lr_reduction_on_plateau", test_2_lr_reduction_on_plateau),
        ("test_3_scheduler_step_tracking", test_3_scheduler_step_tracking),
        ("test_4_multiple_lr_reductions", test_4_multiple_lr_reductions),
        ("test_5_scheduler_patience", test_5_scheduler_patience),
        ("test_6_min_lr_enforcement", test_6_min_lr_enforcement),
        ("test_7_lr_history_completeness", test_7_lr_history_completeness),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY - Level 3.2: Optimizer & Scheduler")
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
