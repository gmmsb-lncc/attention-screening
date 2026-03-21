#!/usr/bin/env python3
"""
Test 3.4: Early Stopping
Tests early stopping mechanism during training.

Tests:
1. Early stopping triggers when patience exceeded
2. Patience parameter behavior
3. Min delta threshold
4. Best model tracking
5. Monitor metric selection (roc_auc, loss, etc.)
6. Monitor mode (max/min)
7. Early stopping history tracking
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


def test_1_early_stopping_triggers():
    """Test 1: Early stopping triggers when patience exceeded."""
    print("\n" + "="*60)
    print("Test 3.4.1: Early Stopping Triggers")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=64, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create model with aggressive early stopping
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=50,  # High max epochs
            patience=3,      # Low patience - should stop early
            min_delta=0.001,
            monitor_metric="roc_auc",
            monitor_mode="max",
            amp_enabled=False
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Check early stopping triggered
        assert history.early_stopped, "Early stopping should have triggered"
        assert history.total_epochs < config.max_epochs, \
            f"Should stop before max_epochs: {history.total_epochs} < {config.max_epochs}"
        
        print(f"✅ Early stopping triggered")
        print(f"✅ Stopped at epoch: {history.total_epochs}/{config.max_epochs}")
        print(f"✅ Best epoch: {history.best_epoch}")
        print(f"✅ Best metric: {history.best_metric_value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_patience_behavior():
    """Test 2: Patience parameter controls stopping."""
    print("\n" + "="*60)
    print("Test 3.4.2: Patience Behavior")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=64, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Test low patience
        model1 = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config1 = TrainingConfig(
            max_epochs=30,
            patience=2,  # Very low patience
            min_delta=0.001,
            amp_enabled=False
        )
        
        trainer1 = ModelTrainer(model=model1, config=config1, device=torch.device('cpu'))
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        trainer1.setup_training(optimizer=optimizer1)
        
        history1 = trainer1.train(train_loader=train_loader, val_loader=val_loader)
        
        print(f"   Low patience (2): stopped at epoch {history1.total_epochs}")
        
        # Test high patience
        model2 = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config2 = TrainingConfig(
            max_epochs=30,
            patience=20,  # High patience
            min_delta=0.001,
            amp_enabled=False
        )
        
        trainer2 = ModelTrainer(model=model2, config=config2, device=torch.device('cpu'))
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        trainer2.setup_training(optimizer=optimizer2)
        
        history2 = trainer2.train(train_loader=train_loader, val_loader=val_loader)
        
        print(f"   High patience (20): stopped at epoch {history2.total_epochs}")
        
        # High patience should train longer (or reach max_epochs)
        assert history2.total_epochs >= history1.total_epochs, \
            "High patience should train at least as long as low patience"
        
        print(f"✅ Patience controls stopping correctly")
        print(f"✅ Low patience: {history1.total_epochs} epochs")
        print(f"✅ High patience: {history2.total_epochs} epochs")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_min_delta_threshold():
    """Test 3: Min delta threshold for improvement."""
    print("\n" + "="*60)
    print("Test 3.4.3: Min Delta Threshold")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=64, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Test with strict min_delta
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=30,
            patience=5,
            min_delta=0.1,  # Large min_delta - requires significant improvement
            monitor_metric="roc_auc",
            monitor_mode="max",
            amp_enabled=False
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # With large min_delta, should stop relatively early
        print(f"✅ Min delta: {config.min_delta}")
        print(f"✅ Stopped at epoch: {history.total_epochs}")
        print(f"✅ Best metric: {history.best_metric_value:.4f}")
        print(f"✅ Early stopped: {history.early_stopped}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_best_model_tracking():
    """Test 4: Best model and epoch tracking."""
    print("\n" + "="*60)
    print("Test 3.4.4: Best Model Tracking")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=64, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Train with early stopping
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=20,
            patience=5,
            min_delta=0.001,
            monitor_metric="roc_auc",
            monitor_mode="max",
            amp_enabled=False
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer.setup_training(optimizer=optimizer)
        
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Check best model tracked
        assert history.best_epoch is not None, "Best epoch should be tracked"
        assert history.best_metric_value is not None, "Best metric should be tracked"
        assert 1 <= history.best_epoch <= history.total_epochs, \
            f"Best epoch {history.best_epoch} should be in range [1, {history.total_epochs}]"
        
        # Find best metric in validation metrics
        # ClassificationMetrics é um dataclass, não um dict - acessar atributos diretamente
        val_metrics_roc_auc = [m.roc_auc for m in history.val_metrics]
        max_roc_auc = max(val_metrics_roc_auc)
        
        # Best metric value should match or be close to max observed
        assert abs(history.best_metric_value - max_roc_auc) < 0.01, \
            f"Best metric {history.best_metric_value} should match max {max_roc_auc}"
        
        print(f"✅ Best epoch: {history.best_epoch}")
        print(f"✅ Best metric: {history.best_metric_value:.4f}")
        print(f"✅ Total epochs: {history.total_epochs}")
        print(f"✅ Max observed metric: {max_roc_auc:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_monitor_metric_selection():
    """Test 5: Different monitor metrics."""
    print("\n" + "="*60)
    print("Test 3.4.5: Monitor Metric Selection")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=64, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Test monitoring roc_auc
        model1 = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config1 = TrainingConfig(
            max_epochs=10,
            patience=5,
            monitor_metric="roc_auc",
            monitor_mode="max",
            amp_enabled=False
        )
        
        trainer1 = ModelTrainer(model=model1, config=config1, device=torch.device('cpu'))
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        trainer1.setup_training(optimizer=optimizer1)
        
        history1 = trainer1.train(train_loader=train_loader, val_loader=val_loader)
        
        print(f"✅ Monitoring roc_auc: best={history1.best_metric_value:.4f} at epoch {history1.best_epoch}")
        
        # Test monitoring accuracy
        model2 = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config2 = TrainingConfig(
            max_epochs=10,
            patience=5,
            monitor_metric="accuracy",
            monitor_mode="max",
            amp_enabled=False
        )
        
        trainer2 = ModelTrainer(model=model2, config=config2, device=torch.device('cpu'))
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        trainer2.setup_training(optimizer=optimizer2)
        
        history2 = trainer2.train(train_loader=train_loader, val_loader=val_loader)
        
        print(f"✅ Monitoring accuracy: best={history2.best_metric_value:.4f} at epoch {history2.best_epoch}")
        
        # Both should complete successfully
        assert history1.best_metric_value is not None, "ROC-AUC monitoring should work"
        assert history2.best_metric_value is not None, "Accuracy monitoring should work"
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_monitor_mode():
    """Test 6: Monitor mode (max/min)."""
    print("\n" + "="*60)
    print("Test 3.4.6: Monitor Mode (max/min)")
    print("="*60)
    
    try:
        # Test max mode (for roc_auc - higher is better)
        config_max = TrainingConfig(
            max_epochs=10,
            patience=5,
            monitor_metric="roc_auc",
            monitor_mode="max",  # Higher is better
            amp_enabled=False
        )
        
        assert config_max.monitor_mode == "max", "Mode should be max"
        print(f"✅ Max mode validated for roc_auc")
        
        # Test min mode would be used for loss (not directly monitored in validation)
        # But we can still validate the config accepts it
        config_min = TrainingConfig(
            max_epochs=10,
            patience=5,
            monitor_metric="roc_auc",  # Still use roc_auc but with min mode
            monitor_mode="min",  # Lower is better (unusual for roc_auc but valid)
            amp_enabled=False
        )
        
        assert config_min.monitor_mode == "min", "Mode should be min"
        print(f"✅ Min mode validated")
        
        # Test invalid mode raises error
        try:
            config_invalid = TrainingConfig(
                max_epochs=10,
                patience=5,
                monitor_metric="roc_auc",
                monitor_mode="invalid",  # Invalid
                amp_enabled=False
            )
            print(f"❌ Should have raised error for invalid mode")
            return False
        except ValueError as e:
            print(f"✅ Correctly rejected invalid mode: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_early_stopping_history():
    """Test 7: Early stopping history tracking."""
    print("\n" + "="*60)
    print("Test 3.4.7: Early Stopping History")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=64, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Train with early stopping
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(
            max_epochs=50,
            patience=3,
            min_delta=0.001,
            monitor_metric="roc_auc",
            monitor_mode="max",
            amp_enabled=False
        )
        
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Check history completeness
        assert isinstance(history.early_stopped, bool), "early_stopped should be bool"
        assert history.total_epochs > 0, "total_epochs should be positive"
        assert len(history.train_losses) == history.total_epochs, \
            f"train_losses length {len(history.train_losses)} != epochs {history.total_epochs}"
        assert len(history.val_losses) == history.total_epochs, \
            f"val_losses length {len(history.val_losses)} != epochs {history.total_epochs}"
        
        print(f"✅ Early stopped: {history.early_stopped}")
        print(f"✅ Total epochs: {history.total_epochs}")
        print(f"✅ Best epoch: {history.best_epoch}")
        print(f"✅ Best metric: {history.best_metric_value:.4f}")
        print(f"✅ History complete with {len(history.train_losses)} epochs")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LEVEL 3.4: EARLY STOPPING TESTS")
    print("="*60)
    
    tests = [
        ("test_1_early_stopping_triggers", test_1_early_stopping_triggers),
        ("test_2_patience_behavior", test_2_patience_behavior),
        ("test_3_min_delta_threshold", test_3_min_delta_threshold),
        ("test_4_best_model_tracking", test_4_best_model_tracking),
        ("test_5_monitor_metric_selection", test_5_monitor_metric_selection),
        ("test_6_monitor_mode", test_6_monitor_mode),
        ("test_7_early_stopping_history", test_7_early_stopping_history),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY - Level 3.4: Early Stopping")
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
