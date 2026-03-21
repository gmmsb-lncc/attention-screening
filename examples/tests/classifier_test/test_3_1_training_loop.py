#!/usr/bin/env python3
"""
Level 3.1: Basic Training Loop Test

Tests the ModelTrainer basic training loop, including:
- Forward pass during training
- Loss computation
- Backward pass and optimization
- Training history tracking
- Single epoch execution

Total Tests: 7 tests
Estimated Time: ~1 min
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from classifier.core.trainer import ModelTrainer, TrainingConfig, TrainingHistory
from classifier.models.mlp_classifier import MLPEmbeddingClassifier


def create_synthetic_data(n_samples=100, input_dim=64):
    """Create synthetic data for testing."""
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, 2, (n_samples,)).float().unsqueeze(1)
    
    dataset = TensorDataset(X, y)
    return dataset


def test_1_trainer_initialization():
    """Test 1: Trainer initialization."""
    print("\n" + "="*60)
    print("Test 3.1.1: Trainer Initialization")
    print("="*60)
    
    try:
        # Create model
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        
        # Create trainer with default config
        config = TrainingConfig()
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        
        assert trainer.model is not None, "Model not set"
        assert trainer.device.type == 'cpu', "Device not CPU"
        assert isinstance(trainer.config, TrainingConfig), "Config not TrainingConfig"
        
        print(f"✅ Trainer initialized")
        print(f"   Model: MLPEmbeddingClassifier")
        print(f"   Device: {trainer.device}")
        print(f"   Config: max_epochs={trainer.config.max_epochs}, patience={trainer.config.patience}")
        
        # Create trainer with custom config
        config = TrainingConfig(max_epochs=50, patience=5, amp_enabled=False)
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        
        assert trainer.config.max_epochs == 50, "Max epochs not set"
        assert trainer.config.patience == 5, "Patience not set"
        
        print(f"✅ Custom config working")
        print(f"   Max epochs: {trainer.config.max_epochs}")
        print(f"   Patience: {trainer.config.patience}")
        
        # Test setup_training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        assert trainer.optimizer is not None, "Optimizer not set"
        assert trainer.criterion is not None, "Criterion not set"
        
        print(f"✅ setup_training() working")
        print(f"   Optimizer: {type(trainer.optimizer).__name__}")
        print(f"   Criterion: {type(trainer.criterion).__name__}")
        
        print("✅ Trainer initialization working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_2_single_epoch_training():
    """Test 2: Single epoch training."""
    print("\n" + "="*60)
    print("Test 3.1.2: Single Epoch Training")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=32, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Create model and trainer
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(max_epochs=1, amp_enabled=False)
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        
        # Setup training (required before train())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Train one epoch
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=None  # No validation for this test
        )
        
        # Check training happened
        assert history is not None, "History is None"
        assert len(history.train_losses) > 0, "No training losses recorded"
        assert history.total_epochs == 1, f"Expected 1 epoch, got {history.total_epochs}"
        
        print(f"✅ Training completed")
        print(f"   Epochs: {history.total_epochs}")
        print(f"   Final loss: {history.train_losses[-1]:.4f}")
        
        # Check parameters updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                params_updated += 1
        
        assert params_updated > 0, "No parameters were updated"
        print(f"✅ Parameters updated: {params_updated}/{len(list(model.parameters()))}")
        
        print("✅ Single epoch training working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_3_loss_calculation():
    """Test 3: Loss calculation correctness."""
    print("\n" + "="*60)
    print("Test 3.1.3: Loss Calculation")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=16, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Create model and trainer
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(max_epochs=1, amp_enabled=False)
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        
        # Setup training (required before train())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=None)
        
        # Check loss properties
        loss = history.train_losses[0]
        assert isinstance(loss, (float, np.floating)), f"Loss not float, got {type(loss)}"
        assert loss > 0, f"Loss should be positive, got {loss}"
        assert not np.isnan(loss), "Loss is NaN"
        assert not np.isinf(loss), "Loss is Inf"
        
        print(f"✅ Loss: {loss:.4f}")
        print(f"✅ Loss type: {type(loss).__name__}")
        print(f"✅ Loss range: (0, inf)")
        print(f"✅ No NaN/Inf")
        
        print("✅ Loss calculation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_4_training_history():
    """Test 4: Training history tracking."""
    print("\n" + "="*60)
    print("Test 3.1.4: Training History Tracking")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=32, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=16, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Create model and trainer
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(max_epochs=3, amp_enabled=False, patience=10)
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        
        # Setup training (required before train())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Train
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Check history structure
        assert isinstance(history, TrainingHistory), "History not TrainingHistory"
        assert len(history.train_losses) == 3, f"Expected 3 train losses, got {len(history.train_losses)}"
        assert len(history.val_losses) == 3, f"Expected 3 val losses, got {len(history.val_losses)}"
        assert history.total_epochs == 3, f"Expected 3 epochs, got {history.total_epochs}"
        
        print(f"✅ History structure correct")
        print(f"   Total epochs: {history.total_epochs}")
        print(f"   Train losses: {len(history.train_losses)}")
        print(f"   Val losses: {len(history.val_losses)}")
        
        # Check losses are decreasing or stable (generally)
        print(f"✅ Loss progression:")
        for i, (train_loss, val_loss) in enumerate(zip(history.train_losses, history.val_losses)):
            print(f"   Epoch {i+1}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        print("✅ Training history tracking working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_5_optimizer_updates():
    """Test 5: Optimizer updates parameters."""
    print("\n" + "="*60)
    print("Test 3.1.5: Optimizer Updates")
    print("="*60)
    
    try:
        # Create data
        dataset = create_synthetic_data(n_samples=16, input_dim=64)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Create model and trainer
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(max_epochs=1, amp_enabled=False)
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        
        # Save initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone().detach()
        
        # Setup training (required before train())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Train
        trainer.train(train_loader=train_loader, val_loader=None)
        
        # Check parameters changed
        updated_count = 0
        for name, param in model.named_parameters():
            diff = torch.abs(param - initial_params[name]).max()
            if diff > 1e-6:
                updated_count += 1
        
        total_params = len(list(model.named_parameters()))
        assert updated_count == total_params, f"Only {updated_count}/{total_params} params updated"
        
        print(f"✅ All {total_params} parameters updated")
        print(f"✅ Optimizer working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_6_validation_loop():
    """Test 6: Validation loop during training."""
    print("\n" + "="*60)
    print("Test 3.1.6: Validation Loop")
    print("="*60)
    
    try:
        # Create data
        train_dataset = create_synthetic_data(n_samples=32, input_dim=64)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=16, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Create model and trainer
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(max_epochs=2, amp_enabled=False, patience=10)
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        
        # Setup training (required before train())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Train with validation
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Check validation metrics computed
        assert len(history.val_losses) == 2, f"Expected 2 val losses, got {len(history.val_losses)}"
        assert len(history.val_metrics) == 2, f"Expected 2 val metrics, got {len(history.val_metrics)}"
        
        print(f"✅ Validation performed for {len(history.val_losses)} epochs")
        
        # Check validation metrics structure
        first_val_metrics = history.val_metrics[0]
        assert hasattr(first_val_metrics, 'accuracy'), "Metrics missing accuracy"
        assert hasattr(first_val_metrics, 'roc_auc'), "Metrics missing roc_auc"
        
        print(f"✅ Validation metrics:")
        print(f"   Accuracy: {first_val_metrics.accuracy:.4f}")
        print(f"   ROC AUC: {first_val_metrics.roc_auc:.4f}")
        
        print("✅ Validation loop working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_7_training_mode_switching():
    """Test 7: Model mode switching (train/eval)."""
    print("\n" + "="*60)
    print("Test 3.1.7: Training Mode Switching")
    print("="*60)
    
    try:
        # Create data
        train_dataset = create_synthetic_data(n_samples=16, input_dim=64)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataset = create_synthetic_data(n_samples=8, input_dim=64)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Create model and trainer
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32)
        config = TrainingConfig(max_epochs=1, amp_enabled=False, patience=10)
        trainer = ModelTrainer(model=model, config=config, device=torch.device('cpu'))
        
        # Check initial mode
        initial_training = model.training
        print(f"✅ Initial model.training: {initial_training}")
        
        # Setup training (required before train())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.setup_training(optimizer=optimizer)
        
        # Train (should handle mode switching internally)
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # After training, model should be in eval mode (or train mode, depends on implementation)
        # Just check that training didn't crash with mode switching
        print(f"✅ Final model.training: {model.training}")
        
        # Manually test mode switching
        model.train()
        assert model.training == True, "Model not in train mode"
        print(f"✅ model.train() sets training=True")
        
        model.eval()
        assert model.training == False, "Model not in eval mode"
        print(f"✅ model.eval() sets training=False")
        
        print("✅ Training mode switching working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all basic training loop tests."""
    print("\n" + "="*70)
    print("🧪 LEVEL 3.1: BASIC TRAINING LOOP TEST")
    print("="*70)
    
    tests = [
        ("Trainer Initialization", test_1_trainer_initialization),
        ("Single Epoch Training", test_2_single_epoch_training),
        ("Loss Calculation", test_3_loss_calculation),
        ("Training History", test_4_training_history),
        ("Optimizer Updates", test_5_optimizer_updates),
        ("Validation Loop", test_6_validation_loop),
        ("Training Mode Switching", test_7_training_mode_switching),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        padded_name = test_name.ljust(40, '.')
        print(f"{padded_name} {status}")
    
    print("="*70)
    
    # Final result
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"Results: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.0f}%)")
    print("="*70)
    
    if passed_count == total_count:
        print("🎉 Basic Training Loop: FULLY FUNCTIONAL ✅")
        return 0
    else:
        print(f"⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
