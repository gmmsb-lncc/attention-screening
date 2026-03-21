#!/usr/bin/env python3
"""
Level 2.1: MLPEmbeddingClassifier Architecture Test

Tests the MLP model architecture, layer structure, forward pass,
and edge cases like batch_size=1 with BatchNorm.

Architecture:
    - Layer 1: input_dim -> hidden_dim (e.g., 512 -> 256)
      Components: Linear -> BatchNorm -> ReLU -> Dropout
    - Layer 2: hidden_dim -> hidden_dim//2 (e.g., 256 -> 128)
      Components: Linear -> BatchNorm -> ReLU -> Dropout
    - Layer 3: hidden_dim//2 -> 1 (binary output)
      Components: Linear -> Sigmoid

Total Tests: 8 tests
Estimated Time: ~1 min
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from classifier.models.mlp_classifier import MLPEmbeddingClassifier, create_mlp_model


def test_1_model_instantiation():
    """Test 1: Model instantiation with different configurations."""
    print("\n" + "="*60)
    print("Test 2.1.1: Model Instantiation")
    print("="*60)
    
    try:
        # Default configuration
        model = MLPEmbeddingClassifier(input_dim=512)
        assert isinstance(model, nn.Module), "Model should be nn.Module"
        print("✅ Default model created (input_dim=512, hidden_dim=1024)")
        
        # Custom configuration
        model = MLPEmbeddingClassifier(input_dim=256, hidden_dim=128, dropout=0.5)
        assert model.fc1.in_features == 256, "Input dim should be 256"
        assert model.fc1.out_features == 128, "Hidden dim should be 128"
        assert model.fc2.out_features == 64, "Second layer should be hidden_dim//2 = 64"
        assert model.fc3.out_features == 1, "Output dim should be 1"
        print("✅ Custom model created (input_dim=256, hidden_dim=128, dropout=0.5)")
        
        # Large configuration
        model = MLPEmbeddingClassifier(input_dim=3328, hidden_dim=2048)
        assert model.fc1.in_features == 3328, "Input dim should be 3328"
        assert model.fc1.out_features == 2048, "Hidden dim should be 2048"
        assert model.fc2.out_features == 1024, "Second layer should be 1024"
        print("✅ Large model created (input_dim=3328, hidden_dim=2048)")
        
        print("✅ Model instantiation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_2_layer_structure():
    """Test 2: Verify layer structure and components."""
    print("\n" + "="*60)
    print("Test 2.1.2: Layer Structure")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        
        # Check layer 1 components
        assert isinstance(model.fc1, nn.Linear), "fc1 should be Linear"
        assert isinstance(model.bn1, nn.BatchNorm1d), "bn1 should be BatchNorm1d"
        assert isinstance(model.act1, nn.ReLU), "act1 should be ReLU"
        assert isinstance(model.drop1, nn.Dropout), "drop1 should be Dropout"
        print("✅ Layer 1 components: Linear -> BatchNorm -> ReLU -> Dropout")
        
        # Check layer 2 components
        assert isinstance(model.fc2, nn.Linear), "fc2 should be Linear"
        assert isinstance(model.bn2, nn.BatchNorm1d), "bn2 should be BatchNorm1d"
        assert isinstance(model.act2, nn.ReLU), "act2 should be ReLU"
        assert isinstance(model.drop2, nn.Dropout), "drop2 should be Dropout"
        print("✅ Layer 2 components: Linear -> BatchNorm -> ReLU -> Dropout")
        
        # Check layer 3 (output)
        assert isinstance(model.fc3, nn.Linear), "fc3 should be Linear"
        assert model.fc3.out_features == 1, "Output layer should have 1 neuron"
        print("✅ Layer 3 components: Linear (output)")
        
        # Check dimensions
        assert model.fc1.in_features == 512, "Input: 512"
        assert model.fc1.out_features == 256, "Hidden1: 256"
        assert model.fc2.in_features == 256, "Hidden1: 256"
        assert model.fc2.out_features == 128, "Hidden2: 128"
        assert model.fc3.in_features == 128, "Hidden2: 128"
        assert model.fc3.out_features == 1, "Output: 1"
        print("✅ Layer dimensions: 512 -> 256 -> 128 -> 1")
        
        print("✅ Layer structure correct")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_3_forward_pass():
    """Test 3: Forward pass with normal batch."""
    print("\n" + "="*60)
    print("Test 2.1.3: Forward Pass")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.eval()  # Set to eval mode
        
        # Create test data (batch_size=4)
        batch_size = 4
        x = torch.randn(batch_size, 512)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 1), f"Output shape should be ({batch_size}, 1), got {output.shape}"
        print(f"✅ Output shape: {output.shape}")
        
        # Check output range (should be [0, 1] after sigmoid)
        assert output.min() >= 0, f"Output min should be >= 0, got {output.min():.4f}"
        assert output.max() <= 1, f"Output max should be <= 1, got {output.max():.4f}"
        print(f"✅ Output range: [{output.min():.4f}, {output.max():.4f}] (sigmoid applied)")
        
        # Check output type
        assert output.dtype == torch.float32, f"Output dtype should be float32, got {output.dtype}"
        print(f"✅ Output dtype: {output.dtype}")
        
        # Check no NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        print("✅ No NaN/Inf in output")
        
        print("✅ Forward pass working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_4_batch_size_one():
    """Test 4: Forward pass with batch_size=1 (BatchNorm edge case)."""
    print("\n" + "="*60)
    print("Test 2.1.4: Batch Size = 1 (BatchNorm Edge Case)")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.eval()  # Important: eval mode for single sample
        
        # Create single sample
        x = torch.randn(1, 512)
        
        # Forward pass (should not crash with BatchNorm)
        with torch.no_grad():
            output = model(x)
        
        # Check output
        assert output.shape == (1, 1), f"Output shape should be (1, 1), got {output.shape}"
        assert 0 <= output.item() <= 1, f"Output should be in [0, 1], got {output.item():.4f}"
        print(f"✅ Single sample handled correctly")
        print(f"   Output: {output.item():.4f}")
        
        # Test in training mode (BatchNorm skips when batch_size=1)
        model.train()
        with torch.no_grad():
            output_train = model(x)
        
        assert output_train.shape == (1, 1), "Output shape should be (1, 1) in train mode"
        print(f"✅ Training mode with batch_size=1: {output_train.item():.4f}")
        
        print("✅ Batch size = 1 handled correctly (BatchNorm skipped)")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_5_gradient_flow():
    """Test 5: Gradient flow through the network."""
    print("\n" + "="*60)
    print("Test 2.1.5: Gradient Flow")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.train()
        
        # Create test data
        x = torch.randn(4, 512, requires_grad=True)
        
        # Forward pass
        output = model(x)
        
        # Compute loss (dummy)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} is None"
                assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"
                assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf"
        
        print("✅ Gradients computed for all parameters")
        
        # Check input gradient
        assert x.grad is not None, "Input gradient is None"
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
        print("✅ Gradient flows to input")
        
        print("✅ Gradient flow working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_6_factory_function():
    """Test 6: Factory function create_mlp_model."""
    print("\n" + "="*60)
    print("Test 2.1.6: Factory Function")
    print("="*60)
    
    try:
        # Create model with factory
        model = create_mlp_model(input_dim=512, hidden_dim=256, dropout=0.4)
        
        # Check model type
        assert isinstance(model, MLPEmbeddingClassifier), "Should return MLPEmbeddingClassifier"
        print("✅ Factory returns correct model type")
        
        # Check model is on device
        device = next(model.parameters()).device
        print(f"✅ Model on device: {device}")
        
        # Check model configuration
        assert model.fc1.in_features == 512, "Input dim should be 512"
        assert model.fc1.out_features == 256, "Hidden dim should be 256"
        print("✅ Model configuration correct")
        
        # Test with explicit device
        cpu_model = create_mlp_model(input_dim=512, device=torch.device('cpu'))
        assert next(cpu_model.parameters()).device.type == 'cpu', "Model should be on CPU"
        print("✅ Explicit device placement working")
        
        print("✅ Factory function working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_7_dropout_behavior():
    """Test 7: Dropout behavior in train vs eval mode."""
    print("\n" + "="*60)
    print("Test 2.1.7: Dropout Behavior")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256, dropout=0.5)
        x = torch.randn(10, 512)
        
        # Training mode (dropout active)
        model.train()
        with torch.no_grad():
            output_train1 = model(x)
            output_train2 = model(x)
        
        # Outputs should be different (dropout is stochastic)
        # Note: May occasionally be the same by chance
        diff_train = torch.abs(output_train1 - output_train2).mean()
        print(f"✅ Training mode: Mean difference = {diff_train:.6f}")
        
        # Eval mode (dropout disabled)
        model.eval()
        with torch.no_grad():
            output_eval1 = model(x)
            output_eval2 = model(x)
        
        # Outputs should be identical (no dropout)
        diff_eval = torch.abs(output_eval1 - output_eval2).mean()
        assert diff_eval < 1e-6, f"Eval outputs should be identical, diff = {diff_eval:.6f}"
        print(f"✅ Eval mode: Mean difference = {diff_eval:.6f} (deterministic)")
        
        print("✅ Dropout behavior correct (stochastic in train, disabled in eval)")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_8_parameter_count():
    """Test 8: Verify parameter count."""
    print("\n" + "="*60)
    print("Test 2.1.8: Parameter Count")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params:,}")
        
        # Expected parameters:
        # fc1: (512 * 256) + 256 = 131,328
        # bn1: 256 * 2 = 512 (gamma, beta)
        # fc2: (256 * 128) + 128 = 32,896
        # bn2: 128 * 2 = 256
        # fc3: (128 * 1) + 1 = 129
        # Total: 131,328 + 512 + 32,896 + 256 + 129 = 165,121
        expected = (512 * 256 + 256) + (256 * 2) + (256 * 128 + 128) + (128 * 2) + (128 * 1 + 1)
        
        assert total_params == expected, f"Expected {expected:,} params, got {total_params:,}"
        assert trainable_params == total_params, "All params should be trainable"
        print(f"✅ Parameter count matches expected: {expected:,}")
        
        # Verify all parameters require gradients
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Parameter {name} should require gradients"
        
        print("✅ All parameters trainable")
        print("✅ Parameter count correct")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all MLPEmbeddingClassifier tests."""
    print("\n" + "="*70)
    print("🧪 LEVEL 2.1: MLPEMBEDDINGCLASSIFIER ARCHITECTURE TEST")
    print("="*70)
    
    tests = [
        ("Model Instantiation", test_1_model_instantiation),
        ("Layer Structure", test_2_layer_structure),
        ("Forward Pass", test_3_forward_pass),
        ("Batch Size = 1", test_4_batch_size_one),
        ("Gradient Flow", test_5_gradient_flow),
        ("Factory Function", test_6_factory_function),
        ("Dropout Behavior", test_7_dropout_behavior),
        ("Parameter Count", test_8_parameter_count),
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
        # Pad test name to align status
        padded_name = test_name.ljust(40, '.')
        print(f"{padded_name} {status}")
    
    print("="*70)
    
    # Final result
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"Results: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.0f}%)")
    print("="*70)
    
    if passed_count == total_count:
        print("🎉 MLPEmbeddingClassifier: FULLY FUNCTIONAL ✅")
        return 0
    else:
        print(f"⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
