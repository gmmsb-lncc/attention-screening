#!/usr/bin/env python3
"""
Level 2.3: Model Forward Pass and Predictions Test

Tests model forward pass in different scenarios, prediction generation,
probability outputs, and edge cases like zero inputs and extreme values.

Key Tests:
    - Forward pass correctness
    - Prediction generation (binary classification)
    - Probability outputs (sigmoid range validation)
    - Different input sizes
    - Zero inputs (edge case)
    - Large inputs (edge case)
    - Model determinism in eval mode

Total Tests: 7 tests
Estimated Time: ~30s
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from classifier.models.mlp_classifier import MLPEmbeddingClassifier, create_mlp_model


def test_1_forward_pass_correctness():
    """Test 1: Verify forward pass produces correct outputs."""
    print("\n" + "="*60)
    print("Test 2.3.1: Forward Pass Correctness")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.eval()
        
        # Create test data
        x = torch.randn(8, 512)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output properties
        assert output.shape == (8, 1), f"Expected shape (8, 1), got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
        assert (output >= 0).all(), "Outputs should be >= 0 (sigmoid)"
        assert (output <= 1).all(), "Outputs should be <= 1 (sigmoid)"
        assert not torch.isnan(output).any(), "Outputs contain NaN"
        assert not torch.isinf(output).any(), "Outputs contain Inf"
        
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"✅ Output mean: {output.mean():.4f}")
        print(f"✅ Forward pass correct")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_2_prediction_generation():
    """Test 2: Generate binary predictions from probabilities."""
    print("\n" + "="*60)
    print("Test 2.3.2: Prediction Generation")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.eval()
        
        # Create test data
        x = torch.randn(10, 512)
        
        # Get probabilities
        with torch.no_grad():
            probs = model(x)
        
        # Generate predictions (threshold = 0.5)
        predictions = (probs >= 0.5).float()
        
        # Check predictions are binary
        unique_preds = torch.unique(predictions)
        assert len(unique_preds) <= 2, "Predictions should be binary (0 or 1)"
        assert all(p in [0, 1] for p in unique_preds.tolist()), "Predictions should be 0 or 1"
        
        print(f"✅ Probabilities: {probs.squeeze()[:5].tolist()}")
        print(f"✅ Predictions: {predictions.squeeze()[:5].tolist()}")
        print(f"✅ Unique predictions: {unique_preds.tolist()}")
        print(f"✅ Prediction generation correct")
        
        # Test with different thresholds
        pred_03 = (probs >= 0.3).float()
        pred_07 = (probs >= 0.7).float()
        
        # More samples should be predicted as 1 with lower threshold
        assert pred_03.sum() >= pred_07.sum(), "Lower threshold should predict more 1s"
        print(f"✅ Threshold 0.3: {pred_03.sum().item():.0f} positives")
        print(f"✅ Threshold 0.5: {predictions.sum().item():.0f} positives")
        print(f"✅ Threshold 0.7: {pred_07.sum().item():.0f} positives")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_3_different_input_sizes():
    """Test 3: Forward pass with different batch sizes."""
    print("\n" + "="*60)
    print("Test 2.3.3: Different Input Sizes")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 512)
            
            with torch.no_grad():
                output = model(x)
            
            # Check output shape
            assert output.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {output.shape}"
            assert (output >= 0).all() and (output <= 1).all(), "Outputs not in [0, 1]"
        
        print(f"✅ Tested batch sizes: {batch_sizes}")
        print(f"✅ All batch sizes handled correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_4_zero_inputs():
    """Test 4: Forward pass with zero inputs (edge case)."""
    print("\n" + "="*60)
    print("Test 2.3.4: Zero Inputs (Edge Case)")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.eval()
        
        # Create zero inputs
        x = torch.zeros(4, 512)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output is valid
        assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"
        assert (output >= 0).all() and (output <= 1).all(), "Outputs not in [0, 1]"
        assert not torch.isnan(output).any(), "Outputs contain NaN"
        assert not torch.isinf(output).any(), "Outputs contain Inf"
        
        print(f"✅ Zero inputs handled")
        print(f"✅ Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"✅ Output mean: {output.mean():.4f}")
        
        # All zeros should produce similar (but not necessarily identical) outputs
        # due to BatchNorm and ReLU
        print(f"✅ Output std: {output.std():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_5_large_inputs():
    """Test 5: Forward pass with large input values (edge case)."""
    print("\n" + "="*60)
    print("Test 2.3.5: Large Input Values (Edge Case)")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.eval()
        
        # Create large inputs
        x = torch.randn(4, 512) * 100  # Scale by 100
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output is still valid (sigmoid should clamp to [0, 1])
        assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"
        assert (output >= 0).all() and (output <= 1).all(), "Outputs not in [0, 1]"
        assert not torch.isnan(output).any(), "Outputs contain NaN"
        assert not torch.isinf(output).any(), "Outputs contain Inf"
        
        print(f"✅ Large inputs handled")
        print(f"✅ Input range: [{x.min():.2f}, {x.max():.2f}]")
        print(f"✅ Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"✅ Sigmoid clamped outputs to [0, 1]")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_6_model_determinism():
    """Test 6: Model is deterministic in eval mode."""
    print("\n" + "="*60)
    print("Test 2.3.6: Model Determinism (Eval Mode)")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.eval()
        
        # Create test data
        x = torch.randn(8, 512)
        
        # Multiple forward passes should give identical results
        outputs = []
        for _ in range(5):
            with torch.no_grad():
                output = model(x)
            outputs.append(output)
        
        # Check all outputs are identical
        for i in range(1, len(outputs)):
            diff = torch.abs(outputs[i] - outputs[0]).max()
            assert diff < 1e-6, f"Outputs not identical: diff = {diff:.9f}"
        
        print(f"✅ 5 forward passes with same input")
        print(f"✅ Max difference: {diff:.9f} (deterministic)")
        print(f"✅ Model is deterministic in eval mode")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_7_probability_distribution():
    """Test 7: Probability distribution analysis."""
    print("\n" + "="*60)
    print("Test 2.3.7: Probability Distribution Analysis")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.eval()
        
        # Create large test batch
        x = torch.randn(1000, 512)
        
        # Get probabilities
        with torch.no_grad():
            probs = model(x)
        
        # Analyze distribution
        probs_np = probs.squeeze().numpy()
        
        # Check distribution properties
        mean = probs_np.mean()
        std = probs_np.std()
        min_val = probs_np.min()
        max_val = probs_np.max()
        median = np.median(probs_np)
        
        print(f"✅ Probability statistics (n=1000):")
        print(f"   Mean:   {mean:.4f}")
        print(f"   Std:    {std:.4f}")
        print(f"   Min:    {min_val:.4f}")
        print(f"   Max:    {max_val:.4f}")
        print(f"   Median: {median:.4f}")
        
        # Check valid range
        assert 0 <= min_val <= 1, f"Min prob outside [0, 1]: {min_val}"
        assert 0 <= max_val <= 1, f"Max prob outside [0, 1]: {max_val}"
        assert 0 <= mean <= 1, f"Mean prob outside [0, 1]: {mean}"
        
        # Count predictions by threshold
        pred_05 = (probs_np >= 0.5).sum()
        pred_03 = (probs_np >= 0.3).sum()
        pred_07 = (probs_np >= 0.7).sum()
        
        print(f"✅ Predictions by threshold:")
        print(f"   >= 0.3: {pred_03} ({pred_03/10:.1f}%)")
        print(f"   >= 0.5: {pred_05} ({pred_05/10:.1f}%)")
        print(f"   >= 0.7: {pred_07} ({pred_07/10:.1f}%)")
        
        print(f"✅ Probability distribution analyzed")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all model forward pass tests."""
    print("\n" + "="*70)
    print("🧪 LEVEL 2.3: MODEL FORWARD PASS AND PREDICTIONS TEST")
    print("="*70)
    
    tests = [
        ("Forward Pass Correctness", test_1_forward_pass_correctness),
        ("Prediction Generation", test_2_prediction_generation),
        ("Different Input Sizes", test_3_different_input_sizes),
        ("Zero Inputs", test_4_zero_inputs),
        ("Large Inputs", test_5_large_inputs),
        ("Model Determinism", test_6_model_determinism),
        ("Probability Distribution", test_7_probability_distribution),
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
        print("🎉 Model Forward Pass: FULLY FUNCTIONAL ✅")
        return 0
    else:
        print(f"⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
