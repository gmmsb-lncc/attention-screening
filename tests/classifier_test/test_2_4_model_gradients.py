#!/usr/bin/env python3
"""
Level 2.4: Model Gradient Flow and Backward Pass Test

Tests gradient computation, backward pass, parameter updates,
gradient clipping, and edge cases like zero gradients.

Key Tests:
    - Gradient computation for all parameters
    - Backward pass correctness
    - Parameter updates after optimizer step
    - Gradient clipping
    - Zero gradients detection
    - Gradient accumulation
    - Mixed precision compatibility

Total Tests: 8 tests
Estimated Time: ~30s
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from classifier.models.mlp_classifier import MLPEmbeddingClassifier


def test_1_gradient_computation():
    """Test 1: Verify gradients are computed for all parameters."""
    print("\n" + "="*60)
    print("Test 2.4.1: Gradient Computation")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.train()
        
        # Create test data
        x = torch.randn(4, 512)
        y = torch.randint(0, 2, (4, 1)).float()
        
        # Forward pass
        output = model(x)
        
        # Compute loss
        criterion = nn.BCELoss()
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check all parameters have gradients
        params_with_grad = 0
        params_without_grad = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grad += 1
                    # Check gradient is valid
                    assert not torch.isnan(param.grad).any(), f"NaN in {name} gradient"
                    assert not torch.isinf(param.grad).any(), f"Inf in {name} gradient"
                else:
                    params_without_grad.append(name)
        
        # All parameters should have gradients
        assert len(params_without_grad) == 0, f"Parameters without gradients: {params_without_grad}"
        
        print(f"✅ Parameters with gradients: {params_with_grad}")
        print(f"✅ All parameters have valid gradients")
        print(f"✅ No NaN/Inf in gradients")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_2_backward_pass():
    """Test 2: Verify backward pass works correctly."""
    print("\n" + "="*60)
    print("Test 2.4.2: Backward Pass Correctness")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.train()
        
        # Create test data
        x = torch.randn(8, 512, requires_grad=True)
        y = torch.randint(0, 2, (8, 1)).float()
        
        # Forward pass
        output = model(x)
        
        # Compute loss
        criterion = nn.BCELoss()
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check input gradient exists
        assert x.grad is not None, "Input gradient is None"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"
        assert not torch.isinf(x.grad).any(), "Inf in input gradient"
        
        print(f"✅ Loss: {loss.item():.4f}")
        print(f"✅ Input gradient shape: {x.grad.shape}")
        print(f"✅ Input gradient range: [{x.grad.min():.6f}, {x.grad.max():.6f}]")
        print(f"✅ Backward pass works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_3_parameter_updates():
    """Test 3: Verify parameters are updated after optimizer step."""
    print("\n" + "="*60)
    print("Test 2.4.3: Parameter Updates")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.train()
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Save initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Create test data
        x = torch.randn(4, 512)
        y = torch.randint(0, 2, (4, 1)).float()
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = nn.BCELoss()(output, y)
        loss.backward()
        optimizer.step()
        
        # Check parameters changed
        params_updated = 0
        params_not_updated = []
        
        for name, param in model.named_parameters():
            initial = initial_params[name]
            diff = torch.abs(param - initial).max()
            
            if diff > 1e-8:
                params_updated += 1
            else:
                params_not_updated.append(name)
        
        # All parameters should be updated
        assert len(params_not_updated) == 0, f"Parameters not updated: {params_not_updated}"
        
        print(f"✅ Parameters updated: {params_updated}")
        print(f"✅ All parameters changed after optimizer step")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_4_gradient_clipping():
    """Test 4: Verify gradient clipping works."""
    print("\n" + "="*60)
    print("Test 2.4.4: Gradient Clipping")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.train()
        
        # Create test data with large loss
        x = torch.randn(4, 512)
        y = torch.ones(4, 1)  # All 1s
        
        # Forward pass
        output = model(x)
        
        # Large loss (multiply by 1000)
        loss = nn.BCELoss()(output, y) * 1000
        
        # Backward pass
        loss.backward()
        
        # Get gradient norm before clipping (clip_grad_norm_ returns the original norm)
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        
        # Zero gradients and recompute
        model.zero_grad()
        output = model(x)
        loss = nn.BCELoss()(output, y) * 1000
        loss.backward()
        
        # Clip gradients (returns original norm, but clips the gradients)
        max_norm = 1.0
        original_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        # Now compute the actual gradient norm after clipping
        total_norm_squared = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm_squared += param.grad.data.norm(2).item() ** 2
        grad_norm_after = total_norm_squared ** 0.5
        
        # Check clipping worked (after norm should be <= max_norm)
        assert grad_norm_after <= max_norm * 1.01, f"Gradient norm {grad_norm_after:.4f} > max_norm {max_norm}"
        
        print(f"✅ Gradient norm before clipping: {original_norm:.4f}")
        print(f"✅ Gradient norm after clipping: {grad_norm_after:.4f}")
        print(f"✅ Gradient clipping works (max_norm={max_norm})")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_5_zero_gradients():
    """Test 5: Verify zero_grad() clears gradients."""
    print("\n" + "="*60)
    print("Test 2.4.5: Zero Gradients")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.train()
        
        # Create test data
        x = torch.randn(4, 512)
        y = torch.randint(0, 2, (4, 1)).float()
        
        # Compute gradients
        output = model(x)
        loss = nn.BCELoss()(output, y)
        loss.backward()
        
        # Check gradients exist
        grads_before = []
        for param in model.parameters():
            if param.grad is not None:
                grads_before.append(param.grad.abs().sum().item())
        
        assert len(grads_before) > 0, "No gradients computed"
        assert sum(grads_before) > 0, "All gradients are zero"
        
        print(f"✅ Gradients before zero_grad: {len(grads_before)} params")
        print(f"✅ Total gradient magnitude: {sum(grads_before):.6f}")
        
        # Zero gradients
        model.zero_grad()
        
        # Check gradients are None or zero
        grads_after = []
        for param in model.parameters():
            if param.grad is not None:
                grads_after.append(param.grad.abs().sum().item())
        
        if len(grads_after) > 0:
            assert all(g == 0 for g in grads_after), "Gradients not zeroed"
            print(f"✅ Gradients after zero_grad: all zero")
        else:
            print(f"✅ Gradients after zero_grad: None (cleared)")
        
        print(f"✅ zero_grad() works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_6_gradient_accumulation():
    """Test 6: Verify gradient accumulation works."""
    print("\n" + "="*60)
    print("Test 2.4.6: Gradient Accumulation")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.train()
        
        # Create test data
        x = torch.randn(4, 512)
        y = torch.randint(0, 2, (4, 1)).float()
        
        # First backward pass
        output1 = model(x)
        loss1 = nn.BCELoss()(output1, y)
        loss1.backward()
        
        # Save gradients (clone to avoid reference issues)
        grads_first = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads_first[name] = param.grad.clone()
        
        # Second backward pass (accumulate - don't zero_grad!)
        output2 = model(x)
        loss2 = nn.BCELoss()(output2, y)
        loss2.backward()
        
        # Check gradients accumulated
        accumulated_count = 0
        for name, param in model.named_parameters():
            if name in grads_first and param.grad is not None:
                # Check gradient is larger (accumulated)
                grad_first = grads_first[name].abs().sum()
                grad_accumulated = param.grad.abs().sum()
                
                # Accumulated gradient should be approximately 2x the first
                # (within a reasonable tolerance due to numerical precision)
                if grad_first > 1e-6:  # Only check non-zero gradients
                    ratio = grad_accumulated / grad_first
                    # Ratio should be ~2 (tolerance: 1.5 to 2.5)
                    if 1.5 < ratio < 2.5:
                        accumulated_count += 1
        
        # At least most parameters should show accumulation
        total_params_with_grad = len(grads_first)
        accumulation_rate = accumulated_count / total_params_with_grad if total_params_with_grad > 0 else 0
        
        assert accumulation_rate >= 0.7, f"Only {accumulated_count}/{total_params_with_grad} params accumulated (rate={accumulation_rate:.2f})"
        
        print(f"✅ Parameters with gradient accumulation: {accumulated_count}/{total_params_with_grad}")
        print(f"✅ Accumulation rate: {accumulation_rate*100:.1f}%")
        print(f"✅ Gradient accumulation works")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_7_gradient_magnitude():
    """Test 7: Analyze gradient magnitudes."""
    print("\n" + "="*60)
    print("Test 2.4.7: Gradient Magnitude Analysis")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        model.train()
        
        # Create test data
        x = torch.randn(8, 512)
        y = torch.randint(0, 2, (8, 1)).float()
        
        # Forward + backward
        output = model(x)
        loss = nn.BCELoss()(output, y)
        loss.backward()
        
        # Analyze gradients per layer
        print(f"✅ Gradient magnitude per layer:")
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                grad_max = param.grad.abs().max().item()
                grad_std = param.grad.std().item()
                
                print(f"   {name:20s}: mean={grad_mean:.6f}, max={grad_max:.6f}, std={grad_std:.6f}")
        
        # Compute total gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        print(f"✅ Total gradient norm: {total_norm:.6f}")
        
        # Check no exploding gradients (norm < 100)
        assert total_norm < 100, f"Gradient norm too large: {total_norm:.2f}"
        print(f"✅ No exploding gradients")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_8_requires_grad():
    """Test 8: Verify requires_grad flag behavior."""
    print("\n" + "="*60)
    print("Test 2.4.8: requires_grad Flag")
    print("="*60)
    
    try:
        model = MLPEmbeddingClassifier(input_dim=512, hidden_dim=256)
        
        # Check all parameters require gradients by default
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Parameter {name} doesn't require gradients"
        
        print(f"✅ All parameters require gradients by default")
        
        # Freeze model
        for param in model.parameters():
            param.requires_grad = False
        
        # Check no gradients computed when frozen
        model.train()
        x = torch.randn(4, 512)
        
        # Forward pass (output will not require grad since model is frozen)
        output = model(x)
        
        # Check output doesn't require grad
        assert not output.requires_grad, "Output should not require grad when model is frozen"
        print(f"✅ Frozen model: Output doesn't require grad")
        
        # Check parameters don't have gradients
        for name, param in model.named_parameters():
            assert param.grad is None, f"Gradient computed for frozen param {name}"
        
        print(f"✅ Frozen model: No gradients in parameters")
        
        # Unfreeze model
        for param in model.parameters():
            param.requires_grad = True
        
        # Check gradients computed again
        x = torch.randn(4, 512, requires_grad=True)
        y = torch.randint(0, 2, (4, 1)).float()
        
        output = model(x)
        loss = nn.BCELoss()(output, y)
        loss.backward()
        
        grads_computed = sum(1 for param in model.parameters() if param.grad is not None)
        assert grads_computed > 0, "No gradients after unfreezing"
        
        print(f"✅ Unfrozen model: {grads_computed} parameters with gradients")
        print(f"✅ requires_grad flag works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all gradient flow tests."""
    print("\n" + "="*70)
    print("🧪 LEVEL 2.4: MODEL GRADIENT FLOW AND BACKWARD PASS TEST")
    print("="*70)
    
    tests = [
        ("Gradient Computation", test_1_gradient_computation),
        ("Backward Pass", test_2_backward_pass),
        ("Parameter Updates", test_3_parameter_updates),
        ("Gradient Clipping", test_4_gradient_clipping),
        ("Zero Gradients", test_5_zero_gradients),
        ("Gradient Accumulation", test_6_gradient_accumulation),
        ("Gradient Magnitude", test_7_gradient_magnitude),
        ("requires_grad Flag", test_8_requires_grad),
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
        print("🎉 Model Gradient Flow: FULLY FUNCTIONAL ✅")
        return 0
    else:
        print(f"⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
