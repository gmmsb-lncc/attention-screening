"""
Test Boltz-2 Integration (CLI-based approach)

Comprehensive integration tests for Boltz-2 CLI-based embedding extraction.

Test Coverage:
1. CLI Installation - Verify boltz command available
2. Basic Embedding - Extract embeddings from simple sequence (no MSA)
3. Dimension Validation - Verify 768-dim output
4. Error Handling - Test invalid sequences, timeouts
5. MSA Integration - Test ColabFold MSA generation (optional)
6. Output Cleanup - Verify temporary files removed

Author: DockTKinase Team
Date: 2025-11-20
"""

import sys
import logging
from pathlib import Path

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.build.embeddings.strategies.boltz_strategy import (
    BoltzStrategy,
    validate_boltz_installation,
    get_boltz_version,
    MODEL_SPECS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST 1: CLI INSTALLATION
# =============================================================================

def test_cli_installation():
    """Test if Boltz CLI is installed and accessible."""
    print("\n" + "=" * 70)
    print("TEST 1: CLI Installation")
    print("=" * 70)
    
    try:
        is_installed = validate_boltz_installation()
        
        if not is_installed:
            print("❌ FAIL: Boltz CLI not found")
            print("\nInstallation Instructions:")
            print("  pip install boltz[cuda]  # For GPU")
            print("  pip install boltz        # For CPU")
            return False
        
        version = get_boltz_version()
        print(f"✓ Boltz CLI installed: {version}")
        
        return True
    
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


# =============================================================================
# TEST 2: BASIC EMBEDDING EXTRACTION
# =============================================================================

def test_basic_embedding():
    """Test basic embedding extraction without MSA (fastest)."""
    print("\n" + "=" * 70)
    print("TEST 2: Basic Embedding Extraction (No MSA)")
    print("=" * 70)
    
    try:
        # Simple test sequence (8 amino acids)
        sequence = "MKFLKFSL"
        
        print(f"Test sequence: {sequence} ({len(sequence)} AA)")
        
        # Initialize strategy (no MSA for speed)
        strategy = BoltzStrategy(logger=logger, use_msa=False)
        
        # Load (initializes CLI environment)
        print("\n1. Initializing Boltz CLI environment...")
        model, tokenizer = strategy.load('boltz2', device=torch.device('cpu'))
        
        assert model is None, "Model should be None (CLI-based)"
        assert tokenizer is None, "Tokenizer should be None (CLI-based)"
        print("✓ CLI environment initialized")
        
        # Generate embedding
        print("\n2. Generating embedding...")
        print("   (This may take 1-3 minutes for first run)")
        embedding = strategy.generate(model, tokenizer, sequence, torch.device('cpu'))
        
        # Validate output
        print("\n3. Validating output...")
        assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
        print(f"✓ Embedding type: {type(embedding)}")
        
        assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"
        print(f"✓ Embedding dtype: {embedding.dtype}")
        
        expected_dim = MODEL_SPECS['boltz2']['output_dim']
        assert embedding.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {embedding.shape}"
        print(f"✓ Embedding shape: {embedding.shape}")
        
        # Check embedding statistics
        mean = embedding.mean()
        std = embedding.std()
        min_val = embedding.min()
        max_val = embedding.max()
        
        print(f"\n4. Embedding statistics:")
        print(f"   Mean: {mean:.4f}")
        print(f"   Std:  {std:.4f}")
        print(f"   Min:  {min_val:.4f}")
        print(f"   Max:  {max_val:.4f}")
        
        # Sanity checks
        assert not np.isnan(embedding).any(), "Embedding contains NaN"
        assert not np.isinf(embedding).any(), "Embedding contains Inf"
        print("✓ No NaN or Inf values")
        
        # Cleanup
        print("\n5. Cleaning up...")
        strategy.cleanup(model, tokenizer)
        print("✓ Cleanup complete")
        
        print("\n✅ TEST PASSED: Basic embedding extraction successful")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 3: DIMENSION VALIDATION
# =============================================================================

def test_dimension_validation():
    """Test that embeddings have correct dimensions for different sequences."""
    print("\n" + "=" * 70)
    print("TEST 3: Dimension Validation")
    print("=" * 70)
    
    try:
        # Test sequences of different lengths
        test_cases = [
            ("MKFL", 4),       # Very short
            ("MKFLKFSL", 8),   # Short
            ("MKFLKFSLKTYCRS", 15),  # Medium
        ]
        
        strategy = BoltzStrategy(logger=logger, use_msa=False)
        strategy.load('boltz2', device=torch.device('cpu'))
        
        expected_dim = MODEL_SPECS['boltz2']['output_dim']
        
        for sequence, length in test_cases:
            print(f"\nTesting sequence: {sequence} ({length} AA)")
            
            embedding = strategy.generate(None, None, sequence, torch.device('cpu'))
            
            assert embedding.shape == (expected_dim,), \
                f"Expected {expected_dim}-dim, got {embedding.shape}"
            
            print(f"✓ Correct dimension: {embedding.shape}")
        
        strategy.cleanup(None, None)
        
        print("\n✅ TEST PASSED: All sequences produce correct dimensions")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 4: ERROR HANDLING
# =============================================================================

def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 70)
    print("TEST 4: Error Handling")
    print("=" * 70)
    
    try:
        strategy = BoltzStrategy(logger=logger, use_msa=False)
        strategy.load('boltz2', device=torch.device('cpu'))
        
        # Test 1: Empty sequence
        print("\n1. Testing empty sequence...")
        try:
            strategy.generate(None, None, "", torch.device('cpu'))
            print("❌ Should have raised ValueError for empty sequence")
            return False
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {e}")
        
        # Test 2: Invalid amino acids
        print("\n2. Testing invalid amino acids...")
        try:
            strategy.generate(None, None, "MKFL123", torch.device('cpu'))
            print("❌ Should have raised ValueError for invalid amino acids")
            return False
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {e}")
        
        # Test 3: Invalid pooling strategy
        print("\n3. Testing invalid pooling strategy...")
        try:
            strategy.generate(None, None, "MKFL", torch.device('cpu'), pooling='invalid')
            print("❌ Should have raised ValueError for invalid pooling")
            return False
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {e}")
        
        strategy.cleanup(None, None)
        
        print("\n✅ TEST PASSED: Error handling works correctly")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 5: MSA INTEGRATION (OPTIONAL)
# =============================================================================

def test_msa_integration():
    """Test MSA generation via ColabFold (optional, takes longer)."""
    print("\n" + "=" * 70)
    print("TEST 5: MSA Integration (Optional)")
    print("=" * 70)
    print("\n⚠️  WARNING: This test takes 3-5 minutes")
    print("⚠️  Requires internet connection to ColabFold server")
    
    # Ask user if they want to run this test
    response = input("\nRun MSA test? (y/n): ").strip().lower()
    
    if response != 'y':
        print("⊘ Skipping MSA test")
        return True  # Not a failure, just skipped
    
    try:
        # Short sequence for faster MSA generation
        sequence = "MKFLKFSL"
        
        print(f"\nTest sequence: {sequence} ({len(sequence)} AA)")
        
        # Initialize strategy WITH MSA
        strategy = BoltzStrategy(
            logger=logger,
            use_msa=True,
            msa_server='https://api.colabfold.com'
        )
        
        print("\n1. Initializing Boltz CLI with MSA support...")
        strategy.load('boltz2', device=torch.device('cpu'))
        
        # Generate embedding with MSA
        print("\n2. Generating embedding with MSA...")
        print("   (This may take 3-5 minutes)")
        embedding = strategy.generate(None, None, sequence, torch.device('cpu'))
        
        # Validate
        print("\n3. Validating output...")
        expected_dim = MODEL_SPECS['boltz2']['output_dim']
        assert embedding.shape == (expected_dim,), f"Expected {expected_dim}-dim"
        print(f"✓ Embedding shape: {embedding.shape}")
        
        # Cleanup
        print("\n4. Cleaning up...")
        strategy.cleanup(None, None)
        
        print("\n✅ TEST PASSED: MSA integration successful")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 6: OUTPUT CLEANUP
# =============================================================================

def test_output_cleanup():
    """Test that temporary files are properly cleaned up."""
    print("\n" + "=" * 70)
    print("TEST 6: Output Cleanup")
    print("=" * 70)
    
    try:
        sequence = "MKFL"
        
        strategy = BoltzStrategy(logger=logger, use_msa=False)
        strategy.load('boltz2', device=torch.device('cpu'))
        
        # Store output directory path
        output_dir = strategy.output_dir
        
        print(f"\nTemporary directory: {output_dir}")
        print(f"✓ Directory exists: {output_dir.exists()}")
        
        # Generate embedding
        print("\nGenerating embedding...")
        strategy.generate(None, None, sequence, torch.device('cpu'))
        
        print(f"✓ Directory still exists: {output_dir.exists()}")
        
        # Cleanup
        print("\nCleaning up...")
        strategy.cleanup(None, None)
        
        # Check if directory was removed
        if output_dir.exists():
            print(f"❌ FAIL: Directory still exists after cleanup: {output_dir}")
            return False
        
        print(f"✓ Directory removed: {output_dir}")
        
        print("\n✅ TEST PASSED: Cleanup removes temporary files")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all Boltz-2 integration tests."""
    print("=" * 70)
    print("BOLTZ-2 INTEGRATION TESTS (CLI-based approach)")
    print("=" * 70)
    print("\nThis test suite validates:")
    print("  1. Boltz CLI installation")
    print("  2. Basic embedding extraction (no MSA)")
    print("  3. Dimension validation")
    print("  4. Error handling")
    print("  5. MSA integration (optional)")
    print("  6. Output cleanup")
    print("\nEstimated time: 5-10 minutes (without MSA)")
    print("=" * 70)
    
    # Run tests
    results = []
    
    # Test 1: CLI Installation (required)
    cli_ok = test_cli_installation()
    results.append(('CLI Installation', cli_ok))
    
    if not cli_ok:
        print("\n⚠️  Boltz CLI not installed. Please install:")
        print("   pip install boltz[cuda]  # For GPU")
        print("   pip install boltz        # For CPU")
        print("\nSkipping remaining tests.")
        return 1
    
    # Test 2: Basic Embedding
    results.append(('Basic Embedding', test_basic_embedding()))
    
    # Test 3: Dimension Validation
    results.append(('Dimension Validation', test_dimension_validation()))
    
    # Test 4: Error Handling
    results.append(('Error Handling', test_error_handling()))
    
    # Test 5: MSA Integration (optional)
    results.append(('MSA Integration', test_msa_integration()))
    
    # Test 6: Output Cleanup
    results.append(('Output Cleanup', test_output_cleanup()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
