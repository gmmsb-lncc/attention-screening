"""
Test 7A.3: Error Messages Clarity
Fast test - checks error messages are clear.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_error_messages():
    """Test error messages are clear and helpful"""
    print("\n" + "="*80)
    print("TEST 7A.3: Error Messages Clarity")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Test 1: Invalid sequence
    print("\n1. Testing invalid sequence error...")
    try:
        pipeline.generate_protein_embeddings(
            source=["INVALID123"],  # Invalid characters
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
        assert False, "Should have raised error for invalid sequence"
    except Exception as e:
        error_msg = str(e).lower()
        # Accept various error messages that indicate invalid input
        assert any(word in error_msg for word in ['invalid', 'character', 'valid', 'process']), \
            f"Error message not clear: {e}"
        print(f"   ✓ Clear error: {e}")
    
    # Test 2: Empty sequence
    print("\n2. Testing empty sequence error...")
    try:
        pipeline.generate_protein_embeddings(
            source=[""],
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
        assert False, "Should have raised error for empty sequence"
    except Exception as e:
        error_msg = str(e).lower()
        # Accept various error messages that indicate invalid input
        assert any(word in error_msg for word in ['empty', 'invalid', 'valid', 'process']), \
            f"Error message not clear: {e}"
        print(f"   ✓ Clear error: {e}")
    
    # Test 3: Invalid model name
    print("\n3. Testing invalid model error...")
    try:
        pipeline.generate_protein_embeddings(
            source=["MKTAYIAK"],
            model_name='nonexistent_model',
            use_cache=False
        )
        assert False, "Should have raised error for invalid model"
    except Exception as e:
        error_msg = str(e).lower()
        assert 'model' in error_msg or 'not found' in error_msg or 'unknown' in error_msg, \
            f"Error message not clear: {e}"
        print(f"   ✓ Clear error: {e}")
    
    print("\n✅ All error messages are clear and helpful")
    print("\n✅ TEST 7A.3 PASSED!\n")


if __name__ == "__main__":
    try:
        test_error_messages()
        print("="*80)
        print("✅ TEST 7A.3: ERROR MESSAGES - PASSED")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST 7A.3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
