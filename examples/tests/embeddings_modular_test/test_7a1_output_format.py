"""
Test 7A.1: Output Format Compatibility
Fast test - checks output format only.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_output_format():
    """Test output format matches expected structure"""
    print("\n" + "="*80)
    print("TEST 7A.1: Output Format Compatibility")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Test numpy array output
    sequences = ["MKTAYIAK", "ARNDCEQ"]
    embeddings = pipeline.generate_protein_embeddings(
        source=sequences,
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    
    # Check type
    assert isinstance(embeddings, np.ndarray), f"Expected ndarray, got {type(embeddings)}"
    
    # Check shape
    assert len(embeddings.shape) == 2, f"Expected 2D array, got shape {embeddings.shape}"
    assert embeddings.shape[0] == len(sequences), \
        f"Expected {len(sequences)} rows, got {embeddings.shape[0]}"
    
    # Check dtype
    assert embeddings.dtype in [np.float32, np.float64], \
        f"Expected float type, got {embeddings.dtype}"
    
    print(f"✅ Output format correct:")
    print(f"   Type: {type(embeddings).__name__}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dtype: {embeddings.dtype}")
    
    print("\n✅ TEST 7A.1 PASSED!\n")


if __name__ == "__main__":
    try:
        test_output_format()
        print("="*80)
        print("✅ TEST 7A.1: OUTPUT FORMAT - PASSED")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST 7A.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
