"""
Test 7B.2: DataFrame Processing
Moderate test - checks DataFrame input handling.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_dataframe():
    """Test DataFrame input processing"""
    print("\n" + "="*80)
    print("TEST 7B.2: DataFrame Processing")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Create test DataFrame
    df = pd.DataFrame({
        'id': ['seq1', 'seq2', 'seq3'],
        'sequence': ['MKTAYIAK', 'ARNDCEQ', 'GHIKLMN']
    })
    
    print("\n1. Testing DataFrame input...")
    embeddings = pipeline.generate_protein_embeddings(
        source=df,
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    
    assert embeddings.shape[0] == 3, f"Expected 3 rows, got {embeddings.shape[0]}"
    assert embeddings.shape[1] == 320, f"Expected 320 dims, got {embeddings.shape[1]}"
    print(f"   ✓ DataFrame processed: {embeddings.shape}")
    
    # Test custom column names
    print("\n2. Testing custom column names...")
    df_custom = pd.DataFrame({
        'protein_id': ['p1', 'p2'],
        'seq': ['MKTAYIAK', 'ARNDCEQ']
    })
    
    embeddings = pipeline.generate_protein_embeddings(
        source=df_custom,
        sequence_column='seq',
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    
    assert embeddings.shape[0] == 2, f"Expected 2 rows, got {embeddings.shape[0]}"
    print(f"   ✓ Custom columns work: {embeddings.shape}")
    
    print("\n✅ DataFrame processing works correctly")
    print("\n✅ TEST 7B.2 PASSED!\n")


if __name__ == "__main__":
    try:
        test_dataframe()
        print("="*80)
        print("✅ TEST 7B.2: DATAFRAME - PASSED")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST 7B.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
