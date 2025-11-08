"""
Level 7B: File I/O Compatibility Tests
Tests file input/output handling.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_7b_1_file_input():
    """Test 7B.1: File input handling (CSV, TSV)"""
    print("\n" + "="*80)
    print("TEST 7B.1: File Input Compatibility")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Use existing test file
    test_file = Path(__file__).parent.parent / 'datasets' / 'kinase_test_small.tsv'
    
    if not test_file.exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("   Skipping file input test")
        print("\n⚠️  TEST 7B.1 SKIPPED\n")
        return
    
    # Load from file
    embeddings = pipeline.generate_protein_embeddings(
        source=str(test_file),
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    
    # Check output
    assert isinstance(embeddings, np.ndarray), "File input should return ndarray"
    assert len(embeddings.shape) == 2, "Should be 2D array"
    
    print(f"✅ File input processed successfully:")
    print(f"   File: {test_file.name}")
    print(f"   Output shape: {embeddings.shape}")
    
    print("\n✅ TEST 7B.1 PASSED!\n")


def test_7b_2_dataframe():
    """Test 7B.2: DataFrame input/output"""
    print("\n" + "="*80)
    print("TEST 7B.2: DataFrame Compatibility")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Create test DataFrame
    df = pd.DataFrame({
        'id': ['prot1', 'prot2', 'prot3'],
        'sequence': ['MKTAYIAK', 'ARNDCEQ', 'GLHIPK']
    })
    
    # Process DataFrame
    embeddings = pipeline.generate_protein_embeddings(
        source=df,
        model_name='esm2_t6_8M_UR50D',
        sequence_column='sequence',
        id_column='id',
        use_cache=False
    )
    
    # Check output
    assert isinstance(embeddings, np.ndarray), "DataFrame input should return ndarray"
    assert embeddings.shape[0] == len(df), \
        f"Expected {len(df)} embeddings, got {embeddings.shape[0]}"
    
    print(f"✅ DataFrame processed successfully:")
    print(f"   Input rows: {len(df)}")
    print(f"   Output shape: {embeddings.shape}")
    
    print("\n✅ TEST 7B.2 PASSED!\n")


def run_all_tests():
    """Run file I/O tests"""
    print("\n" + "="*80)
    print("LEVEL 7B: FILE I/O TESTS")
    print("="*80)
    
    tests = [
        ("7B.1", "File Input", test_7b_1_file_input),
        ("7B.2", "DataFrame", test_7b_2_dataframe),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_id, name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_id} {name} PASSED\n")
            passed += 1
        except Exception as e:
            if "SKIPPED" in str(e):
                print(f"⚠️  {test_id} {name} SKIPPED\n")
                skipped += 1
            else:
                print(f"❌ {test_id} {name} FAILED: {e}\n")
                failed += 1
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*80)
    print("FILE I/O TEST SUMMARY")
    print("="*80)
    print(f"📊 Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped: {skipped}")
    
    if failed == 0:
        print("\n✅ ALL FILE I/O TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
