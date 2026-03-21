"""
Level 7A: Basic Compatibility Tests (Fast)
Tests basic output format and API stability.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_7_1_output_format_compatibility():
    """Test 7.1: Output format matches expected structure"""
    print("\n" + "="*80)
    print("TEST 7.1: Output Format Compatibility")
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
    
    print("\n✅ TEST 7.1 PASSED!\n")


def test_7_2_file_input_compatibility():
    """Test 7.2: File input handling (CSV, TSV)"""
    print("\n" + "="*80)
    print("TEST 7.2: File Input Compatibility")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Use existing test file
    test_file = Path(__file__).parent.parent / 'datasets' / 'kinase_test_small.tsv'
    
    if not test_file.exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("   Skipping file input test")
        print("\n⚠️  TEST 7.2 SKIPPED\n")
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
    
    print("\n✅ TEST 7.2 PASSED!\n")


def test_7_3_dataframe_compatibility():
    """Test 7.3: DataFrame input/output"""
    print("\n" + "="*80)
    print("TEST 7.3: DataFrame Compatibility")
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
    
    print("\n✅ TEST 7.3 PASSED!\n")


def test_7_4_api_interface_stability():
    """Test 7.4: API interface stability"""
    print("\n" + "="*80)
    print("TEST 7.4: API Interface Stability")
    print("="*80)
    
    # Check that main functions exist and have correct signatures
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Check protein embedding method exists
    assert hasattr(pipeline, 'generate_protein_embeddings'), \
        "Missing generate_protein_embeddings method"
    
    # Check ligand embedding method exists
    assert hasattr(pipeline, 'generate_ligand_embeddings'), \
        "Missing generate_ligand_embeddings method"
    
    # Check cache manager exists
    assert hasattr(pipeline, 'cache_manager'), \
        "Missing cache_manager attribute"
    
    # Check model manager exists
    assert hasattr(pipeline, 'model_manager'), \
        "Missing model_manager attribute"
    
    # Test minimal API calls
    emb1 = pipeline.generate_protein_embeddings(["MKTAYIAK"])
    emb2 = pipeline.generate_ligand_embeddings(["CCO"])
    
    assert emb1 is not None, "Protein embedding failed"
    assert emb2 is not None, "Ligand embedding failed"
    
    print("✅ All expected methods present:")
    print("   - generate_protein_embeddings ✅")
    print("   - generate_ligand_embeddings ✅")
    print("   - cache_manager ✅")
    print("   - model_manager ✅")
    
    print("\n✅ TEST 7.4 PASSED!\n")


def test_7_5_error_messages():
    """Test 7.5: Error messages are informative"""
    print("\n" + "="*80)
    print("TEST 7.5: Error Message Quality")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    errors_found = []
    
    # Test 1: Invalid model name
    try:
        pipeline.generate_protein_embeddings(
            source=["MKTAYIAK"],
            model_name="invalid_model_xyz"
        )
        errors_found.append("FAIL: Should raise error for invalid model")
    except Exception as e:
        error_msg = str(e).lower()
        if 'unknown' in error_msg or 'invalid' in error_msg or 'not found' in error_msg:
            print(f"✅ Invalid model error: {e}")
        else:
            errors_found.append(f"Unclear error for invalid model: {e}")
    
    # Test 2: Empty input
    try:
        pipeline.generate_protein_embeddings(source=[])
        errors_found.append("FAIL: Should raise error for empty input")
    except Exception as e:
        error_msg = str(e).lower()
        if 'empty' in error_msg or 'no' in error_msg:
            print(f"✅ Empty input error: {e}")
        else:
            errors_found.append(f"Unclear error for empty input: {e}")
    
    # Test 3: Invalid SMILES
    try:
        embeddings = pipeline.generate_ligand_embeddings(
            source=["INVALID_SMILES_XYZ"],
            model_name='smi_ted_light',
            validate=True
        )
        # If validation filters it, check if result is empty or error
        if embeddings.shape[0] == 0:
            print("✅ Invalid SMILES filtered correctly")
        else:
            errors_found.append(f"Invalid SMILES not filtered: {embeddings.shape}")
    except Exception as e:
        print(f"✅ Invalid SMILES error: {e}")
    
    if errors_found:
        print("\n❌ Error message issues:")
        for error in errors_found:
            print(f"   - {error}")
        raise AssertionError("Error message quality needs improvement")
    
    print("\n✅ TEST 7.5 PASSED!\n")


def test_7_6_performance_baseline():
    """Test 7.6: Performance baseline (not too slow)"""
    print("\n" + "="*80)
    print("TEST 7.6: Performance Baseline")
    print("="*80)
    
    import time
    
    pipeline = EmbeddingPipeline(verbose=False, use_cache=False)
    
    # Test protein embedding speed
    sequences = ["MKTAYIAK"] * 10
    start = time.time()
    emb = pipeline.generate_protein_embeddings(
        source=sequences,
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    protein_time = time.time() - start
    
    # Test ligand embedding speed
    smiles = ["CCO"] * 10
    start = time.time()
    emb = pipeline.generate_ligand_embeddings(
        source=smiles,
        model_name='smi_ted_light',
        use_cache=False
    )
    ligand_time = time.time() - start
    
    print(f"✅ Performance metrics:")
    print(f"   Protein: {protein_time:.3f}s for {len(sequences)} sequences ({protein_time/len(sequences):.3f}s per seq)")
    print(f"   Ligand: {ligand_time:.3f}s for {len(smiles)} molecules ({ligand_time/len(smiles):.3f}s per mol)")
    
    # Sanity check - should not take forever
    assert protein_time < 60, f"Protein embedding too slow: {protein_time}s"
    assert ligand_time < 60, f"Ligand embedding too slow: {ligand_time}s"
    
    print("\n✅ TEST 7.6 PASSED!\n")


def run_all_tests():
    """Run basic compatibility tests (fast subset)"""
    print("\n" + "="*80)
    print("LEVEL 7A: BASIC COMPATIBILITY TESTS (FAST)")
    print("="*80)
    
    tests = [
        ("7.1", "Output Format", test_7_1_output_format_compatibility),
        ("7.4", "API Stability", test_7_4_api_interface_stability),
        ("7.5", "Error Messages", test_7_5_error_messages),
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
    print("BASIC COMPATIBILITY TEST SUMMARY")
    print("="*80)
    print(f"📊 Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped: {skipped}")
    
    if failed == 0:
        print("\n" + "="*80)
        print("✅ ALL BASIC COMPATIBILITY TESTS PASSED!")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print("❌ SOME TESTS FAILED")
        print("="*80)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
