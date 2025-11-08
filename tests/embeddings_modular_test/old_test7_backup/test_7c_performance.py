"""
Level 7C: Performance Tests
Tests performance baseline (slower).
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_7c_1_performance():
    """Test 7C.1: Performance baseline (not too slow)"""
    print("\n" + "="*80)
    print("TEST 7C.1: Performance Baseline")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
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
    
    print("\n✅ TEST 7C.1 PASSED!\n")


def run_all_tests():
    """Run performance tests"""
    print("\n" + "="*80)
    print("LEVEL 7C: PERFORMANCE TESTS")
    print("="*80)
    print("\n⚠️  Note: These tests may take 1-2 minutes\n")
    
    tests = [
        ("7C.1", "Performance Baseline", test_7c_1_performance),
    ]
    
    passed = 0
    failed = 0
    
    for test_id, name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_id} {name} PASSED\n")
            passed += 1
        except Exception as e:
            print(f"❌ {test_id} {name} FAILED: {e}\n")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("PERFORMANCE TEST SUMMARY")
    print("="*80)
    print(f"📊 Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n✅ ALL PERFORMANCE TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
