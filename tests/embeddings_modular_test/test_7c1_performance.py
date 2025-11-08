"""
Test 7C.1: Performance Baseline
Slower test - establishes performance baseline.
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_performance():
    """Test performance baseline"""
    print("\n" + "="*80)
    print("TEST 7C.1: Performance Baseline")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Test sequences
    sequences = ["MKTAYIAKQRQ"] * 10  # 10 identical sequences
    
    # Protein embeddings
    print("\n1. Testing protein embedding speed...")
    start = time.time()
    embeddings = pipeline.generate_protein_embeddings(
        source=sequences,
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    protein_time = time.time() - start
    print(f"   ✓ Protein: {protein_time:.2f}s for {len(sequences)} sequences")
    print(f"   ✓ Rate: {len(sequences)/protein_time:.2f} sequences/sec")
    
    # Ligand embeddings
    print("\n2. Testing ligand embedding speed...")
    smiles = ["CCO"] * 10  # 10 identical SMILES
    start = time.time()
    embeddings = pipeline.generate_ligand_embeddings(
        source=smiles,
        model_name='smi_ted_light',
        use_cache=False
    )
    ligand_time = time.time() - start
    print(f"   ✓ Ligand: {ligand_time:.2f}s for {len(smiles)} SMILES")
    print(f"   ✓ Rate: {len(smiles)/ligand_time:.2f} SMILES/sec")
    
    # Performance baseline (should be reasonably fast with 8M model)
    assert protein_time < 30, f"Protein embeddings too slow: {protein_time:.2f}s"
    assert ligand_time < 10, f"Ligand embeddings too slow: {ligand_time:.2f}s"
    
    print("\n✅ Performance within acceptable baseline")
    print("\n✅ TEST 7C.1 PASSED!\n")


if __name__ == "__main__":
    try:
        test_performance()
        print("="*80)
        print("✅ TEST 7C.1: PERFORMANCE - PASSED")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST 7C.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
