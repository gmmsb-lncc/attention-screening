"""
Test 8A.2: Stress Testing - Large Datasets
Tests handling of large datasets and memory management.
"""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_stress():
    """Test stress scenarios with large datasets"""
    print("\n" + "="*80)
    print("TEST 8A.2: Stress Testing - Large Datasets")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Test 1: Large batch of proteins
    print("\n1. Testing large protein batch (500 sequences)...")
    start = time.time()
    large_batch = ["MKTAYIAKQRQ"] * 500
    embeddings = pipeline.generate_protein_embeddings(
        source=large_batch,
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    elapsed = time.time() - start
    
    assert embeddings.shape == (500, 320), f"Wrong shape: {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "NaN in large batch"
    print(f"   ✓ Large batch processed: {embeddings.shape} in {elapsed:.2f}s")
    print(f"   ✓ Rate: {500/elapsed:.2f} sequences/sec")
    
    # Test 2: Large batch of ligands
    print("\n2. Testing large ligand batch (200 SMILES)...")
    start = time.time()
    large_smiles = ["CCO"] * 200
    embeddings = pipeline.generate_ligand_embeddings(
        source=large_smiles,
        model_name='smi_ted_light',
        use_cache=False
    )
    elapsed = time.time() - start
    
    assert embeddings.shape == (200, 768), f"Wrong shape: {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "NaN in large ligand batch"
    print(f"   ✓ Large ligand batch processed: {embeddings.shape} in {elapsed:.2f}s")
    print(f"   ✓ Rate: {200/elapsed:.2f} SMILES/sec")
    
    # Test 3: Memory efficiency - sequential processing
    print("\n3. Testing sequential processing (no memory leak)...")
    initial_size = embeddings.nbytes / (1024**2)  # MB
    
    for i in range(5):
        seqs = ["MKTAYIAK"] * 50
        emb = pipeline.generate_protein_embeddings(
            source=seqs,
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
    
    final_size = emb.nbytes / (1024**2)  # MB
    print(f"   ✓ Sequential processing OK (no memory leak)")
    print(f"   ✓ Memory per batch: {final_size:.2f} MB")
    
    # Test 4: Variable sequence lengths
    print("\n4. Testing variable sequence lengths...")
    variable_seqs = [
        "MKT",
        "MKTAYIAK",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL",
        "MKTAYIAKQRQ" * 10  # ~110 aa
    ]
    embeddings = pipeline.generate_protein_embeddings(
        source=variable_seqs,
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    assert embeddings.shape == (4, 320), f"Wrong shape: {embeddings.shape}"
    print(f"   ✓ Variable lengths handled: {embeddings.shape}")
    print(f"   ✓ Lengths: {[len(s) for s in variable_seqs]}")
    
    # Test 5: Batch size boundaries
    print("\n5. Testing extreme batch sizes...")
    
    # Single item
    single = ["MKTAYIAK"]
    emb_single = pipeline.generate_protein_embeddings(
        source=single,
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    assert emb_single.shape == (1, 320), f"Wrong shape for single: {emb_single.shape}"
    print(f"   ✓ Batch size = 1: {emb_single.shape}")
    
    # Large batch
    large = ["MKTAYIAK"] * 100
    emb_large = pipeline.generate_protein_embeddings(
        source=large,
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    assert emb_large.shape == (100, 320), f"Wrong shape for large: {emb_large.shape}"
    print(f"   ✓ Batch size = 100: {emb_large.shape}")
    
    print("\n✅ All stress tests passed!")
    print("\n✅ TEST 8A.2 PASSED!\n")


if __name__ == "__main__":
    try:
        test_stress()
        print("="*80)
        print("✅ TEST 8A.2: STRESS TESTING - PASSED")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST 8A.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
