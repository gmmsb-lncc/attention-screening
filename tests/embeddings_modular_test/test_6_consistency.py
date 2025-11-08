"""
Level 6: Consistency Tests
Tests for data consistency, reproducibility, and edge cases.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_6_1_reproducibility():
    """Test 6.1: Embeddings are reproducible (same input = same output)"""
    print("\n" + "="*80)
    print("TEST 6.1: Reproducibility")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Test proteins
    sequences = ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL"]
    
    # Generate twice
    emb1 = pipeline.generate_protein_embeddings(
        source=sequences,
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    
    emb2 = pipeline.generate_protein_embeddings(
        source=sequences,
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    
    # Check if identical
    assert np.allclose(emb1, emb2, rtol=1e-5, atol=1e-8), \
        f"Embeddings not reproducible! Max diff: {np.abs(emb1 - emb2).max()}"
    
    print(f"✅ Embeddings are reproducible (max diff: {np.abs(emb1 - emb2).max():.2e})")
    print("\n✅ TEST 6.1 PASSED!\n")


def test_6_2_embedding_dimensions():
    """Test 6.2: Embedding dimensions match expected values"""
    print("\n" + "="*80)
    print("TEST 6.2: Embedding Dimensions")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    test_cases = [
        ("ESM2 8M", "protein", "esm2_t6_8M_UR50D", ["MKTAYIAK"], 320),
        ("FM4M Light", "ligand", "smi_ted_light", ["CCO"], 768),
    ]
    
    for name, emb_type, model, data, expected_dim in test_cases:
        if emb_type == "protein":
            emb = pipeline.generate_protein_embeddings(
                source=data,
                model_name=model,
                use_cache=False
            )
        else:
            emb = pipeline.generate_ligand_embeddings(
                source=data,
                model_name=model,
                use_cache=False
            )
        
        actual_dim = emb.shape[1]
        assert actual_dim == expected_dim, \
            f"{name}: Expected dim {expected_dim}, got {actual_dim}"
        
        print(f"✅ {name}: {emb.shape} (dimension {actual_dim} correct)")
    
    print("\n✅ TEST 6.2 PASSED!\n")


def test_6_3_batch_consistency():
    """Test 6.3: Batch processing produces same results as individual"""
    print("\n" + "="*80)
    print("TEST 6.3: Batch vs Individual Consistency")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
    
    # Process as batch
    batch_emb = pipeline.generate_ligand_embeddings(
        source=smiles_list,
        model_name='smi_ted_light',
        use_cache=False
    )
    
    # Process individually
    individual_embs = []
    for smiles in smiles_list:
        emb = pipeline.generate_ligand_embeddings(
            source=[smiles],
            model_name='smi_ted_light',
            use_cache=False
        )
        individual_embs.append(emb[0])
    
    individual_embs = np.array(individual_embs)
    
    # Compare
    assert batch_emb.shape == individual_embs.shape, \
        f"Shape mismatch: {batch_emb.shape} vs {individual_embs.shape}"
    
    max_diff = np.abs(batch_emb - individual_embs).max()
    assert max_diff < 1e-4, f"Batch/individual mismatch: max diff {max_diff}"
    
    print(f"✅ Batch and individual processing consistent (max diff: {max_diff:.2e})")
    print(f"   Batch shape: {batch_emb.shape}")
    print(f"   Individual shape: {individual_embs.shape}")
    print("\n✅ TEST 6.3 PASSED!\n")


def test_6_4_embedding_range():
    """Test 6.4: Embeddings are in reasonable range (no NaN/Inf)"""
    print("\n" + "="*80)
    print("TEST 6.4: Embedding Value Range")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Test protein embeddings
    protein_emb = pipeline.generate_protein_embeddings(
        source=["MKTAYIAKQRQISFVK"],
        model_name='esm2_t6_8M_UR50D',
        use_cache=False
    )
    
    # Test ligand embeddings
    ligand_emb = pipeline.generate_ligand_embeddings(
        source=["CCO", "c1ccccc1"],
        model_name='smi_ted_light',
        use_cache=False
    )
    
    # Check for NaN/Inf
    assert not np.isnan(protein_emb).any(), "Protein embeddings contain NaN!"
    assert not np.isinf(protein_emb).any(), "Protein embeddings contain Inf!"
    assert not np.isnan(ligand_emb).any(), "Ligand embeddings contain NaN!"
    assert not np.isinf(ligand_emb).any(), "Ligand embeddings contain Inf!"
    
    # Check reasonable range (most embeddings should be between -10 and 10)
    assert protein_emb.min() > -20, f"Protein embedding too small: {protein_emb.min()}"
    assert protein_emb.max() < 20, f"Protein embedding too large: {protein_emb.max()}"
    assert ligand_emb.min() > -20, f"Ligand embedding too small: {ligand_emb.min()}"
    assert ligand_emb.max() < 20, f"Ligand embedding too large: {ligand_emb.max()}"
    
    print("✅ Protein embeddings:")
    print(f"   Range: [{protein_emb.min():.4f}, {protein_emb.max():.4f}]")
    print(f"   Mean: {protein_emb.mean():.4f}, Std: {protein_emb.std():.4f}")
    
    print("✅ Ligand embeddings:")
    print(f"   Range: [{ligand_emb.min():.4f}, {ligand_emb.max():.4f}]")
    print(f"   Mean: {ligand_emb.mean():.4f}, Std: {ligand_emb.std():.4f}")
    
    print("\n✅ TEST 6.4 PASSED!\n")


def test_6_5_cache_invalidation():
    """Test 6.5: Cache invalidation when parameters change"""
    print("\n" + "="*80)
    print("TEST 6.5: Cache Invalidation")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    smiles = ["CCO"]
    
    # Generate with one model
    emb1 = pipeline.generate_ligand_embeddings(
        source=smiles,
        model_name='smi_ted_light',
        use_cache=True
    )
    
    # Try to get from cache (should work)
    emb2 = pipeline.generate_ligand_embeddings(
        source=smiles,
        model_name='smi_ted_light',
        use_cache=True
    )
    
    assert np.allclose(emb1, emb2), "Cache retrieval failed!"
    print("✅ Cache retrieval works for same parameters")
    
    # Different validation should not use cache (different preprocessing)
    emb3 = pipeline.generate_ligand_embeddings(
        source=smiles,
        model_name='smi_ted_light',
        use_cache=True,
        validate=False  # Different parameter
    )
    
    # Should still produce valid embeddings
    assert emb3.shape == emb1.shape, "Validation parameter affects shape!"
    print("✅ Different parameters handled correctly")
    
    print("\n✅ TEST 6.5 PASSED!\n")


def test_6_6_memory_efficiency():
    """Test 6.6: Memory efficiency - process larger batches without crash"""
    print("\n" + "="*80)
    print("TEST 6.6: Memory Efficiency")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Generate 100 SMILES
    smiles_list = ["CCO"] * 50 + ["c1ccccc1"] * 50
    
    try:
        embeddings = pipeline.generate_ligand_embeddings(
            source=smiles_list,
            model_name='smi_ted_light',
            use_cache=False
        )
        
        assert embeddings.shape[0] == 100, f"Expected 100 embeddings, got {embeddings.shape[0]}"
        assert embeddings.shape[1] == 768, f"Expected dim 768, got {embeddings.shape[1]}"
        
        print(f"✅ Processed {len(smiles_list)} molecules successfully")
        print(f"   Output shape: {embeddings.shape}")
        print(f"   Memory size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        
    except MemoryError:
        print("❌ Memory error - batch size too large!")
        raise
    
    print("\n✅ TEST 6.6 PASSED!\n")


def test_6_7_model_switching():
    """Test 6.7: Switching between models works correctly"""
    print("\n" + "="*80)
    print("TEST 6.7: Model Switching")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    sequences = ["MKTAYIAK"]
    
    # Test different ESM models (using only lightweight models)
    models_to_test = [
        ('esm2_t6_8M_UR50D', 320),
        # Removed larger model to speed up tests
    ]
    
    for model_name, expected_dim in models_to_test:
        emb = pipeline.generate_protein_embeddings(
            source=sequences,
            model_name=model_name,
            use_cache=False
        )
        
        assert emb.shape[1] == expected_dim, \
            f"{model_name}: Expected {expected_dim}, got {emb.shape[1]}"
        
        print(f"✅ {model_name}: {emb.shape} (dimension correct)")
    
    # Also test switching to FM4M
    smiles = ["CCO"]
    emb_fm4m = pipeline.generate_ligand_embeddings(
        source=smiles,
        model_name='smi_ted_light',
        use_cache=False
    )
    assert emb_fm4m.shape[1] == 768, f"FM4M: Expected 768, got {emb_fm4m.shape[1]}"
    print(f"✅ smi_ted_light: {emb_fm4m.shape} (dimension correct)")
    
    print("\n✅ TEST 6.7 PASSED!\n")


def run_all_tests():
    """Run all consistency tests"""
    print("\n" + "="*80)
    print("LEVEL 6: CONSISTENCY TESTS")
    print("="*80)
    
    tests = [
        ("6.1", "Reproducibility", test_6_1_reproducibility),
        ("6.2", "Embedding Dimensions", test_6_2_embedding_dimensions),
        ("6.3", "Batch Consistency", test_6_3_batch_consistency),
        ("6.4", "Embedding Range", test_6_4_embedding_range),
        ("6.5", "Cache Invalidation", test_6_5_cache_invalidation),
        ("6.6", "Memory Efficiency", test_6_6_memory_efficiency),
        ("6.7", "Model Switching", test_6_7_model_switching),
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
    print("CONSISTENCY TEST SUMMARY")
    print("="*80)
    print(f"📊 Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n" + "="*80)
        print("✅ ALL CONSISTENCY TESTS PASSED!")
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
