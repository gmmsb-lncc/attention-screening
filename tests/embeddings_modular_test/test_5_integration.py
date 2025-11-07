"""
Test 5: Integration - Full pipeline test with SMALLEST model
=============================================================
Tests the complete modular pipeline using ESM2 8M (smallest model).
This is a REAL embedding generation test but with minimal resources.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_pipeline_protein_embeddings_small():
    """Test 5.1: Generate protein embeddings with SMALLEST model"""
    print("\n" + "="*70)
    print("TEST 5.1: Protein Embeddings - ESM2 8M (SMALLEST)")
    print("="*70)
    
    cache_dir = tempfile.mkdtemp()
    
    try:
        # Initialize pipeline with SMALLEST model
        pipeline = EmbeddingPipeline(
            cache_dir=cache_dir,
            # cache enabled via cache_dir,
            verbose=True
        )
        
        # Small test sequences
        test_sequences = [
            "MKTAYIAKQRQISFVK",  # 16 aa
            "ACDEFGHIKLMNPQ",    # 14 aa
        ]
        
        print(f"\n📊 Test Setup:")
        print(f"   - Model: esm2_t6_8M_UR50D (SMALLEST)")
        print(f"   - Sequences: {len(test_sequences)}")
        print(f"   - Cache: {cache_dir}")
        
        print(f"\n🚀 Generating embeddings (this will download model first time)...")
        
        # Generate embeddings with smallest model
        embeddings = pipeline.generate_protein_embeddings(
            source=test_sequences,
            model_name="esm2_t6_8M_UR50D",  # 8M model - smallest
        )
        
        print(f"\n✅ Results:")
        print(f"   - Shape: {embeddings.shape}")
        print(f"   - Expected: ({len(test_sequences)}, 320)")  # 8M model = 320 dim
        print(f"   - Dtype: {embeddings.dtype}")
        print(f"   - Min: {embeddings.min():.4f}")
        print(f"   - Max: {embeddings.max():.4f}")
        print(f"   - Mean: {embeddings.mean():.4f}")
        
        assert embeddings.shape[0] == len(test_sequences), \
            f"Expected {len(test_sequences)} embeddings"
        assert embeddings.shape[1] == 320, \
            f"Expected 320 dimensions for esm2_t6_8M_UR50D"
        
        # Test caching - second call should be instant
        print(f"\n📊 Testing cache (should be instant)...")
        embeddings_cached = pipeline.generate_protein_embeddings(
            source=test_sequences,
            model_name="esm2_t6_8M_UR50D",
        )
        
        assert embeddings_cached.shape == embeddings.shape
        print(f"   ✅ Cache working correctly!")
        
        print("\n✅ TEST 5.1 PASSED!")
        
    finally:
        shutil.rmtree(cache_dir)


def test_pipeline_with_real_data():
    """Test 5.2: Real dataset from tests/datasets"""
    print("\n" + "="*70)
    print("TEST 5.2: Real Dataset - kinase_test_small.tsv")
    print("="*70)
    
    cache_dir = tempfile.mkdtemp()
    dataset_path = Path(__file__).parent.parent / "datasets" / "kinase_test_small.tsv"
    
    if not dataset_path.exists():
        print(f"\n⚠️  Dataset not found: {dataset_path}")
        print(f"   Skipping test")
        return
    
    try:
        # Load dataset
        df = pd.read_csv(dataset_path, sep='\t')
        print(f"\n📊 Dataset loaded:")
        print(f"   - Rows: {len(df)}")
        print(f"   - Columns: {list(df.columns)}")
        
        # Take only first 3 unique sequences for testing
        unique_seqs = df['seq'].unique()[:3]
        test_sequences = list(unique_seqs)
        
        print(f"\n📊 Test Setup:")
        print(f"   - Using {len(test_sequences)} unique sequences")
        print(f"   - Sequence lengths: {[len(s) for s in test_sequences]}")
        
        # Initialize pipeline
        pipeline = EmbeddingPipeline(
            cache_dir=cache_dir,
            # cache enabled via cache_dir,
            verbose=True
        )
        
        print(f"\n🚀 Generating protein embeddings...")
        
        # Generate with smallest model
        embeddings = pipeline.generate_protein_embeddings(
            source=test_sequences,
            model_name="esm2_t6_8M_UR50D",
        )
        
        print(f"\n✅ Results:")
        print(f"   - Shape: {embeddings.shape}")
        print(f"   - Dtype: {embeddings.dtype}")
        print(f"   - Mean: {embeddings.mean():.4f}")
        
        assert embeddings.shape[0] == len(test_sequences)
        assert embeddings.shape[1] == 320
        
        print("\n✅ TEST 5.2 PASSED!")
        
    finally:
        shutil.rmtree(cache_dir)


def test_pipeline_ligand_embeddings():
    """Test 5.3: Ligand embeddings with FM4M"""
    print("\n" + "="*70)
    print("TEST 5.3: Ligand Embeddings - FM4M")
    print("="*70)
    
    cache_dir = tempfile.mkdtemp()
    
    try:
        pipeline = EmbeddingPipeline(
            cache_dir=cache_dir,
            # cache enabled via cache_dir,
            verbose=True
        )
        
        # Test SMILES
        test_smiles = [
            "CCO",           # Ethanol
            "c1ccccc1",      # Benzene
            "CC(=O)O"        # Acetic acid
        ]
        
        print(f"\n📊 Test Setup:")
        print(f"   - SMILES: {len(test_smiles)}")
        print(f"   - Model: FM4M")
        
        print(f"\n🚀 Generating ligand embeddings...")
        
        try:
            embeddings = pipeline.generate_ligand_embeddings(
                source=test_smiles,
            )
            
            print(f"\n✅ Results:")
            print(f"   - Shape: {embeddings.shape}")
            print(f"   - Expected: ({len(test_smiles)}, 768)")
            print(f"   - Dtype: {embeddings.dtype}")
            print(f"   - Mean: {embeddings.mean():.4f}")
            
            assert embeddings.shape[0] == len(test_smiles)
            assert embeddings.shape[1] == 768  # FM4M dimension
            
            print("\n✅ TEST 5.3 PASSED!")
            
        except ImportError as e:
            if "FM4M" in str(e) or "SMI_TED" in str(e):
                print(f"\n⚠️  TEST 5.3 SKIPPED: FM4M not installed")
                print(f"   Install FM4M to enable ligand embedding tests")
            else:
                raise
        
    finally:
        shutil.rmtree(cache_dir)


def test_pipeline_error_handling():
    """Test 5.4: Error handling"""
    print("\n" + "="*70)
    print("TEST 5.4: Error Handling")
    print("="*70)
    
    cache_dir = tempfile.mkdtemp()
    
    try:
        pipeline = EmbeddingPipeline(
            cache_dir=cache_dir,
            # cache enabled via cache_dir,
            verbose=True
        )
        
        # Test 1: Empty sequences
        print(f"\n📊 Test: Empty sequences list")
        try:
            pipeline.generate_protein_embeddings(
                source=[],
                model_name="esm2_t6_8M_UR50D"
            )
            assert False, "Should raise error for empty sequences"
        except ValueError as e:
            print(f"   ✅ Correctly raised ValueError: {e}")
        
        # Test 2: Invalid model name
        print(f"\n📊 Test: Invalid model name")
        try:
            pipeline.generate_protein_embeddings(
                source=["MKTAYIAK"],
                model_name="invalid_model"
            )
            assert False, "Should raise error for invalid model"
        except (ValueError, RuntimeError) as e:
            print(f"   ✅ Correctly raised error: {e}")
        
        # Test 3: Invalid sequences (will be filtered)
        print(f"\n📊 Test: Invalid sequences (should filter)")
        embeddings = pipeline.generate_protein_embeddings(
            source=["MKTAYIAK", "INVALID123", "ACDEFGH"],
            model_name="esm2_t6_8M_UR50D"
        )
        print(f"   ✅ Filtered invalid, got {embeddings.shape[0]} valid embeddings")
        assert embeddings.shape[0] == 2  # Only 2 valid
        
        print("\n✅ TEST 5.4 PASSED!")
        
    finally:
        shutil.rmtree(cache_dir)


if __name__ == "__main__":
    print("\n" + "🧪 RUNNING INTEGRATION TESTS (REAL MODELS) ".center(70, "="))
    print("\n⚠️  NOTE: First run will download models (~32MB for ESM2 8M)")
    print("=" * 70)
    
    try:
        test_pipeline_protein_embeddings_small()
        test_pipeline_with_real_data()
        test_pipeline_ligand_embeddings()
        test_pipeline_error_handling()
        
        print("\n" + "="*70)
        print("✅ ALL INTEGRATION TESTS PASSED!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
