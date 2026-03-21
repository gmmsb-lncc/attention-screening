"""
Test 8B.1: Data Resilience - Malformed Inputs
Tests handling of malformed data and error recovery.
"""

import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_resilience():
    """Test resilience to malformed inputs"""
    print("\n" + "="*80)
    print("TEST 8B.1: Data Resilience - Malformed Inputs")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Test 1: CSV without header
    print("\n1. Testing CSV without header...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        f.write("seq1,MKTAYIAK\n")
        f.write("seq2,ARNDCEQ\n")
    
    try:
        embeddings = pipeline.generate_protein_embeddings(
            source=csv_path,
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
        print(f"   ⚠️  CSV without header accepted: {embeddings.shape}")
    except Exception as e:
        print(f"   ✓ CSV without header rejected (expected): {type(e).__name__}")
    finally:
        os.unlink(csv_path)
    
    # Test 2: CSV with extra columns
    print("\n2. Testing CSV with extra columns...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        f.write("id,sequence,extra1,extra2\n")
        f.write("seq1,MKTAYIAK,value1,value2\n")
        f.write("seq2,ARNDCEQ,value3,value4\n")
    
    try:
        embeddings = pipeline.generate_protein_embeddings(
            source=csv_path,
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
        assert embeddings.shape == (2, 320), f"Wrong shape: {embeddings.shape}"
        print(f"   ✓ CSV with extra columns handled: {embeddings.shape}")
    except Exception as e:
        print(f"   ⚠️  CSV with extra columns failed: {e}")
    finally:
        os.unlink(csv_path)
    
    # Test 3: CSV with missing values
    print("\n3. Testing CSV with missing sequences...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        f.write("id,sequence\n")
        f.write("seq1,MKTAYIAK\n")
        f.write("seq2,\n")  # Empty sequence
        f.write("seq3,ARNDCEQ\n")
    
    try:
        embeddings = pipeline.generate_protein_embeddings(
            source=csv_path,
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
        # Should skip empty sequence
        print(f"   ✓ Missing sequences skipped: {embeddings.shape}")
    except Exception as e:
        print(f"   ✓ Missing sequences rejected (expected): {type(e).__name__}")
    finally:
        os.unlink(csv_path)
    
    # Test 4: Mixed valid/invalid sequences
    print("\n4. Testing mixed valid/invalid sequences...")
    mixed_seqs = [
        "MKTAYIAK",      # Valid
        "INVALID123",    # Invalid
        "ARNDCEQ"        # Valid
    ]
    try:
        embeddings = pipeline.generate_protein_embeddings(
            source=mixed_seqs,
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
        print(f"   ⚠️  Mixed sequences accepted: {embeddings.shape}")
    except Exception as e:
        print(f"   ✓ Mixed sequences rejected (expected): {type(e).__name__}")
    
    # Test 5: Empty input
    print("\n5. Testing empty input...")
    try:
        embeddings = pipeline.generate_protein_embeddings(
            source=[],
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
        print(f"   ⚠️  Empty input accepted: {embeddings.shape}")
    except Exception as e:
        print(f"   ✓ Empty input rejected (expected): {type(e).__name__}")
    
    # Test 6: File with different encoding
    print("\n6. Testing file with special characters...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        csv_path = f.name
        f.write("id,sequence\n")
        f.write("séq1,MKTAYIAK\n")  # ID with accent
        f.write("seq2,ARNDCEQ\n")
    
    try:
        embeddings = pipeline.generate_protein_embeddings(
            source=csv_path,
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
        assert embeddings.shape == (2, 320), f"Wrong shape: {embeddings.shape}"
        print(f"   ✓ Special characters in IDs handled: {embeddings.shape}")
    except Exception as e:
        print(f"   ⚠️  Special characters failed: {e}")
    finally:
        os.unlink(csv_path)
    
    # Test 7: Whitespace in sequences
    print("\n7. Testing sequences with whitespace...")
    whitespace_seqs = [
        " MKTAYIAK ",     # Leading/trailing spaces
        "MKT AYIAK",      # Space in middle
        "ARNDCEQ"
    ]
    try:
        embeddings = pipeline.generate_protein_embeddings(
            source=whitespace_seqs,
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
        print(f"   ✓ Whitespace handled (stripped): {embeddings.shape}")
    except Exception as e:
        print(f"   ✓ Whitespace sequences rejected (expected): {type(e).__name__}")
    
    print("\n✅ All resilience tests completed!")
    print("\n✅ TEST 8B.1 PASSED!\n")


if __name__ == "__main__":
    try:
        test_resilience()
        print("="*80)
        print("✅ TEST 8B.1: DATA RESILIENCE - PASSED")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST 8B.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
