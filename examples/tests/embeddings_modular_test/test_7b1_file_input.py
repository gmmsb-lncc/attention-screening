"""
Test 7B.1: File Input Processing
Moderate test - checks CSV/TSV file reading.
"""

import sys
from pathlib import Path
import pandas as pd
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_file_input():
    """Test reading sequences from CSV/TSV files"""
    print("\n" + "="*80)
    print("TEST 7B.1: File Input Processing")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        f.write("id,sequence\n")
        f.write("seq1,MKTAYIAK\n")
        f.write("seq2,ARNDCEQ\n")
    
    try:
        print("\n1. Testing CSV file input...")
        embeddings = pipeline.generate_protein_embeddings(
            source=csv_path,
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
        assert embeddings.shape[0] == 2, f"Expected 2 rows, got {embeddings.shape[0]}"
        print(f"   ✓ CSV processed: {embeddings.shape}")
        
        # Create TSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            tsv_path = f.name
            f.write("id\tsequence\n")
            f.write("seq1\tMKTAYIAK\n")
            f.write("seq2\tARNDCEQ\n")
        
        print("\n2. Testing TSV file input...")
        embeddings = pipeline.generate_protein_embeddings(
            source=tsv_path,
            model_name='esm2_t6_8M_UR50D',
            use_cache=False
        )
        assert embeddings.shape[0] == 2, f"Expected 2 rows, got {embeddings.shape[0]}"
        print(f"   ✓ TSV processed: {embeddings.shape}")
        
        os.unlink(tsv_path)
        
    finally:
        os.unlink(csv_path)
    
    print("\n✅ File input processing works correctly")
    print("\n✅ TEST 7B.1 PASSED!\n")


if __name__ == "__main__":
    try:
        test_file_input()
        print("="*80)
        print("✅ TEST 7B.1: FILE INPUT - PASSED")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST 7B.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
