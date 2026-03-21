"""
Test 2: DataManager - Load sequences from multiple sources
===========================================================
Tests loading sequences/SMILES from lists, FASTA, CSV, TSV, DataFrames.
"""

import sys
from pathlib import Path
import tempfile
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from build.embeddings.core.data_loader import DataManager


def test_load_from_list():
    """Test 2.1: Load sequences from list"""
    print("\n" + "="*70)
    print("TEST 2.1: Load Sequences from List")
    print("="*70)
    
    manager = DataManager(verbose=True)
    
    sequences = [
        "MKTAYIAKQRQISFVKSHFSRQ",
        "ACDEFGHIKLMNPQRSTVWY",
        "MKWVTFISLLFLFSSAYS"
    ]
    
    print(f"\n📊 Loading {len(sequences)} sequences from list...")
    
    loaded_seqs, ids = manager.load_sequences(sequences)
    
    print(f"\n✅ Results:")
    print(f"   - Loaded: {len(loaded_seqs)} sequences")
    print(f"   - IDs: {ids}")
    
    assert len(loaded_seqs) == len(sequences)
    assert ids == [f"seq_{i}" for i in range(len(sequences))]
    
    print("\n✅ TEST 2.1 PASSED!")


def test_load_from_fasta():
    """Test 2.2: Load sequences from FASTA file"""
    print("\n" + "="*70)
    print("TEST 2.2: Load Sequences from FASTA")
    print("="*70)
    
    manager = DataManager(verbose=True)
    
    # Create temporary FASTA file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(">seq1\n")
        f.write("MKTAYIAKQRQISFVKSHFSRQ\n")
        f.write(">seq2\n")
        f.write("ACDEFGHIKLMNPQRSTVWY\n")
        f.write(">seq3 description\n")
        f.write("MKWVTFISLLFLFSSAYS\n")
        fasta_path = f.name
    
    try:
        print(f"\n📊 Loading from FASTA: {fasta_path}")
        
        loaded_seqs, ids = manager.load_sequences(fasta_path)
        
        print(f"\n✅ Results:")
        print(f"   - Loaded: {len(loaded_seqs)} sequences")
        print(f"   - IDs: {ids}")
        
        assert len(loaded_seqs) == 3
        assert ids == ["seq1", "seq2", "seq3"]
        
        print("\n✅ TEST 2.2 PASSED!")
        
    finally:
        Path(fasta_path).unlink()


def test_load_from_csv():
    """Test 2.3: Load sequences from CSV file"""
    print("\n" + "="*70)
    print("TEST 2.3: Load Sequences from CSV")
    print("="*70)
    
    manager = DataManager(verbose=True)
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,sequence\n")
        f.write("prot1,MKTAYIAKQRQISFVKSHFSRQ\n")
        f.write("prot2,ACDEFGHIKLMNPQRSTVWY\n")
        f.write("prot3,MKWVTFISLLFLFSSAYS\n")
        csv_path = f.name
    
    try:
        print(f"\n📊 Loading from CSV: {csv_path}")
        
        loaded_seqs, ids = manager.load_sequences(
            csv_path, 
            sequence_column='sequence',
            id_column='id'
        )
        
        print(f"\n✅ Results:")
        print(f"   - Loaded: {len(loaded_seqs)} sequences")
        print(f"   - IDs: {ids}")
        
        assert len(loaded_seqs) == 3
        assert ids == ["prot1", "prot2", "prot3"]
        
        print("\n✅ TEST 2.3 PASSED!")
        
    finally:
        Path(csv_path).unlink()


def test_load_smiles_from_list():
    """Test 2.4: Load SMILES from list"""
    print("\n" + "="*70)
    print("TEST 2.4: Load SMILES from List")
    print("="*70)
    
    manager = DataManager(verbose=True)
    
    smiles_list = [
        "CCO",           # Ethanol
        "c1ccccc1",      # Benzene
        "CC(=O)O"        # Acetic acid
    ]
    
    print(f"\n📊 Loading {len(smiles_list)} SMILES from list...")
    
    loaded_smiles, ids = manager.load_smiles(smiles_list)
    
    print(f"\n✅ Results:")
    print(f"   - Loaded: {len(loaded_smiles)} SMILES")
    print(f"   - IDs: {ids}")
    
    assert len(loaded_smiles) == len(smiles_list)
    assert ids == [f"seq_{i}" for i in range(len(smiles_list))]  # Uses seq_ prefix
    
    print("\n✅ TEST 2.4 PASSED!")


def test_load_from_dataframe():
    """Test 2.5: Load from pandas DataFrame"""
    print("\n" + "="*70)
    print("TEST 2.5: Load from DataFrame")
    print("="*70)
    
    manager = DataManager(verbose=True)
    
    # Create DataFrame
    df = pd.DataFrame({
        'protein_id': ['P1', 'P2', 'P3'],
        'seq': [
            'MKTAYIAKQRQISFVKSHFSRQ',
            'ACDEFGHIKLMNPQRSTVWY',
            'MKWVTFISLLFLFSSAYS'
        ]
    })
    
    print(f"\n📊 Loading from DataFrame with {len(df)} rows...")
    
    loaded_seqs, ids = manager.load_sequences(
        df,
        sequence_column='seq',
        id_column='protein_id'
    )
    
    print(f"\n✅ Results:")
    print(f"   - Loaded: {len(loaded_seqs)} sequences")
    print(f"   - IDs: {ids}")
    
    assert len(loaded_seqs) == len(df)
    assert ids == ['P1', 'P2', 'P3']
    
    print("\n✅ TEST 2.5 PASSED!")


if __name__ == "__main__":
    print("\n" + "🧪 RUNNING DATA LOADER TESTS ".center(70, "="))
    
    try:
        test_load_from_list()
        test_load_from_fasta()
        test_load_from_csv()
        test_load_smiles_from_list()
        test_load_from_dataframe()
        
        print("\n" + "="*70)
        print("✅ ALL DATA LOADER TESTS PASSED!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
