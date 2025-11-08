"""
Test 1: Validators - Basic input validation
============================================
Tests the simplified validators (proteins and SMILES).
This is the foundation - must work correctly.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from build.embeddings.utils.validators import (
    validate_protein_batch,
    validate_smiles_batch
)


def test_protein_validation():
    """Test 1.1: Validate protein sequences"""
    print("\n" + "="*70)
    print("TEST 1.1: Protein Sequence Validation")
    print("="*70)
    
    # Valid sequences
    valid_sequences = [
        "MKTAYIAKQRQISFVKSHFSRQ",  # Valid
        "ACDEFGHIKLMNPQRSTVWY",    # All 20 amino acids
        "MKWVTFISLLFLFSSAYS"       # Valid
    ]
    
    # Invalid sequences
    invalid_sequences = [
        "INVALID123",              # Numbers
        "MKT*AYIAK",               # Special char
        "",                        # Empty
        "ACBDEFGH"                 # Invalid letter B
    ]
    
    # Mix
    mixed_sequences = valid_sequences + invalid_sequences
    
    print(f"\n📊 Testing {len(mixed_sequences)} sequences:")
    print(f"   - Valid: {len(valid_sequences)}")
    print(f"   - Invalid: {len(invalid_sequences)}")
    
    valid_seqs, valid_indices = validate_protein_batch(mixed_sequences, verbose=True)
    
    print(f"\n✅ Results:")
    print(f"   - Validated: {len(valid_seqs)}/{len(mixed_sequences)}")
    print(f"   - Valid indices: {valid_indices}")
    
    assert len(valid_seqs) == len(valid_sequences), \
        f"Expected {len(valid_sequences)} valid, got {len(valid_seqs)}"
    assert valid_indices == [0, 1, 2], \
        f"Expected indices [0, 1, 2], got {valid_indices}"
    
    print("\n✅ TEST 1.1 PASSED!")


def test_smiles_validation():
    """Test 1.2: Validate SMILES strings"""
    print("\n" + "="*70)
    print("TEST 1.2: SMILES Validation")
    print("="*70)
    
    # Valid SMILES
    valid_smiles = [
        "CCO",                     # Ethanol
        "c1ccccc1",                # Benzene
        "CC(=O)O"                  # Acetic acid
    ]
    
    # Invalid SMILES (only empty/whitespace are filtered by current validator)
    invalid_smiles = [
        "",                        # Empty
        "   ",                     # Whitespace only
    ]
    
    # Mix
    mixed_smiles = valid_smiles + invalid_smiles
    
    print(f"\n📊 Testing {len(mixed_smiles)} SMILES:")
    print(f"   - Valid: {len(valid_smiles)}")
    print(f"   - Invalid: {len(invalid_smiles)}")
    
    valid_smi, valid_indices = validate_smiles_batch(mixed_smiles, verbose=True)
    
    print(f"\n✅ Results:")
    print(f"   - Validated: {len(valid_smi)}/{len(mixed_smiles)}")
    print(f"   - Valid indices: {valid_indices}")
    
    assert len(valid_smi) == len(valid_smiles), \
        f"Expected {len(valid_smiles)} valid, got {len(valid_smi)}"
    assert valid_indices == [0, 1, 2], \
        f"Expected indices [0, 1, 2], got {valid_indices}"
    
    print("\n✅ TEST 1.2 PASSED!")


if __name__ == "__main__":
    print("\n" + "🧪 RUNNING VALIDATOR TESTS ".center(70, "="))
    
    try:
        test_protein_validation()
        test_smiles_validation()
        
        print("\n" + "="*70)
        print("✅ ALL VALIDATOR TESTS PASSED!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
