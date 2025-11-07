"""
Validators for Embeddings

Simple validation functions for sequences and SMILES.
"""

from typing import List, Tuple


def validate_protein_batch(
    sequences: List[str],
    verbose: bool = True
) -> Tuple[List[str], List[int]]:
    """
    Validate protein sequences (basic validation).
    
    Args:
        sequences: List of protein sequences
        verbose: Whether to print warnings
        
    Returns:
        Tuple of (valid_sequences, valid_indices)
    """
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    valid_seqs = []
    valid_indices = []
    
    for i, seq in enumerate(sequences):
        seq_upper = seq.strip().upper()
        if seq_upper and all(aa in valid_amino_acids for aa in seq_upper):
            valid_seqs.append(seq_upper)
            valid_indices.append(i)
        elif verbose:
            print(f"   ⚠️  Skipping invalid sequence at index {i}")
    
    if verbose:
        print(f"   ✅ Validated {len(valid_seqs)}/{len(sequences)} sequences")
    
    return valid_seqs, valid_indices


def validate_smiles_batch(
    smiles_list: List[str],
    verbose: bool = True
) -> Tuple[List[str], List[int]]:
    """
    Validate SMILES strings (basic validation).
    
    Args:
        smiles_list: List of SMILES strings
        verbose: Whether to print warnings
        
    Returns:
        Tuple of (valid_smiles, valid_indices)
    """
    valid_smiles = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        smiles_clean = smiles.strip() if smiles else ""
        if smiles_clean:
            valid_smiles.append(smiles_clean)
            valid_indices.append(i)
        elif verbose:
            print(f"   ⚠️  Skipping empty SMILES at index {i}")
    
    if verbose:
        print(f"   ✅ Validated {len(valid_smiles)}/{len(smiles_list)} SMILES")
    
    return valid_smiles, valid_indices
