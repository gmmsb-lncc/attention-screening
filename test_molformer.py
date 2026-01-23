#!/usr/bin/env python3
"""
Test script for MoLFormer ligand embeddings
"""

import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from src.build.embeddings.ligand_embedding import LigandEmbedding
from src.build.core.config import BuildConfig

def test_molformer():
    """Test MoLFormer embeddings."""
    print("🧪 Testing MoLFormer ligand embeddings...")
    
    # Create a simple config
    config = BuildConfig({
        'ligand_dim': 768,  # MoLFormer dimension
        'batch_size': 8,
        'use_gpu': False  # Use CPU for testing
    })
    
    # Initialize ligand embedding with MoLFormer
    ligand_embedder = LigandEmbedding(
        config=config,
        model_name="MOLFORMER",
        use_parallel=False
    )
    
    # Test SMILES
    test_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]  # Ethanol, Acetic acid, Benzene
    
    print(f"Loading MoLFormer model...")
    try:
        # Generate embeddings
        print(f"Generating embeddings for {len(test_smiles)} SMILES...")
        embeddings = ligand_embedder.generate_batch_embeddings(test_smiles)
        
        print(f"✅ Successfully generated {len(embeddings)} embeddings")
        for i, (smiles, embedding) in enumerate(zip(test_smiles, embeddings)):
            print(f"   {smiles}: shape {embedding.shape}, mean={embedding.mean():.3f}")
        
        # Test per-token embeddings (matrix)
        print(f"\nTesting per-token embeddings for first SMILES: {test_smiles[0]}")
        matrix = ligand_embedder.generate_embedding_matrix(test_smiles[0])
        if matrix is not None:
            print(f"✅ Generated per-token embedding matrix: shape {matrix.shape}")
            print(f"   First token embedding (first 5 dims): {matrix[0][:5]}")
        else:
            print("❌ Could not generate per-token embeddings")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 MoLFormer test completed successfully!")
    return True

if __name__ == "__main__":
    test_molformer()