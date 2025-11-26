"""
Quick OpenFold-3 Validation Test

This script validates that OpenFold-3 can be loaded and used within 
the complete DockTKinase pipeline (just like Boltz-2).

Same objective as Boltz-2:
- Verify model loads correctly
- Generate embeddings for a test sequence
- Validate embedding dimensions (384-dim)
- Ensure it integrates with the pipeline

Usage:
    python tests/test_openfold3_quick.py

Author: DockTKinase Team
Date: 2025-11-26
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.build.embeddings.models.model_registry import ModelRegistry
from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy
from src.build.embeddings.config.msa_config import MsaConfig
import torch
import numpy as np


def main():
    print("\n" + "=" * 70)
    print("OpenFold-3 Quick Validation Test")
    print("=" * 70)
    
    # Test 1: Check model registry
    print("\n1. Checking model registry...")
    model_info = ModelRegistry.get_model_info('openfold3')
    if model_info:
        print(f"   ✓ OpenFold-3 registered")
        print(f"     - Name: {model_info.name}")
        print(f"     - Type: {model_info.type}")
        print(f"     - Embedding dim: {model_info.embedding_dim}")
        print(f"     - Description: {model_info.description}")
    else:
        print(f"   ✗ OpenFold-3 NOT in registry!")
        return 1
    
    # Test 2: Initialize strategy
    print("\n2. Initializing OpenFoldStrategy...")
    try:
        strategy = OpenFoldStrategy(
            msa_config=MsaConfig.no_msa()  # Fastest mode
        )
        print(f"   ✓ Strategy initialized")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return 1
    
    # Test 3: Load model
    print("\n3. Loading OpenFold-3 model...")
    try:
        device = torch.device('cpu')
        model, config = strategy.load('openfold3', device=device)
        print(f"   ✓ Model loaded on {device}")
    except FileNotFoundError as e:
        print(f"   ⚠ SKIP: OpenFold-3 not installed - {e}")
        print(f"   This is expected if OPENFOLD-3/ directory is not present.")
        return 0
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 4: Generate embedding
    print("\n4. Generating embedding for test sequence...")
    test_sequence = "MKFLKFSLMKCDEFGHIKLMNPQRSTV"
    try:
        embedding = strategy.generate(
            model=model,
            auxiliary_objects=None,
            sequence=test_sequence,
            device=device,
            pooling_strategy='mean'
        )
        print(f"   ✓ Embedding generated")
        print(f"     - Sequence: {test_sequence}")
        print(f"     - Length: {len(test_sequence)} AA")
        print(f"     - Output shape: {embedding.shape}")
        print(f"     - Expected: (384,)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        strategy.cleanup(model, None)
        return 1
    
    # Test 5: Validate output
    print("\n5. Validating embedding...")
    try:
        assert isinstance(embedding, np.ndarray), "Should be numpy array"
        assert embedding.shape == (384,), f"Wrong shape: {embedding.shape}"
        assert not np.isnan(embedding).any(), "Contains NaN"
        assert not np.isinf(embedding).any(), "Contains Inf"
        print(f"   ✓ Embedding is valid")
        print(f"     - Mean: {embedding.mean():.6f}")
        print(f"     - Std: {embedding.std():.6f}")
        print(f"     - Min: {embedding.min():.6f}")
        print(f"     - Max: {embedding.max():.6f}")
    except AssertionError as e:
        print(f"   ✗ Validation failed: {e}")
        strategy.cleanup(model, None)
        return 1
    
    # Test 6: Batch test
    print("\n6. Batch test (5 sequences)...")
    try:
        test_seqs = [
            "MKFLKFSL",
            "MKCDEFGHIKLMNPQRSTV",
            "MSDEFGHIKLMNPQRSTV",
            "MKKKKKKKKK",
            "MEEEEEEEEE",
        ]
        
        embeddings = []
        for seq in test_seqs:
            emb = strategy.generate(
                model=model,
                auxiliary_objects=None,
                sequence=seq,
                device=device
            )
            embeddings.append(emb)
            assert emb.shape == (384,), f"Unexpected shape: {emb.shape}"
        
        embeddings = np.array(embeddings)
        print(f"   ✓ Batch processing successful")
        print(f"     - Sequences: {len(embeddings)}")
        print(f"     - Batch shape: {embeddings.shape}")
        print(f"     - Expected: (5, 384)")
        
    except Exception as e:
        print(f"   ✗ Batch test failed: {e}")
        import traceback
        traceback.print_exc()
        strategy.cleanup(model, None)
        return 1
    
    # Test 7: Cleanup
    print("\n7. Cleanup...")
    try:
        strategy.cleanup(model, None)
        print(f"   ✓ Cleanup successful")
    except Exception as e:
        print(f"   ✗ Cleanup failed: {e}")
        return 1
    
    # Success!
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - OpenFold-3 is working correctly!")
    print("=" * 70)
    print("\nOpenFold-3 is now ready to be used in the pipeline:")
    print("  python run_complete_pipeline.py \\")
    print("    --input tests/datasets/kinase_non_human_compounds.tsv \\")
    print("    --output results/openfold3_test \\")
    print("    --protein-model openfold3")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
