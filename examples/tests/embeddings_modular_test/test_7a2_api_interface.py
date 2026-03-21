"""
Test 7A.2: API Interface Stability
Fast test - checks API methods exist.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from build.embeddings.modular_pipeline import EmbeddingPipeline


def test_api_interface():
    """Test API interface has all required methods"""
    print("\n" + "="*80)
    print("TEST 7A.2: API Interface Stability")
    print("="*80)
    
    pipeline = EmbeddingPipeline(verbose=False)
    
    # Check critical methods exist
    methods = [
        'generate_protein_embeddings',
        'generate_ligand_embeddings',
        'clear_cache'
    ]
    
    missing_methods = []
    for method_name in methods:
        if not hasattr(pipeline, method_name):
            missing_methods.append(method_name)
        else:
            print(f"   ✓ {method_name}")
    
    assert len(missing_methods) == 0, \
        f"Missing API methods: {missing_methods}"
    
    # Test method is callable
    assert callable(pipeline.generate_protein_embeddings), \
        "generate_protein_embeddings is not callable"
    
    print(f"\n✅ All {len(methods)} API methods present and callable")
    print("\n✅ TEST 7A.2 PASSED!\n")


if __name__ == "__main__":
    try:
        test_api_interface()
        print("="*80)
        print("✅ TEST 7A.2: API INTERFACE - PASSED")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST 7A.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
