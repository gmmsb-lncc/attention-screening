"""
Master test runner for modular embeddings tests.
Runs all tests in sequence from basic to advanced.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import all test modules
from test_1_validators import (
    test_protein_validation,
    test_smiles_validation
)

from test_2_data_loader import (
    test_load_from_list,
    test_load_from_fasta,
    test_load_from_csv,
    test_load_smiles_from_list,
    test_load_from_dataframe
)

from test_3_model_registry import (
    test_model_registry_basic,
    test_model_info,
    test_model_validation,
    test_all_esm_models_info
)

from test_4_cache import (
    test_cache_manager_init,
    test_memory_cache,
    test_disk_cache,
    test_cache_miss,
    test_clear_cache
)

from test_5_integration import (
    test_pipeline_protein_embeddings_small,
    test_pipeline_with_real_data,
    test_pipeline_ligand_embeddings,
    test_pipeline_error_handling
)


def run_all_tests():
    """Run all tests in sequence"""
    
    print("\n" + "="*70)
    print(" 🧪 MODULAR EMBEDDINGS - COMPLETE TEST SUITE ".center(70, "="))
    print("="*70)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    # Define test suite
    test_suite = [
        ("LEVEL 1: VALIDATORS", [
            ("1.1 Protein Validation", test_protein_validation),
            ("1.2 SMILES Validation", test_smiles_validation),
        ]),
        ("LEVEL 2: DATA LOADER", [
            ("2.1 Load from List", test_load_from_list),
            ("2.2 Load from FASTA", test_load_from_fasta),
            ("2.3 Load from CSV", test_load_from_csv),
            ("2.4 Load SMILES from List", test_load_smiles_from_list),
            ("2.5 Load from DataFrame", test_load_from_dataframe),
        ]),
        ("LEVEL 3: MODEL REGISTRY", [
            ("3.1 Registry Basic Operations", test_model_registry_basic),
            ("3.2 Model Information", test_model_info),
            ("3.3 Model Validation", test_model_validation),
            ("3.4 All ESM Models Info", test_all_esm_models_info),
        ]),
        ("LEVEL 4: CACHE MANAGER", [
            ("4.1 Cache Initialization", test_cache_manager_init),
            ("4.2 Memory Cache", test_memory_cache),
            ("4.3 Disk Cache", test_disk_cache),
            ("4.4 Cache Miss", test_cache_miss),
            ("4.5 Clear Cache", test_clear_cache),
        ]),
        ("LEVEL 5: INTEGRATION (REAL MODELS)", [
            ("5.1 Protein Embeddings (ESM2 8M)", test_pipeline_protein_embeddings_small),
            ("5.2 Real Dataset Test", test_pipeline_with_real_data),
            ("5.3 Ligand Embeddings (FM4M)", test_pipeline_ligand_embeddings),
            ("5.4 Error Handling", test_pipeline_error_handling),
        ]),
    ]
    
    # Run all test levels
    for level_name, tests in test_suite:
        print(f"\n{'='*70}")
        print(f" {level_name} ".center(70, "="))
        print(f"{'='*70}")
        
        for test_name, test_func in tests:
            total_tests += 1
            try:
                test_func()
                passed_tests += 1
                print(f"\n✅ {test_name} PASSED")
            except Exception as e:
                failed_tests.append((test_name, str(e)))
                print(f"\n❌ {test_name} FAILED: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print(" TEST SUMMARY ".center(70, "="))
    print("="*70)
    print(f"\n📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\n❌ FAILED TESTS:")
        for test_name, error in failed_tests:
            print(f"\n   {test_name}:")
            print(f"      {error}")
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED".center(70))
        print("="*70 + "\n")
        return False
    else:
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!".center(70))
        print("="*70)
        print(f"\n🎉 Successfully validated modular embeddings implementation!")
        print(f"   - {total_tests} tests passed")
        print(f"   - All components working correctly")
        print(f"   - Ready for production use\n")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
