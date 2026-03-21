#!/usr/bin/env python3
"""
Run All Classifier Tests
Execute todos os 93 testes do módulo classifier
"""

import subprocess
import sys
import os
from pathlib import Path

def run_test_file(test_file: Path) -> tuple:
    """Executa um arquivo de teste e retorna (sucesso, output)."""
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=120
        )
        # Sucesso baseado apenas em return code
        success = result.returncode == 0
        return success, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)

def main():
    test_dir = Path(__file__).parent
    
    # Lista de arquivos de teste em ordem
    test_files = [
        # Level 1 - Foundation (7 tests)
        "test_1_1_data_loader.py",
        "test_1_2_device_manager.py",
        "test_1_3_metrics_calculator.py",
        "test_1_4_config_manager.py",
        "test_1_5_data_validation.py",
        "test_1_6_import_utils.py",
        "test_1_7_train_test_split.py",
        # Level 2 - Models (3 tests)
        "test_2_1_mlp_classifier.py",
        "test_2_3_model_forward.py",
        "test_2_4_model_gradients.py",
        # Level 3 - Training & Evaluation (42 tests)
        "test_3_1_training_loop.py",
        "test_3_2_optimizer_scheduler.py",
        "test_3_3_amp_training.py",
        "test_3_4_early_stopping.py",
        "test_3_5_gradient_clipping.py",
        "test_3_6_cross_validation.py",
        # Level 4 - Integration (14 tests)
        "test_4_1_end_to_end_pipeline.py",
        "test_4_2_component_integration.py",
        # Level 5 - Edge Cases (7 tests)
        "test_5_edge_cases.py",
        # Level 6 - Performance (3 tests)
        "test_6_performance.py",
        # Level 7 - Serialization (2 tests)
        "test_7_serialization.py",
        # Level 8 - End-to-End (2 tests)
        "test_8_end_to_end.py",
        # Level 9 - Hyperparameter Optimization (3 tests)
        "test_9_hyperparameter_optimization.py",
        # Level 10 - Optional Dependencies (5 tests)
        "test_10_optional_dependencies.py",
        # Level 11 - Robust Train-Test Split (3 tests)
        "test_11_robust_train_test_split.py",
        # Level 12 - CLI Integration (2 tests)
        "test_12_cli_integration.py",
    ]
    
    print("="*70)
    print("RUNNING ALL CLASSIFIER TESTS (93 tests total)")
    print("="*70)
    
    results = []
    total_passed = 0
    total_failed = 0
    
    for i, test_file in enumerate(test_files, 1):
        test_path = test_dir / test_file
        
        if not test_path.exists():
            print(f"\n⚠️  Test {i}/{len(test_files)}: {test_file} NOT FOUND")
            results.append((test_file, False, "FILE NOT FOUND"))
            continue
        
        print(f"\n🔄 Running {i}/{len(test_files)}: {test_file}...")
        success, output = run_test_file(test_path)
        
        if success:
            # Contar testes passando no output
            passed = output.count("PASSED")
            total_passed += passed
            print(f"   ✅ PASSED")
            results.append((test_file, True, None))
        else:
            # Contar testes falhando
            failed = output.count("FAILED")
            total_failed += failed if failed > 0 else 1
            print(f"   ❌ FAILED")
            results.append((test_file, False, "CHECK OUTPUT"))
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL TESTS")
    print("="*70)
    
    passed_files = sum(1 for _, success, _ in results if success)
    failed_files = sum(1 for _, success, _ in results if not success)
    
    for test_file, success, error in results:
        symbol = "✅" if success else "❌"
        status = "PASSED" if success else "FAILED"
        print(f"{symbol} {test_file}: {status}")
        if error and error != "CHECK OUTPUT":
            print(f"   Error: {error}")
    
    print(f"\nTest Files: {len(test_files)}")
    print(f"Passed Files: {passed_files}")
    print(f"Failed Files: {failed_files}")
    
    if total_passed > 0:
        print(f"\nEstimated Tests Passed: ~{total_passed}")
    
    print(f"Success Rate: {(passed_files/len(test_files)*100):.1f}%")
    print("="*70)
    
    if failed_files > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
