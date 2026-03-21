#!/usr/bin/env python3
"""
Master Runner: Level 3 - Training & Evaluation
================================================

Executa todos os testes de Level 3 (Training & Evaluation) sequencialmente.

Level 3 inclui:
- Test 3.1: Basic Training Loop (7 tests)
- Test 3.2: Optimizer & Scheduler (7 tests)
- Test 3.3: AMP Training (7 tests)
- Test 3.4: Early Stopping (7 tests)
- Test 3.5: Gradient Clipping (7 tests)
- Test 3.6: Cross-Validation (7 tests)

Total: 42 tests

Author: Test Suite
Date: 2024
"""

import sys
import time
from pathlib import Path

# Adicionar diretório de testes ao path
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

# Imports dos test modules
import test_3_1_training_loop
import test_3_2_optimizer_scheduler
import test_3_3_amp_training
import test_3_4_early_stopping
import test_3_5_gradient_clipping
import test_3_6_cross_validation


def run_level_3():
    """Executa todos os testes de Level 3."""
    print("\n" + "="*70)
    print("🚀 LEVEL 3: TRAINING & EVALUATION - MASTER RUNNER")
    print("="*70)
    print("Running comprehensive training and evaluation tests...")
    print()
    
    test_modules = [
        ("Level 3.1 - Basic Training Loop", test_3_1_training_loop),
        ("Level 3.2 - Optimizer & Scheduler", test_3_2_optimizer_scheduler),
        ("Level 3.3 - AMP Training", test_3_3_amp_training),
        ("Level 3.4 - Early Stopping", test_3_4_early_stopping),
        ("Level 3.5 - Gradient Clipping", test_3_5_gradient_clipping),
        ("Level 3.6 - Cross-Validation", test_3_6_cross_validation),
    ]
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    module_results = []
    
    overall_start = time.time()
    
    for module_name, test_module in test_modules:
        print(f"\n{'='*70}")
        print(f"📝 Running: {module_name}")
        print(f"{'='*70}")
        
        module_start = time.time()
        
        try:
            # Executar testes do módulo
            if hasattr(test_module, 'run_all_tests'):
                success = test_module.run_all_tests()
            elif hasattr(test_module, 'main'):
                exit_code = test_module.main()
                success = (exit_code == 0)
            else:
                raise AttributeError(f"Module {module_name} has no run_all_tests() or main() function")
            
            module_time = time.time() - module_start
            
            # Obter estatísticas (assumindo 7 testes por módulo)
            module_tests = 7
            if success:
                module_passed = module_tests
                module_failed = 0
            else:
                # Tentar extrair do output (simplificado)
                module_passed = 0
                module_failed = module_tests
            
            total_tests += module_tests
            total_passed += module_passed
            total_failed += module_failed
            
            status = "✅ PASSED" if success else "❌ FAILED"
            module_results.append({
                "name": module_name,
                "status": status,
                "passed": module_passed,
                "failed": module_failed,
                "time": module_time
            })
            
            print(f"\n{status} - {module_name}")
            print(f"Time: {module_time:.2f}s")
            
        except Exception as e:
            module_time = time.time() - module_start
            print(f"\n❌ ERROR in {module_name}: {str(e)}")
            
            module_results.append({
                "name": module_name,
                "status": "❌ ERROR",
                "passed": 0,
                "failed": 7,
                "time": module_time
            })
            
            total_tests += 7
            total_failed += 7
    
    overall_time = time.time() - overall_start
    
    # Sumário final
    print("\n" + "="*70)
    print("📊 LEVEL 3 FINAL SUMMARY")
    print("="*70)
    print(f"\n{'Module':<45} {'Status':<15} {'Time'}")
    print("-"*70)
    
    for result in module_results:
        print(f"{result['name']:<45} {result['status']:<15} {result['time']:.2f}s")
    
    print("-"*70)
    print(f"\n📈 Overall Statistics:")
    print(f"   Total modules: {len(test_modules)}")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {total_passed}")
    print(f"   Failed: {total_failed}")
    print(f"   Success rate: {(total_passed/total_tests*100):.1f}%")
    print(f"   Total time: {overall_time:.2f}s")
    print(f"   Average time per module: {(overall_time/len(test_modules)):.2f}s")
    
    if total_failed == 0:
        print("\n🎉 " + "="*66)
        print("🎉 ALL LEVEL 3 TESTS PASSED!")
        print("🎉 " + "="*66)
        print("\n✨ Training & Evaluation module fully validated")
        print("✅ Ready to proceed to Level 4 (Integration Testing)")
    else:
        print("\n❌ " + "="*66)
        print(f"❌ {total_failed} TESTS FAILED")
        print("❌ " + "="*66)
        print("\n⚠️  Please review failed tests before proceeding")
    
    print("="*70)
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_level_3()
    sys.exit(0 if success else 1)
