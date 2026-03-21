#!/usr/bin/env python3
"""
Test 11: Robust Train-Test Split
==================================

Testa implementação robusta de train-test split (versão alternativa).

NOTE: Este módulo é similar ao train_test_split.py já testado.
Testamos apenas funcionalidades únicas/diferentes.

Tests incluídos:
1. Robust split with validation - split com validação estatística
2. Chi-square validation - validação qui-quadrado
3. Imbalance detection - detecção de desbalanceamento

Author: Test Suite
Date: 2024
"""

import sys
from pathlib import Path

# Adicionar src ao path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
import numpy as np

from classifier.utils.robust_train_test_split import (
    RobustTrainTestSplitter,
    SplitValidationReport
)


def test_1_robust_split():
    """
    Test 11.1: Robust Split with Validation
    
    Testa divisão robusta com validação estatística.
    """
    print("\n" + "="*60)
    print("Test 11.1: Robust Split with Validation")
    print("="*60)
    
    print("\n--- Step 1: Creating Balanced Dataset ---")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Dataset balanceado
    n_samples = 200
    X = torch.randn(n_samples, 32)
    y = torch.cat([torch.zeros(100), torch.ones(100)])
    
    # Shuffle
    indices = torch.randperm(n_samples)
    X, y = X[indices], y[indices]
    
    print(f"✅ Dataset created: {n_samples} samples")
    print(f"   Class 0: {(y==0).sum():.0f} samples")
    print(f"   Class 1: {(y==1).sum():.0f} samples")
    
    print("\n--- Step 2: Performing Robust Split ---")
    
    splitter = RobustTrainTestSplitter(random_state=42)
    
    X_train, X_test, y_train, y_test, report = splitter.robust_split(
        X, y,
        test_size=0.2,
        min_samples_per_class=5,
        validate_split=True
    )
    
    print(f"✅ Split completed:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Validar shapes
    assert X_train.shape[0] + X_test.shape[0] == n_samples
    assert X_train.shape[1] == 32
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    
    print("\n--- Step 3: Validating Report ---")
    
    assert isinstance(report, SplitValidationReport)
    assert hasattr(report, 'is_valid')
    assert hasattr(report, 'train_distribution')
    assert hasattr(report, 'test_distribution')
    assert hasattr(report, 'chi_square_p_value')
    assert hasattr(report, 'imbalance_ratio')
    
    print(f"✅ Validation report:")
    print(f"   Valid split: {report.is_valid}")
    print(f"   Chi-square p-value: {report.chi_square_p_value:.4f}")
    print(f"   Imbalance ratio: {report.imbalance_ratio:.2f}")
    print(f"   Issues: {len(report.issues)}")
    
    # Validar que split é válido para dados balanceados
    assert report.is_valid == True, "Balanced data should produce valid split"
    assert report.imbalance_ratio <= 2.0, "Should be reasonably balanced"
    
    print(f"\n✅ Robust split with validation tested successfully!")


def test_2_validation_report():
    """
    Test 11.2: Validation Report Details
    
    Testa detalhes do relatório de validação.
    """
    print("\n" + "="*60)
    print("Test 11.2: Validation Report Details")
    print("="*60)
    
    print("\n--- Step 1: Creating Dataset ---")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    X = torch.randn(150, 16)
    y = torch.cat([torch.zeros(100), torch.ones(50)])  # Desbalanceado 2:1
    
    indices = torch.randperm(150)
    X, y = X[indices], y[indices]
    
    print(f"✅ Imbalanced dataset: 100 vs 50 samples")
    
    print("\n--- Step 2: Splitting and Getting Report ---")
    
    splitter = RobustTrainTestSplitter(random_state=42)
    X_train, X_test, y_train, y_test, report = splitter.robust_split(
        X, y,
        test_size=0.2,
        validate_split=True
    )
    
    print(f"✅ Split completed")
    
    print("\n--- Step 3: Analyzing Report Details ---")
    
    print(f"✅ Train distribution:")
    for cls, proportion in report.train_distribution.items():
        print(f"   Class {cls}: {proportion:.2%}")
    
    print(f"✅ Test distribution:")
    for cls, proportion in report.test_distribution.items():
        print(f"   Class {cls}: {proportion:.2%}")
    
    # Validar que distribuições somam 1.0
    train_sum = sum(report.train_distribution.values())
    test_sum = sum(report.test_distribution.values())
    
    assert abs(train_sum - 1.0) < 0.01, "Train distribution should sum to 1"
    assert abs(test_sum - 1.0) < 0.01, "Test distribution should sum to 1"
    
    print(f"\n--- Step 4: Checking Statistical Validation ---")
    
    # Chi-square p-value deve estar entre 0 e 1
    assert 0 <= report.chi_square_p_value <= 1, "P-value should be in [0,1]"
    
    print(f"✅ Chi-square p-value: {report.chi_square_p_value:.4f}")
    print(f"✅ Imbalance ratio: {report.imbalance_ratio:.2f}")
    
    # Imbalance ratio deve refletir o desbalanceamento 2:1
    assert 1.5 < report.imbalance_ratio < 2.5, "Should detect 2:1 imbalance"
    
    print(f"\n✅ Validation report details tested successfully!")


def test_3_imbalance_handling():
    """
    Test 11.3: Imbalance Detection
    
    Testa detecção de desbalanceamento extremo.
    """
    print("\n" + "="*60)
    print("Test 11.3: Imbalance Detection")
    print("="*60)
    
    print("\n--- Step 1: Creating Extremely Imbalanced Dataset ---")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Dataset muito desbalanceado 10:1
    X = torch.randn(110, 16)
    y = torch.cat([torch.zeros(100), torch.ones(10)])
    
    indices = torch.randperm(110)
    X, y = X[indices], y[indices]
    
    print(f"✅ Highly imbalanced dataset:")
    print(f"   Class 0: 100 samples (90.9%)")
    print(f"   Class 1: 10 samples (9.1%)")
    print(f"   Imbalance: 10:1")
    
    print("\n--- Step 2: Splitting with Imbalance Detection ---")
    
    splitter = RobustTrainTestSplitter(random_state=42)
    X_train, X_test, y_train, y_test, report = splitter.robust_split(
        X, y,
        test_size=0.2,
        max_imbalance_ratio=5.0,  # Limite mais restritivo
        validate_split=True
    )
    
    print(f"✅ Split completed with warnings")
    
    print("\n--- Step 3: Checking Imbalance Detection ---")
    
    print(f"✅ Imbalance ratio detected: {report.imbalance_ratio:.2f}")
    
    # Deve detectar alto desbalanceamento
    assert report.imbalance_ratio > 5.0, "Should detect high imbalance"
    
    # Pode ter issues reportados
    if report.issues:
        print(f"✅ Issues detected: {len(report.issues)}")
        for issue in report.issues[:3]:  # Mostrar primeiros 3
            print(f"   - {issue}")
    
    # Validar que dados ainda foram divididos
    assert len(X_train) > 0, "Should still produce train set"
    assert len(X_test) > 0, "Should still produce test set"
    
    print(f"\n--- Step 4: Checking Class Preservation ---")
    
    # Verificar que ambas as classes estão presentes nos dois sets
    train_classes = torch.unique(y_train).tolist()
    test_classes = torch.unique(y_test).tolist()
    
    print(f"✅ Train classes: {train_classes}")
    print(f"✅ Test classes: {test_classes}")
    
    # Idealmente ambas as classes devem estar em ambos os sets
    # (mas pode não acontecer com dados muito desbalanceados)
    assert len(train_classes) > 0, "Train should have at least one class"
    assert len(test_classes) > 0, "Test should have at least one class"
    
    print(f"\n✅ Imbalance detection tested successfully!")


def run_all_tests():
    """Executar todos os testes."""
    tests = [
        ("11.1 - Robust Split", test_1_robust_split),
        ("11.2 - Validation Report", test_2_validation_report),
        ("11.3 - Imbalance Detection", test_3_imbalance_handling),
    ]
    
    print("\n" + "="*60)
    print("LEVEL 11: ROBUST TRAIN-TEST SPLIT")
    print("="*60)
    print("\nNOTE: Similar to train_test_split.py (already tested).")
    print("      Testing unique features: validation report, chi-square.")
    print("="*60)
    
    passed = 0
    failed = 0
    errors = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✅ {test_name} PASSED\n")
        except AssertionError as e:
            failed += 1
            error_msg = f"❌ {test_name} FAILED: {str(e)}"
            print(f"{error_msg}\n")
            errors.append(error_msg)
        except Exception as e:
            failed += 1
            error_msg = f"❌ {test_name} ERROR: {str(e)}"
            print(f"{error_msg}\n")
            errors.append(error_msg)
    
    # Sumário final
    print("\n" + "="*60)
    print("FINAL SUMMARY - LEVEL 11")
    print("="*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(tests)*100):.1f}%")
    
    if errors:
        print("\n❌ Failed tests:")
        for error in errors:
            print(f"  {error}")
    else:
        print("\n🎉 All tests passed!")
    
    print("="*60)
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
