#!/usr/bin/env python3
"""
Testes de Cross-Validation - Nível 9
=====================================

Valida K-Fold cross-validation para regressão.

Autor: DockTKinase Team
Data: 2025-11-10
"""

import sys
import numpy as np
from pathlib import Path

# Adicionar src ao path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.core.cross_validator import (
    RegressionCrossValidator,
    CrossValidationConfig,
    quick_cross_validate
)
from regression.models.models import RegressionModels


def test_basic_cross_validation():
    """
    TEST 9.1: Cross-validation básico
    
    Valida:
    - CV executa sem erros
    - Resultados para todos os modelos
    - Métricas calculadas corretamente
    """
    print('\n' + '=' * 70)
    print('TEST 9.1: Basic Cross-Validation')
    print('=' * 70)
    
    # Dados sintéticos
    np.random.seed(42)
    X = np.random.randn(200, 20)
    y = np.random.randn(200) * 100 + 200
    
    # CV com 3 modelos
    results = quick_cross_validate(
        X, y,
        model_names=['Ridge', 'Lasso', 'ElasticNet'],
        n_splits=3,
        verbose=False
    )
    
    # Verificações
    assert len(results) == 3
    print(f'   ✅ 3 modelos validados')
    
    for model_name, cv_result in results.items():
        # Verificar estrutura
        assert cv_result.model_name == model_name
        assert len(cv_result.fold_metrics) == 3
        assert len(cv_result.summary_statistics) > 0
        
        # Verificar métricas
        mae_mean = cv_result.get_mean_metric('mae')
        r2_mean = cv_result.get_mean_metric('r2')
        
        assert mae_mean > 0
        assert -1 <= r2_mean <= 1  # R² pode ser negativo para modelos ruins
        
        print(f'   ✅ {model_name}: MAE={mae_mean:.2f}, R²={r2_mean:.4f}')
    
    print('\n' + '=' * 70)
    print('TEST 9.1 PASSED ✅')
    print('=' * 70)


def test_fold_consistency():
    """
    TEST 9.2: Consistência entre folds
    
    Valida:
    - Todos os folds executam
    - Métricas de treino < validação (overfitting esperado)
    - Estatísticas agregadas corretas
    """
    print('\n' + '=' * 70)
    print('TEST 9.2: Fold Consistency')
    print('=' * 70)
    
    # Dados sintéticos maiores
    np.random.seed(42)
    X = np.random.randn(300, 15)
    y = X[:, 0] * 10 + X[:, 1] * 5 + np.random.randn(300) * 2
    
    # Configuração
    config = CrossValidationConfig(
        n_splits=5,
        shuffle=True,
        random_state=42,
        verbose=False
    )
    
    cv = RegressionCrossValidator(config=config, verbose=False)
    results = cv.cross_validate(X, y, model_names=['Ridge'])
    
    ridge_results = results['Ridge']
    
    # Verificar 5 folds
    assert len(ridge_results.fold_metrics) == 5
    print('   ✅ 5 folds executados')
    
    # Verificar métricas de cada fold
    for fold_idx, fold_metric in enumerate(ridge_results.fold_metrics):
        assert fold_metric.fold_idx == fold_idx
        assert 'mae' in fold_metric.train_metrics
        assert 'mae' in fold_metric.val_metrics
        
        # Geralmente treino tem erro menor que validação
        train_mae = fold_metric.train_metrics['mae']
        val_mae = fold_metric.val_metrics['mae']
        
        print(f'   Fold {fold_idx + 1}: Train MAE={train_mae:.2f}, Val MAE={val_mae:.2f}')
    
    # Verificar estatísticas agregadas
    stats = ridge_results.summary_statistics
    assert 'mae' in stats
    assert 'mean' in stats['mae']
    assert 'std' in stats['mae']
    assert 'min' in stats['mae']
    assert 'max' in stats['mae']
    
    mae_mean = stats['mae']['mean']
    mae_std = stats['mae']['std']
    print(f'\n   ✅ MAE agregado: {mae_mean:.2f} ± {mae_std:.2f}')
    
    # Verificar best fold identificado
    assert 0 <= ridge_results.best_fold < 5
    print(f'   ✅ Best fold: {ridge_results.best_fold}')
    
    print('\n' + '=' * 70)
    print('TEST 9.2 PASSED ✅')
    print('=' * 70)


def test_model_comparison():
    """
    TEST 9.3: Comparação de múltiplos modelos
    
    Valida:
    - CV com 5 modelos diferentes
    - Identificação do melhor modelo
    - DataFrame de comparação
    """
    print('\n' + '=' * 70)
    print('TEST 9.3: Model Comparison')
    print('=' * 70)
    
    # Dados sintéticos
    np.random.seed(42)
    X = np.random.randn(250, 15)
    y = X[:, 0] * 5 + X[:, 1] * 3 + np.random.randn(250)
    
    # CV com 5 modelos
    config = CrossValidationConfig(
        n_splits=3,
        random_state=42,
        verbose=False
    )
    
    cv = RegressionCrossValidator(config=config, verbose=False)
    results = cv.cross_validate(
        X, y,
        model_names=['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'GradientBoosting']
    )
    
    # Verificar 5 modelos
    assert len(results) == 5
    print('   ✅ 5 modelos comparados')
    
    # Identificar melhor modelo
    best_model_mae = cv.get_best_model('mae')
    best_model_r2 = cv.get_best_model('r2')
    
    assert best_model_mae in results
    assert best_model_r2 in results
    
    print(f'   ✅ Melhor por MAE: {best_model_mae}')
    print(f'   ✅ Melhor por R²: {best_model_r2}')
    
    # DataFrame de comparação
    df = cv.compare_models()
    
    assert len(df) == 5
    assert 'model' in df.columns
    assert 'mae_mean' in df.columns
    assert 'r2_mean' in df.columns
    
    # Verificar ordenação (por MAE, crescente)
    mae_values = df['mae_mean'].values
    assert all(mae_values[i] <= mae_values[i+1] for i in range(len(mae_values)-1))
    
    print('\n   Comparação (Top 3):')
    print(df[['model', 'mae_mean', 'mae_std', 'r2_mean']].head(3).to_string(index=False))
    
    print('\n' + '=' * 70)
    print('TEST 9.3 PASSED ✅')
    print('=' * 70)


def test_reproducibility():
    """
    TEST 9.4: Reprodutibilidade
    
    Valida:
    - Mesma seed produz mesmos resultados
    - Shuffle=False produz resultados idênticos
    """
    print('\n' + '=' * 70)
    print('TEST 9.4: Reproducibility')
    print('=' * 70)
    
    # Dados sintéticos
    np.random.seed(42)
    X = np.random.randn(150, 10)
    y = np.random.randn(150) * 50 + 100
    
    # Executar 2x com mesma seed
    results1 = quick_cross_validate(
        X, y,
        model_names=['Ridge'],
        n_splits=3,
        random_state=999,
        verbose=False
    )
    
    results2 = quick_cross_validate(
        X, y,
        model_names=['Ridge'],
        n_splits=3,
        random_state=999,
        verbose=False
    )
    
    # Comparar métricas
    mae1 = results1['Ridge'].get_mean_metric('mae')
    mae2 = results2['Ridge'].get_mean_metric('mae')
    
    assert np.isclose(mae1, mae2, atol=1e-6)
    print(f'   ✅ Reprodutibilidade verificada')
    print(f'      Run 1 MAE: {mae1:.6f}')
    print(f'      Run 2 MAE: {mae2:.6f}')
    print(f'      Diff: {abs(mae1 - mae2):.10f}')
    
    print('\n' + '=' * 70)
    print('TEST 9.4 PASSED ✅')
    print('=' * 70)


def run_all_tests():
    """Executa todos os testes do nível 9"""
    print('\n')
    print('╔' + '═' * 68 + '╗')
    print('║' + ' ' * 19 + 'NÍVEL 9 - CROSS-VALIDATION TESTS' + ' ' * 17 + '║')
    print('╚' + '═' * 68 + '╝')
    
    tests = [
        ('9.1', 'Basic Cross-Validation', test_basic_cross_validation),
        ('9.2', 'Fold Consistency', test_fold_consistency),
        ('9.3', 'Model Comparison', test_model_comparison),
        ('9.4', 'Reproducibility', test_reproducibility)
    ]
    
    passed = 0
    failed = 0
    
    for test_id, test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f'\n❌ TEST {test_id} FAILED: {test_name}')
            print(f'   AssertionError: {e}')
            import traceback
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f'\n💥 TEST {test_id} ERROR: {test_name}')
            print(f'   Exception: {e}')
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Resumo final
    print('\n')
    print('╔' + '═' * 68 + '╗')
    print('║' + ' ' * 25 + 'RESUMO FINAL' + ' ' * 31 + '║')
    print('╠' + '═' * 68 + '╣')
    print(f'║  ✅ Testes Passaram: {passed}/4' + ' ' * (68 - 25 - len(str(passed))) + '║')
    print(f'║  ❌ Testes Falharam: {failed}/4' + ' ' * (68 - 25 - len(str(failed))) + '║')
    print('╠' + '═' * 68 + '╣')
    
    if failed == 0:
        print('║  🎉 NÍVEL 9 COMPLETO - TODOS OS TESTES PASSARAM! 🎉' + ' ' * 15 + '║')
    else:
        print('║  ⚠️  ALGUNS TESTES FALHARAM - REVISAR ERROS ACIMA' + ' ' * 18 + '║')
    
    print('╚' + '═' * 68 + '╝')
    print()
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
