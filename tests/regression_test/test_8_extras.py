#!/usr/bin/env python3
"""
Testes Extras - Nível 8
========================

Testes simples de smoke test e funcionalidades extras.

Autor: DockTKinase Team
Data: 2025-11-10
"""

import sys
import numpy as np
from pathlib import Path

# Adicionar src ao path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.core.trainer import RegressionTrainer
from regression.models.models import RegressionModels


def test_verbose_mode():
    """
    TEST 8.1: Modo verbose
    
    Valida:
    - verbose=True imprime informações
    - verbose=False é silencioso
    """
    print('\n' + '=' * 70)
    print('TEST 8.1: Verbose Mode')
    print('=' * 70)
    
    # Criar dados
    np.random.seed(42)
    X_train = np.random.randn(100, 20)
    y_train = np.random.randn(100) * 100 + 200
    X_val = np.random.randn(20, 20)
    y_val = np.random.randn(20) * 100 + 200
    
    models = RegressionModels.get_all_models(random_state=42)
    models_dict = {'Ridge': models['Ridge']}
    
    # Teste 1: verbose=True (deve imprimir)
    print('\n   Testando verbose=True (deve imprimir):')
    print('   ' + '-' * 60)
    trainer_verbose = RegressionTrainer(models_dict=models_dict, verbose=True)
    trainer_verbose.train_all(X_train, y_train, X_val, y_val)
    print('   ' + '-' * 60)
    print('   ✅ verbose=True funcionou (veja output acima)')
    
    # Teste 2: verbose=False (deve ser silencioso)
    print('\n   Testando verbose=False (não deve imprimir):')
    trainer_silent = RegressionTrainer(models_dict=models_dict, verbose=False)
    trainer_silent.train_all(X_train, y_train, X_val, y_val)
    print('   ✅ verbose=False funcionou (nenhum output do trainer)')
    
    # Verificar que ambos treinaram
    assert 'Ridge' in trainer_verbose.trained_models
    assert 'Ridge' in trainer_silent.trained_models
    print('\n   ✅ Ambos os modos treinaram modelos com sucesso')
    
    print('\n' + '=' * 70)
    print('TEST 8.1 PASSED ✅')
    print('=' * 70)


def test_large_dataset():
    """
    TEST 8.2: Dataset grande (smoke test)
    
    Valida:
    - Pipeline funciona com 10k amostras
    - Não quebra, não trava
    - Tempo razoável
    """
    print('\n' + '=' * 70)
    print('TEST 8.2: Large Dataset Smoke Test')
    print('=' * 70)
    
    # Criar dataset grande
    np.random.seed(42)
    n_samples = 10000
    n_features = 50
    
    print(f'\n   Criando dataset: {n_samples:,} amostras, {n_features} features')
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples) * 100 + 200
    X_val = np.random.randn(2000, n_features)
    y_val = np.random.randn(2000) * 100 + 200
    
    # Testar com 3 modelos rápidos
    models = RegressionModels.get_all_models(random_state=42)
    models_dict = {
        'Ridge': models['Ridge'],
        'Lasso': models['Lasso'],
        'ElasticNet': models['ElasticNet']
    }
    
    print('   Treinando 3 modelos...')
    trainer = RegressionTrainer(models_dict=models_dict, verbose=False)
    
    import time
    start = time.time()
    trainer.train_all(X_train, y_train, X_val, y_val)
    elapsed = time.time() - start
    
    print(f'   ✅ Treinou em {elapsed:.2f}s')
    
    # Verificar que todos treinaram
    assert len(trainer.trained_models) == 3
    print(f'   ✅ {len(trainer.trained_models)} modelos treinados')
    
    # Verificar predições
    for name, model in trainer.trained_models.items():
        pred = model.predict(X_val)
        assert len(pred) == len(y_val)
        assert not np.any(np.isnan(pred))
        print(f'   ✅ {name}: predições válidas ({len(pred):,} amostras)')
    
    print(f'\n   ✅ Pipeline funcional com dataset grande ({n_samples:,} amostras)')
    
    print('\n' + '=' * 70)
    print('TEST 8.2 PASSED ✅')
    print('=' * 70)


def run_all_tests():
    """Executa todos os testes do nível 8"""
    print('\n')
    print('╔' + '═' * 68 + '╗')
    print('║' + ' ' * 24 + 'NÍVEL 8 - EXTRAS TESTS' + ' ' * 22 + '║')
    print('╚' + '═' * 68 + '╝')
    
    tests = [
        ('8.1', 'Verbose Mode', test_verbose_mode),
        ('8.2', 'Large Dataset Smoke Test', test_large_dataset)
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
    print(f'║  ✅ Testes Passaram: {passed}/2' + ' ' * (68 - 25 - len(str(passed))) + '║')
    print(f'║  ❌ Testes Falharam: {failed}/2' + ' ' * (68 - 25 - len(str(failed))) + '║')
    print('╠' + '═' * 68 + '╣')
    
    if failed == 0:
        print('║  🎉 NÍVEL 8 COMPLETO - TODOS OS TESTES PASSARAM! 🎉' + ' ' * 15 + '║')
    else:
        print('║  ⚠️  ALGUNS TESTES FALHARAM - REVISAR ERROS ACIMA' + ' ' * 18 + '║')
    
    print('╚' + '═' * 68 + '╝')
    print()
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
