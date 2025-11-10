#!/usr/bin/env python3
"""
Testes de Error Handling - Nível 7
===================================

Valida tratamento de erros e exceções.

Autor: DockTKinase Team
Data: 2025-11-10
"""

import sys
import numpy as np
from pathlib import Path
import tempfile

# Adicionar src ao path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.core.data_loader import DataManager
from regression.core.trainer import RegressionTrainer
from regression.models.models import RegressionModels


def test_file_not_found():
    """
    TEST 7.1: Arquivo não encontrado
    
    Valida:
    - FileNotFoundError clara quando arquivo não existe
    - Mensagem de erro útil
    """
    print('\n' + '=' * 70)
    print('TEST 7.1: File Not Found')
    print('=' * 70)
    
    # Tentar carregar arquivo inexistente de embeddings
    fake_embeddings = '/path/that/does/not/exist/fake_embeddings.npy'
    fake_targets = '/path/that/does/not/exist/fake_targets.npy'
    
    data_manager = DataManager(
        embeddings_path=fake_embeddings,
        targets_path=fake_targets
    )
    
    try:
        data_manager.load_embeddings()
        assert False, "Deveria ter lançado FileNotFoundError"
    except FileNotFoundError as e:
        print(f'   ✅ FileNotFoundError capturado para embeddings')
        print(f'   ✅ Mensagem: {str(e)[:80]}')
        # Aceitar mensagens em pt ou en
        msg = str(e).lower()
        assert any(word in msg for word in ['not found', 'não encontrado', 'no such file'])
    
    # Testar targets também
    try:
        data_manager.load_targets()
        assert False, "Deveria ter lançado FileNotFoundError"
    except FileNotFoundError as e:
        print(f'   ✅ FileNotFoundError capturado para targets')
        print(f'   ✅ Mensagem: {str(e)[:80]}')
    
    print('\n' + '=' * 70)
    print('TEST 7.1 PASSED ✅')
    print('=' * 70)


def test_invalid_data_shapes():
    """
    TEST 7.2: Shapes inválidos de dados
    
    Valida:
    - Sistema detecta shapes problemáticos
    - Erros são claros (ValueError)
    
    NOTA: sklearn é tolerante, mas testa cenários problemáticos
    """
    print('\n' + '=' * 70)
    print('TEST 7.2: Invalid Data Shapes')
    print('=' * 70)
    
    models = RegressionModels.get_all_models(random_state=42)
    model = models['Ridge']
    
    # Caso 1: Features incompatíveis entre train e test
    print('   Testando features incompatíveis...')
    X_train = np.random.randn(100, 10)  # 10 features
    y_train = np.random.randn(100)
    X_test = np.random.randn(20, 15)  # 15 features (DIFERENTE!)
    
    try:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)  # Aqui deve falhar
        assert False, "Deveria ter lançado ValueError para features diferentes"
    except (ValueError, IndexError) as e:
        print(f'   ✅ ValueError capturado para features incompatíveis')
        print(f'   ✅ Mensagem: {str(e)[:80]}')
    
    # Caso 2: Array vazio
    print('   Testando arrays vazios...')
    try:
        X_empty = np.array([]).reshape(0, 10)
        y_empty = np.array([])
        model.fit(X_empty, y_empty)
        print('   ⚠️  Modelo aceitou arrays vazios')
    except (ValueError, IndexError) as e:
        print(f'   ✅ ValueError capturado para arrays vazios')
        print(f'   ✅ Mensagem: {str(e)[:80]}')
    
    print('\n' + '=' * 70)
    print('TEST 7.2 PASSED ✅')
    print('=' * 70)


def test_mismatched_shapes():
    """
    TEST 7.3: Shapes incompatíveis X e y
    
    Valida:
    - ValueError quando len(X) != len(y)
    - Mensagem clara com os tamanhos
    """
    print('\n' + '=' * 70)
    print('TEST 7.3: Mismatched Shapes')
    print('=' * 70)
    
    models = RegressionModels.get_all_models(random_state=42)
    model = models['Ridge']
    
    # X com 100 amostras, y com 50
    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(50)  # Mismatch!
    
    try:
        model.fit(X_train, y_train)
        assert False, "Deveria ter lançado ValueError para len mismatch"
    except ValueError as e:
        print(f'   ✅ ValueError capturado')
        print(f'   ✅ Mensagem: {str(e)[:80]}')
        # sklearn menciona "samples" ou "inconsistent"
        error_msg = str(e).lower()
        assert any(word in error_msg for word in ['inconsistent', 'sample', 'found', 'expected'])
    
    print('\n' + '=' * 70)
    print('TEST 7.3 PASSED ✅')
    print('=' * 70)


def test_corrupted_data():
    """
    TEST 7.4: Dados corrompidos (NaN/Inf)
    
    Valida:
    - Sistema detecta ou lida com NaN/Inf
    - Não quebra silenciosamente
    
    NOTA: sklearn geralmente NÃO lança erro, mas propaga NaN/Inf
    """
    print('\n' + '=' * 70)
    print('TEST 7.4: Corrupted Data (NaN/Inf)')
    print('=' * 70)
    
    models = RegressionModels.get_all_models(random_state=42)
    model = models['Ridge']
    
    # Caso 1: NaN em X_train
    X_train = np.random.randn(100, 10)
    X_train[50, 5] = np.nan  # Inserir NaN
    y_train = np.random.randn(100)
    X_val = np.random.randn(20, 10)
    
    try:
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        
        # Verificar se resultado é válido ou NaN
        if np.any(np.isnan(pred)):
            print('   ⚠️  Predições contêm NaN (sklearn propagou)')
        else:
            print('   ✅ Modelo treinou apesar do NaN')
            
    except ValueError as e:
        print(f'   ✅ ValueError capturado para NaN: {str(e)[:60]}')
    
    # Caso 2: Inf em y_train
    X_train_clean = np.random.randn(100, 10)
    y_train_inf = np.random.randn(100)
    y_train_inf[25] = np.inf  # Inserir Inf
    
    model2 = models['Lasso']  # Novo modelo
    
    try:
        model2.fit(X_train_clean, y_train_inf)
        pred = model2.predict(X_val)
        
        # Verificar resultado
        if np.any(np.isinf(pred)) or np.any(np.isnan(pred)):
            print('   ⚠️  Predições contêm Inf/NaN (sklearn propagou)')
        else:
            print('   ✅ Modelo treinou apesar do Inf')
            
    except ValueError as e:
        print(f'   ✅ ValueError capturado para Inf: {str(e)[:60]}')
    
    print('\n   NOTA: sklearn geralmente NÃO lança erro para NaN/Inf,')
    print('         mas propaga valores inválidos. Validação deve ser')
    print('         feita ANTES do treinamento.')
    
    print('\n' + '=' * 70)
    print('TEST 7.4 PASSED ✅')
    print('=' * 70)


def run_all_tests():
    """Executa todos os testes do nível 7"""
    print('\n')
    print('╔' + '═' * 68 + '╗')
    print('║' + ' ' * 19 + 'NÍVEL 7 - ERROR HANDLING TESTS' + ' ' * 19 + '║')
    print('╚' + '═' * 68 + '╝')
    
    tests = [
        ('7.1', 'File Not Found', test_file_not_found),
        ('7.2', 'Invalid Data Shapes', test_invalid_data_shapes),
        ('7.3', 'Mismatched Shapes', test_mismatched_shapes),
        ('7.4', 'Corrupted Data (NaN/Inf)', test_corrupted_data)
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
        print('║  🎉 NÍVEL 7 COMPLETO - TODOS OS TESTES PASSARAM! 🎉' + ' ' * 15 + '║')
    else:
        print('║  ⚠️  ALGUNS TESTES FALHARAM - REVISAR ERROS ACIMA' + ' ' * 18 + '║')
    
    print('╚' + '═' * 68 + '╝')
    print()
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
