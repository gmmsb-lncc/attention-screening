#!/usr/bin/env python3
"""
Testes de Serialização - Nível 6
=================================

Valida save/load de modelos com joblib.

Autor: DockTKinase Team
Data: 2025-11-10
"""

import sys
import numpy as np
import joblib
from pathlib import Path
import tempfile
import shutil

# Adicionar src ao path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.models.models import RegressionModels
from regression.core.trainer import RegressionTrainer


def test_save_load_single_model():
    """
    TEST 6.1: Save/Load modelo único
    
    Valida:
    - joblib.dump() salva modelo
    - joblib.load() recupera modelo
    - Predições idênticas após load
    """
    print('\n' + '=' * 70)
    print('TEST 6.1: Save/Load Single Model')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Criar dados sintéticos
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randn(100) * 100 + 200
        X_test = np.random.randn(20, 20)
        
        # Treinar modelo
        models = RegressionModels.get_all_models(random_state=42)
        model = models['Ridge']
        model.fit(X_train, y_train)
        
        # Predição antes de salvar
        pred_before = model.predict(X_test)
        
        # Salvar modelo
        model_path = Path(temp_dir) / 'ridge_model.pkl'
        joblib.dump(model, model_path)
        
        assert model_path.exists()
        
        # Carregar modelo
        loaded_model = joblib.load(model_path)
        
        # Predição depois de carregar
        pred_after = loaded_model.predict(X_test)
        
        # Verificar predições idênticas
        assert np.allclose(pred_before, pred_after)
        
        print(f'✅ Modelo salvo em: {model_path}')
        print(f'✅ Modelo carregado com sucesso')
        print(f'✅ Predições idênticas (diff máximo: {np.max(np.abs(pred_before - pred_after)):.10f})')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 6.1 PASSED ✅')
    print('=' * 70)


def test_save_load_multiple_models():
    """
    TEST 6.2: Save/Load múltiplos modelos
    
    Valida:
    - Loop para salvar 3 modelos
    - Loop para carregar 3 modelos
    - Todos recuperados corretamente
    """
    print('\n' + '=' * 70)
    print('TEST 6.2: Save/Load Multiple Models')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Criar dados sintéticos
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randn(100) * 100 + 200
        X_test = np.random.randn(20, 20)
        
        # Treinar 3 modelos
        all_models = RegressionModels.get_all_models(random_state=42)
        models_to_save = {
            'Ridge': all_models['Ridge'],
            'Lasso': all_models['Lasso'],
            'RandomForest': all_models['RandomForest']
        }
        
        predictions_before = {}
        
        # Treinar e salvar cada modelo
        for name, model in models_to_save.items():
            model.fit(X_train, y_train)
            predictions_before[name] = model.predict(X_test)
            
            # Salvar
            model_path = Path(temp_dir) / f'{name}.pkl'
            joblib.dump(model, model_path)
            
            print(f'   ✅ {name} salvo')
        
        # Carregar todos os modelos
        loaded_models = {}
        for name in models_to_save.keys():
            model_path = Path(temp_dir) / f'{name}.pkl'
            loaded_models[name] = joblib.load(model_path)
            print(f'   ✅ {name} carregado')
        
        # Verificar predições
        for name in models_to_save.keys():
            pred_after = loaded_models[name].predict(X_test)
            assert np.allclose(predictions_before[name], pred_after)
            print(f'   ✅ {name} predições idênticas')
        
        print(f'\n✅ {len(models_to_save)} modelos salvos e carregados')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 6.2 PASSED ✅')
    print('=' * 70)


def test_save_load_pipeline_state():
    """
    TEST 6.3: Save/Load estado completo do pipeline
    
    Valida:
    - Salvar trained_models dict completo
    - Carregar e usar modelos
    - Métricas são consistentes
    """
    print('\n' + '=' * 70)
    print('TEST 6.3: Save/Load Pipeline State')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Criar dados
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randn(100) * 100 + 200
        X_val = np.random.randn(20, 20)
        y_val = np.random.randn(20) * 100 + 200
        
        # Treinar usando Trainer
        all_models = RegressionModels.get_all_models(random_state=42)
        models = {k: v for k, v in list(all_models.items())[:3]}  # Primeiros 3
        
        trainer = RegressionTrainer(models_dict=models, verbose=False)
        trainer.train_all(X_train, y_train, X_val, y_val)
        
        # Salvar estado completo
        pipeline_state = {
            'trained_models': trainer.trained_models,
            'val_results': trainer.val_results,
            'training_times': trainer.training_times
        }
        
        state_path = Path(temp_dir) / 'pipeline_state.pkl'
        joblib.dump(pipeline_state, state_path)
        
        assert state_path.exists()
        print('   ✅ Estado do pipeline salvo')
        
        # Carregar estado
        loaded_state = joblib.load(state_path)
        
        assert 'trained_models' in loaded_state
        assert 'val_results' in loaded_state
        assert 'training_times' in loaded_state
        
        print('   ✅ Estado do pipeline carregado')
        
        # Verificar que modelos funcionam
        for name, model in loaded_state['trained_models'].items():
            pred = model.predict(X_val)
            assert len(pred) == len(y_val)
            print(f'   ✅ {name} funciona após load')
        
        # Verificar métricas
        assert len(loaded_state['val_results']) == 3
        print(f'   ✅ Métricas preservadas: {len(loaded_state["val_results"])} modelos')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 6.3 PASSED ✅')
    print('=' * 70)


def test_model_compatibility():
    """
    TEST 6.4: Compatibilidade entre sessões
    
    Valida:
    - Modelo salvo em uma sessão
    - Carregado em "outra sessão" (sem referências)
    - Funciona sem quebrar
    """
    print('\n' + '=' * 70)
    print('TEST 6.4: Model Compatibility')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Sessão 1: Treinar e salvar
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randn(100) * 100 + 200
        X_test = np.random.randn(20, 20)
        
        models = RegressionModels.get_all_models(random_state=42)
        model = models['Ridge']
        model.fit(X_train, y_train)
        
        model_path = Path(temp_dir) / 'model_for_compatibility.pkl'
        joblib.dump(model, model_path)
        
        # Limpar referência (simular nova sessão)
        del model
        del models
        
        print('   ✅ Sessão 1: Modelo treinado e salvo')
        
        # Sessão 2: Carregar e usar (sem referências anteriores)
        loaded_model = joblib.load(model_path)
        
        # Verificar que é um estimador sklearn válido
        assert hasattr(loaded_model, 'predict')
        assert hasattr(loaded_model, 'score')
        
        # Fazer predições
        predictions = loaded_model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
        
        # Calcular score
        score = loaded_model.score(X_test, np.random.randn(20) * 100 + 200)
        assert isinstance(score, (int, float))
        
        print('   ✅ Sessão 2: Modelo carregado sem referências')
        print(f'   ✅ Predições válidas: {len(predictions)} amostras')
        print(f'   ✅ Score calculado: {score:.4f}')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 6.4 PASSED ✅')
    print('=' * 70)


def run_all_tests():
    """Executa todos os testes do nível 6"""
    print('\n')
    print('╔' + '═' * 68 + '╗')
    print('║' + ' ' * 20 + 'NÍVEL 6 - SERIALIZATION TESTS' + ' ' * 19 + '║')
    print('╚' + '═' * 68 + '╝')
    
    tests = [
        ('6.1', 'Save/Load Single Model', test_save_load_single_model),
        ('6.2', 'Save/Load Multiple Models', test_save_load_multiple_models),
        ('6.3', 'Save/Load Pipeline State', test_save_load_pipeline_state),
        ('6.4', 'Model Compatibility', test_model_compatibility)
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
        print('║  🎉 NÍVEL 6 COMPLETO - TODOS OS TESTES PASSARAM! 🎉' + ' ' * 15 + '║')
    else:
        print('║  ⚠️  ALGUNS TESTES FALHARAM - REVISAR ERROS ACIMA' + ' ' * 18 + '║')
    
    print('╚' + '═' * 68 + '╝')
    print()
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
