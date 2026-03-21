#!/usr/bin/env python3
"""
Testes do RegressionEvaluator - Nível 4.1
==========================================

Valida cálculo de métricas, comparação de modelos e formatação de resultados.

Autor: DockTKinase Team
Data: 2025-11-10
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Adicionar src ao path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.core.evaluator import RegressionEvaluator


def test_evaluator_initialization():
    """
    TEST 4.1.1: Inicialização e métodos estáticos
    
    Valida:
    - RegressionEvaluator tem métodos estáticos
    - Não requer instanciação
    - Todos os métodos principais acessíveis
    """
    print('\n' + '=' * 70)
    print('TEST 4.1.1: Evaluator Initialization')
    print('=' * 70)
    
    # Verificar que é uma classe com métodos estáticos
    assert hasattr(RegressionEvaluator, 'calculate_metrics')
    assert hasattr(RegressionEvaluator, 'compare_models')
    assert hasattr(RegressionEvaluator, 'get_best_model')
    assert hasattr(RegressionEvaluator, 'save_predictions_csv')
    assert hasattr(RegressionEvaluator, 'print_metrics_summary')
    
    # Verificar que são métodos estáticos (podem ser chamados sem instanciar)
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.1])
    
    # Chamar diretamente sem instanciar
    metrics = RegressionEvaluator.calculate_metrics(y_true, y_pred)
    
    assert isinstance(metrics, dict)
    assert 'MAE' in metrics
    assert 'RMSE' in metrics
    assert 'R2' in metrics
    
    print('✅ Métodos estáticos acessíveis')
    print('✅ calculate_metrics funcional')
    print('✅ Retorna dict com métricas principais')
    
    print('\n' + '=' * 70)
    print('TEST 4.1.1 PASSED ✅')
    print('=' * 70)


def test_metrics_calculation():
    """
    TEST 4.1.2: Cálculo detalhado de métricas
    
    Valida:
    - Métricas principais (MAE, RMSE, R2, MedianAE, MAPE)
    - Estatísticas de resíduos (mean, std, max)
    - Percentis de erro (p25, p50, p75, p90, p95, p99)
    - Estrutura completa do dict (21+ campos)
    """
    print('\n' + '=' * 70)
    print('TEST 4.1.2: Metrics Calculation')
    print('=' * 70)
    
    # Dados de teste com padrão conhecido
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 320, 380, 510])
    
    metrics = RegressionEvaluator.calculate_metrics(
        y_true, y_pred, 
        model_name='TestModel'
    )
    
    # Verificar estrutura completa
    expected_keys = [
        'model_name', 'n_samples',
        'MAE', 'MSE', 'RMSE', 'R2', 'MedianAE', 'MAPE',
        'mean_residual', 'std_residual', 'max_error',
        'error_p25', 'error_p50', 'error_p75', 
        'error_p90', 'error_p95', 'error_p99'
    ]
    
    for key in expected_keys:
        assert key in metrics, f'Métrica {key} não encontrada'
    
    # Verificar valores específicos
    assert metrics['model_name'] == 'TestModel'
    assert metrics['n_samples'] == 5
    
    # Verificar MAE manualmente
    # Erros: |10|, |10|, |20|, |20|, |10| → MAE = 14.0
    expected_mae = 14.0
    assert abs(metrics['MAE'] - expected_mae) < 0.01
    
    # Verificar que RMSE > MAE (sempre verdadeiro)
    assert metrics['RMSE'] > metrics['MAE']
    
    # Verificar R2 está no intervalo razoável
    assert 0.5 <= metrics['R2'] <= 1.0
    
    # Verificar percentis estão ordenados
    assert metrics['error_p25'] <= metrics['error_p50']
    assert metrics['error_p50'] <= metrics['error_p75']
    assert metrics['error_p75'] <= metrics['error_p90']
    assert metrics['error_p90'] <= metrics['error_p95']
    assert metrics['error_p95'] <= metrics['error_p99']
    
    print(f'✅ Estrutura: {len(metrics)} campos validados')
    print(f'✅ MAE: {metrics["MAE"]:.2f} (esperado: {expected_mae:.2f})')
    print(f'✅ RMSE: {metrics["RMSE"]:.2f} > MAE')
    print(f'✅ R²: {metrics["R2"]:.4f}')
    print(f'✅ Percentis ordenados corretamente')
    
    print('\n' + '=' * 70)
    print('TEST 4.1.2 PASSED ✅')
    print('=' * 70)


def test_compare_models():
    """
    TEST 4.1.3: Comparação entre múltiplos modelos
    
    Valida:
    - compare_models retorna DataFrame ordenado
    - Ranking por MAE (ascending=True)
    - Ranking por R2 (ascending=False)
    - Coluna 'rank' adicionada corretamente
    """
    print('\n' + '=' * 70)
    print('TEST 4.1.3: Compare Models')
    print('=' * 70)
    
    # Criar métricas para 3 modelos com performance diferente
    y_true = np.linspace(100, 500, 50)
    
    # Modelo A: Melhor (MAE baixo)
    y_pred_a = y_true + np.random.normal(0, 5, 50)
    metrics_a = RegressionEvaluator.calculate_metrics(y_true, y_pred_a, 'ModelA')
    
    # Modelo B: Médio
    y_pred_b = y_true + np.random.normal(0, 15, 50)
    metrics_b = RegressionEvaluator.calculate_metrics(y_true, y_pred_b, 'ModelB')
    
    # Modelo C: Pior (MAE alto)
    y_pred_c = y_true + np.random.normal(0, 30, 50)
    metrics_c = RegressionEvaluator.calculate_metrics(y_true, y_pred_c, 'ModelC')
    
    results = {
        'ModelA': metrics_a,
        'ModelB': metrics_b,
        'ModelC': metrics_c
    }
    
    # Comparar por MAE (menor melhor)
    df_mae = RegressionEvaluator.compare_models(
        results, 
        metric='MAE', 
        ascending=True
    )
    
    # Verificar estrutura do DataFrame
    assert isinstance(df_mae, pd.DataFrame)
    assert 'rank' in df_mae.columns
    assert len(df_mae) == 3
    
    # Verificar ordenação por MAE
    mae_values = df_mae['MAE'].values
    assert all(mae_values[i] <= mae_values[i+1] for i in range(len(mae_values)-1))
    
    # Verificar rankings
    assert df_mae['rank'].tolist() == [1, 2, 3]
    
    # Comparar por R2 (maior melhor)
    df_r2 = RegressionEvaluator.compare_models(
        results, 
        metric='R2', 
        ascending=False
    )
    
    # Verificar ordenação por R2 (decrescente)
    r2_values = df_r2['R2'].values
    assert all(r2_values[i] >= r2_values[i+1] for i in range(len(r2_values)-1))
    
    print('✅ DataFrame retornado corretamente')
    print(f'✅ MAE ordenado: {mae_values}')
    print(f'✅ R² ordenado (desc): {r2_values}')
    print(f'✅ Rankings: {df_mae["rank"].tolist()}')
    
    print('\n' + '=' * 70)
    print('TEST 4.1.3 PASSED ✅')
    print('=' * 70)


def test_best_model_selection():
    """
    TEST 4.1.4: Seleção do melhor modelo
    
    Valida:
    - get_best_model retorna nome correto
    - Seleção por MAE (menor melhor)
    - Seleção por R2 (maior melhor)
    - Seleção por RMSE (menor melhor)
    """
    print('\n' + '=' * 70)
    print('TEST 4.1.4: Best Model Selection')
    print('=' * 70)
    
    # Criar métricas artificiais para controle total
    results = {
        'Ridge': {
            'MAE': 10.0,
            'RMSE': 15.0,
            'R2': 0.95,
            'model_name': 'Ridge'
        },
        'RandomForest': {
            'MAE': 8.0,  # Melhor MAE
            'RMSE': 12.0,  # Melhor RMSE
            'R2': 0.90,
            'model_name': 'RandomForest'
        },
        'Lasso': {
            'MAE': 12.0,
            'RMSE': 18.0,
            'R2': 0.98,  # Melhor R2
            'model_name': 'Lasso'
        }
    }
    
    # Melhor por MAE (menor)
    best_mae = RegressionEvaluator.get_best_model(
        results, 
        metric='MAE', 
        ascending=True
    )
    assert best_mae == 'RandomForest'
    
    # Melhor por R2 (maior)
    best_r2 = RegressionEvaluator.get_best_model(
        results, 
        metric='R2', 
        ascending=False
    )
    assert best_r2 == 'Lasso'
    
    # Melhor por RMSE (menor)
    best_rmse = RegressionEvaluator.get_best_model(
        results, 
        metric='RMSE', 
        ascending=True
    )
    assert best_rmse == 'RandomForest'
    
    print('✅ Melhor MAE: RandomForest (8.0)')
    print('✅ Melhor R²: Lasso (0.98)')
    print('✅ Melhor RMSE: RandomForest (12.0)')
    print('✅ Seleção funcionando corretamente')
    
    print('\n' + '=' * 70)
    print('TEST 4.1.4 PASSED ✅')
    print('=' * 70)


def test_save_predictions_csv():
    """
    TEST 4.1.5: Salvar predições em CSV
    
    Valida:
    - save_predictions_csv cria arquivo CSV
    - Colunas esperadas presentes
    - Valores de erro calculados corretamente
    - Ordenação por erro absoluto (maiores primeiro)
    """
    print('\n' + '=' * 70)
    print('TEST 4.1.5: Save Predictions CSV')
    print('=' * 70)
    
    # Criar diretório temporário
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Dados de teste
        y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        y_pred = np.array([110.0, 190.0, 320.0, 380.0, 510.0])
        
        output_path = Path(temp_dir) / 'predictions.csv'
        
        print(f'   Salvando em: {output_path}')
        
        # Salvar CSV (versão simplificada sem metadados)
        df_predictions = RegressionEvaluator.save_predictions_csv(
            y_true=y_true,
            y_pred=y_pred,
            output_path=output_path,
            model_name='TestModel',
            dataset_name='test'
        )
        
        print(f'   DataFrame retornado: {df_predictions is not None}')
        print(f'   Arquivo existe: {output_path.exists()}')
        
        # Verificar que arquivo foi criado
        assert output_path.exists(), f'Arquivo não foi criado: {output_path}'
        
        # Verificar DataFrame retornado
        assert isinstance(df_predictions, pd.DataFrame), f'Tipo incorreto: {type(df_predictions)}'
        assert len(df_predictions) == 5, f'Tamanho incorreto: {len(df_predictions)}'
        
        # Verificar colunas esperadas
        expected_cols = [
            'dataset', 'model', 'sample_index',
            'true_value_nM', 'predicted_value_nM',
            'absolute_error_nM', 'relative_error_percent'
        ]
        
        print(f'   Colunas esperadas: {expected_cols}')
        print(f'   Colunas presentes: {list(df_predictions.columns)}')
        
        for col in expected_cols:
            assert col in df_predictions.columns, f'Coluna {col} não encontrada'
        
        print('   ✅ Colunas OK')
        
        # Verificar valores
        assert all(df_predictions['dataset'] == 'test'), 'Dataset incorreto'
        print('   ✅ Dataset OK')
        
        assert all(df_predictions['model'] == 'TestModel'), 'Model incorreto'
        print('   ✅ Model OK')
        
        # Verificar cálculo de erros
        errors = np.abs(y_true - y_pred)
        print(f'   Erros calculados: {errors}')
        print(f'   Erros no DF: {df_predictions["absolute_error_nM"].values}')
        
        # O DataFrame está ordenado por erro (maiores primeiro)
        # Então precisamos ordenar os erros esperados também
        sorted_errors = np.sort(errors)[::-1]  # Ordem decrescente
        
        assert np.allclose(
            df_predictions['absolute_error_nM'].values, 
            sorted_errors
        ), f'Erros incorretos. Esperado (sorted): {sorted_errors}, Obtido: {df_predictions["absolute_error_nM"].values}'
        print('   ✅ Erros OK (ordenados por magnitude)')
        
        # Verificar ordenação (maiores erros primeiro)
        sorted_errors = df_predictions['absolute_error_nM'].values
        assert all(sorted_errors[i] >= sorted_errors[i+1] 
                   for i in range(len(sorted_errors)-1))
        
        # Ler CSV salvo
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == 5
        assert 'absolute_error_nM' in df_loaded.columns
        
        print('✅ CSV criado com sucesso')
        print(f'✅ {len(df_predictions)} linhas')
        print(f'✅ {len(df_predictions.columns)} colunas')
        print('✅ Erros calculados corretamente')
        print('✅ Ordenação por erro (maiores primeiro)')
        
    finally:
        # Limpar diretório temporário
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 4.1.5 PASSED ✅')
    print('=' * 70)


def test_metrics_edge_cases():
    """
    TEST 4.1.6: Edge cases e robustez
    
    Valida:
    - Predições perfeitas (R2=1.0, MAE=0)
    - Predições constantes (R2~0)
    - MAPE com zeros em y_true
    - Arrays pequenos (n=1, n=2)
    """
    print('\n' + '=' * 70)
    print('TEST 4.1.6: Metrics Edge Cases')
    print('=' * 70)
    
    # Caso 1: Predições perfeitas
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true.copy()
    
    metrics_perfect = RegressionEvaluator.calculate_metrics(y_true, y_pred)
    
    assert abs(metrics_perfect['MAE']) < 1e-10
    assert abs(metrics_perfect['RMSE']) < 1e-10
    assert abs(metrics_perfect['R2'] - 1.0) < 1e-10
    assert abs(metrics_perfect['mean_residual']) < 1e-10
    
    print('✅ Caso 1: Predições perfeitas')
    print(f'   R²={metrics_perfect["R2"]:.6f}, MAE={metrics_perfect["MAE"]:.6f}')
    
    # Caso 2: Predições constantes (média)
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.full(5, y_true.mean())
    
    metrics_const = RegressionEvaluator.calculate_metrics(y_true, y_pred)
    
    # R2 deve ser ~0 para predições constantes
    assert abs(metrics_const['R2']) < 0.1
    
    print('✅ Caso 2: Predições constantes')
    print(f'   R²={metrics_const["R2"]:.6f} (próximo de 0)')
    
    # Caso 3: MAPE com zeros em y_true
    y_true = np.array([0.0, 100.0, 200.0, 300.0])
    y_pred = np.array([10.0, 110.0, 190.0, 310.0])
    
    metrics_zeros = RegressionEvaluator.calculate_metrics(y_true, y_pred)
    
    # MAPE deve lidar com zeros (retornar valor ou None)
    assert 'MAPE' in metrics_zeros
    if metrics_zeros['MAPE'] is not None:
        assert metrics_zeros['MAPE'] >= 0
    
    print('✅ Caso 3: MAPE com zeros tratado')
    print(f'   MAPE={metrics_zeros["MAPE"]}')
    
    # Caso 4: Array pequeno (n=2)
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    
    metrics_small = RegressionEvaluator.calculate_metrics(y_true, y_pred)
    
    assert metrics_small['n_samples'] == 2
    assert metrics_small['MAE'] == 10.0
    assert 'error_p50' in metrics_small
    
    print('✅ Caso 4: Array pequeno (n=2)')
    print(f'   MAE={metrics_small["MAE"]:.2f}, samples={metrics_small["n_samples"]}')
    
    # Caso 5: Array unitário (n=1)
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    
    metrics_one = RegressionEvaluator.calculate_metrics(y_true, y_pred)
    
    assert metrics_one['n_samples'] == 1
    assert metrics_one['MAE'] == 10.0
    assert metrics_one['max_error'] == 10.0
    
    print('✅ Caso 5: Array unitário (n=1)')
    print(f'   MAE={metrics_one["MAE"]:.2f}, max_error={metrics_one["max_error"]:.2f}')
    
    print('\n' + '=' * 70)
    print('TEST 4.1.6 PASSED ✅')
    print('=' * 70)


def run_all_tests():
    """Executa todos os testes do nível 4.1"""
    print('\n')
    print('╔' + '═' * 68 + '╗')
    print('║' + ' ' * 15 + 'NÍVEL 4.1 - EVALUATOR TESTS' + ' ' * 26 + '║')
    print('╚' + '═' * 68 + '╝')
    
    tests = [
        ('4.1.1', 'Evaluator Initialization', test_evaluator_initialization),
        ('4.1.2', 'Metrics Calculation', test_metrics_calculation),
        ('4.1.3', 'Compare Models', test_compare_models),
        ('4.1.4', 'Best Model Selection', test_best_model_selection),
        ('4.1.5', 'Save Predictions CSV', test_save_predictions_csv),
        ('4.1.6', 'Metrics Edge Cases', test_metrics_edge_cases)
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
    print(f'║  ✅ Testes Passaram: {passed}/6' + ' ' * (68 - 25 - len(str(passed))) + '║')
    print(f'║  ❌ Testes Falharam: {failed}/6' + ' ' * (68 - 25 - len(str(failed))) + '║')
    print('╠' + '═' * 68 + '╣')
    
    if failed == 0:
        print('║  🎉 NÍVEL 4.1 COMPLETO - TODOS OS TESTES PASSARAM! 🎉' + ' ' * 13 + '║')
    else:
        print('║  ⚠️  ALGUNS TESTES FALHARAM - REVISAR ERROS ACIMA' + ' ' * 18 + '║')
    
    print('╚' + '═' * 68 + '╝')
    print()
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
