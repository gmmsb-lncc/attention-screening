#!/usr/bin/env python3
"""
Testes do RegressionPipeline - Nível 5.1
=========================================

Valida pipeline end-to-end, integração de componentes e workflow completo.

Autor: DockTKinase Team
Data: 2025-11-10
"""

import sys
import numpy as np
import json
from pathlib import Path
import tempfile
import shutil

# Adicionar src ao path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.modular_pipeline import RegressionPipeline, run_regression_pipeline


def create_synthetic_data(n_samples=100, n_features=50, temp_dir=None):
    """
    Criar dados sintéticos para testes.
    
    Returns:
        tuple: (embeddings_path, targets_path, temp_dir)
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    temp_path = Path(temp_dir)
    
    # Criar embeddings (features)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    embeddings_path = temp_path / 'embeddings.npy'
    np.save(embeddings_path, X)
    
    # Criar targets (valores contínuos simulando Ki em nM)
    # Relação linear com ruído para R² razoável
    weights = np.random.randn(n_features)
    y_base = X @ weights
    y = np.abs(y_base * 100 + np.random.randn(n_samples) * 50 + 200)  # Ki em nM
    
    targets_path = temp_path / 'targets.npy'
    np.save(targets_path, y)
    
    return str(embeddings_path), str(targets_path), temp_dir


def test_pipeline_initialization():
    """
    TEST 5.1.1: Inicialização do pipeline
    
    Valida:
    - RegressionPipeline inicializa corretamente
    - Parâmetros são armazenados
    - Diretórios de saída são criados
    - Componentes modularizados são instanciados
    """
    print('\n' + '=' * 70)
    print('TEST 5.1.1: Pipeline Initialization')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Criar dados sintéticos
        embeddings_path, targets_path, _ = create_synthetic_data(temp_dir=temp_dir)
        
        output_dir = Path(temp_dir) / 'results'
        
        # Inicializar pipeline
        pipeline = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=str(output_dir),
            models_to_train=['Ridge', 'Lasso'],
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            verbose=False
        )
        
        # Verificar atributos
        assert pipeline.embeddings_path == embeddings_path
        assert pipeline.targets_path == targets_path
        assert pipeline.test_size == 0.2
        assert pipeline.val_size == 0.1
        assert pipeline.random_state == 42
        assert pipeline.models_to_train == ['Ridge', 'Lasso']
        
        # Verificar diretórios criados
        assert output_dir.exists()
        assert (output_dir / 'models').exists()
        assert (output_dir / 'predictions').exists()
        assert (output_dir / 'metrics').exists()
        
        # Verificar componentes instanciados
        assert pipeline.data_manager is not None
        assert pipeline.metrics_calculator is not None
        assert pipeline.evaluator is not None
        
        # Verificar stats iniciais
        assert 'pipeline' in pipeline.stats
        assert 'timestamp' in pipeline.stats
        assert pipeline.stats['random_state'] == 42
        
        print('✅ Pipeline inicializado corretamente')
        print('✅ Parâmetros armazenados')
        print('✅ Diretórios criados')
        print('✅ Componentes instanciados')
        print('✅ Stats inicializados')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 5.1.1 PASSED ✅')
    print('=' * 70)


def test_load_data():
    """
    TEST 5.1.2: Carregamento de dados
    
    Valida:
    - load_data() carrega embeddings e targets
    - Dados são divididos em train/val/test
    - Proporções corretas (70/10/20)
    - Stats são atualizados
    """
    print('\n' + '=' * 70)
    print('TEST 5.1.2: Load Data')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Criar dados sintéticos
        embeddings_path, targets_path, _ = create_synthetic_data(
            n_samples=100, 
            temp_dir=temp_dir
        )
        
        output_dir = Path(temp_dir) / 'results'
        
        pipeline = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=str(output_dir),
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            verbose=False
        )
        
        # Carregar dados
        pipeline.load_data()
        
        # Verificar que dados foram carregados
        assert pipeline.X_train is not None
        assert pipeline.X_val is not None
        assert pipeline.X_test is not None
        assert pipeline.y_train is not None
        assert pipeline.y_val is not None
        assert pipeline.y_test is not None
        
        # Verificar proporções (100 amostras: 70/10/20)
        assert len(pipeline.X_train) == 70
        assert len(pipeline.X_val) == 10
        assert len(pipeline.X_test) == 20
        
        # Verificar shapes consistentes
        assert pipeline.X_train.shape[1] == pipeline.X_val.shape[1] == pipeline.X_test.shape[1]
        assert len(pipeline.y_train) == len(pipeline.X_train)
        assert len(pipeline.y_val) == len(pipeline.X_val)
        assert len(pipeline.y_test) == len(pipeline.X_test)
        
        # Verificar stats atualizados
        assert 'n_samples_total' in pipeline.stats
        assert 'n_samples_train' in pipeline.stats
        assert 'n_samples_val' in pipeline.stats
        assert 'n_samples_test' in pipeline.stats
        assert 'embedding_dim' in pipeline.stats
        assert 'target_stats' in pipeline.stats
        
        assert pipeline.stats['n_samples_total'] == 100
        assert pipeline.stats['n_samples_train'] == 70
        
        print(f'✅ Dados carregados: {len(pipeline.X_train)} train, {len(pipeline.X_val)} val, {len(pipeline.X_test)} test')
        print(f'✅ Proporções corretas: 70/10/20')
        print(f'✅ Shapes consistentes: {pipeline.X_train.shape[1]} features')
        print(f'✅ Stats atualizados')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 5.1.2 PASSED ✅')
    print('=' * 70)


def test_train_models():
    """
    TEST 5.1.3: Treinamento de modelos
    
    Valida:
    - train_models() treina os modelos especificados
    - Métricas de treino e validação são calculadas
    - Modelos treinados são armazenados
    - Stats de treinamento são registrados
    """
    print('\n' + '=' * 70)
    print('TEST 5.1.3: Train Models')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        embeddings_path, targets_path, _ = create_synthetic_data(
            n_samples=100,
            temp_dir=temp_dir
        )
        
        output_dir = Path(temp_dir) / 'results'
        
        pipeline = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=str(output_dir),
            models_to_train=['Ridge', 'Lasso', 'RandomForest'],
            random_state=42,
            verbose=False
        )
        
        # Carregar dados primeiro
        pipeline.load_data()
        
        # Treinar modelos
        val_metrics = pipeline.train_models()
        
        # Verificar que modelos foram treinados
        assert len(pipeline.trained_models) == 3
        assert 'Ridge' in pipeline.trained_models
        assert 'Lasso' in pipeline.trained_models
        assert 'RandomForest' in pipeline.trained_models
        
        # Verificar que cada modelo tem método predict
        for model in pipeline.trained_models.values():
            assert hasattr(model, 'predict')
        
        # Verificar métricas de treino
        assert len(pipeline.train_metrics) == 3
        for model_name, metrics in pipeline.train_metrics.items():
            assert 'MAE' in metrics
            assert 'RMSE' in metrics
            assert 'R2' in metrics
        
        # Verificar métricas de validação
        assert len(pipeline.val_metrics) == 3
        assert len(val_metrics) == 3
        
        # Verificar stats de treinamento
        assert 'training_time' in pipeline.stats
        assert 'n_models_trained' in pipeline.stats
        assert pipeline.stats['n_models_trained'] == 3
        
        print(f'✅ {len(pipeline.trained_models)} modelos treinados')
        print(f'✅ Métricas de treino calculadas')
        print(f'✅ Métricas de validação calculadas')
        print(f'✅ Stats registrados')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 5.1.3 PASSED ✅')
    print('=' * 70)


def test_evaluate_on_test():
    """
    TEST 5.1.4: Avaliação no conjunto de teste
    
    Valida:
    - evaluate_on_test() avalia todos os modelos
    - Métricas de teste são calculadas
    - Predições são válidas
    """
    print('\n' + '=' * 70)
    print('TEST 5.1.4: Evaluate on Test')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        embeddings_path, targets_path, _ = create_synthetic_data(
            n_samples=100,
            temp_dir=temp_dir
        )
        
        output_dir = Path(temp_dir) / 'results'
        
        pipeline = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=str(output_dir),
            models_to_train=['Ridge', 'Lasso'],
            random_state=42,
            verbose=False
        )
        
        pipeline.load_data()
        pipeline.train_models()
        
        # Avaliar no teste
        test_metrics = pipeline.evaluate_on_test()
        
        # Verificar que métricas foram calculadas
        assert len(test_metrics) == 2
        assert 'Ridge' in test_metrics
        assert 'Lasso' in test_metrics
        
        # Verificar estrutura das métricas
        for model_name, metrics in test_metrics.items():
            assert 'MAE' in metrics
            assert 'RMSE' in metrics
            assert 'R2' in metrics
            assert 'model_name' in metrics
            
            # Verificar valores razoáveis
            assert metrics['MAE'] > 0
            assert metrics['RMSE'] > 0
            assert metrics['RMSE'] >= metrics['MAE']  # RMSE sempre >= MAE
        
        # Verificar que test_metrics foi armazenado
        assert len(pipeline.test_metrics) == 2
        
        print(f'✅ {len(test_metrics)} modelos avaliados')
        print(f'✅ Métricas de teste calculadas')
        print(f'✅ Valores razoáveis (MAE, RMSE, R²)')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 5.1.4 PASSED ✅')
    print('=' * 70)


def test_save_results():
    """
    TEST 5.1.5: Salvamento de resultados
    
    Valida:
    - save_results() salva arquivos JSON
    - test_metrics.json criado
    - validation_metrics.json criado
    - pipeline_stats.json criado
    - Estrutura JSON correta
    """
    print('\n' + '=' * 70)
    print('TEST 5.1.5: Save Results')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        embeddings_path, targets_path, _ = create_synthetic_data(
            n_samples=100,
            temp_dir=temp_dir
        )
        
        output_dir = Path(temp_dir) / 'results'
        
        pipeline = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=str(output_dir),
            models_to_train=['Ridge'],
            random_state=42,
            verbose=False
        )
        
        pipeline.load_data()
        pipeline.train_models()
        pipeline.evaluate_on_test()
        
        # Salvar resultados
        pipeline.save_results()
        
        # Verificar arquivos criados
        test_metrics_file = output_dir / 'metrics' / 'test_metrics.json'
        val_metrics_file = output_dir / 'metrics' / 'validation_metrics.json'
        stats_file = output_dir / 'pipeline_stats.json'
        
        assert test_metrics_file.exists()
        assert val_metrics_file.exists()
        assert stats_file.exists()
        
        # Verificar conteúdo dos arquivos
        with open(test_metrics_file) as f:
            test_data = json.load(f)
            assert 'Ridge' in test_data
            assert 'MAE' in test_data['Ridge']
        
        with open(val_metrics_file) as f:
            val_data = json.load(f)
            assert 'Ridge' in val_data
        
        with open(stats_file) as f:
            stats_data = json.load(f)
            assert 'pipeline' in stats_data
            assert 'timestamp' in stats_data
            assert 'n_samples_total' in stats_data
            assert 'test_metrics_summary' in stats_data
        
        print('✅ test_metrics.json criado')
        print('✅ validation_metrics.json criado')
        print('✅ pipeline_stats.json criado')
        print('✅ Estrutura JSON correta')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 5.1.5 PASSED ✅')
    print('=' * 70)


def test_run_complete_pipeline():
    """
    TEST 5.1.6: Pipeline completo end-to-end
    
    Valida:
    - run() executa todas as etapas
    - Retorna métricas de teste
    - Todos os arquivos são criados
    - Workflow completo funciona
    """
    print('\n' + '=' * 70)
    print('TEST 5.1.6: Run Complete Pipeline')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        embeddings_path, targets_path, _ = create_synthetic_data(
            n_samples=100,
            temp_dir=temp_dir
        )
        
        output_dir = Path(temp_dir) / 'results'
        
        pipeline = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=str(output_dir),
            models_to_train=['Ridge', 'Lasso'],
            random_state=42,
            verbose=False
        )
        
        # Executar pipeline completo
        test_metrics = pipeline.run()
        
        # Verificar retorno
        assert isinstance(test_metrics, dict)
        assert len(test_metrics) == 2
        assert 'Ridge' in test_metrics
        assert 'Lasso' in test_metrics
        
        # Verificar que dados foram carregados
        assert pipeline.X_train is not None
        assert len(pipeline.X_train) == 70
        
        # Verificar que modelos foram treinados
        assert len(pipeline.trained_models) == 2
        
        # Verificar que métricas foram calculadas
        assert len(pipeline.train_metrics) == 2
        assert len(pipeline.val_metrics) == 2
        assert len(pipeline.test_metrics) == 2
        
        # Verificar arquivos salvos
        assert (output_dir / 'metrics' / 'test_metrics.json').exists()
        assert (output_dir / 'metrics' / 'validation_metrics.json').exists()
        assert (output_dir / 'pipeline_stats.json').exists()
        
        print('✅ Pipeline executado end-to-end')
        print(f'✅ {len(test_metrics)} modelos avaliados')
        print('✅ Todos os arquivos criados')
        print('✅ Workflow completo funcionando')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 5.1.6 PASSED ✅')
    print('=' * 70)


def test_convenience_function():
    """
    TEST 5.1.7: Função de conveniência run_regression_pipeline
    
    Valida:
    - run_regression_pipeline() funciona
    - Interface simplificada
    - Retorna métricas corretas
    """
    print('\n' + '=' * 70)
    print('TEST 5.1.7: Convenience Function')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        embeddings_path, targets_path, _ = create_synthetic_data(
            n_samples=100,
            temp_dir=temp_dir
        )
        
        output_dir = Path(temp_dir) / 'results'
        
        # Usar função de conveniência
        test_metrics = run_regression_pipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=str(output_dir),
            models=['Ridge'],
            random_state=42,
            verbose=False
        )
        
        # Verificar retorno
        assert isinstance(test_metrics, dict)
        assert 'Ridge' in test_metrics
        assert 'MAE' in test_metrics['Ridge']
        
        # Verificar que arquivos foram criados
        assert (output_dir / 'metrics' / 'test_metrics.json').exists()
        assert (output_dir / 'pipeline_stats.json').exists()
        
        print('✅ Função de conveniência funciona')
        print('✅ Interface simplificada')
        print('✅ Métricas retornadas corretamente')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 5.1.7 PASSED ✅')
    print('=' * 70)


def test_reproducibility():
    """
    TEST 5.1.8: Reprodutibilidade com random_state
    
    Valida:
    - Mesmo random_state produz resultados idênticos
    - Splits são reproduzíveis
    - Predições são reproduzíveis
    """
    print('\n' + '=' * 70)
    print('TEST 5.1.8: Reproducibility')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        embeddings_path, targets_path, _ = create_synthetic_data(
            n_samples=100,
            temp_dir=temp_dir
        )
        
        output_dir1 = Path(temp_dir) / 'results1'
        output_dir2 = Path(temp_dir) / 'results2'
        
        # Executar pipeline 2 vezes com mesmo random_state
        pipeline1 = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=str(output_dir1),
            models_to_train=['Ridge'],
            random_state=999,
            verbose=False
        )
        
        pipeline2 = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=str(output_dir2),
            models_to_train=['Ridge'],
            random_state=999,
            verbose=False
        )
        
        # Executar ambos
        metrics1 = pipeline1.run()
        metrics2 = pipeline2.run()
        
        # Verificar splits idênticos
        assert np.allclose(pipeline1.X_train, pipeline2.X_train)
        assert np.allclose(pipeline1.y_train, pipeline2.y_train)
        assert np.allclose(pipeline1.X_test, pipeline2.X_test)
        assert np.allclose(pipeline1.y_test, pipeline2.y_test)
        
        # Verificar métricas idênticas
        mae1 = metrics1['Ridge']['MAE']
        mae2 = metrics2['Ridge']['MAE']
        assert abs(mae1 - mae2) < 1e-10
        
        r2_1 = metrics1['Ridge']['R2']
        r2_2 = metrics2['Ridge']['R2']
        assert abs(r2_1 - r2_2) < 1e-10
        
        print('✅ Splits reproduzíveis')
        print(f'✅ MAE idêntico: {mae1:.10f}')
        print(f'✅ R² idêntico: {r2_1:.10f}')
        print('✅ Reprodutibilidade completa')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 5.1.8 PASSED ✅')
    print('=' * 70)


def test_multiple_models():
    """
    TEST 5.1.9: Pipeline com múltiplos modelos
    
    Valida:
    - Pipeline treina múltiplos modelos
    - Todos são avaliados corretamente
    - Melhor modelo é identificado
    - Comparação entre modelos funciona
    """
    print('\n' + '=' * 70)
    print('TEST 5.1.9: Multiple Models')
    print('=' * 70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        embeddings_path, targets_path, _ = create_synthetic_data(
            n_samples=150,
            temp_dir=temp_dir
        )
        
        output_dir = Path(temp_dir) / 'results'
        
        # Treinar 5 modelos diferentes
        models_list = ['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'GradientBoosting']
        
        pipeline = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=str(output_dir),
            models_to_train=models_list,
            random_state=42,
            verbose=False
        )
        
        test_metrics = pipeline.run()
        
        # Verificar que todos os modelos foram treinados
        assert len(test_metrics) == 5
        for model_name in models_list:
            assert model_name in test_metrics
        
        # Verificar que todos têm métricas
        for model_name, metrics in test_metrics.items():
            assert 'MAE' in metrics
            assert 'RMSE' in metrics
            assert 'R2' in metrics
            assert metrics['MAE'] > 0
            assert metrics['RMSE'] > 0
        
        # Encontrar melhor modelo por MAE
        best_model = min(test_metrics.items(), key=lambda x: x[1]['MAE'])
        best_name = best_model[0]
        best_mae = best_model[1]['MAE']
        
        # Verificar que o melhor foi identificado
        assert best_name in models_list
        
        # Verificar que stats contém best_model
        with open(output_dir / 'pipeline_stats.json') as f:
            stats = json.load(f)
            assert stats['test_metrics_summary']['best_model'] == best_name
            assert abs(stats['test_metrics_summary']['best_mae'] - best_mae) < 1e-10
        
        print(f'✅ {len(test_metrics)} modelos treinados')
        print(f'✅ Todos avaliados corretamente')
        print(f'✅ Melhor modelo: {best_name} (MAE={best_mae:.4f})')
        print('✅ Comparação entre modelos funciona')
        
    finally:
        shutil.rmtree(temp_dir)
    
    print('\n' + '=' * 70)
    print('TEST 5.1.9 PASSED ✅')
    print('=' * 70)


def run_all_tests():
    """Executa todos os testes do nível 5.1"""
    print('\n')
    print('╔' + '═' * 68 + '╗')
    print('║' + ' ' * 12 + 'NÍVEL 5.1 - PIPELINE INTEGRATION TESTS' + ' ' * 17 + '║')
    print('╚' + '═' * 68 + '╝')
    
    tests = [
        ('5.1.1', 'Pipeline Initialization', test_pipeline_initialization),
        ('5.1.2', 'Load Data', test_load_data),
        ('5.1.3', 'Train Models', test_train_models),
        ('5.1.4', 'Evaluate on Test', test_evaluate_on_test),
        ('5.1.5', 'Save Results', test_save_results),
        ('5.1.6', 'Run Complete Pipeline', test_run_complete_pipeline),
        ('5.1.7', 'Convenience Function', test_convenience_function),
        ('5.1.8', 'Reproducibility', test_reproducibility),
        ('5.1.9', 'Multiple Models', test_multiple_models)
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
    print(f'║  ✅ Testes Passaram: {passed}/9' + ' ' * (68 - 25 - len(str(passed))) + '║')
    print(f'║  ❌ Testes Falharam: {failed}/9' + ' ' * (68 - 25 - len(str(failed))) + '║')
    print('╠' + '═' * 68 + '╣')
    
    if failed == 0:
        print('║  🎉 NÍVEL 5.1 COMPLETO - TODOS OS TESTES PASSARAM! 🎉' + ' ' * 13 + '║')
    else:
        print('║  ⚠️  ALGUNS TESTES FALHARAM - REVISAR ERROS ACIMA' + ' ' * 18 + '║')
    
    print('╚' + '═' * 68 + '╝')
    print()
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
