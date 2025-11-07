#!/usr/bin/env python3
"""
Teste Simples da Modularização de Regressão
============================================

Valida que todos os componentes modulares funcionam corretamente
com dados sintéticos.
"""

import sys
import numpy as np
from pathlib import Path

# Adicionar src ao path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.core import DataManager, RegressionTrainer
from regression.utils import MetricsCalculator
from regression.models import RegressionModels
from regression.modular_pipeline import RegressionPipeline


def test_data_manager():
    """Testar DataManager com dados sintéticos."""
    print("\n" + "="*70)
    print("TESTE 1: DataManager")
    print("="*70)
    
    # Criar dados sintéticos
    n_samples = 100
    embedding_dim = 128
    
    X = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    y = np.random.rand(n_samples) * 100 + 10  # valores entre 10 e 110 nM
    
    # Salvar temporariamente
    temp_dir = Path('tmp')
    temp_dir.mkdir(exist_ok=True)
    
    embeddings_file = temp_dir / 'test_embeddings.npy'
    targets_file = temp_dir / 'test_targets.npy'
    
    np.save(embeddings_file, X)
    np.save(targets_file, y)
    
    # Testar DataManager
    manager = DataManager(str(embeddings_file), str(targets_file))
    
    # Carregar dados
    X_loaded, y_loaded = manager.load_data()
    print(f"✅ Dados carregados: {X_loaded.shape}, {y_loaded.shape}")
    
    # Dividir dados
    X_train, X_val, X_test, y_train, y_val, y_test = manager.split_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    print(f"✅ Split realizado:")
    print(f"   Treino: {len(X_train)} amostras")
    print(f"   Validação: {len(X_val)} amostras")
    print(f"   Teste: {len(X_test)} amostras")
    
    # Estatísticas
    stats = manager.get_stats()
    print(f"✅ Estatísticas obtidas:")
    print(f"   Dimensão: {stats['embedding_dim']}")
    print(f"   Target média: {stats['target_mean']:.2f} nM")
    print(f"   Target std: {stats['target_std']:.2f} nM")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def test_metrics_calculator():
    """Testar MetricsCalculator."""
    print("\n" + "="*70)
    print("TESTE 2: MetricsCalculator")
    print("="*70)
    
    # Dados sintéticos
    y_true = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    y_pred = y_true + np.random.randn(10) * 5
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(y_true, y_pred, 'TestModel')
    
    print(f"✅ Métricas calculadas:")
    print(f"   MAE: {metrics['MAE']:.4f}")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   R²: {metrics['R2']:.4f}")
    print(f"   MedianAE: {metrics['MedianAE']:.4f}")
    
    # Tabela formatada
    table = calculator.format_metrics_table(metrics)
    print(table)
    
    return metrics


def test_regression_models():
    """Testar factory de modelos."""
    print("\n" + "="*70)
    print("TESTE 3: RegressionModels Factory")
    print("="*70)
    
    models = RegressionModels.get_all_models(random_state=42)
    
    print(f"✅ Modelos disponíveis: {len(models)}")
    for name in models.keys():
        print(f"   - {name}")
    
    return models


def test_regression_trainer(X_train, y_train, X_val, y_val):
    """Testar RegressionTrainer."""
    print("\n" + "="*70)
    print("TESTE 4: RegressionTrainer")
    print("="*70)
    
    # Selecionar apenas alguns modelos para teste rápido
    all_models = RegressionModels.get_all_models(random_state=42)
    test_models = {
        'RandomForest': all_models['RandomForest'],
        'Ridge': all_models['Ridge'],
        'KNN': all_models['KNN']
    }
    
    trainer = RegressionTrainer(
        models_dict=test_models,
        verbose=True,
        random_state=42
    )
    
    # Treinar
    val_results = trainer.train_all(X_train, y_train, X_val, y_val)
    
    print(f"\n✅ Treinamento completo!")
    print(f"   Modelos treinados: {len(trainer.trained_models)}")
    
    # Mostrar resultados
    print("\n📊 Resultados de Validação:")
    for model_name, metrics in val_results.items():
        print(f"   {model_name:15s} - MAE: {metrics['MAE']:.4f}, R²: {metrics['R2']:.4f}")
    
    return trainer


def test_complete_pipeline():
    """Testar pipeline completo."""
    print("\n" + "="*70)
    print("TESTE 5: Pipeline Completo")
    print("="*70)
    
    # Criar dados sintéticos
    n_samples = 200
    embedding_dim = 256
    
    X = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Criar targets com alguma relação com X (para R² positivo)
    weights = np.random.randn(embedding_dim)
    y = X @ weights + np.random.randn(n_samples) * 10
    y = np.abs(y) * 10 + 5  # valores positivos em nM
    
    # Salvar
    temp_dir = Path('tmp')
    embeddings_file = temp_dir / 'pipeline_test_embeddings.npy'
    targets_file = temp_dir / 'pipeline_test_targets.npy'
    
    np.save(embeddings_file, X)
    np.save(targets_file, y)
    
    # Criar pipeline
    pipeline = RegressionPipeline(
        embeddings_path=str(embeddings_file),
        targets_path=str(targets_file),
        output_dir='results/test_modular',
        models_to_train=['RandomForest', 'Ridge', 'KNN'],
        random_state=42,
        verbose=True
    )
    
    # Executar
    results = pipeline.run()
    
    print(f"\n✅ Pipeline executado com sucesso!")
    print(f"   Modelos avaliados: {len(results)}")
    
    # Melhor modelo
    best_model = min(results.items(), key=lambda x: x[1]['MAE'])
    print(f"\n🏆 Melhor Modelo: {best_model[0]}")
    print(f"   MAE: {best_model[1]['MAE']:.4f}")
    print(f"   R²: {best_model[1]['R2']:.4f}")
    
    return results


def main():
    """Executar todos os testes."""
    print("\n" + "="*70)
    print("🧪 TESTES DE MODULARIZAÇÃO - REGRESSÃO")
    print("="*70)
    
    try:
        # Teste 1: DataManager
        X_train, X_val, X_test, y_train, y_val, y_test = test_data_manager()
        
        # Teste 2: MetricsCalculator
        test_metrics_calculator()
        
        # Teste 3: RegressionModels
        test_regression_models()
        
        # Teste 4: RegressionTrainer
        test_regression_trainer(X_train, y_train, X_val, y_val)
        
        # Teste 5: Pipeline completo
        test_complete_pipeline()
        
        print("\n" + "="*70)
        print("✅ TODOS OS TESTES PASSARAM!")
        print("="*70)
        print("\n📝 Componentes testados:")
        print("   ✅ DataManager (carregamento e split)")
        print("   ✅ MetricsCalculator (15+ métricas)")
        print("   ✅ RegressionModels (factory de modelos)")
        print("   ✅ RegressionTrainer (treinamento)")
        print("   ✅ RegressionPipeline (pipeline completo)")
        print("\n🎯 Modularização de Regressão: VALIDADA")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
