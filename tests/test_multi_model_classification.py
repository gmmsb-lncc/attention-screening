#!/usr/bin/env python3
"""
Teste do Pipeline Multi-Modelo de Classificação
===============================================

Script de teste para validar a implementação do pipeline
multi-modelo de classificação.
"""

import sys
import numpy as np
from pathlib import Path

# Adicionar path do projeto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from classifier.models.classifiers import ClassificationModels
from classifier.multi_model_pipeline import MultiModelClassificationPipeline


def create_synthetic_data(n_samples=1000, n_features=256, output_dir='tmp'):
    """
    Cria dados sintéticos para teste.
    
    Args:
        n_samples: Número de amostras
        n_features: Número de features
        output_dir: Diretório de saída
        
    Returns:
        Tuple (embeddings_path, labels_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f'Criando dados sintéticos: {n_samples} amostras, {n_features} features')
    
    # Embeddings sintéticos
    embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Labels balanceados (60% classe 0, 40% classe 1)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Salvar
    embeddings_path = output_path / 'synthetic_embeddings.npy'
    labels_path = output_path / 'synthetic_labels.npy'
    
    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    
    print(f'✅ Dados salvos:')
    print(f'   Embeddings: {embeddings_path}')
    print(f'   Labels: {labels_path}')
    print(f'   Distribuição: {np.sum(labels == 0)} negativos, {np.sum(labels == 1)} positivos')
    print()
    
    return str(embeddings_path), str(labels_path)


def test_models_availability():
    """Testa disponibilidade dos modelos."""
    print('=' * 70)
    print('TESTE 1: Disponibilidade de Modelos')
    print('=' * 70)
    print()
    
    ClassificationModels.print_available_models()
    
    available = ClassificationModels.get_available_models()
    print(f'\n✅ Total de modelos disponíveis: {len(available)}')
    print()


def test_pipeline_small(embeddings_path, labels_path):
    """Testa pipeline com conjunto pequeno."""
    print('=' * 70)
    print('TESTE 2: Pipeline Completo (Conjunto Pequeno)')
    print('=' * 70)
    print()
    
    # Testar apenas 3 modelos rápidos
    models_to_test = ['LogisticRegression', 'RandomForest', 'NaiveBayes']
    
    pipeline = MultiModelClassificationPipeline(
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        output_dir='tmp/results_small',
        models_to_train=models_to_test,
        random_state=42,
        verbose=True
    )
    
    results = pipeline.run()
    
    print(f'\n✅ Pipeline executado com sucesso!')
    print(f'   Modelos treinados: {len(results)}')
    
    return results


def test_pipeline_all_models(embeddings_path, labels_path):
    """Testa pipeline com todos os modelos."""
    print('=' * 70)
    print('TESTE 3: Pipeline com Todos os Modelos')
    print('=' * 70)
    print()
    
    pipeline = MultiModelClassificationPipeline(
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        output_dir='tmp/results_all',
        models_to_train=None,  # Todos os modelos
        random_state=42,
        verbose=True
    )
    
    results = pipeline.run()
    
    print(f'\n✅ Pipeline executado com sucesso!')
    print(f'   Modelos treinados: {len(results)}')
    
    # Comparação com MLP
    print('\n📊 Comparação com MLP:')
    if 'MLP' in results:
        mlp_roc = results['MLP']['ROC_AUC']
        print(f'   MLP ROC-AUC: {mlp_roc:.4f}')
        
        # Encontrar melhor modelo
        best_model = max(results.items(), key=lambda x: x[1]['ROC_AUC'])
        print(f'   Melhor modelo: {best_model[0]} (ROC-AUC: {best_model[1]["ROC_AUC"]:.4f})')
        
        improvement = (best_model[1]['ROC_AUC'] - mlp_roc) / mlp_roc * 100
        if improvement > 0:
            print(f'   Melhoria: +{improvement:.1f}%')
    
    return results


def main():
    """Função principal de teste."""
    print('🧪 TESTE DO PIPELINE MULTI-MODELO DE CLASSIFICAÇÃO')
    print('=' * 70)
    print()
    
    # Teste 1: Disponibilidade de modelos
    test_models_availability()
    
    # Criar dados sintéticos
    embeddings_path, labels_path = create_synthetic_data(
        n_samples=1000,
        n_features=256
    )
    
    # Teste 2: Pipeline pequeno (3 modelos)
    test_pipeline_small(embeddings_path, labels_path)
    
    # Teste 3: Pipeline completo (todos os modelos)
    test_pipeline_all_models(embeddings_path, labels_path)
    
    print('=' * 70)
    print('✅ TODOS OS TESTES CONCLUÍDOS COM SUCESSO!')
    print('=' * 70)


if __name__ == '__main__':
    main()
