#!/usr/bin/env python3
"""
Pipeline Multi-Modelo de Classificação - DockTKinase
====================================================

Pipeline completo de classificação modularizado com suporte a múltiplos
algoritmos, equivalente ao pipeline de regressão multi-modelo.

Esta implementação treina 10-11 modelos diferentes e seleciona o melhor.
"""

import time
import json
import warnings
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Suppress harmless sklearn/LGBM warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='.*was fitted with feature names.*')
warnings.filterwarnings('ignore', message=".*parameter 'algorithm' is deprecated.*")
warnings.filterwarnings('ignore', message='.*Liblinear failed to converge.*')
warnings.filterwarnings('ignore', message='An input array is constant')

# Imports dos módulos
try:
    from .models.classifiers import ClassificationModels
    from .core.sklearn_trainer import SklearnClassificationTrainer, ClassificationMetricsCalculator
    from .core.sklearn_data_manager import SklearnDataManager
except ImportError:
    # Fallback para execução direta
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    from models.classifiers import ClassificationModels
    from core.sklearn_trainer import SklearnClassificationTrainer, ClassificationMetricsCalculator
    from core.sklearn_data_manager import SklearnDataManager


class MultiModelClassificationPipeline:
    """
    Pipeline modular de classificação para predição de atividade.
    
    Implementa pipeline completo equivalente ao de regressão:
    1. Carregar dados (embeddings + labels)
    2. Dividir em treino/validação/teste
    3. Treinar múltiplos modelos (10-11 algoritmos)
    4. Avaliar e comparar resultados
    5. Salvar métricas e predições
    
    Modelos suportados:
    - RandomForest, GradientBoosting, LogisticRegression
    - LinearSVC, ExtraTrees, KNN, MLP, NaiveBayes
    - DecisionTree, AdaBoost
    - XGBoost (OBRIGATÓRIO), LightGBM, CatBoost (opcionais)
    """
    
    def __init__(
        self,
        embeddings_path: str,
        labels_path: str,
        output_dir: str = 'results/classification_multi_model',
        models_to_train: Optional[List[str]] = None,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Inicializar pipeline de classificação multi-modelo.
        
        Args:
            embeddings_path: Caminho para embeddings (.npy ou .npz)
            labels_path: Caminho para labels binários (.npy)
            output_dir: Diretório para salvar resultados
            models_to_train: Lista de modelos a treinar (None = todos)
            test_size: Proporção do conjunto de teste (0.1 = 10%)
            val_size: Proporção do conjunto de validação (0.1 = 10%)
            random_state: Seed para reprodutibilidade
            verbose: Mostrar progresso
        """
        self.embeddings_path = embeddings_path
        self.labels_path = labels_path
        self.output_dir = Path(output_dir)
        self.models_to_train = models_to_train
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Criar diretórios de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'predictions').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        
        # Componentes modularizados
        self.data_manager = SklearnDataManager(embeddings_path, labels_path)
        self.metrics_calculator = ClassificationMetricsCalculator()
        
        # Dados (preenchidos no load_data)
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Resultados
        self.trained_models = {}
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
        # Stats do pipeline
        self.stats = {
            'pipeline': 'classification_multi_model',
            'timestamp': datetime.now().isoformat(),
            'random_state': random_state,
            'embeddings_path': str(embeddings_path),
            'labels_path': str(labels_path)
        }
        
    def load_data(self) -> None:
        """
        Carregar embeddings e labels, dividir em treino/val/teste.
        
        Usa stratified split para manter proporção de classes
        em todos os conjuntos.
        """
        if self.verbose:
            print('📊 ETAPA 1: Carregamento e Divisão de Dados')
            print('=' * 70)
        
        start_time = time.time()
        
        # Carregar e dividir dados
        self.X_train, self.X_val, self.X_test, \
        self.y_train, self.y_val, self.y_test = self.data_manager.split_data(
            test_size=self.test_size,
            val_size=self.val_size,
            stratify=True,
            random_state=self.random_state
        )
        
        # Estatísticas
        stats = self.data_manager.get_stats()
        
        # Distribuição de classes
        train_pos = np.sum(self.y_train == 1)
        train_neg = np.sum(self.y_train == 0)
        val_pos = np.sum(self.y_val == 1)
        val_neg = np.sum(self.y_val == 0)
        test_pos = np.sum(self.y_test == 1)
        test_neg = np.sum(self.y_test == 0)
        
        if self.verbose:
            print(f"✅ Dados carregados com sucesso!")
            print(f"   Total de amostras: {stats['n_samples']:,}")
            print(f"   Dimensão embeddings: {stats['embedding_dim']:,}")
            print(f"\n   Treino: {len(self.X_train):,} amostras")
            print(f"      Positivos: {train_pos:,} ({train_pos/len(self.y_train)*100:.1f}%)")
            print(f"      Negativos: {train_neg:,} ({train_neg/len(self.y_train)*100:.1f}%)")
            print(f"\n   Validação: {len(self.X_val):,} amostras")
            print(f"      Positivos: {val_pos:,} ({val_pos/len(self.y_val)*100:.1f}%)")
            print(f"      Negativos: {val_neg:,} ({val_neg/len(self.y_val)*100:.1f}%)")
            print(f"\n   Teste: {len(self.X_test):,} amostras")
            print(f"      Positivos: {test_pos:,} ({test_pos/len(self.y_test)*100:.1f}%)")
            print(f"      Negativos: {test_neg:,} ({test_neg/len(self.y_test)*100:.1f}%)")
            print(f"\n   Tempo: {time.time() - start_time:.2f}s")
            print('=' * 70)
            print()
        
        # Atualizar stats
        self.stats.update({
            'n_samples_total': stats['n_samples'],
            'n_samples_train': len(self.X_train),
            'n_samples_val': len(self.X_val),
            'n_samples_test': len(self.X_test),
            'embedding_dim': stats['embedding_dim'],
            'class_distribution': {
                'train': {'positive': int(train_pos), 'negative': int(train_neg)},
                'val': {'positive': int(val_pos), 'negative': int(val_neg)},
                'test': {'positive': int(test_pos), 'negative': int(test_neg)}
            }
        })
    
    def train_models(self) -> Dict[str, Any]:
        """
        Treinar todos os modelos de classificação.
        
        Returns:
            Dict com métricas de validação de todos os modelos
        """
        if self.verbose:
            print('🤖 ETAPA 2: Treinamento de Modelos')
            print('=' * 70)
        
        # Obter modelos
        all_models = ClassificationModels.get_all_models(
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Filtrar modelos se especificado
        if self.models_to_train:
            models = {k: v for k, v in all_models.items() if k in self.models_to_train}
        else:
            models = all_models
        
        if self.verbose:
            print(f"   Modelos a treinar: {len(models)}")
            print(f"   Modelos: {', '.join(models.keys())}")
        
        # Criar trainer
        trainer = SklearnClassificationTrainer(
            models_dict=models,
            verbose=self.verbose,
            random_state=self.random_state
        )
        
        # Treinar todos
        start_time = time.time()
        trainer.train_all(self.X_train, self.y_train, self.X_val, self.y_val)
        training_time = time.time() - start_time
        
        # Armazenar resultados
        self.trained_models = trainer.trained_models
        self.train_metrics = trainer.train_results
        self.val_metrics = trainer.val_results
        
        if self.verbose:
            print(f"\n✅ Treinamento e validação completos!")
            print(f"   Tempo total: {training_time:.2f}s")
            print(f"   Tempo médio por modelo: {training_time/len(models):.2f}s")
            print('=' * 70)
            print()
        
        self.stats['training_time'] = training_time
        self.stats['n_models_trained'] = len(models)
        
        return self.val_metrics
    
    def evaluate_on_test(self) -> Dict[str, Any]:
        """
        Avaliar todos os modelos no conjunto de teste.
        
        Returns:
            Dict com métricas de teste de todos os modelos
        """
        if self.verbose:
            print('📈 ETAPA 3: Avaliação no Conjunto de Teste')
            print('=' * 70)
        
        for model_name, model in self.trained_models.items():
            if self.verbose:
                print(f"   Avaliando {model_name}...")
            
            # Predições
            y_pred = model.predict(self.X_test)
            
            # Probabilidades (se disponível)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            else:
                y_pred_proba = None
            
            # Calcular métricas
            metrics = self.metrics_calculator.calculate_all_metrics(
                self.y_test,
                y_pred,
                y_pred_proba,
                model_name
            )
            
            self.test_metrics[model_name] = metrics
        
        if self.verbose:
            print("\n✅ Avaliação no conjunto de teste completa!")
            print('=' * 70)
            print()
        
        return self.test_metrics
    
    def print_results_summary(self) -> None:
        """Print summary of results."""
        if not self.test_metrics:
            print("⚠️  No test results available")
            return
        
        print('📊 RESULTS SUMMARY (Test Set)')
        print('=' * 80)
        
        # Sort by ROC-AUC
        sorted_results = sorted(
            self.test_metrics.items(),
            key=lambda x: x[1]['ROC_AUC'],
            reverse=True
        )
        
        # Header
        header = f"{'Model':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'ROC-AUC':>10}"
        print(header)
        print('-' * 80)
        
        # Results
        for i, (model_name, metrics) in enumerate(sorted_results):
            row = (
                f"{model_name:<20} "
                f"{metrics['Accuracy']:>8.4f} "
                f"{metrics['Precision']:>8.4f} "
                f"{metrics['Recall']:>8.4f} "
                f"{metrics['F1']:>8.4f} "
                f"{metrics['ROC_AUC']:>10.4f}"
            )
            
            # Highlight top 3
            if i == 0:
                print(f'🥇 {row}')
            elif i == 1:
                print(f'🥈 {row}')
            elif i == 2:
                print(f'🥉 {row}')
            else:
                print(f'   {row}')
        
        print('=' * 80)
        
        # Best model
        best_model_name = sorted_results[0][0]
        best_metrics = sorted_results[0][1]
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   ROC-AUC: {best_metrics['ROC_AUC']:.4f}")
        print(f"   F1-Score: {best_metrics['F1']:.4f}")
        print(f"   Accuracy: {best_metrics['Accuracy']:.4f}")
        print(f"   Precision: {best_metrics['Precision']:.4f}")
        print(f"   Recall: {best_metrics['Recall']:.4f}")
        print()
    
    def save_results(self) -> None:
        """Save metrics, models and statistics."""
        if self.verbose:
            print('💾 STEP 4: Saving Results')
            print('=' * 70)
        
        # Save test metrics
        metrics_file = self.output_dir / 'metrics' / 'test_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.test_metrics, f, indent=2)
        
        if self.verbose:
            print(f"   ✅ Test metrics saved: {metrics_file}")
        
        # Save validation metrics
        val_metrics_file = self.output_dir / 'metrics' / 'validation_metrics.json'
        with open(val_metrics_file, 'w') as f:
            json.dump(self.val_metrics, f, indent=2)
        
        if self.verbose:
            print(f"   ✅ Validation metrics saved: {val_metrics_file}")
        
        # Save trained models
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            if model is not None:
                model_path = models_dir / f'{model_name}.pkl'
                joblib.dump(model, model_path)
                if self.verbose:
                    print(f"   💾 Model saved: {model_path.name}")
        
        # Save pipeline stats
        stats_file = self.output_dir / 'pipeline_stats.json'
        
        # Find best model based on TEST performance (consistent with displayed ranking)
        best_model_name = max(self.test_metrics.items(), key=lambda x: x[1]['ROC_AUC'])[0]
        best_test_metrics = self.test_metrics[best_model_name]
        
        # Save best model with name
        best_filename = f'{best_model_name}_best_model.pkl'
        best_model_path = models_dir / best_filename
        joblib.dump(self.trained_models[best_model_name], best_model_path)
        
        if self.verbose:
            print(f"   ⭐ Best model saved (selected by test): {best_model_name} → {best_filename}")
        
        self.stats['test_metrics_summary'] = {
            'best_model': best_model_name,
            'best_roc_auc': best_test_metrics['ROC_AUC'],
            'best_f1': best_test_metrics['F1'],
            'best_accuracy': best_test_metrics['Accuracy']
        }
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        if self.verbose:
            print(f"   ✅ Pipeline stats saved: {stats_file}")
            print('=' * 70)
            print()
    
    def run(self) -> Dict[str, Any]:
        """
        Executar pipeline completo.
        
        Returns:
            Dict com métricas de teste
        """
        if self.verbose:
            print('🚀 PIPELINE MULTI-MODELO DE CLASSIFICAÇÃO - DockTKinase')
            print('=' * 70)
            print()
        
        start_time = time.time()
        
        # Etapa 1: Carregar dados
        self.load_data()
        
        # Etapa 2: Treinar modelos
        self.train_models()
        
        # Etapa 3: Avaliar no teste
        self.evaluate_on_test()
        
        # Etapa 4: Salvar resultados
        self.save_results()
        
        # Resumo
        self.print_results_summary()
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f'✅ PIPELINE COMPLETO!')
            print(f'   Tempo total: {total_time:.2f}s ({total_time/60:.2f} min)')
            print(f'   Resultados salvos em: {self.output_dir}')
            print('=' * 70)
        
        return self.test_metrics


# Função de conveniência
def run_multi_model_classification(
    embeddings_path: str,
    labels_path: str,
    output_dir: str = 'results/classification_multi_model',
    models: Optional[List[str]] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Função de conveniência para executar pipeline completo.
    
    Args:
        embeddings_path: Caminho para embeddings
        labels_path: Caminho para labels
        output_dir: Diretório de saída
        models: Lista de modelos (None = todos)
        random_state: Seed
        verbose: Mostrar progresso
        
    Returns:
        Dict com métricas de teste
    """
    pipeline = MultiModelClassificationPipeline(
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        output_dir=output_dir,
        models_to_train=models,
        random_state=random_state,
        verbose=verbose
    )
    
    return pipeline.run()


if __name__ == '__main__':
    print("Pipeline Multi-Modelo de Classificação - DockTKinase")
    print("=" * 70)
    print("\nPara usar este módulo, importe-o:")
    print("\n  from classifier.multi_model_pipeline import MultiModelClassificationPipeline")
    print("\n  pipeline = MultiModelClassificationPipeline(")
    print("      embeddings_path='embeddings.npy',")
    print("      labels_path='labels.npy'")
    print("  )")
    print("  results = pipeline.run()")
    print("\nModelos disponíveis:")
    ClassificationModels.print_available_models()
