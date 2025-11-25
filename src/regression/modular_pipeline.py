#!/usr/bin/env python3
"""
Pipeline Modular de Regressão - DockTKinase
============================================

Pipeline completo de regressão modularizado seguindo o padrão
do classificador modular.

Esta implementação mantém EXATAMENTE a mesma funcionalidade do
pipeline original, mas de forma modularizada e organizada.
"""

import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Import SplitIndices for external stratification
try:
    from src.build.pipeline.split_indices import SplitIndices
except ImportError:
    SplitIndices = None  # Fallback if not available

# Imports dos módulos modularizados
try:
    from .core import RegressionEvaluator, DataManager, RegressionTrainer
    from .utils import MetricsCalculator
    from .models.models import RegressionModels  # UPDATED: usar models/models.py
except ImportError:
    # Fallback para execução direta
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    from core import RegressionEvaluator, DataManager, RegressionTrainer
    from utils import MetricsCalculator
    from models.models import RegressionModels  # UPDATED: usar models/models.py


class RegressionPipeline:
    """
    Pipeline modular de regressão para predição de atividade.
    
    Implementa pipeline completo seguindo o padrão do classificador:
    1. Carregar dados (embeddings + targets)
    2. Dividir em treino/validação/teste
    3. Treinar múltiplos modelos
    4. Avaliar e comparar resultados
    5. Salvar métricas e predições
    
    Mantém compatibilidade 100% com pipeline original.
    """
    
    def __init__(
        self,
        embeddings_path: str,
        targets_path: str,
        output_dir: str = 'results/regression',
        models_to_train: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = True,
        split_indices: Optional['SplitIndices'] = None
    ):
        """
        Inicializar pipeline de regressão.
        
        Args:
            embeddings_path: Caminho para embeddings (.npy ou .npz)
            targets_path: Caminho para targets de regressão (.npy)
            output_dir: Diretório para salvar resultados
            models_to_train: Lista de modelos a treinar (None = todos)
            test_size: Proporção do conjunto de teste (0.2 = 20%)
            val_size: Proporção do conjunto de validação (0.1 = 10%)
            random_state: Seed para reprodutibilidade
            verbose: Mostrar progresso
            split_indices: Optional SplitIndices object with pre-defined train/val/test splits.
                          If provided, these indices will be used instead of random splitting.
                          This ensures consistency with other pipelines (e.g., classification).
        """
        self.embeddings_path = embeddings_path
        self.targets_path = targets_path
        self.output_dir = Path(output_dir)
        self.models_to_train = models_to_train
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose
        self.split_indices = split_indices  # Store external splits if provided
        
        # Criar diretórios de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'predictions').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        
        # Componentes modularizados
        self.data_manager = DataManager(embeddings_path, targets_path)
        self.metrics_calculator = MetricsCalculator()
        self.evaluator = RegressionEvaluator()
        
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
            'pipeline': 'regression_modular',
            'timestamp': datetime.now().isoformat(),
            'random_state': random_state,
            'embeddings_path': str(embeddings_path),
            'targets_path': str(targets_path)
        }
        
    def load_data(self) -> None:
        """
        Carregar embeddings e targets, dividir em treino/val/teste.
        
        Usa stratified split baseado em bins quantílicos para
        manter distribuição similar em todos os conjuntos.
        
        **NEW**: If split_indices was provided during initialization, those indices
        will be used instead of automatic splitting. This ensures consistent splits
        across classification and regression pipelines.
        """
        if self.verbose:
            print('📊 ETAPA 1: Carregamento e Divisão de Dados')
            print('=' * 70)
        
        start_time = time.time()
        
        # Check if external split indices are provided
        if self.split_indices is not None:
            # Use external splits - load data and apply indices
            if self.verbose:
                print("   📌 Using external split indices (from stratification)")
            
            # Load full data first
            X, y = self.data_manager.load_data()
            
            # Apply external indices
            self.X_train = X[self.split_indices.train_idx]
            self.X_val = X[self.split_indices.val_idx]
            self.X_test = X[self.split_indices.test_idx]
            self.y_train = y[self.split_indices.train_idx]
            self.y_val = y[self.split_indices.val_idx]
            self.y_test = y[self.split_indices.test_idx]
        else:
            # Use automatic splitting (default behavior)
            self.X_train, self.X_val, self.X_test, \
            self.y_train, self.y_val, self.y_test = self.data_manager.split_data(
                test_size=self.test_size,
                val_size=self.val_size,
                random_state=self.random_state
            )
        
        # Estatísticas
        stats = self.data_manager.get_stats()
        
        if self.verbose:
            print(f"✅ Dados carregados com sucesso!")
            print(f"   Total de amostras: {stats['n_samples']:,}")
            print(f"   Dimensão embeddings: {stats['embedding_dim']:,}")
            print(f"   Treino: {len(self.X_train):,} amostras")
            print(f"   Validação: {len(self.X_val):,} amostras")
            print(f"   Teste: {len(self.X_test):,} amostras")
            print(f"\n   Target (nM):")
            print(f"      Média: {stats['target_mean']:.2f}")
            print(f"      Std: {stats['target_std']:.2f}")
            print(f"      Min: {stats['target_min']:.2f}")
            print(f"      Max: {stats['target_max']:.2f}")
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
            'target_stats': {
                'mean': stats['target_mean'],
                'std': stats['target_std'],
                'min': stats['target_min'],
                'max': stats['target_max']
            }
        })
    
    def train_models(self) -> Dict[str, Any]:
        """
        Treinar todos os modelos de regressão.
        
        Returns:
            Dict com métricas de validação de todos os modelos
        """
        if self.verbose:
            print('🤖 ETAPA 2: Treinamento de Modelos')
            print('=' * 70)
        
        # Obter modelos
        all_models = RegressionModels.get_all_models(
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
            print()
        
        # Criar trainer
        self.trainer = RegressionTrainer(
            models_dict=models,
            verbose=self.verbose,
            random_state=self.random_state
        )
        
        # Treinar todos
        start_time = time.time()
        self.trainer.train_all(self.X_train, self.y_train, self.X_val, self.y_val)
        training_time = time.time() - start_time
        
        # Armazenar resultados
        self.trained_models = self.trainer.trained_models
        self.train_metrics = self.trainer.train_results
        self.val_metrics = self.trainer.val_results
        
        if self.verbose:
            print(f"\n✅ Treinamento completo!")
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
        # Usar o método evaluate_on_test do trainer que já tem o print formatado correto
        if hasattr(self, 'trainer') and self.trainer:
            self.test_metrics = self.trainer.evaluate_on_test(self.X_test, self.y_test)
        else:
            # Fallback caso o trainer não esteja disponível
            if self.verbose:
                print('📈 ETAPA 3: Avaliação no Conjunto de Teste')
                print('=' * 70)
            
            for model_name, model in self.trained_models.items():
                if self.verbose:
                    print(f"   Avaliando {model_name}...")
                
                # Predições
                y_pred = model.predict(self.X_test)
                
                # Calcular métricas
                metrics = self.metrics_calculator.calculate_all_metrics(
                    self.y_test,
                    y_pred,
                    model_name
                )
                
                self.test_metrics[model_name] = metrics
            
            if self.verbose:
                print("\n✅ Avaliação no conjunto de teste completa!")
                print('=' * 70)
                print()
        
        return self.test_metrics
    
    def print_results_summary(self) -> None:
        """Imprimir resumo dos resultados."""
        if not self.test_metrics:
            print("⚠️  Nenhum resultado de teste disponível")
            return
        
        print('📊 RESUMO DOS RESULTADOS (Conjunto de Teste)')
        print('=' * 80)
        
        # Ordenar por MAE
        sorted_results = sorted(
            self.test_metrics.items(),
            key=lambda x: x[1]['MAE']
        )
        
        # Cabeçalho
        header = f"{'Modelo':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'MedianAE':>10}"
        print(header)
        print('-' * 80)
        
        # Resultados
        for model_name, metrics in sorted_results:
            row = (
                f"{model_name:<20} "
                f"{metrics['MAE']:>10.4f} "
                f"{metrics['RMSE']:>10.4f} "
                f"{metrics['R2']:>10.4f} "
                f"{metrics['MedianAE']:>10.4f}"
            )
            print(row)
        
        print('=' * 80)
        
        # Melhor modelo
        best_model_name = sorted_results[0][0]
        best_metrics = sorted_results[0][1]
        
        print(f"\n🏆 MELHOR MODELO: {best_model_name}")
        print(f"   MAE: {best_metrics['MAE']:.4f} nM")
        print(f"   RMSE: {best_metrics['RMSE']:.4f} nM")
        print(f"   R²: {best_metrics['R2']:.4f}")
        print()
    
    def save_results(self) -> None:
        """Salvar métricas e estatísticas em arquivos JSON."""
        if self.verbose:
            print('💾 ETAPA 4: Salvando Resultados')
            print('=' * 70)
        
        # Salvar métricas de teste
        metrics_file = self.output_dir / 'metrics' / 'test_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.test_metrics, f, indent=2)
        
        if self.verbose:
            print(f"   ✅ Métricas salvas: {metrics_file}")
        
        # Salvar métricas de validação
        val_metrics_file = self.output_dir / 'metrics' / 'validation_metrics.json'
        with open(val_metrics_file, 'w') as f:
            json.dump(self.val_metrics, f, indent=2)
        
        if self.verbose:
            print(f"   ✅ Métricas de validação salvas: {val_metrics_file}")
        
        # Salvar stats do pipeline
        stats_file = self.output_dir / 'pipeline_stats.json'
        self.stats['test_metrics_summary'] = {
            'best_model': min(self.test_metrics.items(), key=lambda x: x[1]['MAE'])[0],
            'best_mae': min(m['MAE'] for m in self.test_metrics.values()),
            'best_r2': max(m['R2'] for m in self.test_metrics.values())
        }
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        if self.verbose:
            print(f"   ✅ Stats do pipeline salvas: {stats_file}")
            print('=' * 70)
            print()
    
    def run(self) -> Dict[str, Any]:
        """
        Executar pipeline completo.
        
        Returns:
            Dict com métricas de teste
        """
        if self.verbose:
            print('🚀 PIPELINE MODULAR DE REGRESSÃO - DockTKinase')
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
def run_regression_pipeline(
    embeddings_path: str,
    targets_path: str,
    output_dir: str = 'results/regression',
    models: Optional[List[str]] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Função de conveniência para executar pipeline completo.
    
    Args:
        embeddings_path: Caminho para embeddings
        targets_path: Caminho para targets
        output_dir: Diretório de saída
        models: Lista de modelos (None = todos)
        random_state: Seed
        verbose: Mostrar progresso
        
    Returns:
        Dict com métricas de teste
    """
    pipeline = RegressionPipeline(
        embeddings_path=embeddings_path,
        targets_path=targets_path,
        output_dir=output_dir,
        models_to_train=models,
        random_state=random_state,
        verbose=verbose
    )
    
    return pipeline.run()


if __name__ == '__main__':
    print("Pipeline Modular de Regressão - DockTKinase")
    print("=" * 70)
    print("\nPara usar este módulo, importe-o:")
    print("\n  from regression.modular_pipeline import RegressionPipeline")
    print("\n  pipeline = RegressionPipeline(")
    print("      embeddings_path='embeddings.npy',")
    print("      targets_path='targets.npy'")
    print("  )")
    print("  results = pipeline.run()")
