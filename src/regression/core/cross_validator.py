#!/usr/bin/env python3
"""
Cross-Validation para Regressão - DockTKinase
==============================================

Implementa K-Fold cross-validation estratificado para modelos de regressão,
seguindo o padrão do módulo classifier.

Autor: DockTKinase Team
Data: 2025-11-10
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Imports locais
try:
    from ..models.models import RegressionModels
    from ..utils.metrics import MetricsCalculator
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.models import RegressionModels
    from utils.metrics import MetricsCalculator


@dataclass
class CrossValidationConfig:
    """
    Configuração para cross-validation.
    
    Attributes:
        n_splits: Número de folds (default: 5)
        shuffle: Embaralhar dados antes de dividir
        random_state: Seed para reprodutibilidade
        verbose: Mostrar progresso
    """
    n_splits: int = 5
    shuffle: bool = True
    random_state: Optional[int] = 42
    verbose: bool = True


@dataclass
class FoldMetrics:
    """
    Métricas de um fold individual.
    
    Attributes:
        fold_idx: Índice do fold
        train_metrics: Métricas no treino
        val_metrics: Métricas na validação
        model_name: Nome do modelo
    """
    fold_idx: int
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    model_name: str


@dataclass
class CrossValidationResults:
    """
    Resultados completos de cross-validation.
    
    Attributes:
        model_name: Nome do modelo
        fold_metrics: Lista de métricas por fold
        summary_statistics: Estatísticas agregadas (mean, std, min, max)
        best_fold: Índice do melhor fold
        config: Configuração usada
    """
    model_name: str
    fold_metrics: List[FoldMetrics] = field(default_factory=list)
    summary_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_fold: int = 0
    config: Optional[CrossValidationConfig] = None
    
    def get_mean_metric(self, metric_name: str) -> float:
        """Obtém média de uma métrica."""
        return self.summary_statistics.get(metric_name, {}).get('mean', 0.0)
    
    def get_std_metric(self, metric_name: str) -> float:
        """Obtém desvio padrão de uma métrica."""
        return self.summary_statistics.get(metric_name, {}).get('std', 0.0)


class RegressionCrossValidator:
    """
    Cross-validator para modelos de regressão.
    
    Implementa K-Fold cross-validation com suporte para múltiplos modelos
    e cálculo de estatísticas agregadas.
    
    Example:
        >>> cv = RegressionCrossValidator(n_splits=5)
        >>> results = cv.cross_validate(X, y, models_dict)
        >>> print(f"Mean MAE: {results['Ridge'].get_mean_metric('mae'):.2f}")
    """
    
    def __init__(
        self,
        config: Optional[CrossValidationConfig] = None,
        verbose: bool = True
    ):
        """
        Inicializar cross-validator.
        
        Args:
            config: Configuração de CV (usa default se None)
            verbose: Mostrar progresso
        """
        self.config = config or CrossValidationConfig()
        self.verbose = verbose or self.config.verbose
        self.metrics_calculator = MetricsCalculator()
        self.results: Dict[str, CrossValidationResults] = {}
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models_dict: Optional[Dict[str, Any]] = None,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, CrossValidationResults]:
        """
        Executar K-Fold cross-validation.
        
        Args:
            X: Features (n_samples, n_features)
            y: Targets (n_samples,)
            models_dict: Dict com modelos {nome: modelo}
                        Se None, usa todos os modelos disponíveis
            model_names: Lista de nomes de modelos para testar
                        Se None e models_dict None, usa todos
        
        Returns:
            Dict com resultados por modelo {nome: CrossValidationResults}
        """
        # Obter modelos
        if models_dict is None:
            all_models = RegressionModels.get_all_models(
                random_state=self.config.random_state
            )
            if model_names:
                models_dict = {k: v for k, v in all_models.items() if k in model_names}
            else:
                models_dict = all_models
        
        if self.verbose:
            print('\n' + '=' * 70)
            print('🔄 CROSS-VALIDATION DE REGRESSÃO')
            print('=' * 70)
            print(f'   Amostras: {len(X):,}')
            print(f'   Features: {X.shape[1]}')
            print(f'   Folds: {self.config.n_splits}')
            print(f'   Modelos: {len(models_dict)}')
            print('=' * 70)
            print()
        
        # Criar KFold
        kfold = KFold(
            n_splits=self.config.n_splits,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state
        )
        
        # Cross-validate cada modelo
        for model_name, model in models_dict.items():
            if self.verbose:
                print(f'📊 Validando {model_name}...')
            
            cv_results = CrossValidationResults(
                model_name=model_name,
                config=self.config
            )
            
            # Iterar pelos folds
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Treinar modelo (clone para cada fold)
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)
                
                # Predições
                y_train_pred = fold_model.predict(X_train)
                y_val_pred = fold_model.predict(X_val)
                
                # Calcular métricas
                train_metrics = self._compute_metrics(y_train, y_train_pred)
                val_metrics = self._compute_metrics(y_val, y_val_pred)
                
                # Armazenar
                fold_metrics = FoldMetrics(
                    fold_idx=fold_idx,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    model_name=model_name
                )
                cv_results.fold_metrics.append(fold_metrics)
                
                if self.verbose:
                    print(f'   Fold {fold_idx + 1}/{self.config.n_splits} - '
                          f'Val MAE: {val_metrics["mae"]:.2f} | '
                          f'Val R²: {val_metrics["r2"]:.4f}')
            
            # Calcular estatísticas agregadas
            cv_results.summary_statistics = self._compute_summary_statistics(
                cv_results.fold_metrics
            )
            
            # Identificar melhor fold (menor MAE)
            best_fold_idx = np.argmin([
                fm.val_metrics['mae'] for fm in cv_results.fold_metrics
            ])
            cv_results.best_fold = best_fold_idx
            
            # Armazenar resultados
            self.results[model_name] = cv_results
            
            if self.verbose:
                print(f'   ✅ {model_name} completo')
                print(f'      Mean MAE: {cv_results.get_mean_metric("mae"):.2f} '
                      f'± {cv_results.get_std_metric("mae"):.2f}')
                print(f'      Mean R²: {cv_results.get_mean_metric("r2"):.4f} '
                      f'± {cv_results.get_std_metric("r2"):.4f}')
                print()
        
        if self.verbose:
            self._print_summary()
        
        return self.results
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcular métricas de regressão.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
        
        Returns:
            Dict com métricas
        """
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
    
    def _compute_summary_statistics(
        self,
        fold_metrics: List[FoldMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcular estatísticas agregadas dos folds.
        
        Args:
            fold_metrics: Lista de métricas por fold
        
        Returns:
            Dict com estatísticas {métrica: {mean, std, min, max}}
        """
        # Coletar métricas de validação de todos os folds
        metrics_dict = {}
        for metric_name in ['mae', 'rmse', 'r2', 'mse']:
            values = [fm.val_metrics[metric_name] for fm in fold_metrics]
            metrics_dict[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return metrics_dict
    
    def _print_summary(self):
        """Imprimir resumo dos resultados."""
        print('=' * 70)
        print('📊 RESUMO DO CROSS-VALIDATION')
        print('=' * 70)
        
        # Ordenar por MAE médio
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get_mean_metric('mae')
        )
        
        print(f'\n{"Modelo":<20} {"MAE (mean±std)":<20} {"R² (mean±std)":<20}')
        print('-' * 70)
        
        for model_name, results in sorted_results:
            mae_mean = results.get_mean_metric('mae')
            mae_std = results.get_std_metric('mae')
            r2_mean = results.get_mean_metric('r2')
            r2_std = results.get_std_metric('r2')
            
            print(f'{model_name:<20} '
                  f'{mae_mean:>7.2f} ± {mae_std:>5.2f}     '
                  f'{r2_mean:>6.4f} ± {r2_std:>5.4f}')
        
        print('=' * 70)
        print()
    
    def get_best_model(self, metric: str = 'mae') -> str:
        """
        Obter nome do melhor modelo baseado em uma métrica.
        
        Args:
            metric: Nome da métrica ('mae', 'rmse', 'r2')
        
        Returns:
            Nome do melhor modelo
        """
        if not self.results:
            raise ValueError("Nenhum resultado disponível. Execute cross_validate primeiro.")
        
        if metric in ['mae', 'rmse', 'mse']:
            # Menor é melhor
            best_model = min(
                self.results.items(),
                key=lambda x: x[1].get_mean_metric(metric)
            )
        else:  # r2
            # Maior é melhor
            best_model = max(
                self.results.items(),
                key=lambda x: x[1].get_mean_metric(metric)
            )
        
        return best_model[0]
    
    def compare_models(self) -> pd.DataFrame:
        """
        Comparar todos os modelos em um DataFrame.
        
        Returns:
            DataFrame com comparação
        """
        if not self.results:
            raise ValueError("Nenhum resultado disponível. Execute cross_validate primeiro.")
        
        data = []
        for model_name, results in self.results.items():
            row = {
                'model': model_name,
                'mae_mean': results.get_mean_metric('mae'),
                'mae_std': results.get_std_metric('mae'),
                'rmse_mean': results.get_mean_metric('rmse'),
                'rmse_std': results.get_std_metric('rmse'),
                'r2_mean': results.get_mean_metric('r2'),
                'r2_std': results.get_std_metric('r2'),
                'best_fold': results.best_fold
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('mae_mean')  # Ordenar por MAE
        
        return df


def quick_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_names: Optional[List[str]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, CrossValidationResults]:
    """
    Função de conveniência para CV rápido.
    
    Args:
        X: Features
        y: Targets
        model_names: Lista de modelos para testar (None = todos)
        n_splits: Número de folds
        random_state: Seed
        verbose: Mostrar progresso
    
    Returns:
        Dict com resultados por modelo
    
    Example:
        >>> results = quick_cross_validate(X, y, model_names=['Ridge', 'Lasso'])
        >>> best = min(results.items(), key=lambda x: x[1].get_mean_metric('mae'))
        >>> print(f"Best: {best[0]}")
    """
    config = CrossValidationConfig(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
        verbose=verbose
    )
    
    cv = RegressionCrossValidator(config=config, verbose=verbose)
    return cv.cross_validate(X, y, model_names=model_names)


if __name__ == '__main__':
    # Teste básico
    print("RegressionCrossValidator - DockTKinase")
    print("=" * 60)
    
    # Dados sintéticos
    np.random.seed(42)
    X = np.random.randn(200, 20)
    y = np.random.randn(200) * 100 + 200
    
    # CV rápido com 3 modelos
    results = quick_cross_validate(
        X, y,
        model_names=['Ridge', 'Lasso', 'RandomForest'],
        n_splits=5
    )
    
    # Comparação
    cv = RegressionCrossValidator()
    cv.results = results
    df = cv.compare_models()
    print("\nComparação:")
    print(df.to_string(index=False))
    
    # Melhor modelo
    best = cv.get_best_model('mae')
    print(f"\nMelhor modelo (MAE): {best}")
