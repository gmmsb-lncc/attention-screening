#!/usr/bin/env python3
"""
Calculador de Métricas para Regressão - DockTKinase
====================================================

Implementa cálculo completo de métricas de regressão,
seguindo o padrão modular do classificador.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error
)


class MetricsCalculator:
    """
    Calculador de métricas de regressão.
    
    Implementa cálculo de 15+ métricas diferentes para avaliação
    completa de modelos de regressão.
    """
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'Model'
    ) -> Dict[str, Any]:
        """
        Calcula todas as métricas de regressão.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            model_name: Nome do modelo
            
        Returns:
            Dict com todas as métricas calculadas
        """
        # Garantir arrays numpy
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Validar tamanhos
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Tamanhos incompatíveis: y_true={len(y_true)}, y_pred={len(y_pred)}"
            )
        
        # Métricas básicas
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        median_ae = median_absolute_error(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)
        max_err = max_error(y_true, y_pred)
        
        # MAPE (cuidado com divisão por zero)
        mape = MetricsCalculator._safe_mape(y_true, y_pred)
        
        # Resíduos
        residuals = y_true - y_pred
        mean_residual = float(np.mean(residuals))
        std_residual = float(np.std(residuals))
        
        # Percentis de erro absoluto
        abs_residuals = np.abs(residuals)
        percentiles = {
            'error_p25': float(np.percentile(abs_residuals, 25)),
            'error_p50': float(np.percentile(abs_residuals, 50)),
            'error_p75': float(np.percentile(abs_residuals, 75)),
            'error_p90': float(np.percentile(abs_residuals, 90)),
            'error_p95': float(np.percentile(abs_residuals, 95)),
            'error_p99': float(np.percentile(abs_residuals, 99))
        }
        
        # Métricas derivadas
        rmse_normalized = rmse / (np.max(y_true) - np.min(y_true)) if np.max(y_true) != np.min(y_true) else 0
        cv_rmse = rmse / np.mean(y_true) if np.mean(y_true) != 0 else float('inf')
        
        # Construir dicionário de métricas
        metrics = {
            'model_name': model_name,
            'n_samples': int(len(y_true)),
            
            # Métricas principais
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'R2': float(r2),
            'MedianAE': float(median_ae),
            'MAPE': float(mape) if mape is not None else None,
            'ExplainedVariance': float(explained_var),
            'MaxError': float(max_err),
            
            # Estatísticas dos resíduos
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            
            # Métricas normalizadas
            'RMSE_normalized': float(rmse_normalized),
            'CV_RMSE': float(cv_rmse) if not np.isinf(cv_rmse) else None,
            
            # Percentis de erro
            **percentiles,
            
            # Estatísticas dos targets
            'target_mean': float(np.mean(y_true)),
            'target_std': float(np.std(y_true)),
            'target_min': float(np.min(y_true)),
            'target_max': float(np.max(y_true)),
            
            # Estatísticas das predições
            'pred_mean': float(np.mean(y_pred)),
            'pred_std': float(np.std(y_pred)),
            'pred_min': float(np.min(y_pred)),
            'pred_max': float(np.max(y_pred))
        }
        
        return metrics
    
    @staticmethod
    def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        """
        Calcula MAPE de forma segura, lidando com zeros.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            
        Returns:
            MAPE ou None se não puder ser calculado
        """
        try:
            return mean_absolute_percentage_error(y_true, y_pred) * 100
        except (ValueError, ZeroDivisionError):
            # Se y_true tem zeros, calcular manualmente excluindo-os
            mask = y_true != 0
            if mask.sum() > 0:
                return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            else:
                return None
    
    @staticmethod
    def format_metrics_table(metrics: Dict[str, Any]) -> str:
        """
        Formata métricas em tabela legível.
        
        Args:
            metrics: Dicionário de métricas
            
        Returns:
            String formatada
        """
        lines = []
        lines.append(f"\n{'=' * 60}")
        lines.append(f"  MÉTRICAS: {metrics.get('model_name', 'Unknown')}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Samples: {metrics['n_samples']:,}")
        lines.append(f"{'-' * 60}")
        lines.append(f"  MAE:     {metrics['MAE']:.4f}")
        lines.append(f"  RMSE:    {metrics['RMSE']:.4f}")
        lines.append(f"  R²:      {metrics['R2']:.4f}")
        lines.append(f"  MedianAE: {metrics['MedianAE']:.4f}")
        if metrics['MAPE'] is not None:
            lines.append(f"  MAPE:    {metrics['MAPE']:.2f}%")
        lines.append(f"{'-' * 60}")
        lines.append(f"  Mean Residual: {metrics['mean_residual']:+.4f}")
        lines.append(f"  Std Residual:  {metrics['std_residual']:.4f}")
        lines.append(f"  Max Error:     {metrics['MaxError']:.4f}")
        lines.append(f"{'=' * 60}\n")
        
        return '\n'.join(lines)
    
    @staticmethod
    def compare_models(metrics_list: list) -> str:
        """
        Compara métricas de múltiplos modelos.
        
        Args:
            metrics_list: Lista de dicionários de métricas
            
        Returns:
            String com tabela comparativa
        """
        if not metrics_list:
            return "Nenhuma métrica para comparar"
        
        lines = []
        lines.append(f"\n{'=' * 80}")
        lines.append(f"  COMPARAÇÃO DE MODELOS")
        lines.append(f"{'=' * 80}")
        
        # Cabeçalho
        header = f"{'Modelo':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'MedianAE':>10}"
        lines.append(header)
        lines.append(f"{'-' * 80}")
        
        # Dados
        for metrics in sorted(metrics_list, key=lambda x: x['MAE']):
            row = (
                f"{metrics['model_name']:<20} "
                f"{metrics['MAE']:>10.4f} "
                f"{metrics['RMSE']:>10.4f} "
                f"{metrics['R2']:>10.4f} "
                f"{metrics['MedianAE']:>10.4f}"
            )
            lines.append(row)
        
        lines.append(f"{'=' * 80}\n")
        
        return '\n'.join(lines)
    
    @staticmethod
    def get_best_model(
        metrics_list: list,
        metric: str = 'MAE',
        minimize: bool = True
    ) -> Dict[str, Any]:
        """
        Encontra o melhor modelo baseado em uma métrica.
        
        Args:
            metrics_list: Lista de métricas
            metric: Nome da métrica para comparação
            minimize: True para minimizar, False para maximizar
            
        Returns:
            Métricas do melhor modelo
        """
        if not metrics_list:
            raise ValueError("Lista de métricas vazia")
        
        if metric not in metrics_list[0]:
            raise ValueError(f"Métrica '{metric}' não encontrada")
        
        if minimize:
            return min(metrics_list, key=lambda x: x[metric])
        else:
            return max(metrics_list, key=lambda x: x[metric])


# Funções de conveniência
def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model'
) -> Dict[str, Any]:
    """
    Função de conveniência para calcular métricas.
    
    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        model_name: Nome do modelo
        
    Returns:
        Dict com métricas
    """
    calculator = MetricsCalculator()
    return calculator.calculate_all_metrics(y_true, y_pred, model_name)


if __name__ == '__main__':
    # Teste básico
    print("MetricsCalculator - DockTKinase")
    print("=" * 60)
    
    # Gerar dados de exemplo
    np.random.seed(42)
    y_true = np.random.randn(100) * 10 + 50
    y_pred = y_true + np.random.randn(100) * 2
    
    # Calcular métricas
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(y_true, y_pred, 'ExampleModel')
    
    # Exibir
    print(calculator.format_metrics_table(metrics))
