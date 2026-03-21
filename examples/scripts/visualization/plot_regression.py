#!/usr/bin/env python3
"""
Módulo para plotar métricas de regressão.
"""

import numpy as np
from typing import Dict
from matplotlib.axes import Axes

from .config import ML_MODEL_ORDER, COLORS


def _plot_regression_metric(
    ax: Axes,
    models: list,
    val_values: list,
    test_values: list,
    metric_name: str,
    xlabel: str,
    invert_x: bool = False
):
    """
    Plota comparação de uma métrica de regressão.
    
    Args:
        ax: Axes do matplotlib
        models: Lista de nomes dos modelos
        val_values: Valores de validação
        test_values: Valores de teste
        metric_name: Nome da métrica
        xlabel: Label do eixo X
        invert_x: Se True, inverte o eixo X (para métricas onde menor é melhor)
    """
    x = np.arange(len(models))
    width = 0.35
    
    ax.barh(x - width/2, val_values, width, label='Validation',
            color=COLORS['regression_val'], alpha=0.8)
    ax.barh(x + width/2, test_values, width, label='Test',
            color=COLORS['regression_test'], alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_title(f'Regression: {metric_name} (Val vs Test)',
                 fontweight='bold', fontsize=12)
    ax.legend(loc='upper right' if invert_x else 'lower right')
    ax.grid(axis='x', alpha=0.3)
    
    if invert_x:
        ax.invert_xaxis()


def _plot_generalization_scatter(
    ax: Axes,
    models: list,
    r2_val: list,
    r2_test: list
):
    """
    Plota scatter plot de R² validação vs teste.
    
    Args:
        ax: Axes do matplotlib
        models: Lista de nomes dos modelos
        r2_val: R² de validação
        r2_test: R² de teste
    """
    ax.scatter(r2_val, r2_test, s=100, alpha=0.7, 
               c=range(len(models)), cmap='viridis')
    
    # Linha diagonal (performance ideal: val = test)
    min_r2 = min(min(r2_val), min(r2_test))
    max_r2 = max(max(r2_val), max(r2_test))
    ax.plot([min_r2, max_r2], [min_r2, max_r2], 'k--', 
            alpha=0.3, linewidth=2, label='Perfect Generalization')
    
    # Anotar modelos com grande diferença
    for i, model in enumerate(models):
        if abs(r2_val[i] - r2_test[i]) > 0.05:
            ax.annotate(model, (r2_val[i], r2_test[i]),
                       fontsize=8, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Validation R²', fontweight='bold')
    ax.set_ylabel('Test R²', fontweight='bold')
    ax.set_title('Regression: Generalization (Val vs Test R²)',
                 fontweight='bold', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)


def plot_regression_metrics(
    reg_val: Dict,
    reg_test: Dict,
    ax_mae: Axes,
    ax_r2: Axes,
    ax_rmse: Axes,
    ax_scatter: Axes
):
    """
    Plota todas as métricas de regressão.
    
    Args:
        reg_val: Métricas de validação
        reg_test: Métricas de teste
        ax_mae: Axes para MAE
        ax_r2: Axes para R²
        ax_rmse: Axes para RMSE
        ax_scatter: Axes para scatter plot de generalização
    """
    # Filtrar modelos que existem em ambos val e test
    models = [m for m in ML_MODEL_ORDER if m in reg_val and m in reg_test]
    
    if not models:
        return
    
    # MAE
    mae_val = [reg_val[m].get('MAE', 0) for m in models]
    mae_test = [reg_test[m].get('MAE', 0) for m in models]
    _plot_regression_metric(ax_mae, models, mae_val, mae_test,
                           'MAE', 'MAE (lower is better)', invert_x=True)
    
    # R²
    r2_val = [reg_val[m].get('R2', 0) for m in models]
    r2_test = [reg_test[m].get('R2', 0) for m in models]
    _plot_regression_metric(ax_r2, models, r2_val, r2_test,
                           'R² Score', 'R² Score')
    
    # RMSE
    rmse_val = [reg_val[m].get('RMSE', 0) for m in models]
    rmse_test = [reg_test[m].get('RMSE', 0) for m in models]
    _plot_regression_metric(ax_rmse, models, rmse_val, rmse_test,
                           'RMSE', 'RMSE (lower is better)', invert_x=True)
    
    # Scatter plot de generalização
    _plot_generalization_scatter(ax_scatter, models, r2_val, r2_test)
