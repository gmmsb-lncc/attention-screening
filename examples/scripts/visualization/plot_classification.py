#!/usr/bin/env python3
"""
Módulo para plotar métricas de classificação.
"""

import numpy as np
from typing import Dict
from matplotlib.axes import Axes

from .config import ML_MODEL_ORDER, COLORS


def _plot_metric_comparison(
    ax: Axes,
    models: list,
    val_values: list,
    test_values: list,
    metric_name: str,
    xlabel: str,
    xlim: tuple = None,
    invert_x: bool = False
):
    """
    Plota comparação de uma métrica entre validação e teste.
    
    Args:
        ax: Axes do matplotlib
        models: Lista de nomes dos modelos
        val_values: Valores de validação
        test_values: Valores de teste
        metric_name: Nome da métrica para o título
        xlabel: Label do eixo X
        xlim: Limites do eixo X (opcional)
        invert_x: Se True, inverte o eixo X
    """
    x = np.arange(len(models))
    width = 0.35
    
    ax.barh(x - width/2, val_values, width, label='Validation', 
            color=COLORS['validation'], alpha=0.8)
    ax.barh(x + width/2, test_values, width, label='Test', 
            color=COLORS['test'], alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_title(f'Classification: {metric_name} (Val vs Test)', 
                 fontweight='bold', fontsize=12)
    ax.legend(loc='lower right' if not invert_x else 'upper right')
    ax.grid(axis='x', alpha=0.3)
    
    if xlim:
        ax.set_xlim(xlim)
        if xlim[0] < 1.0:
            ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    if invert_x:
        ax.invert_xaxis()


def plot_classification_metrics(
    clf_val: Dict,
    clf_test: Dict,
    ax_roc: Axes,
    ax_f1: Axes,
    ax_acc: Axes,
    ax_mcc: Axes
):
    """
    Plota todas as métricas de classificação.
    
    Args:
        clf_val: Métricas de validação
        clf_test: Métricas de teste
        ax_roc: Axes para ROC-AUC
        ax_f1: Axes para F1-Score
        ax_acc: Axes para Accuracy
        ax_mcc: Axes para MCC
    """
    # Filtrar modelos que existem em ambos val e test
    models = [m for m in ML_MODEL_ORDER if m in clf_val and m in clf_test]
    
    if not models:
        return
    
    # ROC-AUC
    roc_val = [clf_val[m].get('ROC_AUC', 0) for m in models]
    roc_test = [clf_test[m].get('ROC_AUC', 0) for m in models]
    _plot_metric_comparison(ax_roc, models, roc_val, roc_test, 
                           'ROC-AUC', 'ROC-AUC', xlim=(0.5, 1.0))
    
    # F1-Score
    f1_val = [clf_val[m].get('F1', 0) for m in models]
    f1_test = [clf_test[m].get('F1', 0) for m in models]
    _plot_metric_comparison(ax_f1, models, f1_val, f1_test,
                           'F1-Score', 'F1-Score', xlim=(0.5, 1.0))
    
    # Accuracy
    acc_val = [clf_val[m].get('Accuracy', 0) for m in models]
    acc_test = [clf_test[m].get('Accuracy', 0) for m in models]
    _plot_metric_comparison(ax_acc, models, acc_val, acc_test,
                           'Accuracy', 'Accuracy', xlim=(0.5, 1.0))
    
    # MCC
    mcc_val = [clf_val[m].get('MCC', 0) for m in models]
    mcc_test = [clf_test[m].get('MCC', 0) for m in models]
    _plot_metric_comparison(ax_mcc, models, mcc_val, mcc_test,
                           'MCC', 'MCC', xlim=(0.0, 1.0))
