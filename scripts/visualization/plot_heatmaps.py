#!/usr/bin/env python3
"""
Módulo para criar heatmaps comparativos entre modelos de proteína e ML models.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List
from matplotlib.axes import Axes

from .config import ML_MODEL_ORDER


def create_heatmap(
    ax: Axes,
    data: pd.DataFrame,
    title: str,
    cmap: str = 'RdYlGn',
    vmin: float = None,
    vmax: float = None,
    fmt: str = '.3f',
    annot_kws: dict = None
):
    """
    Cria um heatmap com formatação padrão.
    
    Args:
        ax: Axes do matplotlib
        data: DataFrame com dados (rows=ML models, cols=protein models)
        title: Título do heatmap
        cmap: Colormap
        vmin: Valor mínimo da escala
        vmax: Valor máximo da escala
        fmt: Formato dos números
        annot_kws: Configurações de anotação
    """
    if annot_kws is None:
        annot_kws = {'size': 8}
    
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=fmt,
        cbar_kws={'label': title},
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor='gray',
        annot_kws=annot_kws
    )
    
    ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
    ax.set_xlabel('Protein Model', fontweight='bold')
    ax.set_ylabel('ML Model', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)


def prepare_heatmap_data(
    all_metrics: Dict[str, Dict],
    metric_name: str,
    protein_models: List[str]
) -> pd.DataFrame:
    """
    Prepara dados para heatmap.
    
    Args:
        all_metrics: Dict com métricas {protein_model: {ml_model: metrics}}
        metric_name: Nome da métrica a extrair
        protein_models: Lista de modelos de proteína
        
    Returns:
        DataFrame com ML models nas linhas e protein models nas colunas
    """
    data = []
    
    for ml_model in ML_MODEL_ORDER:
        row = []
        for protein_model in protein_models:
            value = all_metrics.get(protein_model, {}).get(ml_model, {}).get(metric_name, np.nan)
            row.append(value)
        data.append(row)
    
    df = pd.DataFrame(data, index=ML_MODEL_ORDER, columns=protein_models)
    
    # Remover linhas/colunas totalmente vazias
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)
    
    return df


def plot_classification_heatmaps(
    all_clf_test: Dict[str, Dict],
    protein_models: List[str],
    ax_roc: Axes,
    ax_f1: Axes
):
    """
    Plota heatmaps de métricas de classificação.
    
    Args:
        all_clf_test: Métricas de teste {protein_model: {ml_model: metrics}}
        protein_models: Lista de modelos de proteína
        ax_roc: Axes para ROC-AUC
        ax_f1: Axes para F1-Score
    """
    # ROC-AUC
    roc_data = prepare_heatmap_data(all_clf_test, 'ROC_AUC', protein_models)
    create_heatmap(
        ax_roc,
        roc_data,
        'Classification: ROC-AUC (Test)',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0
    )
    
    # F1-Score
    f1_data = prepare_heatmap_data(all_clf_test, 'F1', protein_models)
    create_heatmap(
        ax_f1,
        f1_data,
        'Classification: F1-Score (Test)',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0
    )


def plot_regression_heatmaps(
    all_reg_test: Dict[str, Dict],
    protein_models: List[str],
    ax_mae: Axes,
    ax_r2: Axes
):
    """
    Plota heatmaps de métricas de regressão.
    
    Args:
        all_reg_test: Métricas de teste {protein_model: {ml_model: metrics}}
        protein_models: Lista de modelos de proteína
        ax_mae: Axes para MAE
        ax_r2: Axes para R²
    """
    # MAE (inverted colormap - menor é melhor)
    mae_data = prepare_heatmap_data(all_reg_test, 'MAE', protein_models)
    create_heatmap(
        ax_mae,
        mae_data,
        'Regression: MAE (Test)',
        cmap='RdYlGn_r',  # Invertido
        vmin=mae_data.min().min() * 0.9,
        vmax=mae_data.max().max() * 1.1
    )
    
    # R²
    r2_data = prepare_heatmap_data(all_reg_test, 'R2', protein_models)
    create_heatmap(
        ax_r2,
        r2_data,
        'Regression: R² Score (Test)',
        cmap='RdYlGn',
        vmin=-0.5,
        vmax=1.0
    )
