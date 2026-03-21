#!/usr/bin/env python3
"""
Módulo para criar visualizações estatísticas (boxplots, rankings).
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from matplotlib.axes import Axes

from .config import ML_MODEL_ORDER, COLORS


def plot_classification_boxplot(
    all_clf_test: Dict[str, Dict],
    protein_models: List[str],
    ax: Axes
):
    """
    Plota boxplot de ROC-AUC para classificação.
    
    Args:
        all_clf_test: Métricas de teste {protein_model: {ml_model: metrics}}
        protein_models: Lista de modelos de proteína
        ax: Axes do matplotlib
    """
    # Preparar dados
    data = []
    labels = []
    
    for ml_model in ML_MODEL_ORDER:
        values = []
        for protein_model in protein_models:
            value = all_clf_test.get(protein_model, {}).get(ml_model, {}).get('ROC_AUC', np.nan)
            if not np.isnan(value):
                values.append(value)
        
        if values:
            data.append(values)
            labels.append(ml_model)
    
    # Criar boxplot
    bp = ax.boxplot(data, labels=labels, vert=False, patch_artist=True)
    
    # Colorir boxes
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['validation'])
        patch.set_alpha(0.6)
    
    ax.set_xlabel('ROC-AUC', fontweight='bold')
    ax.set_ylabel('ML Model', fontweight='bold')
    ax.set_title('Classification: ROC-AUC Distribution Across Protein Models',
                 fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0.5, 1.0])
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.3, linewidth=2)


def plot_regression_boxplot(
    all_reg_test: Dict[str, Dict],
    protein_models: List[str],
    ax: Axes
):
    """
    Plota boxplot de R² para regressão.
    
    Args:
        all_reg_test: Métricas de teste {protein_model: {ml_model: metrics}}
        protein_models: Lista de modelos de proteína
        ax: Axes do matplotlib
    """
    # Preparar dados
    data = []
    labels = []
    
    for ml_model in ML_MODEL_ORDER:
        values = []
        for protein_model in protein_models:
            value = all_reg_test.get(protein_model, {}).get(ml_model, {}).get('R2', np.nan)
            if not np.isnan(value):
                values.append(value)
        
        if values:
            data.append(values)
            labels.append(ml_model)
    
    # Criar boxplot
    bp = ax.boxplot(data, labels=labels, vert=False, patch_artist=True)
    
    # Colorir boxes
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['regression_val'])
        patch.set_alpha(0.6)
    
    ax.set_xlabel('R² Score', fontweight='bold')
    ax.set_ylabel('ML Model', fontweight='bold')
    ax.set_title('Regression: R² Distribution Across Protein Models',
                 fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)


def create_ranking_table(
    all_clf_test: Dict[str, Dict],
    all_reg_test: Dict[str, Dict],
    protein_models: List[str],
    ax: Axes
):
    """
    Cria tabela com ranking dos top-3 ML models.
    
    Args:
        all_clf_test: Métricas de classificação
        all_reg_test: Métricas de regressão
        protein_models: Lista de modelos de proteína
        ax: Axes do matplotlib
    """
    ax.axis('off')
    
    # Calcular média de ROC-AUC por ML model
    clf_means = {}
    for ml_model in ML_MODEL_ORDER:
        values = [
            all_clf_test.get(p, {}).get(ml_model, {}).get('ROC_AUC', np.nan)
            for p in protein_models
        ]
        values = [v for v in values if not np.isnan(v)]
        if values:
            clf_means[ml_model] = np.mean(values)
    
    # Calcular média de R² por ML model
    reg_means = {}
    for ml_model in ML_MODEL_ORDER:
        values = [
            all_reg_test.get(p, {}).get(ml_model, {}).get('R2', np.nan)
            for p in protein_models
        ]
        values = [v for v in values if not np.isnan(v)]
        if values:
            reg_means[ml_model] = np.mean(values)
    
    # Top-3 classificação
    top_clf = sorted(clf_means.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Top-3 regressão
    top_reg = sorted(reg_means.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Criar tabela
    table_data = [
        ['Rank', 'Classification (ROC-AUC)', 'Mean', 'Regression (R²)', 'Mean'],
        ['1st', top_clf[0][0], f"{top_clf[0][1]:.4f}", 
         top_reg[0][0], f"{top_reg[0][1]:.4f}"],
        ['2nd', top_clf[1][0], f"{top_clf[1][1]:.4f}",
         top_reg[1][0], f"{top_reg[1][1]:.4f}"],
        ['3rd', top_clf[2][0], f"{top_clf[2][1]:.4f}",
         top_reg[2][0], f"{top_reg[2][1]:.4f}"],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     bbox=[0.05, 0.2, 0.9, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Colorir header
    for i in range(5):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorir medalhas
    medals = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze
    for i in range(1, 4):
        table[(i, 0)].set_facecolor(medals[i-1])
        table[(i, 0)].set_text_props(weight='bold')
    
    ax.set_title('Top-3 ML Models (Average Across All Protein Models)',
                 fontweight='bold', fontsize=12, pad=10)
