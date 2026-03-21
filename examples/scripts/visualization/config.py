#!/usr/bin/env python3
"""
Configurações globais para visualizações.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Ordem dos modelos ML (do mais simples ao mais complexo)
ML_MODEL_ORDER = [
    'NaiveBayes', 'Ridge', 'Lasso', 'ElasticNet',
    'DecisionTree', 'LogisticRegression', 'LinearSVC', 'LinearSVR',
    'KNN', 'AdaBoost', 'GradientBoosting',
    'LightGBM', 'XGBoost', 'RandomForest', 'ExtraTrees', 'MLP'
]

# Cores para visualizações
COLORS = {
    'validation': '#3498db',  # Azul
    'test': '#e74c3c',        # Vermelho
    'regression_val': '#9b59b6',  # Roxo
    'regression_test': '#e67e22',  # Laranja
}


def setup_plot_style():
    """Configura estilo padrão para plots."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (20, 12)
    plt.rcParams['font.size'] = 10
