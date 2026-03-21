"""
Módulo de visualização para comparação de modelos de proteína.

Estrutura modular seguindo princípios SOLID, KISS e Clean Code.
Inclui módulos para benchmark de modelos de ML e DL.
"""

__version__ = "1.0.0"

from .metrics_loader import (
    load_all_metrics,
    load_classification_metrics,
    load_regression_metrics
)
from .plot_classification import plot_classification_metrics
from .plot_regression import plot_regression_metrics
from .plot_summary import create_summary_table
from .plot_heatmaps import plot_classification_heatmaps, plot_regression_heatmaps
from .plot_statistics import (
    plot_classification_boxplot,
    plot_regression_boxplot,
    create_ranking_table
)
from .config import ML_MODEL_ORDER, setup_plot_style

__all__ = [
    'load_all_metrics',
    'load_classification_metrics',
    'load_regression_metrics',
    'plot_classification_metrics',
    'plot_regression_metrics',
    'create_summary_table',
    'plot_classification_heatmaps',
    'plot_regression_heatmaps',
    'plot_classification_boxplot',
    'plot_regression_boxplot',
    'create_ranking_table',
    'ML_MODEL_ORDER',
    'setup_plot_style',
]
