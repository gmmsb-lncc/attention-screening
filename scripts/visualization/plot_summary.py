#!/usr/bin/env python3
"""
Módulo para criar tabelas de resumo.
"""

from typing import Dict
from matplotlib.axes import Axes


def create_summary_table(
    clf_val: Dict,
    clf_test: Dict,
    reg_val: Dict,
    reg_test: Dict,
    ax: Axes
):
    """
    Cria tabela com resumo dos melhores modelos.
    
    Args:
        clf_val: Métricas de classificação (validação)
        clf_test: Métricas de classificação (teste)
        reg_val: Métricas de regressão (validação)
        reg_test: Métricas de regressão (teste)
        ax: Axes do matplotlib
    """
    ax.axis('off')
    
    # Encontrar melhores modelos
    best_clf_val = max(clf_val.items(), 
                       key=lambda x: x[1].get('ROC_AUC', 0),
                       default=('N/A', {}))
    best_clf_test = max(clf_test.items(),
                        key=lambda x: x[1].get('ROC_AUC', 0),
                        default=('N/A', {}))
    best_reg_val = max(reg_val.items(),
                       key=lambda x: x[1].get('R2', -999),
                       default=('N/A', {}))
    best_reg_test = max(reg_test.items(),
                        key=lambda x: x[1].get('R2', -999),
                        default=('N/A', {}))
    
    # Dados da tabela
    table_data = [
        ['Task', 'Split', 'Best Model', 'Primary Metric'],
        ['Classification', 'Validation', best_clf_val[0],
         f"ROC-AUC: {best_clf_val[1].get('ROC_AUC', 0):.4f}"],
        ['Classification', 'Test', best_clf_test[0],
         f"ROC-AUC: {best_clf_test[1].get('ROC_AUC', 0):.4f}"],
        ['Regression', 'Validation', best_reg_val[0],
         f"R²: {best_reg_val[1].get('R2', 0):.4f}"],
        ['Regression', 'Test', best_reg_test[0],
         f"R²: {best_reg_test[1].get('R2', 0):.4f}"],
    ]
    
    # Criar tabela
    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                     bbox=[0.05, 0.1, 0.9, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Colorir header
    for i in range(4):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorir linhas alternadas
    for i in range(1, 5):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax.set_title('Summary: Best Models per Split',
                 fontweight='bold', fontsize=12, pad=10)
