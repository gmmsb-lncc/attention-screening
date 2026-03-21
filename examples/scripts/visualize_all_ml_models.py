#!/usr/bin/env python3
"""
Visualiza todos os 12 modelos ML (classificação e regressão) para um modelo de proteína.
Compara métricas de validação vs teste para detectar overfitting.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

# Ordem dos modelos ML (do mais simples ao mais complexo)
ML_MODEL_ORDER = [
    'NaiveBayes', 'Ridge', 'Lasso', 'ElasticNet',
    'DecisionTree', 'LogisticRegression', 'LinearSVC', 'LinearSVR',
    'KNN', 'AdaBoost', 'GradientBoosting',
    'LightGBM', 'XGBoost', 'RandomForest', 'ExtraTrees', 'MLP'
]

def load_metrics(model_name, base_path):
    """Carrega métricas de validação e teste para classificação e regressão."""
    
    # Classification
    clf_val_path = base_path / "classifier/metrics/validation_metrics.json"
    clf_test_path = base_path / "classifier/metrics/test_metrics.json"
    
    clf_val = {}
    clf_test = {}
    if clf_val_path.exists():
        with open(clf_val_path) as f:
            clf_val = json.load(f)
    if clf_test_path.exists():
        with open(clf_test_path) as f:
            clf_test = json.load(f)
    
    # Regression
    reg_val_path = base_path / "regression/metrics/validation_metrics.json"
    reg_test_path = base_path / "regression/metrics/test_metrics.json"
    
    reg_val = {}
    reg_test = {}
    if reg_val_path.exists():
        with open(reg_val_path) as f:
            reg_val = json.load(f)
    if reg_test_path.exists():
        with open(reg_test_path) as f:
            reg_test = json.load(f)
    
    return clf_val, clf_test, reg_val, reg_test

def plot_classification_metrics(clf_val, clf_test, ax_roc, ax_f1, ax_acc, ax_mcc):
    """Plota métricas de classificação (Val vs Test)."""
    
    # Filtrar modelos que existem em ambos val e test
    models = [m for m in ML_MODEL_ORDER if m in clf_val and m in clf_test]
    
    if not models:
        return
    
    # ROC-AUC
    roc_val = [clf_val[m].get('ROC_AUC', 0) for m in models]
    roc_test = [clf_test[m].get('ROC_AUC', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax_roc.barh(x - width/2, roc_val, width, label='Validation', color='#3498db', alpha=0.8)
    ax_roc.barh(x + width/2, roc_test, width, label='Test', color='#e74c3c', alpha=0.8)
    ax_roc.set_yticks(x)
    ax_roc.set_yticklabels(models, fontsize=9)
    ax_roc.set_xlabel('ROC-AUC', fontweight='bold')
    ax_roc.set_title('Classification: ROC-AUC (Val vs Test)', fontweight='bold', fontsize=12)
    ax_roc.legend(loc='lower right')
    ax_roc.set_xlim([0.5, 1.0])
    ax_roc.axvline(x=0.9, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax_roc.grid(axis='x', alpha=0.3)
    
    # F1-Score
    f1_val = [clf_val[m].get('F1', 0) for m in models]
    f1_test = [clf_test[m].get('F1', 0) for m in models]
    
    ax_f1.barh(x - width/2, f1_val, width, label='Validation', color='#3498db', alpha=0.8)
    ax_f1.barh(x + width/2, f1_test, width, label='Test', color='#e74c3c', alpha=0.8)
    ax_f1.set_yticks(x)
    ax_f1.set_yticklabels(models, fontsize=9)
    ax_f1.set_xlabel('F1-Score', fontweight='bold')
    ax_f1.set_title('Classification: F1-Score (Val vs Test)', fontweight='bold', fontsize=12)
    ax_f1.legend(loc='lower right')
    ax_f1.set_xlim([0.5, 1.0])
    ax_f1.grid(axis='x', alpha=0.3)
    
    # Accuracy
    acc_val = [clf_val[m].get('Accuracy', 0) for m in models]
    acc_test = [clf_test[m].get('Accuracy', 0) for m in models]
    
    ax_acc.barh(x - width/2, acc_val, width, label='Validation', color='#3498db', alpha=0.8)
    ax_acc.barh(x + width/2, acc_test, width, label='Test', color='#e74c3c', alpha=0.8)
    ax_acc.set_yticks(x)
    ax_acc.set_yticklabels(models, fontsize=9)
    ax_acc.set_xlabel('Accuracy', fontweight='bold')
    ax_acc.set_title('Classification: Accuracy (Val vs Test)', fontweight='bold', fontsize=12)
    ax_acc.legend(loc='lower right')
    ax_acc.set_xlim([0.5, 1.0])
    ax_acc.grid(axis='x', alpha=0.3)
    
    # MCC
    mcc_val = [clf_val[m].get('MCC', 0) for m in models]
    mcc_test = [clf_test[m].get('MCC', 0) for m in models]
    
    ax_mcc.barh(x - width/2, mcc_val, width, label='Validation', color='#3498db', alpha=0.8)
    ax_mcc.barh(x + width/2, mcc_test, width, label='Test', color='#e74c3c', alpha=0.8)
    ax_mcc.set_yticks(x)
    ax_mcc.set_yticklabels(models, fontsize=9)
    ax_mcc.set_xlabel('MCC', fontweight='bold')
    ax_mcc.set_title('Classification: MCC (Val vs Test)', fontweight='bold', fontsize=12)
    ax_mcc.legend(loc='lower right')
    ax_mcc.set_xlim([0.0, 1.0])
    ax_mcc.grid(axis='x', alpha=0.3)

def plot_regression_metrics(reg_val, reg_test, ax_mae, ax_r2, ax_rmse, ax_scatter):
    """Plota métricas de regressão (Val vs Test)."""
    
    # Filtrar modelos que existem em ambos val e test
    models = [m for m in ML_MODEL_ORDER if m in reg_val and m in reg_test]
    
    if not models:
        return
    
    # MAE
    mae_val = [reg_val[m].get('MAE', 0) for m in models]
    mae_test = [reg_test[m].get('MAE', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax_mae.barh(x - width/2, mae_val, width, label='Validation', color='#9b59b6', alpha=0.8)
    ax_mae.barh(x + width/2, mae_test, width, label='Test', color='#e67e22', alpha=0.8)
    ax_mae.set_yticks(x)
    ax_mae.set_yticklabels(models, fontsize=9)
    ax_mae.set_xlabel('MAE (lower is better)', fontweight='bold')
    ax_mae.set_title('Regression: MAE (Val vs Test)', fontweight='bold', fontsize=12)
    ax_mae.legend(loc='upper right')
    ax_mae.grid(axis='x', alpha=0.3)
    ax_mae.invert_xaxis()  # Inverter para que melhor (menor MAE) fique à direita
    
    # R²
    r2_val = [reg_val[m].get('R2', 0) for m in models]
    r2_test = [reg_test[m].get('R2', 0) for m in models]
    
    ax_r2.barh(x - width/2, r2_val, width, label='Validation', color='#9b59b6', alpha=0.8)
    ax_r2.barh(x + width/2, r2_test, width, label='Test', color='#e67e22', alpha=0.8)
    ax_r2.set_yticks(x)
    ax_r2.set_yticklabels(models, fontsize=9)
    ax_r2.set_xlabel('R² Score', fontweight='bold')
    ax_r2.set_title('Regression: R² Score (Val vs Test)', fontweight='bold', fontsize=12)
    ax_r2.legend(loc='lower right')
    ax_r2.grid(axis='x', alpha=0.3)
    
    # RMSE
    rmse_val = [reg_val[m].get('RMSE', 0) for m in models]
    rmse_test = [reg_test[m].get('RMSE', 0) for m in models]
    
    ax_rmse.barh(x - width/2, rmse_val, width, label='Validation', color='#9b59b6', alpha=0.8)
    ax_rmse.barh(x + width/2, rmse_test, width, label='Test', color='#e67e22', alpha=0.8)
    ax_rmse.set_yticks(x)
    ax_rmse.set_yticklabels(models, fontsize=9)
    ax_rmse.set_xlabel('RMSE (lower is better)', fontweight='bold')
    ax_rmse.set_title('Regression: RMSE (Val vs Test)', fontweight='bold', fontsize=12)
    ax_rmse.legend(loc='upper right')
    ax_rmse.grid(axis='x', alpha=0.3)
    ax_rmse.invert_xaxis()  # Inverter para que melhor (menor RMSE) fique à direita
    
    # Scatter: Val R² vs Test R² (detectar overfitting)
    ax_scatter.scatter(r2_val, r2_test, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
    
    # Linha diagonal (performance ideal: val = test)
    min_r2 = min(min(r2_val), min(r2_test))
    max_r2 = max(max(r2_val), max(r2_test))
    ax_scatter.plot([min_r2, max_r2], [min_r2, max_r2], 'k--', alpha=0.3, linewidth=2, label='Perfect Generalization')
    
    # Anotar modelos
    for i, model in enumerate(models):
        if abs(r2_val[i] - r2_test[i]) > 0.05:  # Anotar apenas modelos com grande diferença
            ax_scatter.annotate(model, (r2_val[i], r2_test[i]), 
                              fontsize=8, alpha=0.7,
                              xytext=(5, 5), textcoords='offset points')
    
    ax_scatter.set_xlabel('Validation R²', fontweight='bold')
    ax_scatter.set_ylabel('Test R²', fontweight='bold')
    ax_scatter.set_title('Regression: Generalization (Val vs Test R²)', fontweight='bold', fontsize=12)
    ax_scatter.legend(loc='lower right')
    ax_scatter.grid(alpha=0.3)

def create_summary_table(clf_val, clf_test, reg_val, reg_test, ax):
    """Cria tabela com resumo dos melhores modelos."""
    
    ax.axis('off')
    
    # Encontrar melhores modelos
    best_clf_val = max(clf_val.items(), key=lambda x: x[1].get('ROC_AUC', 0), default=('N/A', {}))
    best_clf_test = max(clf_test.items(), key=lambda x: x[1].get('ROC_AUC', 0), default=('N/A', {}))
    best_reg_val = max(reg_val.items(), key=lambda x: x[1].get('R2', -999), default=('N/A', {}))
    best_reg_test = max(reg_test.items(), key=lambda x: x[1].get('R2', -999), default=('N/A', {}))
    
    table_data = [
        ['Task', 'Split', 'Best Model', 'Primary Metric'],
        ['Classification', 'Validation', best_clf_val[0], f"ROC-AUC: {best_clf_val[1].get('ROC_AUC', 0):.4f}"],
        ['Classification', 'Test', best_clf_test[0], f"ROC-AUC: {best_clf_test[1].get('ROC_AUC', 0):.4f}"],
        ['Regression', 'Validation', best_reg_val[0], f"R²: {best_reg_val[1].get('R2', 0):.4f}"],
        ['Regression', 'Test', best_reg_test[0], f"R²: {best_reg_test[1].get('R2', 0):.4f}"],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                    bbox=[0.05, 0.1, 0.9, 0.7])  # [x, y, width, height] dentro do subplot
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
    
    ax.set_title('Summary: Best Models per Split', fontweight='bold', fontsize=12, pad=10)

def main():
    parser = argparse.ArgumentParser(description='Visualiza todos os 12 modelos ML para um modelo de proteína')
    parser.add_argument('--model', type=str, required=True,
                       help='Nome do modelo de proteína (ex: esmc-600m-2024-12)')
    parser.add_argument('--output-dir', type=str, 
                       default='results/benchmark_visualizations',
                       help='Diretório de saída para as visualizações')
    
    args = parser.parse_args()
    
    # Paths
    base_path = Path(f"results/protein_model_benchmark_non_human_v2/{args.model}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not base_path.exists():
        print(f"❌ Erro: Diretório não encontrado: {base_path}")
        return
    
    print(f"📊 Carregando métricas de: {args.model}")
    
    # Carregar métricas
    clf_val, clf_test, reg_val, reg_test = load_metrics(args.model, base_path)
    
    print(f"   ✅ Classification: {len(clf_val)} modelos (val), {len(clf_test)} modelos (test)")
    print(f"   ✅ Regression: {len(reg_val)} modelos (val), {len(reg_test)} modelos (test)")
    
    # Criar figura principal (3x3 grid)
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Classification plots (primeira linha)
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_f1 = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[0, 2])
    
    # MCC (segunda linha, primeira coluna)
    ax_mcc = fig.add_subplot(gs[1, 0])
    
    # Regression plots (segunda linha)
    ax_mae = fig.add_subplot(gs[1, 1])
    ax_r2 = fig.add_subplot(gs[1, 2])
    
    # RMSE e Scatter (terceira linha)
    ax_rmse = fig.add_subplot(gs[2, 0])
    ax_scatter = fig.add_subplot(gs[2, 1])
    
    # Summary table (terceira linha, última coluna)
    ax_summary = fig.add_subplot(gs[2, 2])
    
    # Plotar métricas
    print(f"📈 Gerando visualizações...")
    plot_classification_metrics(clf_val, clf_test, ax_roc, ax_f1, ax_acc, ax_mcc)
    plot_regression_metrics(reg_val, reg_test, ax_mae, ax_r2, ax_rmse, ax_scatter)
    create_summary_table(clf_val, clf_test, reg_val, reg_test, ax_summary)
    
    # Título principal
    fig.suptitle(f'ML Models Benchmark: {args.model}\n(Validation vs Test Performance)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Salvar
    output_path = output_dir / f"all_ml_models_{args.model}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualização salva: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    main()
