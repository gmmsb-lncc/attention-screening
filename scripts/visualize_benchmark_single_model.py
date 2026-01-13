#!/usr/bin/env python3
"""
Visualização de Benchmark - Modelo Individual
==============================================

Plota métricas de classificação e regressão para um único modelo.

Usage:
    python scripts/visualize_benchmark_single_model.py \
        --model esmc-600m-2024-12 \
        --output results/benchmark_viz/
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configuração visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_model_results(base_path: Path, model_name: str) -> dict:
    """Carrega resultados do modelo."""
    model_path = base_path / model_name / "integrated_results.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {model_path}")
    
    with open(model_path) as f:
        return json.load(f)


def plot_single_model_metrics(data: dict, model_name: str, output_dir: Path):
    """
    Plota métricas de classificação e regressão para um modelo.
    
    Layout: 2x2 grid
    - [0,0]: Classification metrics (barplot)
    - [0,1]: Confusion Matrix heatmap
    - [1,0]: Regression Test vs Val MAE
    - [1,1]: Regression Test vs Val R²
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Título geral
    fig.suptitle(f'Benchmark: {model_name}', fontsize=20, fontweight='bold', y=0.98)
    
    # ========================================================================
    # [0,0] Classification Metrics
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    clf = data.get('classifier', {})
    if clf.get('success'):
        best_metrics = clf.get('best_metrics', {})
        
        metrics = {
            'ROC-AUC': best_metrics.get('ROC_AUC', 0),
            'Accuracy': best_metrics.get('Accuracy', 0),
            'F1-Score': best_metrics.get('F1', 0),
            'Precision': best_metrics.get('Precision', 0),
            'Recall': best_metrics.get('Recall', 0),
            'MCC': best_metrics.get('MCC', 0),
            'Specificity': best_metrics.get('Specificity', 0)
        }
        
        # Normalizar MCC para [0, 1] para visualização
        metrics['MCC (norm)'] = (metrics.pop('MCC') + 1) / 2
        
        names = list(metrics.keys())
        values = list(metrics.values())
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        
        bars = ax1.barh(names, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Adicionar valores nas barras
        for bar, val in zip(bars, values):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
        
        ax1.set_xlim(0, 1.1)
        ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Classification Metrics\nBest: {clf.get("best_model", "N/A")}', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3)
        ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Random (0.5)')
        ax1.legend(loc='lower right')
    else:
        ax1.text(0.5, 0.5, 'Classification: No Data', ha='center', va='center', fontsize=14)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
    
    # ========================================================================
    # [0,1] Confusion Matrix
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    if clf.get('success'):
        best_metrics = clf.get('best_metrics', {})
        
        # Extrair matriz de confusão
        tn = best_metrics.get('True_Negatives', 0)
        fp = best_metrics.get('False_Positives', 0)
        fn = best_metrics.get('False_Negatives', 0)
        tp = best_metrics.get('True_Positives', 0)
        
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        
        # Heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', 
                   cbar=True, square=True, ax=ax2, 
                   xticklabels=['Inactive (0)', 'Active (1)'],
                   yticklabels=['Inactive (0)', 'Active (1)'],
                   annot_kws={'fontsize': 16, 'fontweight': 'bold'})
        
        ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True', fontsize=12, fontweight='bold')
        ax2.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold', pad=15)
        
        # Adicionar porcentagens
        total = confusion_matrix.sum()
        for i in range(2):
            for j in range(2):
                percentage = confusion_matrix[i, j] / total * 100
                ax2.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
    else:
        ax2.text(0.5, 0.5, 'Confusion Matrix: No Data', ha='center', va='center', fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    # ========================================================================
    # [1,0] Regression: Val vs Test MAE
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    reg = data.get('regression', {})
    if reg.get('success'):
        val_mae = reg.get('best_val_mae', 0)
        test_mae = reg.get('best_test_mae', 0)
        
        x = ['Validation', 'Test']
        y = [val_mae, test_mae]
        colors_reg = ['#3498db', '#e74c3c']
        
        bars = ax3.bar(x, y, color=colors_reg, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Valores nas barras
        for bar, val in zip(bars, y):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        ax3.set_ylabel('MAE (pChEMBL units)', fontsize=12, fontweight='bold')
        ax3.set_title(f'Regression: Mean Absolute Error\nBest: {reg.get("best_model", "N/A")}', 
                     fontsize=14, fontweight='bold', pad=15)
        ax3.grid(axis='y', alpha=0.3)
        
        # Linha de referência
        max_val = max(y) * 1.1
        ax3.set_ylim(0, max_val)
        ax3.axhline(1.0, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='MAE = 1.0')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Regression MAE: No Data', ha='center', va='center', fontsize=14)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    
    # ========================================================================
    # [1,1] Regression: Val vs Test R²
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    if reg.get('success'):
        val_r2 = reg.get('best_val_r2', 0)
        test_r2 = reg.get('best_test_r2', 0)
        
        x = ['Validation', 'Test']
        y = [val_r2, test_r2]
        
        # Cores baseadas em performance
        colors_r2 = ['#27ae60' if v > 0.3 else '#f39c12' if v > 0 else '#e74c3c' for v in y]
        
        bars = ax4.bar(x, y, color=colors_r2, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Valores nas barras
        for bar, val in zip(bars, y):
            offset = 0.02 if val > 0 else -0.05
            va = 'bottom' if val > 0 else 'top'
            ax4.text(bar.get_x() + bar.get_width()/2, val + offset, 
                    f'{val:.4f}', ha='center', va=va, fontsize=13, fontweight='bold')
        
        ax4.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax4.set_title(f'Regression: Coefficient of Determination (R²)\nBest: {reg.get("best_model", "N/A")}', 
                     fontsize=14, fontweight='bold', pad=15)
        ax4.grid(axis='y', alpha=0.3)
        
        # Linhas de referência
        ax4.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='No skill (R²=0)')
        ax4.axhline(0.5, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Good (R²=0.5)')
        ax4.legend()
        
        # Ajustar limites
        min_val = min(y) - 0.1
        max_val = max(1.0, max(y) + 0.1)
        ax4.set_ylim(min_val, max_val)
    else:
        ax4.text(0.5, 0.5, 'Regression R²: No Data', ha='center', va='center', fontsize=14)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
    
    # ========================================================================
    # Salvar
    # ========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'benchmark_{model_name}_single.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualização salva: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualizar benchmark de um modelo')
    parser.add_argument('--model', type=str, required=True, 
                       help='Nome do modelo (ex: esmc-600m-2024-12)')
    parser.add_argument('--base-path', type=str, 
                       default='results/protein_model_benchmark_non_human_v2',
                       help='Caminho base dos resultados')
    parser.add_argument('--output', type=str, default='results/benchmark_visualizations',
                       help='Diretório de saída')
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    output_dir = Path(args.output)
    
    print(f"📊 Carregando resultados do modelo: {args.model}")
    data = load_model_results(base_path, args.model)
    
    print(f"🎨 Gerando visualização...")
    plot_single_model_metrics(data, args.model, output_dir)
    
    print(f"✅ Concluído!")


if __name__ == '__main__':
    main()
