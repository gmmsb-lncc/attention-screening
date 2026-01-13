#!/usr/bin/env python3
"""
Visualização de Benchmark - Comparação entre Modelos
=====================================================

Compara métricas de classificação e regressão entre todos os modelos.

Usage:
    python scripts/visualize_benchmark_comparison.py \
        --output results/benchmark_viz/
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Configuração visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_all_models(base_path: Path, models: list) -> tuple:
    """Carrega resultados de todos os modelos."""
    classification_data = []
    regression_data = []
    
    for model in models:
        model_path = base_path / model / "integrated_results.json"
        
        if not model_path.exists():
            print(f"⚠️  Pulando {model}: arquivo não encontrado")
            continue
        
        with open(model_path) as f:
            data = json.load(f)
        
        # Classification
        clf = data.get('classifier', {})
        if clf.get('success'):
            best_metrics = clf.get('best_metrics', {})
            classification_data.append({
                'Model': model,
                'Best_Classifier': clf.get('best_model', 'N/A'),
                'ROC_AUC': best_metrics.get('ROC_AUC', 0),
                'Accuracy': best_metrics.get('Accuracy', 0),
                'F1': best_metrics.get('F1', 0),
                'Precision': best_metrics.get('Precision', 0),
                'Recall': best_metrics.get('Recall', 0),
                'MCC': best_metrics.get('MCC', 0),
                'Specificity': best_metrics.get('Specificity', 0)
            })
        
        # Regression
        reg = data.get('regression', {})
        if reg.get('success'):
            regression_data.append({
                'Model': model,
                'Best_Regressor': reg.get('best_model', 'N/A'),
                'Test_MAE': reg.get('best_test_mae', 999),
                'Test_R2': reg.get('best_test_r2', -999),
                'Val_MAE': reg.get('best_val_mae', 999),
                'Val_R2': reg.get('best_val_r2', -999)
            })
    
    return pd.DataFrame(classification_data), pd.DataFrame(regression_data)


def create_comparison_plots(df_clf: pd.DataFrame, df_reg: pd.DataFrame, output_dir: Path):
    """
    Cria visualizações comparando todos os modelos.
    
    Layout: 3x2 grid
    - [0,0]: Classification ROC-AUC
    - [0,1]: Classification Accuracy & F1
    - [1,0]: Classification MCC & Specificity
    - [1,1]: Regression Test MAE
    - [2,0]: Regression Test R²
    - [2,1]: Generalization: Val vs Test (scatter)
    """
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    fig.suptitle('Protein Model Benchmark Comparison\n(Non-Human Kinase Dataset)', 
                fontsize=22, fontweight='bold', y=0.995)
    
    # Preparar labels curtos
    df_clf['Model_Short'] = df_clf['Model'].apply(lambda x: x.replace('esm2_', 'ESM2-').replace('esmc-', 'ESMC-').replace('_UR50D', ''))
    df_reg['Model_Short'] = df_reg['Model'].apply(lambda x: x.replace('esm2_', 'ESM2-').replace('esmc-', 'ESMC-').replace('_UR50D', ''))
    
    # Excluir boltz2 (performance muito baixa)
    df_clf = df_clf[df_clf['Model'] != 'boltz2']
    df_reg = df_reg[df_reg['Model'] != 'boltz2']
    
    # Cores por família
    def get_color(model):
        if 'esm2' in model.lower():
            return '#3498db'  # Azul
        elif 'esmc' in model.lower():
            return '#e74c3c'  # Vermelho
        else:
            return '#95a5a6'  # Cinza
    
    colors_clf = [get_color(m) for m in df_clf['Model']]
    colors_reg = [get_color(m) for m in df_reg['Model']]
    
    # ========================================================================
    # [0,0] Classification: ROC-AUC
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    bars = ax1.barh(df_clf['Model_Short'], df_clf['ROC_AUC'], 
                   color=colors_clf, edgecolor='black', linewidth=1.2)
    
    # Valores nas barras
    for bar, val in zip(bars, df_clf['ROC_AUC']):
        ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax1.set_title('Classification: ROC-AUC (Test Set)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlim(0.96, 0.975)
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(0.97, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Excellent (0.97)')
    ax1.legend()
    
    # ========================================================================
    # [0,1] Classification: Accuracy & F1
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    x = np.arange(len(df_clf))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, df_clf['Accuracy'], width, 
                   label='Accuracy', color='#2ecc71', edgecolor='black', linewidth=1.2)
    bars2 = ax2.bar(x + width/2, df_clf['F1'], width, 
                   label='F1-Score', color='#f39c12', edgecolor='black', linewidth=1.2)
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Classification: Accuracy vs F1-Score', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_clf['Model_Short'], rotation=45, ha='right', fontsize=10)
    ax2.set_ylim(0.88, 0.93)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # ========================================================================
    # [1,0] Classification: MCC & Specificity
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    bars1 = ax3.bar(x - width/2, df_clf['MCC'], width, 
                   label='MCC', color='#9b59b6', edgecolor='black', linewidth=1.2)
    bars2 = ax3.bar(x + width/2, df_clf['Specificity'], width, 
                   label='Specificity', color='#1abc9c', edgecolor='black', linewidth=1.2)
    
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Classification: MCC vs Specificity', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_clf['Model_Short'], rotation=45, ha='right', fontsize=10)
    ax3.set_ylim(0.80, 0.91)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # ========================================================================
    # [1,1] Regression: Test MAE
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Ordenar por MAE (menor = melhor)
    df_reg_sorted = df_reg.sort_values('Test_MAE', ascending=True)
    colors_sorted = [get_color(m) for m in df_reg_sorted['Model']]
    
    bars = ax4.barh(df_reg_sorted['Model_Short'], df_reg_sorted['Test_MAE'], 
                   color=colors_sorted, edgecolor='black', linewidth=1.2)
    
    # Valores nas barras
    for bar, val in zip(bars, df_reg_sorted['Test_MAE']):
        ax4.text(val + 0.03, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('MAE (pChEMBL units)', fontsize=12, fontweight='bold')
    ax4.set_title('Regression: Test MAE (Lower is Better)', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(axis='x', alpha=0.3)
    ax4.axvline(0.5, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Excellent (<0.5)')
    ax4.axvline(1.0, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='Good (<1.0)')
    ax4.legend()
    ax4.invert_yaxis()  # Melhor no topo
    
    # ========================================================================
    # [2,0] Regression: Test R²
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Ordenar por R² (maior = melhor)
    df_reg_sorted_r2 = df_reg.sort_values('Test_R2', ascending=False)
    colors_sorted_r2 = [get_color(m) for m in df_reg_sorted_r2['Model']]
    
    bars = ax5.barh(df_reg_sorted_r2['Model_Short'], df_reg_sorted_r2['Test_R2'], 
                   color=colors_sorted_r2, edgecolor='black', linewidth=1.2)
    
    # Valores nas barras
    for bar, val in zip(bars, df_reg_sorted_r2['Test_R2']):
        offset = 0.01 if val > 0 else -0.02
        ax5.text(val + offset, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    ax5.set_xlabel('R² Score', fontsize=12, fontweight='bold')
    ax5.set_title('Regression: Test R² (Higher is Better)', fontsize=14, fontweight='bold', pad=15)
    ax5.grid(axis='x', alpha=0.3)
    ax5.axvline(0, color='red', linestyle='--', alpha=0.6, linewidth=2, label='No Skill (0)')
    ax5.axvline(0.5, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Good (0.5)')
    ax5.legend()
    ax5.invert_yaxis()  # Melhor no topo
    
    # ========================================================================
    # [2,1] Generalization: Val vs Test (Scatter)
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Scatter: Val R² vs Test R²
    colors_scatter = [get_color(m) for m in df_reg['Model']]
    
    scatter = ax6.scatter(df_reg['Val_R2'], df_reg['Test_R2'], 
                         s=200, c=colors_scatter, alpha=0.7, 
                         edgecolors='black', linewidths=2)
    
    # Labels
    for i, row in df_reg.iterrows():
        ax6.annotate(row['Model_Short'], 
                    (row['Val_R2'], row['Test_R2']),
                    fontsize=9, ha='right', va='bottom',
                    xytext=(-5, 5), textcoords='offset points')
    
    # Linha diagonal (generalização perfeita)
    lims = [
        min(ax6.get_xlim()[0], ax6.get_ylim()[0]),
        max(ax6.get_xlim()[1], ax6.get_ylim()[1])
    ]
    ax6.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect Generalization')
    
    ax6.set_xlabel('Validation R²', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Test R²', fontsize=12, fontweight='bold')
    ax6.set_title('Regression: Generalization (Val vs Test R²)', fontsize=14, fontweight='bold', pad=15)
    ax6.grid(alpha=0.3)
    ax6.legend()
    ax6.axhline(0, color='red', linestyle=':', alpha=0.4)
    ax6.axvline(0, color='red', linestyle=':', alpha=0.4)
    
    # ========================================================================
    # Legenda de cores (família de modelos)
    # ========================================================================
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='ESM-2 Family'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='ESM-C Family')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
              fontsize=12, frameon=True, fancybox=True, shadow=True,
              bbox_to_anchor=(0.5, -0.01))
    
    # ========================================================================
    # Salvar
    # ========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'benchmark_comparison_all_models.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualização salva: {output_file}")
    plt.close()


def create_heatmap_comparison(df_clf: pd.DataFrame, df_reg: pd.DataFrame, output_dir: Path):
    """Cria heatmap comparando todas as métricas de todos os modelos."""
    
    # Excluir boltz2
    df_clf = df_clf[df_clf['Model'] != 'boltz2']
    df_reg = df_reg[df_reg['Model'] != 'boltz2']
    
    # Preparar dados
    models = df_clf['Model'].tolist()
    
    # Matriz de métricas
    metrics_matrix = []
    metric_names = []
    
    # Classification metrics
    for metric in ['ROC_AUC', 'Accuracy', 'F1', 'Precision', 'Recall', 'MCC', 'Specificity']:
        metrics_matrix.append(df_clf[metric].values)
        metric_names.append(f'CLF: {metric}')
    
    # Regression metrics (inverter MAE para que maior = melhor)
    # Normalizar MAE: 1 / (1 + MAE)
    mae_normalized = 1 / (1 + df_reg['Test_MAE'].values)
    metrics_matrix.append(mae_normalized)
    metric_names.append('REG: Test MAE (norm)')
    
    metrics_matrix.append(df_reg['Test_R2'].values)
    metric_names.append('REG: Test R²')
    
    # Converter para array
    data = np.array(metrics_matrix)
    
    # Labels curtos
    model_labels = [m.replace('esm2_', 'ESM2-').replace('esmc-', 'ESMC-').replace('_UR50D', '') for m in models]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
               xticklabels=model_labels, yticklabels=metric_names,
               cbar_kws={'label': 'Score (normalized)'}, 
               linewidths=0.5, linecolor='gray',
               vmin=0, vmax=1, ax=ax)
    
    ax.set_title('Protein Model Benchmark: All Metrics Heatmap\n(Green=Better, Red=Worse)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=13, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Salvar
    output_file = output_dir / 'benchmark_heatmap_all_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap salvo: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Comparar benchmarks de todos os modelos')
    parser.add_argument('--base-path', type=str, 
                       default='results/protein_model_benchmark_non_human_v2',
                       help='Caminho base dos resultados')
    parser.add_argument('--output', type=str, default='results/benchmark_visualizations',
                       help='Diretório de saída')
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    output_dir = Path(args.output)
    
    models = [
        "boltz2",
        "esm2_t6_8M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esmc-300m-2024-12",
        "esmc-600m-2024-12"
    ]
    
    print(f"📊 Carregando resultados de {len(models)} modelos...")
    df_clf, df_reg = load_all_models(base_path, models)
    
    print(f"✅ Carregados: {len(df_clf)} modelos com classificação, {len(df_reg)} com regressão")
    
    print(f"\n🎨 Gerando visualização de comparação...")
    create_comparison_plots(df_clf, df_reg, output_dir)
    
    print(f"\n🎨 Gerando heatmap de todas as métricas...")
    create_heatmap_comparison(df_clf, df_reg, output_dir)
    
    print(f"\n✅ Concluído!")


if __name__ == '__main__':
    main()
