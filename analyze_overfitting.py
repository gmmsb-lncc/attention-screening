#!/usr/bin/env python3
"""
DockTKinase - Análise de Overfitting
=====================================

Análise robusta para detectar overfitting em modelos KNN e MLP.

Visualizações geradas:
1. Gap Analysis (Train → Val → Test)
2. Variância entre Seeds
3. Heatmap de Gaps
4. Box plots por classificador
5. Curvas de comparação Train vs Test
6. Radar chart de métricas

Autor: DockTKinase Team
Data: Janeiro 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def load_results(results_path: Path) -> Dict:
    """Carrega resultados do JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def extract_metrics(data: Dict) -> Dict:
    """Extrai métricas organizadas por modelo/classificador/seed."""
    
    metrics = {}
    
    for esm_model in data['models']:
        metrics[esm_model] = {'KNN': [], 'MLP': []}
        
        for seed_result in data['models'][esm_model]['seed_results']:
            seed = seed_result['seed']
            
            for clf in ['KNN', 'MLP']:
                clf_data = seed_result['classifiers'][clf]
                
                metrics[esm_model][clf].append({
                    'seed': seed,
                    'train': clf_data['train_metrics'],
                    'val': clf_data['val_metrics'],
                    'test': clf_data['test_metrics'],
                })
    
    return metrics


def calculate_gaps(metrics: Dict) -> Dict:
    """Calcula gaps entre train/val/test."""
    
    gaps = {}
    
    for esm_model in metrics:
        gaps[esm_model] = {}
        
        for clf in ['KNN', 'MLP']:
            train_aucs = [m['train']['roc_auc'] for m in metrics[esm_model][clf]]
            val_aucs = [m['val']['roc_auc'] for m in metrics[esm_model][clf]]
            test_aucs = [m['test']['roc_auc'] for m in metrics[esm_model][clf]]
            
            gaps[esm_model][clf] = {
                'train_mean': np.mean(train_aucs),
                'train_std': np.std(train_aucs),
                'val_mean': np.mean(val_aucs),
                'val_std': np.std(val_aucs),
                'test_mean': np.mean(test_aucs),
                'test_std': np.std(test_aucs),
                'gap_train_test': (np.mean(train_aucs) - np.mean(test_aucs)) * 100,
                'gap_val_test': (np.mean(val_aucs) - np.mean(test_aucs)) * 100,
                'gap_train_val': (np.mean(train_aucs) - np.mean(val_aucs)) * 100,
                'train_values': train_aucs,
                'val_values': val_aucs,
                'test_values': test_aucs,
            }
    
    return gaps


# =============================================================================
# VISUALIZAÇÃO 1: Gap Analysis Bar Chart
# =============================================================================

def plot_gap_analysis(gaps: Dict, output_dir: Path):
    """Gráfico de barras mostrando gaps Train→Val→Test."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    esm_models = list(gaps.keys())
    model_labels = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 3B']
    
    for clf_idx, clf in enumerate(['KNN', 'MLP']):
        ax = axes[clf_idx]
        
        x = np.arange(len(esm_models))
        width = 0.25
        
        train_means = [gaps[m][clf]['train_mean'] for m in esm_models]
        val_means = [gaps[m][clf]['val_mean'] for m in esm_models]
        test_means = [gaps[m][clf]['test_mean'] for m in esm_models]
        
        train_stds = [gaps[m][clf]['train_std'] for m in esm_models]
        val_stds = [gaps[m][clf]['val_std'] for m in esm_models]
        test_stds = [gaps[m][clf]['test_std'] for m in esm_models]
        
        bars1 = ax.bar(x - width, train_means, width, label='Train', 
                       color='#e74c3c', alpha=0.8, yerr=train_stds, capsize=3)
        bars2 = ax.bar(x, val_means, width, label='Validation', 
                       color='#f39c12', alpha=0.8, yerr=val_stds, capsize=3)
        bars3 = ax.bar(x + width, test_means, width, label='Test', 
                       color='#27ae60', alpha=0.8, yerr=test_stds, capsize=3)
        
        # Adicionar valores
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Modelo ESM-2', fontweight='bold')
        ax.set_ylabel('ROC-AUC', fontweight='bold')
        ax.set_title(f'{clf}: Train vs Validation vs Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend(loc='lower right')
        ax.set_ylim(0.9, 1.02)
        ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Threshold 0.95')
        
        # Adicionar gaps como texto
        for i, model in enumerate(esm_models):
            gap = gaps[model][clf]['gap_train_test']
            color = '#e74c3c' if gap > 5 else '#f39c12' if gap > 3 else '#27ae60'
            ax.annotate(f'Gap: {gap:.1f}%', 
                       xy=(i, 0.905), ha='center', fontsize=9, 
                       fontweight='bold', color=color)
    
    plt.suptitle('Análise de Overfitting: Comparação Train/Val/Test\n(5 seeds, média ± std)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_gap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo: overfitting_gap_analysis.png")


# =============================================================================
# VISUALIZAÇÃO 2: Heatmap de Gaps
# =============================================================================

def plot_gap_heatmap(gaps: Dict, output_dir: Path):
    """Heatmap mostrando gaps por modelo/classificador."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    esm_models = list(gaps.keys())
    model_labels = ['8M', '150M', '3B']
    classifiers = ['KNN', 'MLP']
    
    gap_types = [
        ('gap_train_test', 'Gap Train→Test (%)', 'Reds'),
        ('gap_val_test', 'Gap Val→Test (%)', 'Oranges'),
        ('gap_train_val', 'Gap Train→Val (%)', 'Purples'),
    ]
    
    for ax_idx, (gap_key, title, cmap) in enumerate(gap_types):
        ax = axes[ax_idx]
        
        # Criar matriz de gaps
        data = np.zeros((len(classifiers), len(esm_models)))
        
        for i, clf in enumerate(classifiers):
            for j, model in enumerate(esm_models):
                data[i, j] = gaps[model][clf][gap_key]
        
        # Heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=max(6, np.max(data)))
        
        # Adicionar valores
        for i in range(len(classifiers)):
            for j in range(len(esm_models)):
                val = data[i, j]
                color = 'white' if val > 3 else 'black'
                ax.text(j, i, f'{val:.2f}%', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color=color)
        
        ax.set_xticks(range(len(esm_models)))
        ax.set_xticklabels(model_labels)
        ax.set_yticks(range(len(classifiers)))
        ax.set_yticklabels(classifiers)
        ax.set_xlabel('Modelo ESM-2', fontweight='bold')
        ax.set_ylabel('Classificador', fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('%', fontweight='bold')
    
    plt.suptitle('Heatmap de Gaps: Indicadores de Overfitting', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo: overfitting_heatmap.png")


# =============================================================================
# VISUALIZAÇÃO 3: Box Plots por Seed
# =============================================================================

def plot_boxplots(gaps: Dict, output_dir: Path):
    """Box plots mostrando distribuição entre seeds."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    esm_models = list(gaps.keys())
    model_labels = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 3B']
    
    colors = {'Train': '#e74c3c', 'Val': '#f39c12', 'Test': '#27ae60'}
    
    for i, (model, label) in enumerate(zip(esm_models, model_labels)):
        for j, clf in enumerate(['KNN', 'MLP']):
            ax = axes[j, i]
            
            data = [
                gaps[model][clf]['train_values'],
                gaps[model][clf]['val_values'],
                gaps[model][clf]['test_values'],
            ]
            
            bp = ax.boxplot(data, labels=['Train', 'Val', 'Test'], patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors.values()):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Adicionar pontos individuais
            for k, (d, c) in enumerate(zip(data, colors.values())):
                x = np.random.normal(k+1, 0.04, len(d))
                ax.scatter(x, d, color=c, alpha=0.6, s=50, zorder=3, edgecolor='white')
            
            ax.set_ylabel('ROC-AUC', fontweight='bold')
            ax.set_title(f'{label} - {clf}', fontsize=11, fontweight='bold')
            ax.set_ylim(0.9, 1.01)
            ax.grid(axis='y', alpha=0.3)
            
            # Calcular e mostrar gap
            gap = gaps[model][clf]['gap_train_test']
            color = '#e74c3c' if gap > 5 else '#f39c12' if gap > 3 else '#27ae60'
            ax.annotate(f'Gap T→Te: {gap:.1f}%', xy=(0.98, 0.02), xycoords='axes fraction',
                       ha='right', fontsize=9, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Distribuição de ROC-AUC por Seed (5 seeds)\nBox Plot: Mediana, Quartis, Outliers', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo: overfitting_boxplots.png")


# =============================================================================
# VISUALIZAÇÃO 4: Comparação Lado a Lado KNN vs MLP
# =============================================================================

def plot_knn_vs_mlp(gaps: Dict, output_dir: Path):
    """Comparação direta KNN vs MLP para detectar diferenças de overfitting."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    esm_models = list(gaps.keys())
    model_labels = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 3B']
    
    for i, (model, label) in enumerate(zip(esm_models, model_labels)):
        ax = axes[i]
        
        # Dados
        categories = ['Train', 'Val', 'Test']
        knn_vals = [gaps[model]['KNN']['train_mean'], 
                    gaps[model]['KNN']['val_mean'], 
                    gaps[model]['KNN']['test_mean']]
        mlp_vals = [gaps[model]['MLP']['train_mean'], 
                    gaps[model]['MLP']['val_mean'], 
                    gaps[model]['MLP']['test_mean']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, knn_vals, width, label='KNN', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, mlp_vals, width, label='MLP', color='#e74c3c', alpha=0.8)
        
        # Conectar Train→Val→Test com linhas
        ax.plot(x - width/2, knn_vals, 'o-', color='#2980b9', linewidth=2, markersize=8)
        ax.plot(x + width/2, mlp_vals, 'o-', color='#c0392b', linewidth=2, markersize=8)
        
        # Valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Conjunto', fontweight='bold')
        ax.set_ylabel('ROC-AUC', fontweight='bold')
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim(0.9, 1.02)
        
        # Mostrar gaps
        knn_gap = gaps[model]['KNN']['gap_train_test']
        mlp_gap = gaps[model]['MLP']['gap_train_test']
        
        ax.annotate(f'KNN Gap: {knn_gap:.1f}%\nMLP Gap: {mlp_gap:.1f}%', 
                   xy=(0.02, 0.02), xycoords='axes fraction',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('KNN vs MLP: Progressão Train → Val → Test\n(Queda acentuada indica overfitting)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_knn_vs_mlp.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo: overfitting_knn_vs_mlp.png")


# =============================================================================
# VISUALIZAÇÃO 5: Radar Chart de Métricas
# =============================================================================

def plot_radar_chart(metrics: Dict, output_dir: Path):
    """Radar chart comparando múltiplas métricas Train vs Test."""
    
    from math import pi
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'ROC-AUC']
    
    # Ângulos
    angles = [n / float(len(metric_names)) * 2 * pi for n in range(len(metric_names))]
    angles += angles[:1]  # Fechar o polígono
    
    esm_models = list(metrics.keys())
    colors_train = ['#e74c3c', '#c0392b', '#a93226']
    colors_test = ['#27ae60', '#1e8449', '#196f3d']
    
    for clf_idx, clf in enumerate(['KNN', 'MLP']):
        ax = axes[clf_idx]
        
        for i, model in enumerate(esm_models):
            # Médias de Train e Test
            train_vals = []
            test_vals = []
            
            for metric in metric_names:
                train_vals.append(np.mean([m['train'][metric] for m in metrics[model][clf]]))
                test_vals.append(np.mean([m['test'][metric] for m in metrics[model][clf]]))
            
            # Fechar polígono
            train_vals += train_vals[:1]
            test_vals += test_vals[:1]
            
            # Plotar
            model_short = model.replace('esm2_', '').replace('_UR50D', '')
            ax.plot(angles, train_vals, 'o-', linewidth=2, label=f'{model_short} Train', 
                   color=colors_train[i], alpha=0.7)
            ax.fill(angles, train_vals, alpha=0.1, color=colors_train[i])
            
            ax.plot(angles, test_vals, 'o--', linewidth=2, label=f'{model_short} Test', 
                   color=colors_test[i], alpha=0.7)
            ax.fill(angles, test_vals, alpha=0.1, color=colors_test[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0.7, 1.0)
        ax.set_title(f'{clf}: Train vs Test (todas as métricas)', fontsize=12, fontweight='bold', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    
    plt.suptitle('Radar Chart: Comparação de Métricas Train vs Test\n(Área maior = melhor performance)', 
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo: overfitting_radar.png")


# =============================================================================
# VISUALIZAÇÃO 6: Diagnóstico Final
# =============================================================================

def plot_diagnostic_summary(gaps: Dict, output_dir: Path):
    """Resumo visual com diagnóstico de overfitting."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    esm_models = list(gaps.keys())
    model_labels = ['ESM-2\n8M', 'ESM-2\n150M', 'ESM-2\n3B']
    
    # Criar tabela de diagnóstico
    data = []
    colors = []
    
    headers = ['Modelo', 'Clf', 'Train AUC', 'Val AUC', 'Test AUC', 
               'Gap T→Te', 'Gap V→Te', 'Std Seeds', 'Diagnóstico']
    
    for model, label in zip(esm_models, model_labels):
        for clf in ['KNN', 'MLP']:
            g = gaps[model][clf]
            
            gap_tt = g['gap_train_test']
            gap_vt = g['gap_val_test']
            test_std = g['test_std'] * 100  # Em %
            
            # Diagnóstico
            if gap_tt > 5 and gap_vt > 2:
                diag = '🔴 OVERFITTING'
                row_color = '#ffcccc'
            elif gap_tt > 5 and gap_vt < 1:
                diag = '🟡 Normal (KNN)'
                row_color = '#fff3cd'
            elif gap_tt > 3:
                diag = '🟡 Leve Gap'
                row_color = '#fff3cd'
            else:
                diag = '🟢 Robusto'
                row_color = '#d4edda'
            
            row = [
                label.replace('\n', ' '),
                clf,
                f'{g["train_mean"]:.4f}',
                f'{g["val_mean"]:.4f}',
                f'{g["test_mean"]:.4f}',
                f'{gap_tt:.2f}%',
                f'{gap_vt:.2f}%',
                f'±{test_std:.2f}%',
                diag
            ]
            data.append(row)
            colors.append(row_color)
    
    # Criar tabela
    ax.axis('off')
    
    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#2c3e50'] * len(headers)
    )
    
    # Estilizar tabela
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Cores do header
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Cores das linhas
    for i, color in enumerate(colors):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
    
    # Legenda
    legend_elements = [
        mpatches.Patch(facecolor='#d4edda', edgecolor='black', label='🟢 Robusto (Gap <3%)'),
        mpatches.Patch(facecolor='#fff3cd', edgecolor='black', label='🟡 Leve/Normal (Gap 3-5%)'),
        mpatches.Patch(facecolor='#ffcccc', edgecolor='black', label='🔴 Overfitting (Gap >5% + Val>>Test)'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
             ncol=3, fontsize=10)
    
    plt.title('📊 Diagnóstico de Overfitting: Resumo Final\n(Baseado em 5 seeds)', 
             fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_diagnostic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo: overfitting_diagnostic.png")


# =============================================================================
# VISUALIZAÇÃO 7: Scatter Gap vs Performance
# =============================================================================

def plot_gap_vs_performance(gaps: Dict, output_dir: Path):
    """Scatter plot: Gap Train-Test vs Test Performance."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    esm_models = list(gaps.keys())
    
    markers = {'KNN': 'o', 'MLP': 's'}
    colors = {'esm2_t6_8M_UR50D': '#3498db', 
              'esm2_t30_150M_UR50D': '#e74c3c', 
              'esm2_t36_3B_UR50D': '#2ecc71'}
    
    for model in esm_models:
        for clf in ['KNN', 'MLP']:
            g = gaps[model][clf]
            
            x = g['gap_train_test']
            y = g['test_mean']
            
            model_short = model.replace('esm2_', '').replace('_UR50D', '')
            
            ax.scatter(x, y, s=200, marker=markers[clf], color=colors[model],
                      edgecolor='black', linewidth=2, alpha=0.8,
                      label=f'{model_short} {clf}')
            
            # Adicionar label
            ax.annotate(f'{model_short}\n{clf}', (x, y), 
                       textcoords='offset points', xytext=(10, 5),
                       fontsize=9, fontweight='bold')
    
    # Zonas de diagnóstico
    ax.axvspan(0, 3, alpha=0.1, color='green', label='Zona Robusta')
    ax.axvspan(3, 5, alpha=0.1, color='yellow', label='Zona de Atenção')
    ax.axvspan(5, 10, alpha=0.1, color='red', label='Zona de Overfitting')
    
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.7)
    ax.axvline(x=5, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Gap Train→Test (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test ROC-AUC', fontsize=12, fontweight='bold')
    ax.set_title('Gap vs Performance: Identificação de Overfitting\n(Ideal: baixo gap + alta performance)', 
                fontsize=14, fontweight='bold')
    
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0.92, 0.98)
    ax.grid(True, alpha=0.3)
    
    # Legenda customizada
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='KNN'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, label='MLP'),
        mpatches.Patch(facecolor='green', alpha=0.3, label='Robusto (<3%)'),
        mpatches.Patch(facecolor='yellow', alpha=0.3, label='Atenção (3-5%)'),
        mpatches.Patch(facecolor='red', alpha=0.3, label='Overfitting (>5%)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_gap_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo: overfitting_gap_vs_performance.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("📊 ANÁLISE DE OVERFITTING - DockTKinase")
    print("=" * 70)
    print()
    
    # Paths
    results_path = Path('results/baseline_multiseed/baseline_multiseed_results.json')
    output_dir = Path('results/overfitting_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar dados
    print("📂 Carregando resultados...")
    data = load_results(results_path)
    
    # Extrair métricas
    print("📈 Extraindo métricas...")
    metrics = extract_metrics(data)
    
    # Calcular gaps
    print("🔍 Calculando gaps...")
    gaps = calculate_gaps(metrics)
    
    # Gerar visualizações
    print("\n🎨 Gerando visualizações...")
    print("-" * 40)
    
    plot_gap_analysis(gaps, output_dir)
    plot_gap_heatmap(gaps, output_dir)
    plot_boxplots(gaps, output_dir)
    plot_knn_vs_mlp(gaps, output_dir)
    plot_radar_chart(metrics, output_dir)
    plot_diagnostic_summary(gaps, output_dir)
    plot_gap_vs_performance(gaps, output_dir)
    
    # Imprimir resumo
    print("\n" + "=" * 70)
    print("📋 RESUMO DOS RESULTADOS")
    print("=" * 70)
    
    for model in gaps:
        print(f"\n🧬 {model}")
        for clf in ['KNN', 'MLP']:
            g = gaps[model][clf]
            print(f"   {clf}:")
            print(f"      Train: {g['train_mean']:.4f} ± {g['train_std']:.4f}")
            print(f"      Val:   {g['val_mean']:.4f} ± {g['val_std']:.4f}")
            print(f"      Test:  {g['test_mean']:.4f} ± {g['test_std']:.4f}")
            print(f"      Gap Train→Test: {g['gap_train_test']:.2f}%")
            print(f"      Gap Val→Test:   {g['gap_val_test']:.2f}%")
    
    print("\n" + "=" * 70)
    print(f"✅ Visualizações salvas em: {output_dir}")
    print("=" * 70)
    
    # Listar arquivos gerados
    print("\n📁 Arquivos gerados:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"   • {f.name}")


if __name__ == '__main__':
    main()
