#!/usr/bin/env python3
"""
Gráfico de Comparação entre Modelos ESM-2 (KNN vs MLP)
======================================================

Cria gráfico de barras comparando AUC-ROC dos modelos 8M, 150M e 3B
para KNN e MLP usando split aleatório com 5 seeds.

Autor: DockTKinase Project
Data: Janeiro 2026
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configurações
RESULTS_FILE = Path('results/baseline_multiseed/baseline_multiseed_results.json')
OUTPUT_DIR = Path('results/baseline_multiseed')
OUTPUT_FILE = OUTPUT_DIR / 'model_comparison_random.png'

# Cores
COLORS = {
    'KNN': '#3498db',  # Azul
    'MLP': '#e74c3c',  # Vermelho
}

# Modelos
MODELS = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']
MODEL_LABELS = ['8M', '150M', '3B']


def load_results():
    """Carrega resultados do arquivo JSON."""
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    return data['models']


def extract_stats(results, esm_model, classifier, metric='roc_auc'):
    """Extrai média e desvio padrão para um modelo/classificador."""
    model_data = results.get(esm_model, {})
    aggregated = model_data.get('aggregated', {})
    clf_data = aggregated.get(classifier, {})
    
    metric_data = clf_data.get(metric, {})
    mean_value = metric_data.get('mean', 0.0)
    std_value = metric_data.get('std', 0.0)
    
    return mean_value, std_value


def create_comparison_plot():
    """Cria gráfico de barras comparando modelos."""
    
    # Carregar dados
    print('📂 Carregando resultados...')
    results = load_results()
    
    # Extrair estatísticas para ambas métricas
    data_auc = {
        'KNN': {'means': [], 'stds': []},
        'MLP': {'means': [], 'stds': []},
    }
    
    data_mcc = {
        'KNN': {'means': [], 'stds': []},
        'MLP': {'means': [], 'stds': []},
    }
    
    for esm_model in MODELS:
        for classifier in ['KNN', 'MLP']:
            # AUC-ROC
            mean_auc, std_auc = extract_stats(results, esm_model, classifier, 'roc_auc')
            data_auc[classifier]['means'].append(mean_auc)
            data_auc[classifier]['stds'].append(std_auc)
            
            # MCC
            mean_mcc, std_mcc = extract_stats(results, esm_model, classifier, 'mcc')
            data_mcc[classifier]['means'].append(mean_mcc)
            data_mcc[classifier]['stds'].append(std_mcc)
            
            print(f'   {esm_model} - {classifier}: AUC={mean_auc:.4f}±{std_auc:.4f}, MCC={mean_mcc:.4f}±{std_mcc:.4f}')
    
    # Configurar figura com 2 subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(MODEL_LABELS))
    width = 0.35
    
    # ========== SUBPLOT 1: AUC-ROC ==========
    bars1_auc = ax1.bar(
        x - width/2, 
        data_auc['KNN']['means'], 
        width,
        yerr=data_auc['KNN']['stds'],
        label='KNN',
        color=COLORS['KNN'],
        alpha=0.8,
        capsize=5,
        error_kw={'linewidth': 2}
    )
    
    bars2_auc = ax1.bar(
        x + width/2,
        data_auc['MLP']['means'],
        width,
        yerr=data_auc['MLP']['stds'],
        label='MLP',
        color=COLORS['MLP'],
        alpha=0.8,
        capsize=5,
        error_kw={'linewidth': 2}
    )
    
    # ========== SUBPLOT 2: MCC ==========
    bars1_mcc = ax2.bar(
        x - width/2, 
        data_mcc['KNN']['means'], 
        width,
        yerr=data_mcc['KNN']['stds'],
        label='KNN',
        color=COLORS['KNN'],
        alpha=0.8,
        capsize=5,
        error_kw={'linewidth': 2}
    )
    
    bars2_mcc = ax2.bar(
        x + width/2,
        data_mcc['MLP']['means'],
        width,
        yerr=data_mcc['MLP']['stds'],
        label='MLP',
        color=COLORS['MLP'],
        alpha=0.8,
        capsize=5,
        error_kw={'linewidth': 2}
    )
    
    # Adicionar valores nas barras (acima das barras de erro)
    def add_value_labels(ax, bars, stds, offset_multiplier=1.5):
        """Adiciona valores acima das barras de erro."""
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            # Posicionar acima da barra de erro
            y_pos = height + std + (0.01 * offset_multiplier)
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                y_pos,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
    
    add_value_labels(ax1, bars1_auc, data_auc['KNN']['stds'])
    add_value_labels(ax1, bars2_auc, data_auc['MLP']['stds'])
    add_value_labels(ax2, bars1_mcc, data_mcc['KNN']['stds'])
    add_value_labels(ax2, bars2_mcc, data_mcc['MLP']['stds'])
    
    # ========== CONFIGURAÇÕES AUC-ROC ==========
    ax1.set_ylabel('AUC-ROC (Test)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Modelo ESM-2', fontsize=13, fontweight='bold')
    ax1.set_title('AUC-ROC', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODEL_LABELS, fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    
    # ========== CONFIGURAÇÕES MCC ==========
    ax2.set_ylabel('MCC (Test)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Modelo ESM-2', fontsize=13, fontweight='bold')
    ax2.set_title('Matthews Correlation Coefficient', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODEL_LABELS, fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    
    # Título geral
    fig.suptitle(
        'Comparação de Performance: KNN vs MLP\n(Split Aleatório, 5 seeds)',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    # Layout
    plt.tight_layout()
    
    # Salvar
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f'\n✅ Gráfico salvo em: {OUTPUT_FILE}')
    
    plt.close()


if __name__ == '__main__':
    print('=' * 70)
    print('📊 GRÁFICO DE COMPARAÇÃO: KNN vs MLP')
    print('=' * 70)
    print()
    
    create_comparison_plot()
    
    print()
    print('=' * 70)
    print('✅ CONCLUÍDO')
    print('=' * 70)
