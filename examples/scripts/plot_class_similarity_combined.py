#!/usr/bin/env python3
"""
Visualização Combinada de Similaridade Entre Classes
=====================================================

Cria um único gráfico com os 3 modelos para facilitar comparação visual.

Autor: DockTKinase Project
Data: Janeiro 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configurações
RESULTS_FILE = Path('results/class_similarity_analysis/class_similarity_stats.json')
OUTPUT_DIR = Path('results/class_similarity_analysis')

MODELS = {
    'esm2_t6_8M_UR50D': '8M',
    'esm2_t30_150M_UR50D': '150M',
    'esm2_t36_3B_UR50D': '3B',
}


def create_combined_plot():
    """Cria gráfico combinado com overlay de todos os modelos."""
    
    # Carregar dados
    with open(RESULTS_FILE, 'r') as f:
        all_results_raw = json.load(f)
    
    all_results = {}
    for model_name, results in all_results_raw.items():
        all_results[model_name] = {
            k: np.array(v) for k, v in results.items()
        }
    
    # Criar figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {
        'esm2_t6_8M_UR50D': '#3498db',
        'esm2_t30_150M_UR50D': '#e74c3c',
        'esm2_t36_3B_UR50D': '#2ecc71',
    }
    
    bins = np.linspace(0.98, 1.0, 50)
    alpha = 0.5
    
    scenarios = [
        ('train_pos_test_pos', 'Treino POS → Teste POS\n(Mesma Classe)', axes[0, 0]),
        ('train_neg_test_pos', 'Treino NEG → Teste POS\n(Classes Diferentes)', axes[0, 1]),
        ('train_pos_test_neg', 'Treino POS → Teste NEG\n(Classes Diferentes)', axes[1, 0]),
        ('train_neg_test_neg', 'Treino NEG → Teste NEG\n(Mesma Classe)', axes[1, 1]),
    ]
    
    for key, title, ax in scenarios:
        for model_name, model_label in MODELS.items():
            data = all_results[model_name][key]
            ax.hist(data, bins=bins, color=colors[model_name], alpha=alpha, 
                   label=f'{model_label} (μ={data.mean():.4f})', edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Similaridade de Cossenos', fontsize=12)
        ax.set_ylabel('Frequência', fontsize=12)
        ax.set_xlim(0.98, 1.0)
        ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axvline(0.99, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Threshold 0.99')
    
    # Título geral
    fig.suptitle(
        'Similaridade de Cossenos: Treino vs Teste por Classe\n'
        '(Split Aleatório - Data Leakage Evidente)',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    plt.tight_layout()
    
    # Salvar
    output_file = OUTPUT_DIR / 'class_similarity_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✅ Gráfico combinado salvo: {output_file}')
    
    plt.show()
    plt.close()


if __name__ == '__main__':
    create_combined_plot()
