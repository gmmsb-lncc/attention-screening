#!/usr/bin/env python3
"""
Visualização Alternativa - Boxplots de Similaridade por Classe
================================================================

Cria visualizações mais claras usando boxplots para mostrar
a distribuição de similaridades entre classes.

Autor: DockTKinase Project
Data: Janeiro 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configurações
STATS_FILE = Path('results/class_similarity_analysis/class_similarity_separate_stats.json')
OUTPUT_DIR = Path('results/class_similarity_analysis')

# Cores
COLORS = {
    'same': '#2ecc71',      # Verde para mesma classe
    'diff': '#e74c3c',      # Vermelho para classes diferentes
}

MODEL_NAMES = {
    'esm2_t6_8M_UR50D': '8M',
    'esm2_t30_150M_UR50D': '150M',
    'esm2_t36_3B_UR50D': '3B',
}


def load_stats():
    """Carrega estatísticas do JSON."""
    with open(STATS_FILE, 'r') as f:
        return json.load(f)


def create_boxplot_comparison(stats, component_name):
    """
    Cria boxplot comparando same-class vs different-class.
    
    Args:
        stats: dict com estatísticas
        component_name: 'protein' ou 'ligand'
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f'Distribuição de Similaridade - {component_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    component_stats = stats[component_name]
    
    for idx, (model_key, model_short) in enumerate(MODEL_NAMES.items()):
        ax = axes[idx]
        
        if model_key not in component_stats:
            continue
        
        model_data = component_stats[model_key]
        
        # Preparar dados para boxplot
        same_class = [
            model_data['train_pos_test_pos']['mean'],
            model_data['train_neg_test_neg']['mean'],
        ]
        
        diff_class = [
            model_data['train_pos_test_neg']['mean'],
            model_data['train_neg_test_pos']['mean'],
        ]
        
        # Criar boxplot
        positions = [1, 2]
        box_data = [same_class, diff_class]
        
        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Colorir boxes
        colors_list = [COLORS['same'], COLORS['diff']]
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Adicionar pontos individuais
        for pos, data in zip(positions, box_data):
            y = data
            x = np.random.normal(pos, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.6, s=100, color='black', zorder=3)
        
        # Linha horizontal para comparação visual
        ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='0.95')
        ax.axhline(y=0.90, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='0.90')
        
        # Labels
        ax.set_xticks(positions)
        ax.set_xticklabels(['Mesma\nClasse', 'Classes\nDiferentes'], fontsize=11)
        ax.set_ylabel('Similaridade de Cosseno', fontsize=11)
        ax.set_title(f'Modelo {model_short}', fontsize=13, fontweight='bold')
        ax.set_ylim(0.7, 1.01)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar separabilidade como texto
        same_mean = np.mean(same_class)
        diff_mean = np.mean(diff_class)
        separability = same_mean - diff_mean
        
        ax.text(0.5, 0.05, f'Δ = {separability:.4f}', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               ha='center')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f'{component_name}_boxplot_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✅ Salvo: {output_file}')
    plt.close()


def create_heatmap(stats, component_name):
    """
    Cria heatmap mostrando média de similaridades.
    
    Args:
        stats: dict com estatísticas
        component_name: 'protein' ou 'ligand'
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Matriz de Similaridade Média - {component_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    component_stats = stats[component_name]
    
    for idx, (model_key, model_short) in enumerate(MODEL_NAMES.items()):
        ax = axes[idx]
        
        if model_key not in component_stats:
            continue
        
        model_data = component_stats[model_key]
        
        # Criar matriz 2x2
        matrix = np.array([
            [model_data['train_pos_test_pos']['mean'], 
             model_data['train_pos_test_neg']['mean']],
            [model_data['train_neg_test_pos']['mean'], 
             model_data['train_neg_test_neg']['mean']]
        ])
        
        # Heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.95, vmax=1.0, aspect='auto')
        
        # Adicionar valores
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{matrix[i, j]:.4f}',
                             ha="center", va="center", color="black", 
                             fontsize=14, fontweight='bold')
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Teste\nPOS', 'Teste\nNEG'], fontsize=10)
        ax.set_yticklabels(['Treino\nPOS', 'Treino\nNEG'], fontsize=10)
        ax.set_title(f'Modelo {model_short}', fontsize=13, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Similaridade', fontsize=10)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f'{component_name}_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✅ Salvo: {output_file}')
    plt.close()


def create_violin_plot(stats, component_name):
    """
    Cria violin plot para visualizar distribuições completas.
    
    Args:
        stats: dict com estatísticas
        component_name: 'protein' ou 'ligand'
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Distribuição de Similaridade (Violin Plot) - {component_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    component_stats = stats[component_name]
    
    for idx, (model_key, model_short) in enumerate(MODEL_NAMES.items()):
        ax = axes[idx]
        
        if model_key not in component_stats:
            continue
        
        model_data = component_stats[model_key]
        
        # Simular distribuições baseadas em mean/std
        scenarios = [
            ('POS→POS', 'train_pos_test_pos', COLORS['same']),
            ('NEG→POS', 'train_neg_test_pos', COLORS['diff']),
            ('POS→NEG', 'train_pos_test_neg', COLORS['diff']),
            ('NEG→NEG', 'train_neg_test_neg', COLORS['same']),
        ]
        
        positions = [1, 2, 3, 4]
        data_list = []
        colors_list = []
        
        for label, key, color in scenarios:
            mean = model_data[key]['mean']
            std = model_data[key]['std']
            # Simular dados normais
            simulated = np.random.normal(mean, std, 1000)
            simulated = np.clip(simulated, 0, 1)  # Limitar a [0, 1]
            data_list.append(simulated)
            colors_list.append(color)
        
        # Violin plot
        parts = ax.violinplot(data_list, positions=positions, widths=0.7,
                             showmeans=True, showmedians=True)
        
        # Colorir violins
        for pc, color in zip(parts['bodies'], colors_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Labels
        ax.set_xticks(positions)
        ax.set_xticklabels([s[0] for s in scenarios], fontsize=9, rotation=45)
        ax.set_ylabel('Similaridade de Cosseno', fontsize=11)
        ax.set_title(f'Modelo {model_short}', fontsize=13, fontweight='bold')
        ax.set_ylim(0.7, 1.01)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Linha de separação
        ax.axvline(x=2.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.text(1.5, 0.72, 'Mesma Classe', ha='center', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor=COLORS['same'], alpha=0.3))
        ax.text(3.5, 0.72, 'Classes Diferentes', ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor=COLORS['diff'], alpha=0.3))
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f'{component_name}_violin_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✅ Salvo: {output_file}')
    plt.close()


def create_range_comparison(stats):
    """
    Cria gráfico de barras mostrando range (min-max) para cada cenário.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Comparação de Ranges de Similaridade', fontsize=16, fontweight='bold')
    
    components = [('protein', 'PROTEÍNA', axes[0]), ('ligand', 'LIGANTE', axes[1])]
    
    for component_name, component_label, ax in components:
        component_stats = stats[component_name]
        
        scenarios = [
            ('POS→POS', 'train_pos_test_pos', COLORS['same']),
            ('NEG→POS', 'train_neg_test_pos', COLORS['diff']),
            ('POS→NEG', 'train_pos_test_neg', COLORS['diff']),
            ('NEG→NEG', 'train_neg_test_neg', COLORS['same']),
        ]
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        for idx, (model_key, model_short) in enumerate(MODEL_NAMES.items()):
            if model_key not in component_stats:
                continue
            
            model_data = component_stats[model_key]
            
            means = []
            mins = []
            maxs = []
            
            for label, key, color in scenarios:
                means.append(model_data[key]['mean'])
                mins.append(model_data[key]['min'])
                maxs.append(model_data[key]['max'])
            
            means = np.array(means)
            mins = np.array(mins)
            maxs = np.array(maxs)
            
            # Plotar barras com error bars
            offset = (idx - 1) * width
            ax.bar(x + offset, means, width, label=model_short,
                  yerr=[means - mins, maxs - means],
                  capsize=5, alpha=0.7)
        
        ax.set_ylabel('Similaridade de Cosseno', fontsize=11)
        ax.set_title(component_label, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in scenarios], fontsize=10)
        ax.legend(title='Modelo', fontsize=10)
        ax.set_ylim(0.7, 1.01)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'range_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✅ Salvo: {output_file}')
    plt.close()


def main():
    print('\n' + '='*70)
    print('GERANDO VISUALIZAÇÕES ALTERNATIVAS')
    print('='*70)
    
    if not STATS_FILE.exists():
        print(f'❌ Arquivo não encontrado: {STATS_FILE}')
        print('Execute primeiro: python analyze_class_similarity_separate.py')
        return
    
    stats = load_stats()
    
    print('\n📊 Criando Boxplots...')
    create_boxplot_comparison(stats, 'protein')
    create_boxplot_comparison(stats, 'ligand')
    
    print('\n🔥 Criando Heatmaps...')
    create_heatmap(stats, 'protein')
    create_heatmap(stats, 'ligand')
    
    print('\n🎻 Criando Violin Plots...')
    create_violin_plot(stats, 'protein')
    create_violin_plot(stats, 'ligand')
    
    print('\n📏 Criando Range Comparison...')
    create_range_comparison(stats)
    
    print('\n✅ TODAS AS VISUALIZAÇÕES FORAM CRIADAS!')
    print(f'   📁 Diretório: {OUTPUT_DIR}')
    print('\nArquivos gerados:')
    print('   - protein_boxplot_comparison.png')
    print('   - ligand_boxplot_comparison.png')
    print('   - protein_heatmap.png')
    print('   - ligand_heatmap.png')
    print('   - protein_violin_plot.png')
    print('   - ligand_violin_plot.png')
    print('   - range_comparison.png')


if __name__ == '__main__':
    main()
