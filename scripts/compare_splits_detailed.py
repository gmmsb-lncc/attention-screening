#!/usr/bin/env python3
"""
Análise detalhada comparando Random Split vs Stratified Split.
Gera visualizações mostrando qual método teve melhor performance.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

def load_results():
    """Carrega resultados de ambos os splits."""
    base_path = Path('results/baseline_multiseed')
    
    with open(base_path / 'baseline_multiseed_results.json') as f:
        random_data = json.load(f)
    
    with open(base_path / 'stratified_multiseed_results.json') as f:
        stratified_data = json.load(f)
    
    return random_data, stratified_data

def extract_metrics(data, split_type):
    """Extrai métricas organizadas por modelo e classificador."""
    results = []
    
    for model_name, model_data in data['models'].items():
        # Simplificar nome do modelo
        if '8M' in model_name:
            short_name = 'ESM 8M'
        elif '150M' in model_name:
            short_name = 'ESM 150M'
        elif '3B' in model_name:
            short_name = 'ESM 3B'
        else:
            short_name = model_name
        
        # Extrair métricas de cada seed
        for seed_result in model_data['seed_results']:
            seed = seed_result['seed']
            
            for clf_name, clf_data in seed_result['classifiers'].items():
                train_auc = clf_data['train_metrics']['roc_auc']
                val_auc = clf_data['val_metrics']['roc_auc']
                test_auc = clf_data['test_metrics']['roc_auc']
                
                results.append({
                    'model': short_name,
                    'classifier': clf_name,
                    'seed': seed,
                    'split': split_type,
                    'train_auc': train_auc,
                    'val_auc': val_auc,
                    'test_auc': test_auc,
                    'gap_train_test': (train_auc - test_auc) * 100,
                    'gap_val_test': (val_auc - test_auc) * 100
                })
    
    return results

def calculate_summary_stats(results):
    """Calcula estatísticas resumidas por modelo/classificador/split."""
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    summary = df.groupby(['model', 'classifier', 'split']).agg({
        'test_auc': ['mean', 'std'],
        'gap_train_test': ['mean', 'std'],
        'gap_val_test': ['mean', 'std']
    }).round(4)
    
    return df, summary

def create_comparison_visualization(df, output_path):
    """Cria visualização completa comparando Random vs Stratified."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    models = ['ESM 8M', 'ESM 150M', 'ESM 3B']
    classifiers = ['KNN', 'MLP']
    colors = {'Random': '#3498db', 'Stratified': '#e74c3c'}
    
    # ============================================================
    # 1. Test AUC Comparison (barplot com error bars)
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    x_pos = np.arange(len(models) * len(classifiers))
    width = 0.35
    
    random_means = []
    random_stds = []
    strat_means = []
    strat_stds = []
    labels = []
    
    for model in models:
        for clf in classifiers:
            labels.append(f'{model}\n{clf}')
            
            random_data = df[(df['model'] == model) & 
                            (df['classifier'] == clf) & 
                            (df['split'] == 'Random')]['test_auc']
            strat_data = df[(df['model'] == model) & 
                           (df['classifier'] == clf) & 
                           (df['split'] == 'Stratified')]['test_auc']
            
            random_means.append(random_data.mean())
            random_stds.append(random_data.std())
            strat_means.append(strat_data.mean())
            strat_stds.append(strat_data.std())
    
    ax1.bar(x_pos - width/2, random_means, width, yerr=random_stds,
            label='Random', color=colors['Random'], alpha=0.8, capsize=5)
    ax1.bar(x_pos + width/2, strat_means, width, yerr=strat_stds,
            label='Stratified', color=colors['Stratified'], alpha=0.8, capsize=5)
    
    ax1.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    ax1.set_title('Test AUC: Random vs Stratified Split', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.93, 0.97])
    
    # Adicionar valores no topo
    for i, (r_mean, s_mean) in enumerate(zip(random_means, strat_means)):
        diff = (r_mean - s_mean) * 100
        if abs(diff) > 0.1:
            ax1.text(i, max(r_mean, s_mean) + 0.002, 
                    f'Δ={diff:+.2f}%', 
                    ha='center', fontsize=7, fontweight='bold')
    
    # ============================================================
    # 2. Delta (Diferença) - Heatmap
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 2])
    
    delta_matrix = np.zeros((len(models), len(classifiers)))
    
    for i, model in enumerate(models):
        for j, clf in enumerate(classifiers):
            random_auc = df[(df['model'] == model) & 
                           (df['classifier'] == clf) & 
                           (df['split'] == 'Random')]['test_auc'].mean()
            strat_auc = df[(df['model'] == model) & 
                          (df['classifier'] == clf) & 
                          (df['split'] == 'Stratified')]['test_auc'].mean()
            delta_matrix[i, j] = (random_auc - strat_auc) * 100
    
    im = ax2.imshow(delta_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax2.set_xticks(range(len(classifiers)))
    ax2.set_xticklabels(classifiers)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models)
    ax2.set_title('Δ AUC (Random - Stratified) %', 
                  fontsize=12, fontweight='bold', pad=10)
    
    # Adicionar valores nas células
    for i in range(len(models)):
        for j in range(len(classifiers)):
            text = ax2.text(j, i, f'{delta_matrix[i, j]:.2f}%',
                           ha="center", va="center", 
                           color="white" if abs(delta_matrix[i, j]) > 0.2 else "black",
                           fontweight='bold', fontsize=10)
    
    plt.colorbar(im, ax=ax2, label='Δ AUC (%)')
    
    # ============================================================
    # 3. Gap Train→Test Comparison
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    gap_data = []
    gap_labels = []
    gap_colors = []
    
    for model in models:
        for clf in classifiers:
            random_gap = df[(df['model'] == model) & 
                           (df['classifier'] == clf) & 
                           (df['split'] == 'Random')]['gap_train_test'].mean()
            strat_gap = df[(df['model'] == model) & 
                          (df['classifier'] == clf) & 
                          (df['split'] == 'Stratified')]['gap_train_test'].mean()
            
            gap_data.extend([random_gap, strat_gap])
            gap_labels.extend([f'{model}\n{clf}\nRandom', f'{model}\n{clf}\nStrat'])
            gap_colors.extend([colors['Random'], colors['Stratified']])
    
    bars = ax3.barh(range(len(gap_data)), gap_data, color=gap_colors, alpha=0.7)
    ax3.set_yticks(range(len(gap_data)))
    ax3.set_yticklabels(gap_labels, fontsize=7)
    ax3.set_xlabel('Gap Train→Test (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Gap Train→Test: Random vs Stratified', 
                  fontsize=12, fontweight='bold')
    ax3.axvline(x=3, color='orange', linestyle='--', linewidth=2, label='Limiar 3%')
    ax3.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Limiar 5%')
    ax3.legend(fontsize=8)
    ax3.grid(axis='x', alpha=0.3)
    
    # ============================================================
    # 4. Boxplot - Distribuição Test AUC
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    box_data = []
    box_labels = []
    box_positions = []
    pos = 0
    
    for model in models:
        for clf in classifiers:
            random_aucs = df[(df['model'] == model) & 
                            (df['classifier'] == clf) & 
                            (df['split'] == 'Random')]['test_auc'].values
            strat_aucs = df[(df['model'] == model) & 
                           (df['classifier'] == clf) & 
                           (df['split'] == 'Stratified')]['test_auc'].values
            
            box_data.extend([random_aucs, strat_aucs])
            box_labels.extend([f'{model}-{clf}\nR', f'{model}-{clf}\nS'])
            box_positions.extend([pos, pos + 0.5])
            pos += 1.2
    
    bp = ax4.boxplot(box_data, positions=box_positions, widths=0.4,
                     patch_artist=True, showmeans=True)
    
    # Colorir boxes
    for i, patch in enumerate(bp['boxes']):
        if i % 2 == 0:
            patch.set_facecolor(colors['Random'])
        else:
            patch.set_facecolor(colors['Stratified'])
        patch.set_alpha(0.6)
    
    ax4.set_xticks([i + 0.25 for i in range(0, len(box_data), 2)])
    ax4.set_xticklabels([f'{m}\n{c}' for m in models for c in classifiers], 
                        fontsize=8)
    ax4.set_ylabel('Test AUC', fontsize=11, fontweight='bold')
    ax4.set_title('Distribuição Test AUC (5 seeds)', 
                  fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 5. Scatter: Random vs Stratified
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 2])
    
    for clf in classifiers:
        clf_data = df[df['classifier'] == clf]
        
        random_aucs = []
        strat_aucs = []
        
        for model in models:
            for seed in df['seed'].unique():
                random_auc = clf_data[(clf_data['model'] == model) & 
                                     (clf_data['seed'] == seed) & 
                                     (clf_data['split'] == 'Random')]['test_auc'].values
                strat_auc = clf_data[(clf_data['model'] == model) & 
                                    (clf_data['seed'] == seed) & 
                                    (clf_data['split'] == 'Stratified')]['test_auc'].values
                
                if len(random_auc) > 0 and len(strat_auc) > 0:
                    random_aucs.append(random_auc[0])
                    strat_aucs.append(strat_auc[0])
        
        ax5.scatter(random_aucs, strat_aucs, label=clf, alpha=0.6, s=50)
    
    # Linha de identidade
    lims = [0.93, 0.97]
    ax5.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='y=x (identidade)')
    ax5.set_xlim(lims)
    ax5.set_ylim(lims)
    ax5.set_xlabel('Random Split AUC', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Stratified Split AUC', fontsize=11, fontweight='bold')
    ax5.set_title('Correlação: Random vs Stratified', 
                  fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    ax5.set_aspect('equal')
    
    # ============================================================
    # 6. Tabela de Estatísticas Resumidas
    # ============================================================
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Calcular estatísticas
    stats_data = []
    
    for model in models:
        for clf in classifiers:
            random_df = df[(df['model'] == model) & 
                          (df['classifier'] == clf) & 
                          (df['split'] == 'Random')]
            strat_df = df[(df['model'] == model) & 
                         (df['classifier'] == clf) & 
                         (df['split'] == 'Stratified')]
            
            r_auc_mean = random_df['test_auc'].mean()
            r_auc_std = random_df['test_auc'].std()
            s_auc_mean = strat_df['test_auc'].mean()
            s_auc_std = strat_df['test_auc'].std()
            
            delta_auc = (r_auc_mean - s_auc_mean) * 100
            
            # Determinar qual é melhor
            if abs(delta_auc) < 0.1:
                winner = '≈ (Empate)'
            elif delta_auc > 0:
                winner = '✓ Random'
            else:
                winner = '✓ Stratified'
            
            stats_data.append([
                model,
                clf,
                f'{r_auc_mean:.4f} ± {r_auc_std:.4f}',
                f'{s_auc_mean:.4f} ± {s_auc_std:.4f}',
                f'{delta_auc:+.3f}%',
                winner
            ])
    
    headers = ['Modelo', 'Classificador', 'Random AUC', 'Stratified AUC', 
               'Δ (R-S)', 'Melhor']
    
    table = ax6.table(cellText=stats_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.2, 0.2, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Estilizar header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorir células baseado no vencedor
    for i in range(len(stats_data)):
        if '✓ Random' in stats_data[i][-1]:
            table[(i+1, len(headers)-1)].set_facecolor('#d4edff')
        elif '✓ Stratified' in stats_data[i][-1]:
            table[(i+1, len(headers)-1)].set_facecolor('#ffd4d4')
        else:
            table[(i+1, len(headers)-1)].set_facecolor('#f0f0f0')
    
    ax6.set_title('Resumo Estatístico: Comparação Random vs Stratified', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # ============================================================
    # Título Geral
    # ============================================================
    fig.suptitle('Análise Comparativa Completa: Random Split vs Stratified Split\n' + 
                 'Baseline Multiseed (5 seeds: 42, 123, 420, 777, 2024)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✅ Visualização salva em: {output_path}')
    
    return stats_data

def print_analysis_summary(df):
    """Imprime resumo textual da análise."""
    print('\n' + '='*80)
    print('ANÁLISE COMPARATIVA: RANDOM vs STRATIFIED SPLIT')
    print('='*80)
    
    models = df['model'].unique()
    classifiers = df['classifier'].unique()
    
    random_wins = 0
    strat_wins = 0
    ties = 0
    
    print('\n📊 RESULTADOS POR MODELO/CLASSIFICADOR:\n')
    
    for model in models:
        print(f'\n🧬 {model}:')
        for clf in classifiers:
            random_auc = df[(df['model'] == model) & 
                           (df['classifier'] == clf) & 
                           (df['split'] == 'Random')]['test_auc'].mean()
            strat_auc = df[(df['model'] == model) & 
                          (df['classifier'] == clf) & 
                          (df['split'] == 'Stratified')]['test_auc'].mean()
            
            delta = (random_auc - strat_auc) * 100
            
            print(f'   {clf}:')
            print(f'      Random:      {random_auc:.4f}')
            print(f'      Stratified:  {strat_auc:.4f}')
            print(f'      Δ (R-S):     {delta:+.3f}%', end='')
            
            if abs(delta) < 0.1:
                print(' → EMPATE')
                ties += 1
            elif delta > 0:
                print(' → ✓ RANDOM MELHOR')
                random_wins += 1
            else:
                print(' → ✓ STRATIFIED MELHOR')
                strat_wins += 1
    
    print('\n' + '='*80)
    print('PLACAR FINAL:')
    print('='*80)
    print(f'   Random ganhou:      {random_wins} vezes')
    print(f'   Stratified ganhou:  {strat_wins} vezes')
    print(f'   Empates:            {ties} vezes')
    print('='*80)
    
    # Análise de gaps
    print('\n📈 ANÁLISE DE GAPS (Overfitting):\n')
    
    for model in models:
        for clf in classifiers:
            random_gap = df[(df['model'] == model) & 
                           (df['classifier'] == clf) & 
                           (df['split'] == 'Random')]['gap_train_test'].mean()
            strat_gap = df[(df['model'] == model) & 
                          (df['classifier'] == clf) & 
                          (df['split'] == 'Stratified')]['gap_train_test'].mean()
            
            print(f'{model} - {clf}:')
            print(f'   Random Gap T→Te:      {random_gap:.2f}%')
            print(f'   Stratified Gap T→Te:  {strat_gap:.2f}%')
            print(f'   Diferença:            {(random_gap - strat_gap):.2f}%\n')

if __name__ == '__main__':
    print('🔍 Carregando resultados...')
    random_data, stratified_data = load_results()
    
    print('📊 Extraindo métricas...')
    random_results = extract_metrics(random_data, 'Random')
    stratified_results = extract_metrics(stratified_data, 'Stratified')
    
    all_results = random_results + stratified_results
    
    print('📈 Calculando estatísticas...')
    df, summary = calculate_summary_stats(all_results)
    
    print('🎨 Gerando visualização...')
    output_path = 'results/baseline_multiseed/detailed_comparison_random_vs_stratified.png'
    stats = create_comparison_visualization(df, output_path)
    
    print_analysis_summary(df)
    
    print('\n✅ Análise completa!')
