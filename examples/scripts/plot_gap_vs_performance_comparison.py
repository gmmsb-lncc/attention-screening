#!/usr/bin/env python3
"""
Gera gráfico Gap vs Performance comparando Random e Stratified splits.
Similar ao gráfico de análise de overfitting, mas comparando ambos os métodos.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

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
    results = {}
    
    for model_name, model_data in data['models'].items():
        # Simplificar nome do modelo
        if '8M' in model_name:
            short_name = '8M'
        elif '150M' in model_name:
            short_name = '150M'
        elif '3B' in model_name:
            short_name = '3B'
        else:
            short_name = model_name
        
        for clf_name in ['KNN', 'MLP']:
            key = f'{short_name}_{clf_name}'
            
            train_aucs = []
            val_aucs = []
            test_aucs = []
            
            # Coletar métricas de todos os seeds
            for seed_result in model_data['seed_results']:
                clf_data = seed_result['classifiers'][clf_name]
                train_aucs.append(clf_data['train_metrics']['roc_auc'])
                val_aucs.append(clf_data['val_metrics']['roc_auc'])
                test_aucs.append(clf_data['test_metrics']['roc_auc'])
            
            # Calcular médias
            train_mean = np.mean(train_aucs)
            val_mean = np.mean(val_aucs)
            test_mean = np.mean(test_aucs)
            
            gap_train_test = (train_mean - test_mean) * 100
            
            results[key] = {
                'model': short_name,
                'classifier': clf_name,
                'split': split_type,
                'train_auc': train_mean,
                'val_auc': val_mean,
                'test_auc': test_mean,
                'gap_train_test': gap_train_test
            }
    
    return results

def create_gap_vs_performance_plot(random_results, stratified_results, output_path):
    """Cria gráfico Gap vs Performance similar ao da análise de overfitting."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Definir zonas coloridas
    # Zona verde: gap < 3% (robusto)
    ax.axvspan(0, 3, alpha=0.15, color='green', label='Robusto (<3%)')
    
    # Zona amarela: 3% < gap < 5% (atenção)
    ax.axvspan(3, 5, alpha=0.15, color='yellow', label='Atenção (3-5%)')
    
    # Zona vermelha: gap > 5% (overfitting)
    ax.axvspan(5, 8, alpha=0.15, color='red', label='Overfitting (>5%)')
    
    # Linhas verticais de limiar
    ax.axvline(x=3, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Cores e marcadores
    colors_clf = {'KNN': 'tab:blue', 'MLP': 'tab:orange'}
    markers_split = {'Random': 'o', 'Stratified': 's'}
    sizes_split = {'Random': 180, 'Stratified': 120}
    
    # Plotar pontos
    plotted_labels = set()
    
    for key, data in random_results.items():
        label_clf = data['classifier'] if data['classifier'] not in plotted_labels else None
        if label_clf:
            plotted_labels.add(data['classifier'])
        
        ax.scatter(data['gap_train_test'], data['test_auc'],
                  c=colors_clf[data['classifier']], 
                  marker=markers_split['Random'],
                  s=sizes_split['Random'], 
                  alpha=0.7,
                  edgecolors='black', linewidths=1.5,
                  label=label_clf)
    
    # Plotar stratified por cima (menor)
    for key, data in stratified_results.items():
        ax.scatter(data['gap_train_test'], data['test_auc'],
                  c=colors_clf[data['classifier']], 
                  marker=markers_split['Stratified'],
                  s=sizes_split['Stratified'], 
                  alpha=0.9,
                  edgecolors='darkgreen', linewidths=2)
    
    # Adicionar anotações
    offset_x = 0.15
    offset_y = 0.0008
    
    for key, r_data in random_results.items():
        s_data = stratified_results[key]
        
        # Posição média entre random e stratified
        x_pos = (r_data['gap_train_test'] + s_data['gap_train_test']) / 2
        y_pos = max(r_data['test_auc'], s_data['test_auc']) + offset_y
        
        # Nome simplificado
        model_label = f"{r_data['model']}\n{r_data['classifier']}"
        
        ax.annotate(model_label, 
                   xy=(x_pos, y_pos),
                   fontsize=9, 
                   ha='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            edgecolor='gray',
                            alpha=0.8))
        
        # Linha conectando random e stratified
        ax.plot([r_data['gap_train_test'], s_data['gap_train_test']],
               [r_data['test_auc'], s_data['test_auc']],
               'gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Configurações dos eixos
    ax.set_xlabel('Gap Train → Test (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test ROC-AUC', fontsize=13, fontweight='bold')
    ax.set_title('Gap vs Performance: Identificação de Overfitting\n' +
                 'Comparação Random (●) vs Stratified (■) Split\n' +
                 '(Ideal: baixo gap + alta performance)',
                 fontsize=15, fontweight='bold', pad=20)
    
    # Limites
    ax.set_xlim([2, 6.5])
    ax.set_ylim([0.92, 0.98])
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Criar legenda customizada
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor='gray', markersize=12, 
               markeredgecolor='black', markeredgewidth=1.5,
               label='Random Split'),
        Line2D([0], [0], marker='s', color='w', 
               markerfacecolor='gray', markersize=10, 
               markeredgecolor='darkgreen', markeredgewidth=2,
               label='Stratified Split'),
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor='tab:blue', markersize=10,
               label='KNN'),
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor='tab:orange', markersize=10,
               label='MLP'),
        Patch(facecolor='green', alpha=0.3, label='Robusto (<3%)'),
        Patch(facecolor='yellow', alpha=0.3, label='Atenção (3-5%)'),
        Patch(facecolor='red', alpha=0.3, label='Overfitting (>5%)')
    ]
    
    ax.legend(handles=legend_elements, 
             loc='lower left', 
             fontsize=10,
             framealpha=0.95,
             edgecolor='black')
    
    # Adicionar nota explicativa
    note_text = ("Nota: Círculos (●) = Random | Quadrados (■) = Stratified\n"
                 "Linhas pontilhadas conectam os mesmos modelos nos dois métodos de split")
    ax.text(0.98, 0.02, note_text,
           transform=ax.transAxes,
           fontsize=8,
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✅ Gráfico salvo em: {output_path}')
    
    # Imprimir análise textual
    print('\n' + '='*80)
    print('ANÁLISE GAP vs PERFORMANCE: RANDOM vs STRATIFIED')
    print('='*80)
    
    for key in sorted(random_results.keys()):
        r_data = random_results[key]
        s_data = stratified_results[key]
        
        print(f'\n📊 {r_data["model"]} - {r_data["classifier"]}:')
        print(f'   Random:      Gap={r_data["gap_train_test"]:.2f}%  |  Test AUC={r_data["test_auc"]:.4f}')
        print(f'   Stratified:  Gap={s_data["gap_train_test"]:.2f}%  |  Test AUC={s_data["test_auc"]:.4f}')
        
        gap_diff = r_data['gap_train_test'] - s_data['gap_train_test']
        auc_diff = (r_data['test_auc'] - s_data['test_auc']) * 100
        
        print(f'   Δ Gap:       {gap_diff:+.2f}% ({"Random maior" if gap_diff > 0 else "Stratified maior"})')
        print(f'   Δ AUC:       {auc_diff:+.3f}% ({"Random melhor" if auc_diff > 0 else "Stratified melhor"})')
        
        # Diagnóstico
        if r_data['gap_train_test'] < 3 and s_data['gap_train_test'] < 3:
            print('   ✅ Ambos ROBUSTOS')
        elif r_data['gap_train_test'] > 5 or s_data['gap_train_test'] > 5:
            print('   ⚠️  Pelo menos um com OVERFITTING')
        else:
            print('   ⚠️  Ambos em zona de ATENÇÃO')
    
    print('\n' + '='*80)
    print('RESUMO GERAL')
    print('='*80)
    
    # Contar por zona
    random_robust = sum(1 for d in random_results.values() if d['gap_train_test'] < 3)
    random_attention = sum(1 for d in random_results.values() if 3 <= d['gap_train_test'] < 5)
    random_overfit = sum(1 for d in random_results.values() if d['gap_train_test'] >= 5)
    
    strat_robust = sum(1 for d in stratified_results.values() if d['gap_train_test'] < 3)
    strat_attention = sum(1 for d in stratified_results.values() if 3 <= d['gap_train_test'] < 5)
    strat_overfit = sum(1 for d in stratified_results.values() if d['gap_train_test'] >= 5)
    
    print(f'\nRandom Split:')
    print(f'   Robusto (<3%):       {random_robust}/6')
    print(f'   Atenção (3-5%):      {random_attention}/6')
    print(f'   Overfitting (>5%):   {random_overfit}/6')
    
    print(f'\nStratified Split:')
    print(f'   Robusto (<3%):       {strat_robust}/6')
    print(f'   Atenção (3-5%):      {strat_attention}/6')
    print(f'   Overfitting (>5%):   {strat_overfit}/6')
    
    # Performance média
    random_avg_auc = np.mean([d['test_auc'] for d in random_results.values()])
    strat_avg_auc = np.mean([d['test_auc'] for d in stratified_results.values()])
    
    print(f'\nTest AUC Médio:')
    print(f'   Random:      {random_avg_auc:.4f}')
    print(f'   Stratified:  {strat_avg_auc:.4f}')
    print(f'   Diferença:   {(random_avg_auc - strat_avg_auc)*100:+.3f}%')
    
    print('='*80)

if __name__ == '__main__':
    print('🔍 Carregando resultados...')
    random_data, stratified_data = load_results()
    
    print('📊 Extraindo métricas...')
    random_results = extract_metrics(random_data, 'Random')
    stratified_results = extract_metrics(stratified_data, 'Stratified')
    
    print('🎨 Gerando gráfico Gap vs Performance...')
    output_path = 'results/overfitting_analysis/gap_vs_performance_random_stratified.png'
    create_gap_vs_performance_plot(random_results, stratified_results, output_path)
    
    print('\n✅ Análise completa!')
