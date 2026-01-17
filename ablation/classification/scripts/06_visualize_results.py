#!/usr/bin/env python3
"""
Script 06: Visualização dos Resultados de Ablação
Cria gráficos comparativos entre as 4 combinações de representações.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Configuração
BASE_DIR = Path('/media/leon/ssd2tb/docktkinase/ablation/classification')
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Estilo
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_results():
    """Carrega resultados do CSV."""
    csv_path = RESULTS_DIR / 'classification_summary.csv'
    df = pd.read_csv(csv_path)
    return df


def extract_combination_info(combination_name):
    """Extrai informação sobre a combinação (C1-C4, modelo ESM, etc)."""
    parts = combination_name.split('_')
    
    # C1, C2, C3, C4
    combination = parts[0]
    
    # Proteína: ESM ou OneHot
    if 'ESM' in combination_name:
        protein_type = 'ESM-2'
        if 'esm2_t6_8M' in combination_name:
            protein_model = '8M'
        elif 'esm2_t30_150M' in combination_name:
            protein_model = '150M'
        elif 'esm2_t36_3B' in combination_name:
            protein_model = '3B'
        else:
            protein_model = 'Unknown'
    else:
        protein_type = 'One-Hot'
        protein_model = 'N/A'
    
    # Ligante: SMITED ou Morgan
    if 'SMITED' in combination_name:
        ligand_type = 'SMI-TED'
    else:
        ligand_type = 'Morgan FP'
    
    return {
        'combination': combination,
        'protein_type': protein_type,
        'protein_model': protein_model,
        'ligand_type': ligand_type,
        'full_name': f"{combination} ({protein_type} + {ligand_type})"
    }


def plot_combination_comparison(df):
    """
    Gráfico principal: comparação entre C1-C4 para cada métrica.
    Layout 2x3: Accuracy, Precision, Recall, F1-Score, ROC-AUC, MCC
    """
    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc', 'test_mcc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC']
    
    # Adicionar informação de combinação
    df['comb_info'] = df['combination'].apply(extract_combination_info)
    df['comb_type'] = df['comb_info'].apply(lambda x: x['combination'])
    df['protein_repr'] = df['comb_info'].apply(lambda x: x['protein_type'])
    df['ligand_repr'] = df['comb_info'].apply(lambda x: x['ligand_type'])
    
    # Cores para cada combinação
    colors = {
        'C1': '#e74c3c',  # Vermelho - ESM+SMITED (mais complexo)
        'C2': '#f39c12',  # Laranja - ESM+Morgan
        'C3': '#3498db',  # Azul - OneHot+SMITED
        'C4': '#27ae60',  # Verde - OneHot+Morgan (mais simples)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Ablation Study: Comparação de Representações\n(C1: ESM+SMITED | C2: ESM+Morgan | C3: OneHot+SMITED | C4: OneHot+Morgan)',
                 fontsize=14, fontweight='bold', y=1.00)
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Agrupar por combinação e classificador
        grouped = df.groupby(['comb_type', 'classifier'])[metric].agg(['mean', 'std']).reset_index()
        
        # Separar KNN e MLP
        knn_data = grouped[grouped['classifier'] == 'KNN']
        mlp_data = grouped[grouped['classifier'] == 'MLP']
        
        x = np.arange(len(['C1', 'C2', 'C3', 'C4']))
        width = 0.35
        
        # Ordenar por C1, C2, C3, C4
        knn_means = []
        knn_stds = []
        mlp_means = []
        mlp_stds = []
        
        for comb in ['C1', 'C2', 'C3', 'C4']:
            knn_row = knn_data[knn_data['comb_type'] == comb]
            mlp_row = mlp_data[mlp_data['comb_type'] == comb]
            
            if len(knn_row) > 0:
                knn_means.append(knn_row['mean'].values[0])
                knn_stds.append(knn_row['std'].values[0])
            else:
                knn_means.append(0)
                knn_stds.append(0)
            
            if len(mlp_row) > 0:
                mlp_means.append(mlp_row['mean'].values[0])
                mlp_stds.append(mlp_row['std'].values[0])
            else:
                mlp_means.append(0)
                mlp_stds.append(0)
        
        # Plotar barras com erro
        bars1 = ax.bar(x - width/2, knn_means, width, yerr=knn_stds,
                       label='KNN', alpha=0.8, capsize=5,
                       color=[colors[c] for c in ['C1', 'C2', 'C3', 'C4']],
                       edgecolor='black', linewidth=1.5)
        
        bars2 = ax.bar(x + width/2, mlp_means, width, yerr=mlp_stds,
                       label='MLP', alpha=0.6, capsize=5,
                       color=[colors[c] for c in ['C1', 'C2', 'C3', 'C4']],
                       edgecolor='black', linewidth=1.5, hatch='//')
        
        # Adicionar valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['C1\n(ESM+SMITED)', 'C2\n(ESM+Morgan)', 
                           'C3\n(OneHot+SMITED)', 'C4\n(OneHot+Morgan)'],
                          fontsize=9)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Remover bordas superiores e direitas
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Legenda única no canto superior direito (fora dos plots)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
               fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'ablation_comparison_all_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Gráfico comparativo salvo: {output_path}")


def plot_auc_focus(df):
    """Gráfico focado em ROC-AUC com barras de erro."""
    # Adicionar informação de combinação
    df['comb_info'] = df['combination'].apply(extract_combination_info)
    df['comb_type'] = df['comb_info'].apply(lambda x: x['combination'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Agrupar por combinação e classificador
    grouped = df.groupby(['comb_type', 'classifier'])['test_auc'].agg(['mean', 'std']).reset_index()
    
    knn_data = grouped[grouped['classifier'] == 'KNN']
    mlp_data = grouped[grouped['classifier'] == 'MLP']
    
    x = np.arange(4)
    width = 0.35
    
    knn_means = [knn_data[knn_data['comb_type'] == c]['mean'].values[0] if len(knn_data[knn_data['comb_type'] == c]) > 0 else 0 for c in ['C1', 'C2', 'C3', 'C4']]
    knn_stds = [knn_data[knn_data['comb_type'] == c]['std'].values[0] if len(knn_data[knn_data['comb_type'] == c]) > 0 else 0 for c in ['C1', 'C2', 'C3', 'C4']]
    mlp_means = [mlp_data[mlp_data['comb_type'] == c]['mean'].values[0] if len(mlp_data[mlp_data['comb_type'] == c]) > 0 else 0 for c in ['C1', 'C2', 'C3', 'C4']]
    mlp_stds = [mlp_data[mlp_data['comb_type'] == c]['std'].values[0] if len(mlp_data[mlp_data['comb_type'] == c]) > 0 else 0 for c in ['C1', 'C2', 'C3', 'C4']]
    
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
    
    bars1 = ax.bar(x - width/2, knn_means, width, yerr=knn_stds,
                   label='KNN', alpha=0.8, capsize=8,
                   color=colors, edgecolor='black', linewidth=2)
    
    bars2 = ax.bar(x + width/2, mlp_means, width, yerr=mlp_stds,
                   label='MLP', alpha=0.6, capsize=8,
                   color=colors, edgecolor='black', linewidth=2, hatch='//')
    
    # Valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('ROC-AUC', fontsize=14, fontweight='bold')
    ax.set_xlabel('Combinação de Representações', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Study: ROC-AUC por Combinação\n(média ± std sobre 5 seeds)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['C1\nESM-2 + SMI-TED\n(mais complexo)', 
                       'C2\nESM-2 + Morgan FP', 
                       'C3\nOne-Hot + SMI-TED',
                       'C4\nOne-Hot + Morgan FP\n(mais simples)'],
                      fontsize=11)
    ax.set_ylim(0.92, 0.97)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='lower right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'ablation_auc_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Gráfico ROC-AUC salvo: {output_path}")


def plot_protein_ligand_contribution(df):
    """
    Gráfico 2x2 mostrando contribuição de proteína e ligante.
    Eixo X: Proteína (One-Hot vs ESM-2)
    Eixo Y: Ligante (Morgan vs SMI-TED)
    """
    df['comb_info'] = df['combination'].apply(extract_combination_info)
    df['protein_repr'] = df['comb_info'].apply(lambda x: x['protein_type'])
    df['ligand_repr'] = df['comb_info'].apply(lambda x: x['ligand_type'])
    
    # Calcular média por proteína/ligante/classificador
    grouped = df.groupby(['protein_repr', 'ligand_repr', 'classifier'])['test_auc'].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, classifier in enumerate(['KNN', 'MLP']):
        ax = axes[idx]
        data = grouped[grouped['classifier'] == classifier].pivot(
            index='ligand_repr', columns='protein_repr', values='test_auc'
        )
        
        # Reordenar colunas e índices
        data = data.reindex(index=['Morgan FP', 'SMI-TED'], 
                           columns=['One-Hot', 'ESM-2'])
        
        # Heatmap
        sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlGn', 
                   vmin=0.93, vmax=0.96, cbar_kws={'label': 'ROC-AUC'},
                   linewidths=2, linecolor='black', ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
        
        ax.set_title(f'{classifier} - ROC-AUC\n(média sobre seeds)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Representação da Proteína', fontsize=11, fontweight='bold')
        ax.set_ylabel('Representação do Ligante', fontsize=11, fontweight='bold')
        
        # Adicionar anotações de combinação
        ax.text(0.5, 0.5, 'C4', ha='center', va='top', fontsize=10, color='blue', weight='bold')
        ax.text(1.5, 0.5, 'C2', ha='center', va='top', fontsize=10, color='blue', weight='bold')
        ax.text(0.5, 1.5, 'C3', ha='center', va='top', fontsize=10, color='blue', weight='bold')
        ax.text(1.5, 1.5, 'C1', ha='center', va='top', fontsize=10, color='blue', weight='bold')
    
    plt.suptitle('Contribuição de Proteína vs Ligante para ROC-AUC', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'protein_ligand_contribution_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Heatmap de contribuição salvo: {output_path}")


def create_summary_table(df):
    """Cria tabela resumo com médias e desvios padrão."""
    df['comb_info'] = df['combination'].apply(extract_combination_info)
    df['comb_type'] = df['comb_info'].apply(lambda x: x['combination'])
    
    # Agrupar por combinação e classificador
    summary = df.groupby(['comb_type', 'classifier']).agg({
        'test_accuracy': ['mean', 'std'],
        'test_precision': ['mean', 'std'],
        'test_recall': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'test_auc': ['mean', 'std'],
        'test_mcc': ['mean', 'std']
    }).round(4)
    
    # Salvar como CSV
    output_path = FIGURES_DIR / 'ablation_summary_table.csv'
    summary.to_csv(output_path)
    
    print(f"✓ Tabela resumo salva: {output_path}")
    
    # Imprimir no console
    print("\n" + "="*80)
    print("RESUMO DOS RESULTADOS (média ± std)")
    print("="*80)
    print(summary.to_string())
    print("="*80)


def main():
    print("="*80)
    print("VISUALIZAÇÃO DOS RESULTADOS DE ABLAÇÃO")
    print("="*80)
    print()
    
    # Carregar dados
    print("📊 Carregando resultados...")
    df = load_results()
    print(f"   Carregados {len(df)} resultados")
    print()
    
    # Gráfico 1: Comparação completa (2x3)
    print("📈 Criando gráfico comparativo de todas as métricas...")
    plot_combination_comparison(df)
    print()
    
    # Gráfico 2: Foco em ROC-AUC
    print("📈 Criando gráfico focado em ROC-AUC...")
    plot_auc_focus(df)
    print()
    
    # Gráfico 3: Heatmap de contribuição
    print("📈 Criando heatmap de contribuição proteína/ligante...")
    plot_protein_ligand_contribution(df)
    print()
    
    # Tabela resumo
    print("📋 Gerando tabela resumo...")
    create_summary_table(df)
    print()
    
    print("="*80)
    print("✅ VISUALIZAÇÕES CONCLUÍDAS")
    print("="*80)
    print(f"\nFiguras salvas em: {FIGURES_DIR}")
    print(f"   - ablation_comparison_all_metrics.png (comparação completa)")
    print(f"   - ablation_auc_comparison.png (foco em ROC-AUC)")
    print(f"   - protein_ligand_contribution_heatmap.png (contribuição)")
    print(f"   - ablation_summary_table.csv (tabela resumo)")


if __name__ == '__main__':
    main()
