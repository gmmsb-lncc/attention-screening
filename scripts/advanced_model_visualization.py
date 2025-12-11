#!/usr/bin/env python3
"""
Visualizações avançadas para comparação de modelos de proteína.

Implementa visualizações científicas rigorosas:
- Radar charts para comparação multidimensional
- Heatmaps para correlação de métricas
- Scatter matrices para análise de trade-offs
- Pareto charts para eficiência computacional
- Box plots para distribuição de métricas por algoritmo

Usage:
    python scripts/advanced_model_visualization.py --results-dir results/protein_model_comparison
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from math import pi

# Configurar estilo científico
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


def create_radar_chart(df: pd.DataFrame, output_path: Path):
    """
    Radar chart para comparação multidimensional de métricas.
    
    Visualiza 6 métricas principais em formato polar para cada modelo,
    permitindo comparação direta de perfis de performance.
    
    Args:
        df: DataFrame com métricas de classificação
        output_path: Caminho para salvar figura
    """
    # Selecionar métricas normalizadas [0,1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'MCC']
    
    # Número de variáveis
    num_vars = len(metrics)
    
    # Calcular ângulos para cada eixo
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Fechar o círculo
    
    # Inicializar figura
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Cores distintas para cada modelo
    colors = sns.color_palette("husl", len(df))
    
    # Plotar cada modelo
    for idx, (_, row) in enumerate(df.iterrows()):
        values = row[metrics].values.flatten().tolist()
        values += values[:1]  # Fechar o polígono
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Protein_Model'], 
                color=colors[idx], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Configurar eixos
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Adicionar círculo de referência em 0.9 (excelente performance)
    ax.plot(angles, [0.9]*len(angles), 'k--', linewidth=1, alpha=0.3, label='Excelente (0.9)')
    
    # Título e legenda
    plt.title('Perfil Multidimensional de Performance\n(Métricas de Classificação)', 
             size=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: {output_path}")
    plt.close()


def create_performance_heatmap(df_class: pd.DataFrame, df_reg: pd.DataFrame, output_path: Path):
    """
    Heatmap de métricas normalizadas para todos os modelos.
    
    Visualização tipo matriz que facilita identificação de padrões
    e comparação direta entre modelos e métricas.
    
    Args:
        df_class: DataFrame com métricas de classificação
        df_reg: DataFrame com métricas de regressão
        output_path: Caminho para salvar figura
    """
    # Preparar dados
    models = df_class['Protein_Model'].values
    
    # Selecionar métricas principais
    class_metrics = ['Accuracy', 'F1', 'ROC_AUC', 'MCC']
    
    # Normalizar métricas de regressão para [0,1]
    # R² pode ser negativo, então normalizamos para [-1,1] -> [0,1]
    reg_metrics_data = []
    for model in models:
        reg_row = df_reg[df_reg['Protein_Model'] == model].iloc[0]
        # Normalizar R² de [-2, 1] para [0, 1] aproximadamente
        r2_norm = (reg_row['Test_R2'] + 2) / 3
        r2_norm = np.clip(r2_norm, 0, 1)
        # Normalizar MAE (inverter: menor é melhor)
        mae_norm = 1 / (1 + reg_row['Test_MAE'])
        reg_metrics_data.append([r2_norm, mae_norm])
    
    # Construir matriz de dados
    data_matrix = []
    for idx, model in enumerate(models):
        class_row = df_class[df_class['Protein_Model'] == model].iloc[0]
        row_data = [class_row[m] for m in class_metrics] + reg_metrics_data[idx]
        data_matrix.append(row_data)
    
    # Criar DataFrame
    all_metrics = class_metrics + ['R² (norm)', 'MAE⁻¹ (norm)']
    df_heatmap = pd.DataFrame(data_matrix, columns=all_metrics, index=models)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, linewidths=0.5, cbar_kws={'label': 'Score Normalizado'},
                ax=ax)
    
    # Configurar
    ax.set_title('Matriz de Performance por Modelo e Métrica\n(Valores Normalizados [0,1])', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Métricas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Modelo de Proteína', fontsize=12, fontweight='bold')
    
    # Rotacionar labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: {output_path}")
    plt.close()


def create_tradeoff_analysis(df_class: pd.DataFrame, df_reg: pd.DataFrame, output_path: Path):
    """
    Análise de trade-offs entre diferentes aspectos de performance.
    
    4 scatter plots mostrando relações importantes:
    - F1 vs R² (classificação vs regressão)
    - Accuracy vs MAE
    - Dimensão vs F1 (custo computacional vs performance)
    - Dimensão vs R² (custo computacional vs regressão)
    
    Args:
        df_class: DataFrame com métricas de classificação
        df_reg: DataFrame com métricas de regressão
        output_path: Caminho para salvar figura
    """
    # Merge dos dados
    df = pd.merge(df_class[['Protein_Model', 'Accuracy', 'F1', 'ROC_AUC', 'Embedding_Dim']], 
                  df_reg[['Protein_Model', 'Test_R2', 'Test_MAE']], 
                  on='Protein_Model')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Análise de Trade-offs: Performance vs Custo Computacional', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = sns.color_palette("husl", len(df))
    
    # 1. F1 vs R² (classificação vs regressão)
    ax = axes[0, 0]
    for idx, row in df.iterrows():
        ax.scatter(row['F1'], row['Test_R2'], s=300, alpha=0.6, 
                  color=colors[idx], edgecolors='black', linewidth=2)
        ax.annotate(row['Protein_Model'], (row['F1'], row['Test_R2']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='F1 = 0.9')
    ax.set_xlabel('F1 Score (Classificação)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test R² (Regressão)', fontsize=11, fontweight='bold')
    ax.set_title('Trade-off: Classificação vs Regressão', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Accuracy vs MAE (outra perspectiva)
    ax = axes[0, 1]
    for idx, row in df.iterrows():
        ax.scatter(row['Accuracy'], row['Test_MAE'], s=300, alpha=0.6,
                  color=colors[idx], edgecolors='black', linewidth=2)
        ax.annotate(row['Protein_Model'], (row['Accuracy'], row['Test_MAE']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='Acc = 0.9')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='MAE = 1.0')
    ax.set_xlabel('Accuracy (Classificação)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test MAE (Regressão, menor é melhor)', fontsize=11, fontweight='bold')
    ax.set_title('Trade-off: Accuracy vs MAE', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.invert_yaxis()  # Inverter eixo Y (menor MAE é melhor)
    
    # 3. Embedding Dimension vs F1 (eficiência classificação)
    ax = axes[1, 0]
    for idx, row in df.iterrows():
        ax.scatter(row['Embedding_Dim'], row['F1'], s=300, alpha=0.6,
                  color=colors[idx], edgecolors='black', linewidth=2)
        ax.annotate(row['Protein_Model'], (row['Embedding_Dim'], row['F1']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Linha de pareto ideal (aproximação)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='F1 = 0.9')
    ax.set_xlabel('Dimensão do Embedding\n(Custo Computacional)', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score (Performance)', fontsize=11, fontweight='bold')
    ax.set_title('Eficiência: Custo vs Performance (Classificação)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Destacar região ótima (alto F1, baixa dimensão)
    ax.axvspan(1000, 2000, alpha=0.1, color='green', label='Região Eficiente')
    
    # 4. Embedding Dimension vs R² (eficiência regressão)
    ax = axes[1, 1]
    for idx, row in df.iterrows():
        ax.scatter(row['Embedding_Dim'], row['Test_R2'], s=300, alpha=0.6,
                  color=colors[idx], edgecolors='black', linewidth=2)
        ax.annotate(row['Protein_Model'], (row['Embedding_Dim'], row['Test_R2']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='R² = 0.3')
    ax.set_xlabel('Dimensão do Embedding\n(Custo Computacional)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test R² (Performance)', fontsize=11, fontweight='bold')
    ax.set_title('Eficiência: Custo vs Performance (Regressão)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: {output_path}")
    plt.close()


def create_classifier_distribution(results: Dict[str, Dict], output_path: Path):
    """
    Box plots mostrando distribuição de performance por tipo de classificador.
    
    Analisa quais algoritmos (RandomForest, XGBoost, etc.) performam melhor
    em cada modelo de proteína.
    
    Args:
        results: Dicionário com resultados completos
        output_path: Caminho para salvar figura
    """
    # Coletar métricas de todos os classificadores
    data_list = []
    
    for model_name, model_data in results.items():
        if 'classifier' not in model_data or not model_data['classifier']['success']:
            continue
        
        individual = model_data['classifier']['individual_results']
        
        for clf_name, metrics in individual.items():
            data_list.append({
                'Protein_Model': model_name,
                'Classifier': clf_name,
                'F1': metrics['F1'],
                'ROC_AUC': metrics['ROC_AUC'],
                'Accuracy': metrics['Accuracy']
            })
    
    df = pd.DataFrame(data_list)
    
    # Criar figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distribuição de Performance por Algoritmo de Classificação', 
                 fontsize=14, fontweight='bold')
    
    metrics_to_plot = ['Accuracy', 'F1', 'ROC_AUC']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Box plot
        sns.boxplot(data=df, x='Classifier', y=metric, ax=ax, 
                   palette='Set2', linewidth=1.5)
        
        # Swarm plot sobreposto para ver pontos individuais
        sns.swarmplot(data=df, x='Classifier', y=metric, ax=ax,
                     color='black', alpha=0.5, size=4)
        
        ax.set_xlabel('Algoritmo', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'Distribuição de {metric}', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Linha de referência
        if metric in ['Accuracy', 'F1']:
            ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: {output_path}")
    plt.close()


def create_regressor_distribution(results: Dict[str, Dict], output_path: Path):
    """
    Box plots mostrando distribuição de performance por tipo de regressor.
    
    Args:
        results: Dicionário com resultados completos
        output_path: Caminho para salvar figura
    """
    # Coletar métricas de todos os regressores
    data_list = []
    
    for model_name, model_data in results.items():
        if 'regression' not in model_data or not model_data['regression']['success']:
            continue
        
        individual = model_data['regression']['individual_results']
        
        for reg_name, metrics in individual.items():
            data_list.append({
                'Protein_Model': model_name,
                'Regressor': reg_name,
                'Pearson_R': metrics['Pearson_R'],
                'Pearson_P': metrics['Pearson_P'],
                'RMSE': metrics['RMSE']
            })
    
    df = pd.DataFrame(data_list)
    
    # Criar figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distribuição de Performance por Algoritmo de Regressão', 
                 fontsize=14, fontweight='bold')
    
    metrics_to_plot = [
        ('Pearson_R', 'Pearson R (correlação, -1 a 1)'),
        ('Pearson_P', 'Pearson P-value (menor é melhor)'),
        ('RMSE', 'RMSE (menor é melhor)')
    ]
    
    for idx, (metric, label) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Box plot
        sns.boxplot(data=df, x='Regressor', y=metric, ax=ax,
                   palette='Set3', linewidth=1.5)
        
        # Swarm plot
        sns.swarmplot(data=df, x='Regressor', y=metric, ax=ax,
                     color='black', alpha=0.5, size=4)
        
        ax.set_xlabel('Algoritmo', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'Distribuição de {metric}', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Linhas de referência
        if metric == 'Pearson_R':
            ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Forte (0.7)')
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Moderada (0.5)')
        elif metric == 'Pearson_P':
            ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, linewidth=1, label='α=0.05')
            ax.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, linewidth=1, label='α=0.01')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: {output_path}")
    plt.close()


def create_pareto_chart(df_class: pd.DataFrame, df_reg: pd.DataFrame, output_path: Path):
    """
    Pareto chart mostrando fronteira de eficiência.
    
    Identifica modelos que oferecem melhor trade-off entre
    performance e custo computacional (dimensão).
    
    Args:
        df_class: DataFrame com métricas de classificação
        df_reg: DataFrame com métricas de regressão
        output_path: Caminho para salvar figura
    """
    # Merge
    df = pd.merge(
        df_class[['Protein_Model', 'F1', 'Embedding_Dim']], 
        df_reg[['Protein_Model', 'Test_R2']], 
        on='Protein_Model'
    )
    
    # Score composto (média de F1 e R² normalizado)
    df['R2_norm'] = (df['Test_R2'] + 2) / 3  # Normalizar [-2,1] -> [0,1]
    df['R2_norm'] = df['R2_norm'].clip(0, 1)
    df['Combined_Score'] = (df['F1'] + df['R2_norm']) / 2
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = sns.color_palette("husl", len(df))
    
    # Scatter principal
    for idx, row in df.iterrows():
        ax.scatter(row['Embedding_Dim'], row['Combined_Score'], 
                  s=500, alpha=0.7, color=colors[idx], 
                  edgecolors='black', linewidth=2, zorder=3)
        
        # Label
        ax.annotate(row['Protein_Model'], 
                   (row['Embedding_Dim'], row['Combined_Score']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    # Identificar fronteira de Pareto
    # Ordenar por dimensão
    df_sorted = df.sort_values('Embedding_Dim')
    pareto_points = []
    max_score = -np.inf
    
    for idx, row in df_sorted.iterrows():
        if row['Combined_Score'] > max_score:
            pareto_points.append((row['Embedding_Dim'], row['Combined_Score']))
            max_score = row['Combined_Score']
    
    # Desenhar fronteira de Pareto
    if len(pareto_points) > 1:
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, alpha=0.7, 
               label='Fronteira de Pareto (Eficiente)', zorder=2)
        ax.fill_between(pareto_x, 0, pareto_y, alpha=0.1, color='green',
                       label='Região Dominada')
    
    # Configurar
    ax.set_xlabel('Dimensão do Embedding (Custo Computacional)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Score Combinado\n(Média de F1 e R² Normalizado)', 
                 fontsize=12, fontweight='bold')
    ax.set_title('Análise de Pareto: Eficiência vs Performance\n' + 
                '(Modelos na fronteira são ótimos de Pareto)', 
                fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(fontsize=11, loc='lower right')
    
    # Adicionar anotações sobre regiões
    ax.text(0.05, 0.95, '← Mais Eficiente\n(Menor Custo)', 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax.text(0.95, 0.05, 'Maior Performance →', 
           transform=ax.transAxes, fontsize=10,
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: {output_path}")
    plt.close()


def load_json_results(files: List[str]) -> Dict[str, Dict]:
    """Carrega resultados dos JSONs originais."""
    results = {}
    for file in files:
        path = Path(file)
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                model_name = data['config'].get('esm_model', 'unknown')
                results[model_name] = data
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Visualizações avançadas para comparação de modelos'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Diretório com resultados da comparação'
    )
    parser.add_argument(
        '--json-files',
        nargs='+',
        help='Arquivos JSON originais para análises detalhadas'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print("\n" + "="*80)
    print("🎨 VISUALIZAÇÕES AVANÇADAS - COMPARAÇÃO DE MODELOS")
    print("="*80 + "\n")
    
    # Carregar CSVs
    print("📂 Carregando dados...")
    df_class = pd.read_csv(results_dir / 'classification_metrics.csv')
    df_reg = pd.read_csv(results_dir / 'regression_metrics.csv')
    print(f"✓ {len(df_class)} modelos carregados\n")
    
    # Criar visualizações
    print("🎨 Gerando visualizações avançadas...\n")
    
    create_radar_chart(df_class, results_dir / 'radar_chart.png')
    create_performance_heatmap(df_class, df_reg, results_dir / 'performance_heatmap.png')
    create_tradeoff_analysis(df_class, df_reg, results_dir / 'tradeoff_analysis.png')
    create_pareto_chart(df_class, df_reg, results_dir / 'pareto_chart.png')
    
    # Se JSONs foram fornecidos, criar distribuições
    if args.json_files:
        print("\n📊 Gerando análises de distribuição...\n")
        results = load_json_results(args.json_files)
        create_classifier_distribution(results, results_dir / 'classifier_distribution.png')
        create_regressor_distribution(results, results_dir / 'regressor_distribution.png')
    
    print("\n" + "="*80)
    print("✅ VISUALIZAÇÕES CONCLUÍDAS!")
    print("="*80 + "\n")
    print(f"📊 Resultados salvos em: {results_dir}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
