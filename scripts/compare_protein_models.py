#!/usr/bin/env python3
"""
Script para comparar resultados de diferentes modelos de proteína.

Analisa e visualiza métricas de classificação e regressão para benchmarks
de modelos ESM-2 (diferentes tamanhos), Boltz2 e ESMC.

Usage:
    python scripts/compare_protein_models.py --results-dir /path/to/results/
    python scripts/compare_protein_models.py --files file1.json file2.json file3.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_results(files: List[str]) -> Dict[str, Dict]:
    """
    Carrega resultados de múltiplos arquivos JSON.
    
    Args:
        files: Lista de caminhos para arquivos JSON
        
    Returns:
        Dict com nome do modelo -> dados completos
    """
    results = {}
    
    for file in files:
        path = Path(file)
        if not path.exists():
            print(f"⚠️  Arquivo não encontrado: {file}")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Extrair nome do modelo da config
        model_name = data['config'].get('esm_model', 'unknown')
        results[model_name] = data
        print(f"✓ Carregado: {model_name} ({path.name})")
    
    return results


def extract_classification_metrics(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Extrai métricas de classificação de todos os modelos.
    
    Args:
        results: Dict com resultados por modelo
        
    Returns:
        DataFrame com métricas organizadas
    """
    data = []
    
    for model_name, model_data in results.items():
        if 'classifier' not in model_data or not model_data['classifier']['success']:
            continue
            
        best_metrics = model_data['classifier']['best_metrics']
        best_model = model_data['classifier']['best_model']
        
        data.append({
            'Protein_Model': model_name,
            'Best_Classifier': best_model,
            'Accuracy': best_metrics['Accuracy'],
            'Precision': best_metrics['Precision'],
            'Recall': best_metrics['Recall'],
            'F1': best_metrics['F1'],
            'ROC_AUC': best_metrics['ROC_AUC'],
            'MCC': best_metrics['MCC'],
            'Specificity': best_metrics['Specificity'],
            'Embedding_Dim': model_data['build']['embedding_dim']
        })
    
    return pd.DataFrame(data)


def extract_regression_metrics(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Extrai métricas de regressão de todos os modelos.
    
    Args:
        results: Dict com resultados por modelo
        
    Returns:
        DataFrame com métricas organizadas
    """
    data = []
    
    for model_name, model_data in results.items():
        if 'regression' not in model_data or not model_data['regression']['success']:
            continue
            
        reg_data = model_data['regression']
        
    data.append({
        'Protein_Model': model_name,
        'Best_Regressor': reg_data['best_model'],
        'Test_MAE': reg_data['best_test_mae'],
        'Test_R2': reg_data['best_test_r2'],
        'Val_MAE': reg_data['best_val_mae'],
        'Val_R2': reg_data['best_val_r2'],
        'Embedding_Dim': model_data['build']['embedding_dim']
    })
    
    # Adicionar métricas de Pearson dos resultados de teste
    test_results = reg_data.get('test_results', {})
    best_model = reg_data['best_model']
    if best_model in test_results:
        data[-1]['Pearson_R'] = test_results[best_model].get('Pearson_R', 0)
        data[-1]['Pearson_P'] = test_results[best_model].get('Pearson_P', 1)
        data[-1]['RMSE'] = test_results[best_model].get('RMSE', 0)
    else:
        data[-1]['Pearson_R'] = 0
        data[-1]['Pearson_P'] = 1
        data[-1]['RMSE'] = 0
    
    return pd.DataFrame(data)
def plot_classification_comparison(df: pd.DataFrame, output_path: Path):
    """
    Plota comparação de métricas de classificação.
    
    Args:
        df: DataFrame com métricas
        output_path: Caminho para salvar figura
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparação de Classificação entre Modelos de Proteína', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'MCC']
    colors = sns.color_palette("husl", len(df))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Ordenar por métrica
        df_sorted = df.sort_values(metric, ascending=False)
        
        bars = ax.barh(df_sorted['Protein_Model'], df_sorted[metric], color=colors)
        ax.set_xlabel(metric, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Adicionar valores nas barras
        for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
        
        # Destacar melhor modelo
        best_idx = df_sorted[metric].idxmax()
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: {output_path}")
    plt.close()


def plot_regression_comparison(df: pd.DataFrame, output_path: Path):
    """
    Plota comparação de métricas de regressão.
    
    Args:
        df: DataFrame com métricas
        output_path: Caminho para salvar figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparação de Regressão entre Modelos de Proteína', 
                 fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("husl", len(df))
    
    # Test MAE
    ax = axes[0, 0]
    df_sorted = df.sort_values('Test_MAE')
    bars = ax.barh(df_sorted['Protein_Model'], df_sorted['Test_MAE'], color=colors)
    ax.set_xlabel('Test MAE (menor é melhor)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, df_sorted['Test_MAE']):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=9)
    bars[0].set_edgecolor('gold')
    bars[0].set_linewidth(3)
    
    # Test R²
    ax = axes[0, 1]
    df_sorted = df.sort_values('Test_R2', ascending=False)
    bars = ax.barh(df_sorted['Protein_Model'], df_sorted['Test_R2'], color=colors)
    ax.set_xlabel('Test R² (maior é melhor)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, df_sorted['Test_R2']):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=9)
    bars[0].set_edgecolor('gold')
    bars[0].set_linewidth(3)
    
    # Val MAE
    ax = axes[1, 0]
    df_sorted = df.sort_values('Val_MAE')
    bars = ax.barh(df_sorted['Protein_Model'], df_sorted['Val_MAE'], color=colors)
    ax.set_xlabel('Validation MAE (menor é melhor)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, df_sorted['Val_MAE']):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=9)
    
    # Val R²
    ax = axes[1, 1]
    df_sorted = df.sort_values('Val_R2', ascending=False)
    bars = ax.barh(df_sorted['Protein_Model'], df_sorted['Val_R2'], color=colors)
    ax.set_xlabel('Validation R² (maior é melhor)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, df_sorted['Val_R2']):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: {output_path}")
    plt.close()


def plot_embedding_dimensions(df_class: pd.DataFrame, df_reg: pd.DataFrame, output_path: Path):
    """
    Plota relação entre dimensão de embedding e performance.
    
    Args:
        df_class: DataFrame com métricas de classificação
        df_reg: DataFrame com métricas de regressão
        output_path: Caminho para salvar figura
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dimensão de Embedding vs Performance', 
                 fontsize=14, fontweight='bold')
    
    # Classificação: Embedding vs F1
    ax = axes[0]
    ax.scatter(df_class['Embedding_Dim'], df_class['F1'], 
              s=200, alpha=0.6, c=range(len(df_class)), cmap='viridis')
    for _, row in df_class.iterrows():
        ax.annotate(row['Protein_Model'], 
                   (row['Embedding_Dim'], row['F1']),
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, alpha=0.8)
    ax.set_xlabel('Dimensão Total do Embedding', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Classificação')
    ax.grid(True, alpha=0.3)
    
    # Regressão: Embedding vs R²
    ax = axes[1]
    ax.scatter(df_reg['Embedding_Dim'], df_reg['Test_R2'], 
              s=200, alpha=0.6, c=range(len(df_reg)), cmap='viridis')
    for _, row in df_reg.iterrows():
        ax.annotate(row['Protein_Model'], 
                   (row['Embedding_Dim'], row['Test_R2']),
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, alpha=0.8)
    ax.set_xlabel('Dimensão Total do Embedding', fontweight='bold')
    ax.set_ylabel('Test R²', fontweight='bold')
    ax.set_title('Regressão')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: {output_path}")
    plt.close()


def plot_overall_ranking(df_class: pd.DataFrame, df_reg: pd.DataFrame, output_path: Path):
    """
    Plota ranking geral considerando todas as métricas.
    
    Args:
        df_class: DataFrame com métricas de classificação
        df_reg: DataFrame com métricas de regressão
        output_path: Caminho para salvar figura
    """
    # Calcular score composto
    scores = []
    
    for model in df_class['Protein_Model']:
        class_row = df_class[df_class['Protein_Model'] == model].iloc[0]
        reg_row = df_reg[df_reg['Protein_Model'] == model].iloc[0]
        
        # Score de classificação (média de métricas chave)
        class_score = (class_row['F1'] + class_row['ROC_AUC'] + class_row['MCC']) / 3
        
        # Score de regressão (normalizar R² para [0,1] e usar MAE invertido)
        reg_score = (reg_row['Test_R2'] + 1) / 2  # Normalizar R² de [-1,1] para [0,1]
        mae_score = 1 / (1 + reg_row['Test_MAE'])  # Inverter MAE (menor é melhor)
        reg_score = (reg_score + mae_score) / 2
        
        # Score final (média ponderada)
        final_score = 0.5 * class_score + 0.5 * reg_score
        
        scores.append({
            'Model': model,
            'Classification_Score': class_score,
            'Regression_Score': reg_score,
            'Overall_Score': final_score,
            'Embedding_Dim': class_row['Embedding_Dim']
        })
    
    df_scores = pd.DataFrame(scores).sort_values('Overall_Score', ascending=False)
    
    # Plotar
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_scores))
    width = 0.25
    
    bars1 = ax.bar(x - width, df_scores['Classification_Score'], width, 
                   label='Classification', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, df_scores['Regression_Score'], width, 
                   label='Regression', color='coral', alpha=0.8)
    bars3 = ax.bar(x + width, df_scores['Overall_Score'], width, 
                   label='Overall', color='gold', alpha=0.8)
    
    ax.set_xlabel('Modelo de Proteína', fontweight='bold')
    ax.set_ylabel('Score Normalizado', fontweight='bold')
    ax.set_title('Ranking Geral de Modelos', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_scores['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Adicionar valores
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: {output_path}")
    plt.close()
    
    return df_scores


def generate_summary_table(df_class: pd.DataFrame, df_reg: pd.DataFrame, 
                          df_scores: pd.DataFrame, output_path: Path):
    """
    Gera tabela resumo em formato markdown.
    
    Args:
        df_class: DataFrame com métricas de classificação
        df_reg: DataFrame com métricas de regressão
        df_scores: DataFrame com scores gerais
        output_path: Caminho para salvar tabela
    """
    with open(output_path, 'w') as f:
        f.write("# Comparação de Modelos de Proteína - Resultados\n\n")
        f.write(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Ranking geral
        f.write("## 🏆 Ranking Geral\n\n")
        f.write("| Posição | Modelo | Overall Score | Classification | Regression | Embedding Dim |\n")
        f.write("|---------|--------|---------------|----------------|------------|---------------|\n")
        for idx, row in df_scores.iterrows():
            pos = idx + 1
            medal = "🥇" if pos == 1 else "🥈" if pos == 2 else "🥉" if pos == 3 else f"{pos}º"
            f.write(f"| {medal} | {row['Model']} | {row['Overall_Score']:.3f} | "
                   f"{row['Classification_Score']:.3f} | {row['Regression_Score']:.3f} | "
                   f"{row['Embedding_Dim']} |\n")
        
        # Classificação detalhada
        f.write("\n## 📊 Classificação (Métricas Detalhadas)\n\n")
        f.write("| Modelo | Best Classifier | Accuracy | F1 | ROC-AUC | MCC | Precision | Recall |\n")
        f.write("|--------|----------------|----------|-------|---------|-----|-----------|--------|\n")
        for _, row in df_class.sort_values('F1', ascending=False).iterrows():
            f.write(f"| {row['Protein_Model']} | {row['Best_Classifier']} | "
                   f"{row['Accuracy']:.3f} | {row['F1']:.3f} | {row['ROC_AUC']:.3f} | "
                   f"{row['MCC']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} |\n")
        
        # Regressão detalhada
        f.write("\n## 📈 Regressão (Métricas Detalhadas)\n\n")
        f.write("| Modelo | Best Regressor | Test MAE | Test R² | Val MAE | Val R² |\n")
        f.write("|--------|---------------|----------|---------|---------|--------|\n")
        for _, row in df_reg.sort_values('Test_R2', ascending=False).iterrows():
            f.write(f"| {row['Protein_Model']} | {row['Best_Regressor']} | "
                   f"{row['Test_MAE']:.3f} | {row['Test_R2']:.3f} | "
                   f"{row['Val_MAE']:.3f} | {row['Val_R2']:.3f} |\n")
        
        # Análise
        f.write("\n## 💡 Análise\n\n")
        
        best_overall = df_scores.iloc[0]
        best_class = df_class.sort_values('F1', ascending=False).iloc[0]
        best_reg = df_reg.sort_values('Test_R2', ascending=False).iloc[0]
        
        f.write(f"- **Melhor modelo geral**: {best_overall['Model']} "
               f"(score: {best_overall['Overall_Score']:.3f})\n")
        f.write(f"- **Melhor classificação**: {best_class['Protein_Model']} "
               f"(F1: {best_class['F1']:.3f}, usando {best_class['Best_Classifier']})\n")
        f.write(f"- **Melhor regressão**: {best_reg['Protein_Model']} "
               f"(R²: {best_reg['Test_R2']:.3f}, usando {best_reg['Best_Regressor']})\n")
        
        f.write("\n### Dimensão de Embedding\n\n")
        f.write(f"- Menor dimensão: {df_class['Embedding_Dim'].min()} "
               f"({df_class[df_class['Embedding_Dim'] == df_class['Embedding_Dim'].min()]['Protein_Model'].values[0]})\n")
        f.write(f"- Maior dimensão: {df_class['Embedding_Dim'].max()} "
               f"({df_class[df_class['Embedding_Dim'] == df_class['Embedding_Dim'].max()]['Protein_Model'].values[0]})\n")
        f.write(f"- Dimensão média: {df_class['Embedding_Dim'].mean():.0f}\n")
    
    print(f"✓ Salvo: {output_path}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Comparar resultados de diferentes modelos de proteína'
    )
    parser.add_argument(
        '--files', 
        nargs='+',
        help='Lista de arquivos JSON com resultados'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        help='Diretório contendo arquivos JSON de resultados'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/comparison',
        help='Diretório para salvar visualizações (default: results/comparison)'
    )
    
    args = parser.parse_args()
    
    # Coletar arquivos
    files = []
    if args.files:
        files = args.files
    elif args.results_dir:
        results_dir = Path(args.results_dir)
        files = list(results_dir.glob('**/integrated_results*.json'))
        files = [str(f) for f in files]
    else:
        print("❌ Forneça --files ou --results-dir")
        return 1
    
    if not files:
        print("❌ Nenhum arquivo JSON encontrado")
        return 1
    
    print(f"\n{'='*80}")
    print("🔬 COMPARAÇÃO DE MODELOS DE PROTEÍNA")
    print(f"{'='*80}\n")
    
    # Carregar resultados
    print("📂 Carregando resultados...")
    results = load_results(files)
    
    if not results:
        print("❌ Nenhum resultado válido carregado")
        return 1
    
    print(f"\n✓ {len(results)} modelos carregados\n")
    
    # Criar diretório de saída
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Diretório de saída: {output_dir}\n")
    
    # Extrair métricas
    print("📊 Extraindo métricas...")
    df_class = extract_classification_metrics(results)
    df_reg = extract_regression_metrics(results)
    
    if df_class.empty or df_reg.empty:
        print("❌ Falha ao extrair métricas")
        return 1
    
    print(f"✓ Classificação: {len(df_class)} modelos")
    print(f"✓ Regressão: {len(df_reg)} modelos\n")
    
    # Gerar visualizações
    print("📈 Gerando visualizações...\n")
    
    plot_classification_comparison(df_class, output_dir / 'classification_comparison.png')
    plot_regression_comparison(df_reg, output_dir / 'regression_comparison.png')
    plot_embedding_dimensions(df_class, df_reg, output_dir / 'embedding_dimensions.png')
    df_scores = plot_overall_ranking(df_class, df_reg, output_dir / 'overall_ranking.png')
    
    # Gerar tabela resumo
    print("\n📝 Gerando relatório...")
    generate_summary_table(df_class, df_reg, df_scores, output_dir / 'SUMMARY.md')
    
    # Salvar DataFrames como CSV
    df_class.to_csv(output_dir / 'classification_metrics.csv', index=False)
    df_reg.to_csv(output_dir / 'regression_metrics.csv', index=False)
    df_scores.to_csv(output_dir / 'overall_scores.csv', index=False)
    print(f"✓ Salvo: {output_dir / 'classification_metrics.csv'}")
    print(f"✓ Salvo: {output_dir / 'regression_metrics.csv'}")
    print(f"✓ Salvo: {output_dir / 'overall_scores.csv'}")
    
    print(f"\n{'='*80}")
    print("✅ COMPARAÇÃO CONCLUÍDA!")
    print(f"{'='*80}\n")
    print(f"📊 Resultados salvos em: {output_dir}")
    print(f"📄 Veja o relatório completo em: {output_dir / 'SUMMARY.md'}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
