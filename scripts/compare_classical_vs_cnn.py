#!/usr/bin/env python3
"""
Script para comparação entre Modelos Clássicos e CNN+Atenção Cruzada.

Compara as performances de algoritmos clássicos (XGBoost, Random Forest, etc.)
com a arquitetura CNN+Atenção Cruzada em múltiplos modelos de proteína.

Usage:
    python scripts/compare_classical_vs_cnn.py \
        --classical results/integrated_results_*.json \
        --cnn results/results_*.json \
        --output results/classical_vs_cnn_comparison
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Configuração de estilo
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_classical_results(files: List[str]) -> Dict:
    """Carrega resultados de modelos clássicos (ML algorithms)."""
    results = {}
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extrair nome do modelo do arquivo ou config
            model_name = Path(file_path).stem.replace('integrated_results_', '')
            if not model_name or model_name == 'integrated_results':
                model_name = data.get('config', {}).get('protein_model', 'unknown')
            
            # Extrair métricas dos melhores classificadores
            classifier = data.get('classifier', {})
            regression = data.get('regression', {})
            
            if classifier.get('success') and regression.get('success'):
                best_class_metrics = classifier.get('best_metrics', {})
                
                # Regressão: pode ter best_metrics OU usar test_results do melhor modelo
                best_reg_metrics = regression.get('best_metrics', {})
                if not best_reg_metrics:
                    # Obter métricas do melhor modelo em test_results
                    best_model = regression.get('best_model', '')
                    test_results = regression.get('test_results', {})
                    if best_model and best_model in test_results:
                        model_results = test_results[best_model]
                        
                        # Converter 'None' string para 0
                        def safe_float(val, default=0):
                            if val is None or val == 'None' or val == 'none':
                                return default
                            try:
                                return float(val)
                            except (ValueError, TypeError):
                                return default
                        
                        best_reg_metrics = {
                            'Pearson_R': safe_float(model_results.get('Pearson_R', 0)),
                            'Pearson_P': safe_float(model_results.get('Pearson_P', 1), 1),
                            'Spearman_R': safe_float(model_results.get('Spearman_R', 0)),
                            'R2': safe_float(model_results.get('R2', 0)),
                            'RMSE': safe_float(model_results.get('RMSE', 999), 999),
                            'MAE': safe_float(model_results.get('MAE', 999), 999),
                        }
                    else:
                        # Fallback para valores diretos
                        best_reg_metrics = {
                            'R2': regression.get('best_r2', 0),
                            'RMSE': regression.get('best_rmse', 999),
                            'MAE': regression.get('best_mae', 999),
                        }
                
                results[model_name] = {
                    'type': 'classical',
                    'classification': {
                        'best_model': classifier.get('best_model', 'Unknown'),
                        'accuracy': best_class_metrics.get('Accuracy', best_class_metrics.get('accuracy', 0)),
                        'f1': best_class_metrics.get('F1', best_class_metrics.get('f1', 0)),
                        'roc_auc': best_class_metrics.get('ROC_AUC', best_class_metrics.get('roc_auc', 0)),
                        'mcc': best_class_metrics.get('MCC', best_class_metrics.get('mcc', 0)),
                        'precision': best_class_metrics.get('Precision', best_class_metrics.get('precision', 0)),
                        'recall': best_class_metrics.get('Recall', best_class_metrics.get('recall', 0)),
                    },
                    'regression': {
                        'best_model': regression.get('best_model', 'Unknown'),
                        'pearson_r': best_reg_metrics.get('Pearson_R', best_reg_metrics.get('pearson_r', 0)),
                        'pearson_p': best_reg_metrics.get('Pearson_P', best_reg_metrics.get('pearson_p', 1)),
                        'spearman_r': best_reg_metrics.get('Spearman_R', best_reg_metrics.get('spearman_r', 0)),
                        'r2': best_reg_metrics.get('R2', best_reg_metrics.get('r2', 0)),
                        'rmse': best_reg_metrics.get('RMSE', best_reg_metrics.get('rmse', 999)),
                        'mae': best_reg_metrics.get('MAE', best_reg_metrics.get('mae', 999)),
                    },
                    'individual_results': {
                        'classification': classifier.get('individual_results', {}),
                        'regression': regression.get('individual_results', {})
                    }
                }
                print(f"✓ Classical: {model_name} ({classifier['best_model']} / {regression['best_model']})")
            
        except Exception as e:
            print(f"⚠️  Erro ao carregar {file_path}: {e}")
    
    return results


def load_cnn_results(files: List[str]) -> Dict:
    """Carrega resultados de CNN+Atenção Cruzada."""
    results = {}
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            model_name = Path(file_path).stem.replace('results_', '')
            
            metrics = data.get('metrics', {})
            
            results[model_name] = {
                'type': 'cnn',
                'classification': {
                    'best_model': 'CNN+CrossAttention',
                    'accuracy': metrics.get('classification', {}).get('accuracy', 0),
                    'f1': metrics.get('classification', {}).get('f1', 0),
                    'roc_auc': metrics.get('classification', {}).get('roc_auc', 0),
                    'mcc': metrics.get('classification', {}).get('mcc', 0),
                    'precision': metrics.get('classification', {}).get('precision', 0),
                    'recall': metrics.get('classification', {}).get('recall', 0),
                },
                'regression': {
                    'best_model': 'CNN+CrossAttention',
                    'pearson_r': metrics.get('regression', {}).get('pearson_r', 0),
                    'pearson_p': metrics.get('regression', {}).get('pearson_p', 1),
                    'spearman_r': metrics.get('regression', {}).get('spearman_r', 0),
                    'r2': metrics.get('regression', {}).get('r2', 0),
                    'rmse': metrics.get('regression', {}).get('rmse', 999),
                    'mae': metrics.get('regression', {}).get('mae', 999),
                }
            }
            print(f"✓ CNN: {model_name}")
            
        except Exception as e:
            print(f"⚠️  Erro ao carregar {file_path}: {e}")
    
    return results


def create_comparison_dataframe(classical: Dict, cnn: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cria DataFrames para comparação."""
    
    class_data = []
    reg_data = []
    
    # Processar modelos clássicos
    for model, data in classical.items():
        class_data.append({
            'Protein_Model': model,
            'Type': 'Classical ML',
            'Algorithm': data['classification']['best_model'],
            'Accuracy': data['classification']['accuracy'],
            'F1': data['classification']['f1'],
            'ROC_AUC': data['classification']['roc_auc'],
            'MCC': data['classification']['mcc'],
            'Precision': data['classification']['precision'],
            'Recall': data['classification']['recall'],
        })
        
        reg_data.append({
            'Protein_Model': model,
            'Type': 'Classical ML',
            'Algorithm': data['regression']['best_model'],
            'Pearson_R': data['regression']['pearson_r'],
            'Pearson_P': data['regression']['pearson_p'],
            'Spearman_R': data['regression']['spearman_r'],
            'R2': data['regression']['r2'],
            'RMSE': data['regression']['rmse'],
            'MAE': data['regression']['mae'],
        })
    
    # Processar CNN
    for model, data in cnn.items():
        class_data.append({
            'Protein_Model': model,
            'Type': 'CNN+Attention',
            'Algorithm': 'CNN+CrossAttention',
            'Accuracy': data['classification']['accuracy'],
            'F1': data['classification']['f1'],
            'ROC_AUC': data['classification']['roc_auc'],
            'MCC': data['classification']['mcc'],
            'Precision': data['classification']['precision'],
            'Recall': data['classification']['recall'],
        })
        
        reg_data.append({
            'Protein_Model': model,
            'Type': 'CNN+Attention',
            'Algorithm': 'CNN+CrossAttention',
            'Pearson_R': data['regression']['pearson_r'],
            'Pearson_P': data['regression']['pearson_p'],
            'Spearman_R': data['regression']['spearman_r'],
            'R2': data['regression']['r2'],
            'RMSE': data['regression']['rmse'],
            'MAE': data['regression']['mae'],
        })
    
    df_class = pd.DataFrame(class_data)
    df_reg = pd.DataFrame(reg_data)
    
    return df_class, df_reg


def plot_classification_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plota comparação de classificação: Classical vs CNN."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Métricas principais
    metrics = ['F1', 'Accuracy', 'ROC_AUC', 'MCC']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.suptitle('Comparação: Modelos Clássicos vs CNN+Atenção Cruzada - Classificação', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = {'Classical ML': '#3498db', 'CNN+Attention': '#e74c3c'}
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Agrupar por Protein_Model
        df_pivot = df.pivot_table(
            values=metric, 
            index='Protein_Model', 
            columns='Type',
            aggfunc='mean'
        ).fillna(0)
        
        # Plotar barras agrupadas
        x = np.arange(len(df_pivot.index))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_pivot.get('Classical ML', [0]*len(x)), 
                       width, label='Classical ML', color=colors['Classical ML'], alpha=0.8)
        bars2 = ax.bar(x + width/2, df_pivot.get('CNN+Attention', [0]*len(x)), 
                       width, label='CNN+Attention', color=colors['CNN+Attention'], alpha=0.8)
        
        # Adicionar valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Configurações
        ax.set_title(metric, fontweight='bold', fontsize=13, pad=10)
        ax.set_ylabel('Score', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(df_pivot.index, rotation=45, ha='right', fontsize=9)
        
        if metric == 'MCC':
            ax.set_ylim(-0.2, 1.05)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        else:
            ax.set_ylim(0, 1.05)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)
        
        # Linhas de referência
        if metric != 'MCC':
            ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.2, linewidth=1)
            ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.2, linewidth=1)
    
    plt.tight_layout()
    output_path = output_dir / 'classification_classical_vs_cnn.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_regression_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plota comparação de regressão: Classical vs CNN."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = ['Pearson_R', 'R2', 'RMSE', 'MAE']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.suptitle('Comparação: Modelos Clássicos vs CNN+Atenção Cruzada - Regressão', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = {'Classical ML': '#3498db', 'CNN+Attention': '#e74c3c'}
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        df_pivot = df.pivot_table(
            values=metric, 
            index='Protein_Model', 
            columns='Type',
            aggfunc='mean'
        ).fillna(0)
        
        x = np.arange(len(df_pivot.index))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_pivot.get('Classical ML', [0]*len(x)), 
                       width, label='Classical ML', color=colors['Classical ML'], alpha=0.8)
        bars2 = ax.bar(x + width/2, df_pivot.get('CNN+Attention', [0]*len(x)), 
                       width, label='CNN+Attention', color=colors['CNN+Attention'], alpha=0.8)
        
        # Adicionar valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(metric, fontweight='bold', fontsize=13, pad=10)
        ax.set_ylabel('Score', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(df_pivot.index, rotation=45, ha='right', fontsize=9)
        
        # Ajustar limites baseado na métrica
        if metric in ['Pearson_R', 'R2']:
            y_min = min(df[metric].min() - 0.1, -0.1)
            y_max = max(df[metric].max() + 0.1, 1.0)
            ax.set_ylim(y_min, y_max)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        else:  # RMSE, MAE
            ax.set_ylim(0, df[metric].max() * 1.2)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'regression_classical_vs_cnn.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_aggregated_comparison(df_class: pd.DataFrame, df_reg: pd.DataFrame, output_dir: Path) -> Path:
    """Plota comparação agregada (média por tipo de modelo)."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance Média: Classical ML vs CNN+Atenção Cruzada', 
                 fontsize=16, fontweight='bold')
    
    colors = {'Classical ML': '#3498db', 'CNN+Attention': '#e74c3c'}
    
    # Classificação
    ax = axes[0]
    class_metrics = ['Accuracy', 'F1', 'ROC_AUC', 'MCC']
    class_means = df_class.groupby('Type')[class_metrics].mean()
    
    x = np.arange(len(class_metrics))
    width = 0.35
    
    for i, (type_name, row) in enumerate(class_means.iterrows()):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, row.values, width, label=type_name, 
                     color=colors[type_name], alpha=0.8)
        
        for bar, val in zip(bars, row.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title('Classificação', fontweight='bold', fontsize=14, pad=15)
    ax.set_ylabel('Score Médio', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_metrics, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Regressão
    ax = axes[1]
    reg_metrics = ['Pearson_R', 'R2']
    reg_means = df_reg.groupby('Type')[reg_metrics].mean()
    
    x = np.arange(len(reg_metrics))
    
    for i, (type_name, row) in enumerate(reg_means.iterrows()):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, row.values, width, label=type_name,
                     color=colors[type_name], alpha=0.8)
        
        for bar, val in zip(bars, row.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title('Regressão', fontweight='bold', fontsize=14, pad=15)
    ax.set_ylabel('Score Médio', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(reg_metrics, fontsize=11)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylim(reg_means.values.min() - 0.1, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'aggregated_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_best_algorithm_distribution(classical: Dict, output_dir: Path) -> Path:
    """Plota distribuição dos melhores algoritmos clássicos."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Contar algoritmos
    class_algos = {}
    reg_algos = {}
    
    for model, data in classical.items():
        class_algo = data['classification']['best_model']
        reg_algo = data['regression']['best_model']
        
        class_algos[class_algo] = class_algos.get(class_algo, 0) + 1
        reg_algos[reg_algo] = reg_algos.get(reg_algo, 0) + 1
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Distribuição dos Melhores Algoritmos Clássicos por Protein Model', 
                 fontsize=16, fontweight='bold')
    
    # Classificação
    ax = axes[0]
    if class_algos:
        algos = list(class_algos.keys())
        counts = list(class_algos.values())
        colors_palette = sns.color_palette('Set2', len(algos))
        
        bars = ax.barh(algos, counts, color=colors_palette, alpha=0.8)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{count}', va='center', fontsize=11, fontweight='bold')
        
        ax.set_title('Classificação', fontweight='bold', fontsize=14, pad=15)
        ax.set_xlabel('Número de Protein Models', fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
    
    # Regressão
    ax = axes[1]
    if reg_algos:
        algos = list(reg_algos.keys())
        counts = list(reg_algos.values())
        colors_palette = sns.color_palette('Set3', len(algos))
        
        bars = ax.barh(algos, counts, color=colors_palette, alpha=0.8)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{count}', va='center', fontsize=11, fontweight='bold')
        
        ax.set_title('Regressão', fontweight='bold', fontsize=14, pad=15)
        ax.set_xlabel('Número de Protein Models', fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'best_algorithms_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_summary_report(df_class: pd.DataFrame, df_reg: pd.DataFrame, 
                           classical: Dict, cnn: Dict, output_dir: Path) -> Path:
    """Gera relatório de resumo em markdown."""
    
    output_path = output_dir / 'comparison_summary.md'
    
    with open(output_path, 'w') as f:
        f.write('# Comparação: Modelos Clássicos vs CNN+Atenção Cruzada\n\n')
        f.write(f'**Data:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('---\n\n')
        
        # Estatísticas gerais
        f.write('## 📊 Estatísticas Gerais\n\n')
        f.write(f'- **Protein Models Clássicos:** {len(classical)}\n')
        f.write(f'- **Protein Models CNN:** {len(cnn)}\n')
        f.write(f'- **Total de Comparações:** {len(df_class)}\n\n')
        
        # Performance média
        f.write('## 🎯 Performance Média\n\n')
        f.write('### Classificação\n\n')
        class_means = df_class.groupby('Type')[['Accuracy', 'F1', 'ROC_AUC', 'MCC']].mean()
        f.write(class_means.to_markdown() + '\n\n')
        
        f.write('### Regressão\n\n')
        reg_means = df_reg.groupby('Type')[['Pearson_R', 'R2', 'RMSE', 'MAE']].mean()
        f.write(reg_means.to_markdown() + '\n\n')
        
        # Melhores modelos
        f.write('## 🏆 Melhores Modelos\n\n')
        f.write('### Classificação (por F1)\n')
        best_class = df_class.nlargest(3, 'F1')[['Protein_Model', 'Type', 'Algorithm', 'F1', 'Accuracy', 'ROC_AUC']]
        f.write(best_class.to_markdown(index=False) + '\n\n')
        
        f.write('### Regressão (por Pearson R)\n')
        best_reg = df_reg.nlargest(3, 'Pearson_R')[['Protein_Model', 'Type', 'Algorithm', 'Pearson_R', 'R2', 'RMSE']]
        f.write(best_reg.to_markdown(index=False) + '\n\n')
        
        # Análise comparativa
        f.write('## 🔍 Análise Comparativa\n\n')
        
        # Vencedores por métrica
        f.write('### Vencedor por Métrica (Média Geral)\n\n')
        
        metrics_comparison = []
        for metric in ['Accuracy', 'F1', 'ROC_AUC', 'MCC']:
            means = df_class.groupby('Type')[metric].mean()
            winner = means.idxmax()
            diff = means.max() - means.min()
            metrics_comparison.append({
                'Métrica': metric,
                'Vencedor': winner,
                'Diferença': f'{diff:.3f}'
            })
        
        for metric in ['Pearson_R', 'R2']:
            means = df_reg.groupby('Type')[metric].mean()
            winner = means.idxmax()
            diff = means.max() - means.min()
            metrics_comparison.append({
                'Métrica': metric,
                'Vencedor': winner,
                'Diferença': f'{diff:.3f}'
            })
        
        df_comparison = pd.DataFrame(metrics_comparison)
        f.write(df_comparison.to_markdown(index=False) + '\n\n')
        
        # Distribuição de algoritmos clássicos
        if classical:
            f.write('### Algoritmos Clássicos Mais Utilizados\n\n')
            
            class_algos = {}
            reg_algos = {}
            
            for model, data in classical.items():
                class_algo = data['classification']['best_model']
                reg_algo = data['regression']['best_model']
                
                class_algos[class_algo] = class_algos.get(class_algo, 0) + 1
                reg_algos[reg_algo] = reg_algos.get(reg_algo, 0) + 1
            
            f.write('**Classificação:**\n')
            for algo, count in sorted(class_algos.items(), key=lambda x: x[1], reverse=True):
                f.write(f'- {algo}: {count} modelos\n')
            
            f.write('\n**Regressão:**\n')
            for algo, count in sorted(reg_algos.items(), key=lambda x: x[1], reverse=True):
                f.write(f'- {algo}: {count} modelos\n')
        
        f.write('\n---\n\n')
        f.write('## 💡 Conclusões\n\n')
        
        # Análise automática
        class_winner = df_class.groupby('Type')['F1'].mean().idxmax()
        reg_winner = df_reg.groupby('Type')['Pearson_R'].mean().idxmax()
        
        f.write(f'- **Classificação:** {class_winner} apresenta melhor performance média em F1-Score\n')
        f.write(f'- **Regressão:** {reg_winner} apresenta melhor performance média em Pearson R\n\n')
        
        # Análise por protein model
        protein_models = set(df_class['Protein_Model'].unique())
        common_models = protein_models.intersection(df_reg['Protein_Model'].unique())
        
        if common_models:
            f.write(f'\n**Análise por Protein Model:** {len(common_models)} modelos comparados\n\n')
    
    return output_path


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Comparação entre Modelos Clássicos e CNN+Atenção Cruzada'
    )
    parser.add_argument(
        '--classical',
        nargs='+',
        required=True,
        help='Arquivos JSON com resultados de modelos clássicos'
    )
    parser.add_argument(
        '--cnn',
        nargs='+',
        required=True,
        help='Arquivos JSON com resultados de CNN+Atenção'
    )
    parser.add_argument(
        '--output',
        default='results/classical_vs_cnn',
        help='Diretório de saída'
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('🔬 Comparação: Modelos Clássicos vs CNN+Atenção Cruzada')
    print('=' * 60)
    
    # 1. Carregar dados
    print('\n📂 Carregando resultados...')
    classical = load_classical_results(args.classical)
    cnn = load_cnn_results(args.cnn)
    
    if not classical:
        print('❌ Nenhum resultado clássico carregado!')
        return 1
    
    if not cnn:
        print('❌ Nenhum resultado CNN carregado!')
        return 1
    
    print(f'✅ {len(classical)} modelos clássicos e {len(cnn)} modelos CNN carregados')
    
    # 2. Criar DataFrames
    print('\n📊 Criando DataFrames de comparação...')
    df_class, df_reg = create_comparison_dataframe(classical, cnn)
    
    # 3. Gerar visualizações
    print('\n📈 Gerando visualizações...')
    
    plots = []
    
    print('  → Comparação de classificação...')
    plots.append(plot_classification_comparison(df_class, output_dir))
    
    print('  → Comparação de regressão...')
    plots.append(plot_regression_comparison(df_reg, output_dir))
    
    print('  → Comparação agregada...')
    plots.append(plot_aggregated_comparison(df_class, df_reg, output_dir))
    
    print('  → Distribuição de algoritmos...')
    plots.append(plot_best_algorithm_distribution(classical, output_dir))
    
    # 4. Salvar CSVs
    print('\n💾 Salvando dados em CSV...')
    df_class.to_csv(output_dir / 'classification_comparison.csv', index=False)
    df_reg.to_csv(output_dir / 'regression_comparison.csv', index=False)
    
    # 5. Gerar relatório
    print('\n📄 Gerando relatório de resumo...')
    report_path = generate_summary_report(df_class, df_reg, classical, cnn, output_dir)
    
    # Resumo final
    print('\n' + '=' * 60)
    print('✅ Comparação concluída!')
    print(f'\n📁 Resultados salvos em: {output_dir}')
    print(f'📊 {len(plots)} visualizações geradas')
    print(f'📄 2 arquivos CSV gerados')
    print(f'📝 Relatório: {report_path.name}')
    
    # Insights rápidos
    print('\n💡 Insights Rápidos:')
    
    class_means = df_class.groupby('Type')[['F1', 'Accuracy']].mean()
    reg_means = df_reg.groupby('Type')[['Pearson_R', 'R2']].mean()
    
    print('\n  📊 Classificação (F1 médio):')
    for type_name, row in class_means.iterrows():
        print(f'    {type_name}: F1={row["F1"]:.3f} | Acc={row["Accuracy"]:.3f}')
    
    print('\n  📈 Regressão (Pearson R médio):')
    for type_name, row in reg_means.iterrows():
        print(f'    {type_name}: Pearson={row["Pearson_R"]:.3f} | R²={row["R2"]:.3f}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
