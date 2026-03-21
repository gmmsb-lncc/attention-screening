#!/usr/bin/env python3
"""
Gráfico de Barras: Comparação KNN vs MLP (Split Aleatório)
==========================================================

Plota gráfico de barras comparando KNN e MLP para os modelos ESM-2
(8M, 150M, 3B) com barras de erro representando o desvio padrão
das 5 seeds.

Autor: DockTKinase Project
Data: Janeiro 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configurações de estilo
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_results(json_path: Path) -> dict:
    """Carrega resultados do arquivo JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_metrics(results: dict) -> dict:
    """
    Extrai métricas de test AUC para cada modelo e classificador.
    
    Returns:
        {
            'esm2_t6_8M_UR50D': {
                'KNN': {'mean': float, 'std': float, 'values': list},
                'MLP': {'mean': float, 'std': float, 'values': list}
            },
            ...
        }
    """
    metrics = {}
    
    for esm_model, model_data in results.items():
        if esm_model in ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']:
            metrics[esm_model] = {}
            
            for classifier in ['KNN', 'MLP']:
                if classifier in model_data:
                    clf_data = model_data[classifier]
                    
                    # Extrair valores de test AUC de cada seed
                    test_aucs = []
                    for seed_result in clf_data.get('per_seed_results', []):
                        test_auc = seed_result.get('test_metrics', {}).get('roc_auc', None)
                        if test_auc is not None:
                            test_aucs.append(test_auc)
                    
                    if test_aucs:
                        metrics[esm_model][classifier] = {
                            'mean': np.mean(test_aucs),
                            'std': np.std(test_aucs),
                            'values': test_aucs
                        }
    
    return metrics


def create_comparison_barplot(metrics: dict, output_path: Path):
    """
    Cria gráfico de barras comparando KNN vs MLP.
    
    Args:
        metrics: Dicionário com métricas extraídas
        output_path: Caminho para salvar o gráfico
    """
    # Ordem dos modelos
    models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']
    model_labels = ['8M (320D)', '150M (640D)', '3B (2560D)']
    
    # Extrair dados
    knn_means = []
    knn_stds = []
    mlp_means = []
    mlp_stds = []
    
    for model in models:
        knn_data = metrics.get(model, {}).get('KNN', {'mean': 0, 'std': 0})
        mlp_data = metrics.get(model, {}).get('MLP', {'mean': 0, 'std': 0})
        
        knn_means.append(knn_data['mean'])
        knn_stds.append(knn_data['std'])
        mlp_means.append(mlp_data['mean'])
        mlp_stds.append(mlp_data['std'])
    
    # Configuração do gráfico
    x = np.arange(len(model_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Barras
    bars1 = ax.bar(x - width/2, knn_means, width, yerr=knn_stds,
                   label='KNN (k=5)', color='#3498db', alpha=0.8,
                   capsize=5, error_kw={'linewidth': 1.5})
    bars2 = ax.bar(x + width/2, mlp_means, width, yerr=mlp_stds,
                   label='MLP (256→128)', color='#e74c3c', alpha=0.8,
                   capsize=5, error_kw={'linewidth': 1.5})
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Labels e título
    ax.set_xlabel('Modelo ESM-2', fontweight='bold')
    ax.set_ylabel('ROC-AUC (Test Set)', fontweight='bold')
    ax.set_title('Comparação KNN vs MLP - Split Aleatório\n(Média ± Desvio Padrão de 5 Seeds)',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.93, 0.98])
    
    # Adicionar linha horizontal de referência
    ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='0.95 baseline')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f'✅ Gráfico salvo: {output_path}')
    plt.close()


def create_detailed_table(metrics: dict):
    """Imprime tabela detalhada com as métricas."""
    print('\n' + '=' * 80)
    print('TABELA DETALHADA: KNN vs MLP (Split Aleatório)')
    print('=' * 80)
    print(f'{"Modelo ESM-2":<20} {"Classificador":<15} {"Mean AUC":<12} {"Std":<10} {"Min":<10} {"Max":<10}')
    print('-' * 80)
    
    models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']
    model_labels = {
        'esm2_t6_8M_UR50D': '8M (320D)',
        'esm2_t30_150M_UR50D': '150M (640D)',
        'esm2_t36_3B_UR50D': '3B (2560D)'
    }
    
    for model in models:
        for classifier in ['KNN', 'MLP']:
            data = metrics.get(model, {}).get(classifier, None)
            if data:
                values = data['values']
                print(f"{model_labels[model]:<20} {classifier:<15} "
                      f"{data['mean']:<12.4f} {data['std']:<10.4f} "
                      f"{min(values):<10.4f} {max(values):<10.4f}")
    
    print('-' * 80)
    
    # Placar: qual modelo ganhou
    print('\n' + '=' * 80)
    print('PLACAR: Melhor AUC médio por modelo ESM-2')
    print('=' * 80)
    
    for model in models:
        knn_data = metrics.get(model, {}).get('KNN', {'mean': 0})
        mlp_data = metrics.get(model, {}).get('MLP', {'mean': 0})
        
        knn_mean = knn_data['mean']
        mlp_mean = mlp_data['mean']
        
        winner = 'MLP' if mlp_mean > knn_mean else 'KNN'
        diff = abs(mlp_mean - knn_mean)
        
        print(f"{model_labels[model]:<20} Vencedor: {winner:<5} (diferença: {diff:.4f})")
    
    print('=' * 80 + '\n')


def main():
    # Paths
    results_file = Path('results/baseline_multiseed/baseline_multiseed_results.json')
    output_dir = Path('results/baseline_multiseed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_plot = output_dir / 'knn_vs_mlp_barplot_random.png'
    
    print('=' * 80)
    print('📊 GRÁFICO DE COMPARAÇÃO: KNN vs MLP (Split Aleatório)')
    print('=' * 80)
    print()
    
    # Carregar resultados
    print(f'📂 Carregando resultados de: {results_file}')
    results = load_results(results_file)
    
    # Extrair métricas
    print('🔍 Extraindo métricas...')
    metrics = extract_metrics(results)
    
    # Criar tabela
    create_detailed_table(metrics)
    
    # Criar gráfico
    print('📈 Gerando gráfico de barras...')
    create_comparison_barplot(metrics, output_plot)
    
    print()
    print('✅ Processo concluído!')
    print('=' * 80)


if __name__ == '__main__':
    main()
