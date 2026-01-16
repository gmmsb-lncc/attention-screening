#!/usr/bin/env python3
"""
Análise de Similaridade Train-Test
===================================

Calcula a similaridade de cossenos máxima entre cada amostra do conjunto
de teste e todas as amostras do conjunto de treino. Gera histograma para
verificar se os conjuntos são altamente semelhantes (data leakage potencial).

Se a maioria dos pontos de teste têm alta similaridade (>0.95) com o treino,
isso indica que os conjuntos são muito parecidos, o que pode inflar as métricas.

Autor: DockTKinase Project
Data: Janeiro 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configurações
EMBEDDINGS_DIR = Path('results/protein_model_benchmark_non_human_v2')
OUTPUT_DIR = Path('results/similarity_analysis')
RANDOM_SEED = 42
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# Modelos a analisar
MODELS = {
    'esm2_t6_8M_UR50D': '8M (320D)',
    'esm2_t30_150M_UR50D': '150M (640D)',
    'esm2_t36_3B_UR50D': '3B (2560D)',
}


def load_embeddings(model_name):
    """Carrega embeddings e labels para um modelo."""
    build_dir = EMBEDDINGS_DIR / model_name / 'build'
    
    embeddings = np.load(build_dir / 'embedding_matrix.npy', allow_pickle=True)
    labels = np.load(build_dir / 'binary_labels.npy', allow_pickle=True)
    
    # Garantir formato correto
    if embeddings.ndim == 1:
        embeddings = np.vstack(embeddings)
    if labels.ndim > 1:
        labels = labels.ravel()
    
    # Filtrar válidos
    valid_mask = np.isin(labels, [0, 1])
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask].astype(int)
    
    return embeddings, labels


def split_data(embeddings, labels, seed=RANDOM_SEED):
    """Realiza split train/test."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        embeddings, labels,
        test_size=TEST_SIZE + VAL_SIZE,
        random_state=seed,
        stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
        random_state=seed,
        stratify=y_temp
    )
    
    return X_train, X_test, y_train, y_test


def calculate_max_similarities(X_train, X_test, batch_size=100):
    """
    Calcula a similaridade de cossenos máxima para cada ponto de teste.
    
    Para cada amostra em X_test, calcula a similaridade com todas as
    amostras em X_train e retorna o valor máximo.
    
    Args:
        X_train: Embeddings de treino [n_train, dim]
        X_test: Embeddings de teste [n_test, dim]
        batch_size: Tamanho do batch para processar (evitar OOM)
    
    Returns:
        max_similarities: Array [n_test] com similaridades máximas
    """
    n_test = X_test.shape[0]
    max_similarities = np.zeros(n_test)
    
    print(f'   Calculando similaridades: {n_test} amostras de teste vs {X_train.shape[0]} de treino')
    
    # Processar em batches para não explodir memória
    for i in tqdm(range(0, n_test, batch_size), desc='   Progresso'):
        end_idx = min(i + batch_size, n_test)
        batch_test = X_test[i:end_idx]
        
        # Calcular similaridade do batch com todo o treino
        # Shape: [batch_size, n_train]
        similarities = cosine_similarity(batch_test, X_train)
        
        # Pegar máximo para cada amostra do batch
        max_similarities[i:end_idx] = similarities.max(axis=1)
    
    return max_similarities


def plot_similarity_histogram(max_sims_dict, output_file):
    """Cria histograma de similaridades máximas."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (model_name, label) in zip(axes, MODELS.items()):
        max_sims = max_sims_dict[model_name]
        
        # Histograma
        n, bins, patches = ax.hist(
            max_sims,
            bins=50,
            range=(0.5, 1.0),
            alpha=0.7,
            color='steelblue',
            edgecolor='black',
            linewidth=0.5
        )
        
        # Estatísticas
        mean_sim = np.mean(max_sims)
        median_sim = np.median(max_sims)
        min_sim = np.min(max_sims)
        max_sim = np.max(max_sims)
        
        # Percentis
        p95 = np.percentile(max_sims, 95)
        p99 = np.percentile(max_sims, 99)
        
        # Linha vertical na média e mediana
        ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_sim:.3f}')
        ax.axvline(median_sim, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median_sim:.3f}')
        
        # Área de alto risco (>0.95)
        ax.axvspan(0.95, 1.0, alpha=0.2, color='red', label='Alto risco (>0.95)')
        
        # Configurações
        ax.set_xlabel('Similaridade Máxima de Cossenos', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequência', fontsize=10, fontweight='bold')
        ax.set_title(f'{label}\nMin={min_sim:.3f}, Max={max_sim:.3f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Texto com estatísticas
        stats_text = f'P95: {p95:.3f}\nP99: {p99:.3f}'
        ax.text(
            0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.suptitle(
        'Distribuição de Similaridade Máxima: Teste vs Treino\n'
        '(Split Aleatório, Seed=42)',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    # Salvar
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'\n✅ Gráfico salvo em: {output_file}')
    
    return fig


def generate_similarity_report(max_sims_dict):
    """Gera relatório de similaridades."""
    
    print('\n' + '=' * 70)
    print('📊 RELATÓRIO DE SIMILARIDADE TRAIN-TEST')
    print('=' * 70)
    
    for model_name, label in MODELS.items():
        max_sims = max_sims_dict[model_name]
        
        # Estatísticas
        mean_sim = np.mean(max_sims)
        median_sim = np.median(max_sims)
        std_sim = np.std(max_sims)
        min_sim = np.min(max_sims)
        max_sim = np.max(max_sims)
        
        # Percentis
        p90 = np.percentile(max_sims, 90)
        p95 = np.percentile(max_sims, 95)
        p99 = np.percentile(max_sims, 99)
        
        # Contagem de amostras com alta similaridade
        high_sim_90 = np.sum(max_sims > 0.90) / len(max_sims) * 100
        high_sim_95 = np.sum(max_sims > 0.95) / len(max_sims) * 100
        high_sim_99 = np.sum(max_sims > 0.99) / len(max_sims) * 100
        
        print(f'\n🧬 {label}')
        print('-' * 70)
        print(f'   Média:    {mean_sim:.4f}')
        print(f'   Mediana:  {median_sim:.4f}')
        print(f'   Std:      {std_sim:.4f}')
        print(f'   Min:      {min_sim:.4f}')
        print(f'   Max:      {max_sim:.4f}')
        print(f'   P90:      {p90:.4f}')
        print(f'   P95:      {p95:.4f}')
        print(f'   P99:      {p99:.4f}')
        print(f'   ')
        print(f'   Amostras com similaridade > 0.90: {high_sim_90:.2f}%')
        print(f'   Amostras com similaridade > 0.95: {high_sim_95:.2f}%')
        print(f'   Amostras com similaridade > 0.99: {high_sim_99:.2f}%')
        
        # Interpretação
        if mean_sim > 0.95:
            print(f'   ⚠️  ALERTA: Média muito alta! Possível data leakage ou overfitting.')
        elif mean_sim > 0.90:
            print(f'   ⚠️  Atenção: Média alta. Conjuntos muito similares.')
        else:
            print(f'   ✅ Similaridade aceitável.')
    
    print('\n' + '=' * 70)


def main():
    """Função principal."""
    
    print('=' * 70)
    print('🔍 ANÁLISE DE SIMILARIDADE TRAIN-TEST')
    print('=' * 70)
    print()
    
    max_sims_dict = {}
    
    for model_name, label in MODELS.items():
        print(f'\n📊 Processando: {label}')
        print('-' * 70)
        
        # Carregar dados
        print('   Carregando embeddings...')
        embeddings, labels = load_embeddings(model_name)
        print(f'   Total: {len(embeddings):,} amostras, {embeddings.shape[1]} dims')
        
        # Split
        print('   Realizando split...')
        X_train, X_test, y_train, y_test = split_data(embeddings, labels)
        print(f'   Train: {len(X_train):,}, Test: {len(X_test):,}')
        
        # Calcular similaridades
        max_similarities = calculate_max_similarities(X_train, X_test, batch_size=100)
        max_sims_dict[model_name] = max_similarities
        
        print(f'   ✅ Concluído: Média={np.mean(max_similarities):.4f}')
    
    # Gerar gráfico
    print('\n' + '=' * 70)
    print('📊 GERANDO HISTOGRAMAS')
    print('=' * 70)
    
    output_file = OUTPUT_DIR / 'similarity_histogram.png'
    plot_similarity_histogram(max_sims_dict, output_file)
    
    # Gerar relatório
    generate_similarity_report(max_sims_dict)
    
    # Salvar dados
    results_file = OUTPUT_DIR / 'similarity_stats.json'
    results = {}
    for model_name, max_sims in max_sims_dict.items():
        results[model_name] = {
            'mean': float(np.mean(max_sims)),
            'median': float(np.median(max_sims)),
            'std': float(np.std(max_sims)),
            'min': float(np.min(max_sims)),
            'max': float(np.max(max_sims)),
            'p90': float(np.percentile(max_sims, 90)),
            'p95': float(np.percentile(max_sims, 95)),
            'p99': float(np.percentile(max_sims, 99)),
            'pct_above_90': float(np.sum(max_sims > 0.90) / len(max_sims) * 100),
            'pct_above_95': float(np.sum(max_sims > 0.95) / len(max_sims) * 100),
            'pct_above_99': float(np.sum(max_sims > 0.99) / len(max_sims) * 100),
        }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n💾 Estatísticas salvas em: {results_file}')
    
    print()
    print('=' * 70)
    print('✅ ANÁLISE CONCLUÍDA')
    print('=' * 70)


if __name__ == '__main__':
    main()
