#!/usr/bin/env python3
"""
Análise de Similaridade por Classes (Positivo/Negativo)
========================================================

Calcula similaridade de cossenos entre:
- Treino POS → Teste POS/NEG
- Treino NEG → Teste POS/NEG

Gera histogramas 2x2 para visualizar separabilidade entre classes.

Autor: DockTKinase Project
Data: Janeiro 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Configurações
EMBEDDINGS_DIR = Path('results/protein_model_benchmark_non_human_v2')
OUTPUT_DIR = Path('results/class_similarity_analysis')
RANDOM_SEED = 42
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# Modelos
MODELS = {
    'esm2_t6_8M_UR50D': '8M',
    'esm2_t30_150M_UR50D': '150M',
    'esm2_t36_3B_UR50D': '3B',
}


def load_embeddings(model_name):
    """Carrega embeddings e labels."""
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


def calculate_class_similarities(X_train, y_train, X_test, y_test):
    """
    Calcula similaridades entre classes.
    
    Returns:
        dict com 4 arrays:
        - 'train_pos_test_pos': similaridades de treino POS para teste POS
        - 'train_pos_test_neg': similaridades de treino POS para teste NEG
        - 'train_neg_test_pos': similaridades de treino NEG para teste POS
        - 'train_neg_test_neg': similaridades de treino NEG para teste NEG
    """
    print('   Separando por classes...')
    
    # Separar treino por classe
    X_train_pos = X_train[y_train == 1]
    X_train_neg = X_train[y_train == 0]
    
    # Separar teste por classe
    X_test_pos = X_test[y_test == 1]
    X_test_neg = X_test[y_test == 0]
    
    print(f'   Treino: {len(X_train_pos)} POS, {len(X_train_neg)} NEG')
    print(f'   Teste:  {len(X_test_pos)} POS, {len(X_test_neg)} NEG')
    
    results = {}
    
    # Treino POS → Teste POS
    print('   Calculando: Treino POS → Teste POS...')
    sim_matrix = cosine_similarity(X_test_pos, X_train_pos)
    results['train_pos_test_pos'] = sim_matrix.max(axis=1)  # Similaridade máxima para cada teste
    
    # Treino POS → Teste NEG
    print('   Calculando: Treino POS → Teste NEG...')
    sim_matrix = cosine_similarity(X_test_neg, X_train_pos)
    results['train_pos_test_neg'] = sim_matrix.max(axis=1)
    
    # Treino NEG → Teste POS
    print('   Calculando: Treino NEG → Teste POS...')
    sim_matrix = cosine_similarity(X_test_pos, X_train_neg)
    results['train_neg_test_pos'] = sim_matrix.max(axis=1)
    
    # Treino NEG → Teste NEG
    print('   Calculando: Treino NEG → Teste NEG...')
    sim_matrix = cosine_similarity(X_test_neg, X_train_neg)
    results['train_neg_test_neg'] = sim_matrix.max(axis=1)
    
    return results


def plot_class_similarities(all_results):
    """Cria histogramas 2x2 para cada modelo."""
    
    for model_name, model_label in MODELS.items():
        if model_name not in all_results:
            continue
        
        results = all_results[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Cores
        color_pos_pos = '#2ecc71'  # Verde (mesma classe)
        color_pos_neg = '#e74c3c'  # Vermelho (classes diferentes)
        color_neg_pos = '#e74c3c'  # Vermelho (classes diferentes)
        color_neg_neg = '#3498db'  # Azul (mesma classe)
        
        bins = 50
        alpha = 0.7
        
        # [0,0] Treino POS → Teste POS
        ax = axes[0, 0]
        data = results['train_pos_test_pos']
        ax.hist(data, bins=bins, color=color_pos_pos, alpha=alpha, edgecolor='black')
        ax.set_title('Treino POS → Teste POS', fontsize=13, fontweight='bold')
        ax.set_xlabel('Similaridade de Cossenos', fontsize=11)
        ax.set_ylabel('Frequência', fontsize=11)
        ax.axvline(data.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Média: {data.mean():.4f}')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # [0,1] Treino NEG → Teste POS
        ax = axes[0, 1]
        data = results['train_neg_test_pos']
        ax.hist(data, bins=bins, color=color_neg_pos, alpha=alpha, edgecolor='black')
        ax.set_title('Treino NEG → Teste POS', fontsize=13, fontweight='bold')
        ax.set_xlabel('Similaridade de Cossenos', fontsize=11)
        ax.set_ylabel('Frequência', fontsize=11)
        ax.axvline(data.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Média: {data.mean():.4f}')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # [1,0] Treino POS → Teste NEG
        ax = axes[1, 0]
        data = results['train_pos_test_neg']
        ax.hist(data, bins=bins, color=color_pos_neg, alpha=alpha, edgecolor='black')
        ax.set_title('Treino POS → Teste NEG', fontsize=13, fontweight='bold')
        ax.set_xlabel('Similaridade de Cossenos', fontsize=11)
        ax.set_ylabel('Frequência', fontsize=11)
        ax.axvline(data.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Média: {data.mean():.4f}')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # [1,1] Treino NEG → Teste NEG
        ax = axes[1, 1]
        data = results['train_neg_test_neg']
        ax.hist(data, bins=bins, color=color_neg_neg, alpha=alpha, edgecolor='black')
        ax.set_title('Treino NEG → Teste NEG', fontsize=13, fontweight='bold')
        ax.set_xlabel('Similaridade de Cossenos', fontsize=11)
        ax.set_ylabel('Frequência', fontsize=11)
        ax.axvline(data.mean(), color='darkblue', linestyle='--', linewidth=2, label=f'Média: {data.mean():.4f}')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Título geral
        fig.suptitle(
            f'Similaridade Entre Classes: Treino vs Teste - {model_label}\n(Split Aleatório, Seed={RANDOM_SEED})',
            fontsize=15, fontweight='bold', y=0.995
        )
        
        plt.tight_layout()
        
        # Salvar
        output_file = OUTPUT_DIR / f'class_similarity_{model_name}.png'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'   ✅ Gráfico salvo: {output_file}')
        
        plt.close()


def print_statistics(all_results):
    """Imprime estatísticas detalhadas."""
    print('\n' + '=' * 70)
    print('📊 ESTATÍSTICAS DE SIMILARIDADE POR CLASSE')
    print('=' * 70)
    
    for model_name, model_label in MODELS.items():
        if model_name not in all_results:
            continue
        
        results = all_results[model_name]
        
        print(f'\n🧬 {model_label}')
        print('-' * 70)
        
        for key, label in [
            ('train_pos_test_pos', 'Treino POS → Teste POS (mesma classe)'),
            ('train_neg_test_pos', 'Treino NEG → Teste POS (classes diferentes)'),
            ('train_pos_test_neg', 'Treino POS → Teste NEG (classes diferentes)'),
            ('train_neg_test_neg', 'Treino NEG → Teste NEG (mesma classe)'),
        ]:
            data = results[key]
            print(f'\n   {label}:')
            print(f'      Média:    {data.mean():.4f}')
            print(f'      Mediana:  {np.median(data):.4f}')
            print(f'      Std:      {data.std():.4f}')
            print(f'      Min:      {data.min():.4f}')
            print(f'      Max:      {data.max():.4f}')
            print(f'      P90:      {np.percentile(data, 90):.4f}')
            print(f'      P95:      {np.percentile(data, 95):.4f}')
            print(f'      P99:      {np.percentile(data, 99):.4f}')
        
        # Análise de separabilidade
        print(f'\n   📈 Análise de Separabilidade:')
        same_class_mean = (results['train_pos_test_pos'].mean() + results['train_neg_test_neg'].mean()) / 2
        diff_class_mean = (results['train_pos_test_neg'].mean() + results['train_neg_test_pos'].mean()) / 2
        separability = same_class_mean - diff_class_mean
        
        print(f'      Média mesma classe:      {same_class_mean:.4f}')
        print(f'      Média classes diferentes: {diff_class_mean:.4f}')
        print(f'      Separabilidade (diff):   {separability:.4f}')
        
        if separability > 0.1:
            print(f'      ✅ Classes BEM SEPARADAS')
        elif separability > 0.05:
            print(f'      ⚠️  Classes MODERADAMENTE separadas')
        else:
            print(f'      ❌ Classes POUCO separadas (overlap alto)')


def main():
    """Executa análise completa."""
    print('=' * 70)
    print('🔍 ANÁLISE DE SIMILARIDADE POR CLASSE')
    print('=' * 70)
    
    # Verificar se resultados já existem
    results_file = OUTPUT_DIR / 'class_similarity_stats.json'
    
    if results_file.exists():
        print('\n✅ Carregando resultados existentes...')
        with open(results_file, 'r') as f:
            # Converter listas de volta para arrays
            all_results_raw = json.load(f)
            all_results = {}
            for model_name, results in all_results_raw.items():
                all_results[model_name] = {
                    k: np.array(v) for k, v in results.items()
                }
        print('   Dados carregados!')
    else:
        print('\n⏳ Calculando similaridades (será feito apenas uma vez)...')
        all_results = {}
        
        for model_name, model_label in MODELS.items():
            print(f'\n📂 Processando: {model_label}')
            print('-' * 70)
            
            # Carregar dados
            embeddings, labels = load_embeddings(model_name)
            print(f'   Total: {len(embeddings)} amostras')
            print(f'   Dimensões: {embeddings.shape[1]}')
            print(f'   Positivos: {(labels == 1).sum()} ({(labels == 1).mean() * 100:.1f}%)')
            
            # Split
            X_train, X_test, y_train, y_test = split_data(embeddings, labels)
            print(f'   Treino: {len(X_train)}, Teste: {len(X_test)}')
            
            # Calcular similaridades
            results = calculate_class_similarities(X_train, y_train, X_test, y_test)
            all_results[model_name] = results
        
        # Salvar resultados
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results_to_save = {}
        for model_name, results in all_results.items():
            results_to_save[model_name] = {
                k: v.tolist() for k, v in results.items()
            }
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f'\n💾 Resultados salvos em: {results_file}')
    
    # Imprimir estatísticas
    print_statistics(all_results)
    
    # Criar gráficos
    print('\n' + '=' * 70)
    print('📈 GERANDO GRÁFICOS...')
    print('=' * 70)
    plot_class_similarities(all_results)
    
    print('\n' + '=' * 70)
    print('✅ ANÁLISE CONCLUÍDA')
    print('=' * 70)


if __name__ == '__main__':
    main()
