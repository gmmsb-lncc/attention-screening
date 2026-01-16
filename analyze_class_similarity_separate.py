#!/usr/bin/env python3
"""
Análise de Similaridade por Classes - SEPARADO (Proteína vs Ligante)
=====================================================================

Calcula similaridade de cossenos separadamente para:
1. Apenas embeddings de PROTEÍNA
2. Apenas embeddings de LIGANTE

Para cada componente, calcula:
- Treino POS → Teste POS/NEG
- Treino NEG → Teste POS/NEG

Gera 2 figuras (uma para proteína, outra para ligante) com histogramas 2x2.

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
LIGAND_DIM = 768

# Modelos e dimensões
MODELS = {
    'esm2_t6_8M_UR50D': {'name': '8M', 'protein_dim': 320},
    'esm2_t30_150M_UR50D': {'name': '150M', 'protein_dim': 640},
    'esm2_t36_3B_UR50D': {'name': '3B', 'protein_dim': 2560},
}


def load_embeddings(model_name):
    """Carrega embeddings completos e labels."""
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


def separate_protein_ligand(embeddings, protein_dim):
    """
    Separa embeddings em componentes proteína e ligante.
    
    Args:
        embeddings: array [N, protein_dim + 768]
        protein_dim: dimensão da proteína (320/640/2560)
    
    Returns:
        protein_embeddings: array [N, protein_dim]
        ligand_embeddings: array [N, 768]
    """
    protein_embeddings = embeddings[:, :protein_dim]
    ligand_embeddings = embeddings[:, protein_dim:]
    
    assert ligand_embeddings.shape[1] == LIGAND_DIM, \
        f"Esperado ligante com {LIGAND_DIM} dims, obteve {ligand_embeddings.shape[1]}"
    
    return protein_embeddings, ligand_embeddings


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
        dict com 4 arrays de similaridades máximas
    """
    # Separar treino por classe
    X_train_pos = X_train[y_train == 1]
    X_train_neg = X_train[y_train == 0]
    
    # Separar teste por classe
    X_test_pos = X_test[y_test == 1]
    X_test_neg = X_test[y_test == 0]
    
    results = {}
    
    # POS → POS
    sim_matrix = cosine_similarity(X_test_pos, X_train_pos)
    results['train_pos_test_pos'] = sim_matrix.max(axis=1)
    
    # NEG → POS
    sim_matrix = cosine_similarity(X_test_pos, X_train_neg)
    results['train_neg_test_pos'] = sim_matrix.max(axis=1)
    
    # POS → NEG
    sim_matrix = cosine_similarity(X_test_neg, X_train_pos)
    results['train_pos_test_neg'] = sim_matrix.max(axis=1)
    
    # NEG → NEG
    sim_matrix = cosine_similarity(X_test_neg, X_train_neg)
    results['train_neg_test_neg'] = sim_matrix.max(axis=1)
    
    return results


def plot_component_similarities(all_results, component_name, output_file):
    """
    Gera figura com 3 modelos x 4 cenários para um componente.
    
    Args:
        all_results: dict {model_name: similarity_dict}
        component_name: 'Proteína' ou 'Ligante'
        output_file: caminho para salvar PNG
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(f'Similaridade de Cosseno - {component_name} Only', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    scenarios = [
        ('train_pos_test_pos', 'Treino POS → Teste POS\n(Mesma classe)', '#2ecc71'),
        ('train_neg_test_pos', 'Treino NEG → Teste POS\n(Classes diferentes)', '#e74c3c'),
        ('train_pos_test_neg', 'Treino POS → Teste NEG\n(Classes diferentes)', '#e67e22'),
        ('train_neg_test_neg', 'Treino NEG → Teste NEG\n(Mesma classe)', '#3498db'),
    ]
    
    model_names = list(all_results.keys())
    
    for row_idx, model_name in enumerate(model_names):
        model_short = MODELS[model_name]['name']
        similarities = all_results[model_name]
        
        for col_idx, (key, title, color) in enumerate(scenarios):
            ax = axes[row_idx, col_idx]
            data = similarities[key]
            
            # Histograma com densidade normalizada
            counts, bins, patches = ax.hist(data, bins=50, color=color, alpha=0.7, 
                                           edgecolor='black', linewidth=0.5, density=True)
            
            # Normalizar frequências para [0, 1]
            if counts.max() > 0:
                patches_array = np.array([p.get_height() for p in patches])
                for patch, normalized_height in zip(patches, patches_array / patches_array.max()):
                    patch.set_height(normalized_height)
            
            # Estatísticas
            mean_val = data.mean()
            median_val = np.median(data)
            std_val = data.std()
            
            # Título e labels
            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            
            ax.set_xlabel('Similaridade de Cosseno', fontsize=9)
            ax.set_ylabel('Frequência Normalizada', fontsize=9)
            
            # Adicionar label do modelo no lado esquerdo
            if col_idx == 0:
                ax.text(-0.35, 0.5, f'Modelo {model_short}', 
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       rotation=90, va='center', ha='center')
            
            # Texto com estatísticas
            stats_text = f'μ={mean_val:.4f}\nσ={std_val:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Linha vertical na média
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'   ✅ Salvo: {output_file}')
    plt.close()


def print_statistics(results, model_name, component_name):
    """Imprime estatísticas de separabilidade."""
    model_short = MODELS[model_name]['name']
    
    print(f'\n🧬 Modelo {model_short} - {component_name}')
    print('=' * 60)
    
    for key, label in [
        ('train_pos_test_pos', 'Treino POS → Teste POS (mesma classe)'),
        ('train_neg_test_pos', 'Treino NEG → Teste POS (diferente)'),
        ('train_pos_test_neg', 'Treino POS → Teste NEG (diferente)'),
        ('train_neg_test_neg', 'Treino NEG → Teste NEG (mesma classe)'),
    ]:
        data = results[key]
        print(f'{label:45s}: μ={data.mean():.4f} | σ={data.std():.4f} | '
              f'min={data.min():.4f} | max={data.max():.4f}')
    
    # Separabilidade
    same_class_mean = (results['train_pos_test_pos'].mean() + 
                       results['train_neg_test_neg'].mean()) / 2
    diff_class_mean = (results['train_pos_test_neg'].mean() + 
                       results['train_neg_test_pos'].mean()) / 2
    separability = same_class_mean - diff_class_mean
    
    print(f'\n📊 Separabilidade: {separability:.4f}')
    if separability > 0.05:
        print('   ✅ Classes BEM separadas')
    elif separability > 0.02:
        print('   ⚠️  Classes MODERADAMENTE separadas')
    else:
        print('   ❌ Classes POUCO separadas (overlap alto)')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print('\n' + '='*70)
    print('ANÁLISE DE SIMILARIDADE POR COMPONENTE (PROTEÍNA vs LIGANTE)')
    print('='*70)
    
    # Armazenar resultados
    protein_results = {}
    ligand_results = {}
    
    # Processar cada modelo
    for model_name, model_info in MODELS.items():
        model_short = model_info['name']
        protein_dim = model_info['protein_dim']
        
        print(f'\n📦 Processando modelo {model_short}...')
        
        # Carregar embeddings completos
        embeddings, labels = load_embeddings(model_name)
        print(f'   Carregados: {len(embeddings)} amostras')
        
        # Separar proteína e ligante
        protein_emb, ligand_emb = separate_protein_ligand(embeddings, protein_dim)
        print(f'   Proteína: {protein_emb.shape}, Ligante: {ligand_emb.shape}')
        
        # Split
        X_train, X_test, y_train, y_test = split_data(embeddings, labels)
        print(f'   Split: {len(X_train)} treino, {len(X_test)} teste')
        
        # Separar componentes nos splits
        X_train_prot, X_train_lig = separate_protein_ligand(X_train, protein_dim)
        X_test_prot, X_test_lig = separate_protein_ligand(X_test, protein_dim)
        
        # Calcular similaridades - PROTEÍNA
        print(f'   Calculando similaridades - Proteína...')
        protein_sim = calculate_class_similarities(X_train_prot, y_train, X_test_prot, y_test)
        protein_results[model_name] = protein_sim
        print_statistics(protein_sim, model_name, 'PROTEÍNA')
        
        # Calcular similaridades - LIGANTE
        print(f'   Calculando similaridades - Ligante...')
        ligand_sim = calculate_class_similarities(X_train_lig, y_train, X_test_lig, y_test)
        ligand_results[model_name] = ligand_sim
        print_statistics(ligand_sim, model_name, 'LIGANTE')
    
    # Gerar visualizações
    print(f'\n📊 Gerando visualizações...')
    
    protein_plot = OUTPUT_DIR / 'class_similarity_protein_only.png'
    plot_component_similarities(protein_results, 'Proteína', protein_plot)
    
    ligand_plot = OUTPUT_DIR / 'class_similarity_ligand_only.png'
    plot_component_similarities(ligand_results, 'Ligante', ligand_plot)
    
    # Salvar estatísticas
    stats_output = OUTPUT_DIR / 'class_similarity_separate_stats.json'
    stats = {
        'protein': {
            model: {k: {
                'mean': float(v.mean()),
                'std': float(v.std()),
                'min': float(v.min()),
                'max': float(v.max()),
                'median': float(np.median(v))
            } for k, v in results.items()}
            for model, results in protein_results.items()
        },
        'ligand': {
            model: {k: {
                'mean': float(v.mean()),
                'std': float(v.std()),
                'min': float(v.min()),
                'max': float(v.max()),
                'median': float(np.median(v))
            } for k, v in results.items()}
            for model, results in ligand_results.items()
        }
    }
    
    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'   ✅ Estatísticas salvas: {stats_output}')
    
    print(f'\n✅ ANÁLISE COMPLETA!')
    print(f'   📁 Proteína: {protein_plot}')
    print(f'   📁 Ligante: {ligand_plot}')
    print(f'   📁 Stats: {stats_output}')


if __name__ == '__main__':
    main()
