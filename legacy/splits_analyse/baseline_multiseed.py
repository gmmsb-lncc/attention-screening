#!/usr/bin/env python3
"""
DockTKinase - Baseline Pipeline Multi-Seed
============================================

Executa baseline com 5 seeds diferentes para avaliação estatística robusta.
Suporta split ALEATÓRIO e ESTRATIFICADO (clustering adaptativo).

Configuração:
- Seeds: [42, 123, 420, 777, 2024]
- Split: 80/10/10
- Modelos ESM-2: 8M, 150M, 650M
- Classificadores: KNN, MLP

Saídas:
- Resultados individuais por seed
- Média ± desvio padrão de cada métrica
- Visualizações comparativas

Uso:
    # Split aleatório (padrão)
    python baseline_multiseed.py --split random
    
    # Split estratificado
    python baseline_multiseed.py --split stratified
    
    # Ambos
    python baseline_multiseed.py --split both
    
    # Reutilizar resultados existentes
    python baseline_multiseed.py --split both --skip-existing

Autor: DockTKinase Team
Data: Janeiro 2026
"""

import os
import sys
import json
import time
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Suprimir warnings
warnings.filterwarnings('ignore')

# Adicionar path do projeto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Imports sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# Seeds para experimentos reproduzíveis
SEEDS = [42, 123, 420, 777, 2024]

# Modelos ESM-2
ESM_MODELS = [
    'esm2_t6_8M_UR50D',      # 8M parâmetros, 320-dim
    'esm2_t30_150M_UR50D',   # 150M parâmetros, 640-dim
    'esm2_t33_650M_UR50D',   # 650M parâmetros, 1280-dim
]

ESM_DIMS = {
    'esm2_t6_8M_UR50D': 320,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t33_650M_UR50D': 1280,
}

# Proporções do split
TEST_SIZE = 0.1
VAL_SIZE = 0.1


# =============================================================================
# MODELOS
# =============================================================================

def create_knn_model() -> Pipeline:
    """Cria modelo KNN com StandardScaler."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='cosine',
            n_jobs=-1
        ))
    ])


def create_mlp_model(random_state: int) -> Pipeline:
    """Cria modelo MLP com StandardScaler."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
            verbose=False
        ))
    ])


# =============================================================================
# DATA LOADING
# =============================================================================

def load_embeddings(embeddings_path: Path, labels_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Carrega embeddings e labels."""
    embeddings = np.load(embeddings_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    
    if embeddings.ndim == 1:
        embeddings = np.vstack(embeddings)
    if labels.ndim > 1:
        labels = labels.ravel()
    
    # Filtrar labels inválidos
    valid_mask = np.isin(labels, [0, 1])
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask].astype(int)
    
    return embeddings, labels


# =============================================================================
# SPLIT ALEATÓRIO
# =============================================================================

def random_split(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple:
    """Divide dados em treino/val/teste com split ALEATÓRIO."""
    # Primeiro split: separar teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    
    # Segundo split: separar validação
    val_size_adj = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adj, random_state=seed, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, None


# =============================================================================
# SPLIT ESTRATIFICADO (CLUSTERING)
# =============================================================================

def stratified_cluster_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    target_cluster_ratio: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Realiza split ESTRATIFICADO baseado em clusters de similaridade.
    
    Este é o método DockTKinase que agrupa amostras similares e garante
    que cada cluster contribua proporcionalmente para treino/val/teste.
    
    Args:
        embeddings: Matriz de embeddings [N, D]
        labels: Labels binários
        test_size: Proporção para teste
        val_size: Proporção para validação
        random_state: Seed
        target_cluster_ratio: Proporção alvo de clusters (~1% do dataset)
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, cluster_metrics
    """
    np.random.seed(random_state)
    n_samples = len(embeddings)
    
    # Calcular similaridade de cosseno (amostrar se muito grande)
    if n_samples > 5000:
        sample_idx = np.random.choice(n_samples, 5000, replace=False)
        sample_emb = embeddings[sample_idx]
        sim_matrix_sample = cosine_similarity(sample_emb)
        triu_idx = np.triu_indices(len(sample_emb), k=1)
        similarities = sim_matrix_sample[triu_idx]
    else:
        sim_matrix = cosine_similarity(embeddings)
        triu_idx = np.triu_indices(n_samples, k=1)
        similarities = sim_matrix[triu_idx]
    
    # Estatísticas de similaridade
    sim_stats = {
        'min': float(np.min(similarities)),
        'max': float(np.max(similarities)),
        'mean': float(np.mean(similarities)),
        'std': float(np.std(similarities)),
    }
    
    # Definir threshold adaptativo
    target_clusters = max(10, int(n_samples * target_cluster_ratio))
    
    # Calcular matriz completa de distância
    sim_matrix_full = cosine_similarity(embeddings)
    distance_matrix = np.clip(1 - sim_matrix_full, 0, 2)
    
    # Busca binária para encontrar threshold que produz ~target clusters
    low, high = sim_stats['min'], sim_stats['max']
    best_threshold = (low + high) / 2
    best_n_clusters = 0
    
    for iteration in range(20):
        mid = (low + high) / 2
        distance_thresh = 1 - mid
        
        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_thresh,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = model.fit_predict(distance_matrix)
            n_clusters = len(np.unique(cluster_labels))
            
            if abs(n_clusters - target_clusters) < abs(best_n_clusters - target_clusters):
                best_n_clusters = n_clusters
                best_threshold = mid
            
            if n_clusters < target_clusters:
                low = mid
            elif n_clusters > target_clusters:
                high = mid
            else:
                break
                
            if high - low < 0.001:
                break
        except:
            break
    
    # Clustering final com melhor threshold
    distance_thresh = 1 - best_threshold
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_thresh,
        metric='precomputed',
        linkage='average'
    )
    cluster_labels = model.fit_predict(distance_matrix)
    n_clusters = len(np.unique(cluster_labels))
    
    # Agrupar amostras por cluster
    clusters = {}
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(idx)
    
    # Dividir cada cluster proporcionalmente
    train_idx, val_idx, test_idx = [], [], []
    
    for cluster_id, cluster_samples in clusters.items():
        n_cluster = len(cluster_samples)
        
        if n_cluster >= 3:
            n_test = max(1, int(n_cluster * test_size))
            n_val = max(1, int(n_cluster * val_size))
            n_train = n_cluster - n_test - n_val
            
            if n_train < 1:
                n_train = 1
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
        elif n_cluster == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = 1, 0, 0
        
        np.random.shuffle(cluster_samples)
        train_idx.extend(cluster_samples[:n_train])
        val_idx.extend(cluster_samples[n_train:n_train + n_val])
        test_idx.extend(cluster_samples[n_train + n_val:])
    
    # Shuffle final
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    
    # Métricas do clustering
    cluster_metrics = {
        'n_clusters': n_clusters,
        'threshold': best_threshold,
        'similarity_stats': sim_stats,
        'cluster_sizes': {
            'min': int(np.min([len(c) for c in clusters.values()])),
            'max': int(np.max([len(c) for c in clusters.values()])),
            'mean': float(np.mean([len(c) for c in clusters.values()])),
        }
    }
    
    X_train = embeddings[train_idx]
    X_val = embeddings[val_idx]
    X_test = embeddings[test_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, cluster_metrics


# =============================================================================
# MÉTRICAS
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calcula todas as métricas."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_and_evaluate(
    model_name: str,
    model: Pipeline,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    """Treina modelo e avalia."""
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Predições
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        'model_name': model_name,
        'train_time': train_time,
        'train_metrics': calculate_metrics(y_train, y_train_pred, y_train_prob),
        'val_metrics': calculate_metrics(y_val, y_val_pred, y_val_prob),
        'test_metrics': calculate_metrics(y_test, y_test_pred, y_test_prob),
    }


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_single_seed(
    seed: int,
    embeddings_dir: Path,
    esm_model: str,
    split_method: str = 'random',
    verbose: bool = True
) -> Dict[str, Any]:
    """Executa experimento com uma seed."""
    
    # Carregar dados
    model_dir = embeddings_dir / esm_model / "build"
    embeddings_path = model_dir / "embedding_matrix.npy"
    labels_path = model_dir / "binary_labels.npy"
    
    X, y = load_embeddings(embeddings_path, labels_path)
    
    # Split conforme método escolhido
    if split_method == 'random':
        X_train, X_val, X_test, y_train, y_val, y_test, cluster_info = random_split(X, y, seed)
    else:  # stratified
        X_train, X_val, X_test, y_train, y_val, y_test, cluster_info = stratified_cluster_split(
            X, y, TEST_SIZE, VAL_SIZE, seed
        )
    
    results = {
        'seed': seed,
        'split_method': split_method,
        'n_train': len(y_train),
        'n_val': len(y_val),
        'n_test': len(y_test),
        'cluster_info': cluster_info,
        'classifiers': {}
    }
    
    # Treinar KNN
    knn = create_knn_model()
    knn_results = train_and_evaluate(
        'KNN', knn, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results['classifiers']['KNN'] = knn_results
    
    if verbose:
        print(f"      KNN: ROC-AUC={knn_results['test_metrics']['roc_auc']:.4f} ({knn_results['train_time']:.1f}s)")
    
    # Treinar MLP
    mlp = create_mlp_model(seed)
    mlp_results = train_and_evaluate(
        'MLP', mlp, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results['classifiers']['MLP'] = mlp_results
    
    if verbose:
        print(f"      MLP: ROC-AUC={mlp_results['test_metrics']['roc_auc']:.4f} ({mlp_results['train_time']:.1f}s)")
    
    return results


def aggregate_results(all_results: List[Dict], metric_name: str = 'test_metrics') -> Dict:
    """Agrega resultados de múltiplas seeds."""
    aggregated = {}
    
    for clf in ['KNN', 'MLP']:
        clf_metrics = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'roc_auc', 'balanced_accuracy']:
            values = [r['classifiers'][clf][metric_name][metric] for r in all_results]
            clf_metrics[metric] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
        
        aggregated[clf] = clf_metrics
    
    return aggregated


def load_existing_results(output_dir: Path, split_method: str) -> Optional[Dict]:
    """Carrega resultados existentes se disponíveis."""
    if split_method == 'random':
        results_file = output_dir / 'baseline_multiseed_results.json'
    else:
        results_file = output_dir / 'stratified_multiseed_results.json'
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def run_pipeline(
    embeddings_dir: Path,
    output_dir: Path,
    split_method: str,
    n_seeds: int = 5,
    skip_existing: bool = False
) -> Dict:
    """Executa pipeline completo para um método de split."""
    
    seeds = SEEDS[:n_seeds]
    
    # Verificar resultados existentes
    if skip_existing:
        existing = load_existing_results(output_dir, split_method)
        if existing is not None:
            print(f"\n   ✅ Resultados {split_method.upper()} já existem - pulando...")
            return existing
    
    split_label = "ALEATÓRIO" if split_method == 'random' else "ESTRATIFICADO (Clustering)"
    
    print("\n" + "=" * 70)
    print(f"🎯 BASELINE MULTI-SEED PIPELINE - Split {split_label}")
    print("=" * 70)
    print()
    print(f"📋 Configuração:")
    print(f"   Seeds: {seeds}")
    print(f"   Split: 80% treino / 10% val / 10% teste ({split_label})")
    print(f"   Modelos ESM-2: {', '.join(ESM_MODELS)}")
    print(f"   Classificadores: KNN, MLP")
    print()
    
    all_results = {
        'pipeline': f'baseline_multiseed_{split_method}',
        'split_method': split_method,
        'timestamp': datetime.now().isoformat(),
        'seeds': seeds,
        'n_seeds': len(seeds),
        'models': {}
    }
    
    total_start = time.time()
    
    for esm_model in ESM_MODELS:
        print("=" * 70)
        print(f"🧬 Modelo: {esm_model}")
        print("=" * 70)
        
        seed_results = []
        
        for seed in seeds:
            print(f"\n   🎲 Seed {seed}:")
            result = run_single_seed(seed, embeddings_dir, esm_model, split_method, verbose=True)
            seed_results.append(result)
        
        # Agregar resultados
        aggregated = aggregate_results(seed_results)
        
        all_results['models'][esm_model] = {
            'seed_results': seed_results,
            'aggregated': aggregated
        }
        
        # Resumo do modelo
        print(f"\n   📊 Resumo {esm_model}:")
        for clf in ['KNN', 'MLP']:
            auc = aggregated[clf]['roc_auc']
            print(f"      {clf}: ROC-AUC = {auc['mean']:.4f} ± {auc['std']:.4f}")
    
    total_time = time.time() - total_start
    all_results['total_time_seconds'] = total_time
    
    # Salvar resultados
    if split_method == 'random':
        results_path = output_dir / 'baseline_multiseed_results.json'
    else:
        results_path = output_dir / 'stratified_multiseed_results.json'
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    print(f"\n💾 Resultados salvos em: {results_path}")
    print(f"⏱️  Tempo: {total_time:.2f}s ({total_time/60:.2f} min)")
    
    return all_results


# =============================================================================
# VISUALIZAÇÕES
# =============================================================================

def create_single_visualization(results: Dict, output_dir: Path, split_method: str):
    """Cria visualização para um único método de split."""
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC']
    
    esm_models = list(results['models'].keys())
    model_labels = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 650M'][:len(esm_models)]
    
    split_title = "Aleatório" if split_method == 'random' else "Estratificado"
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Baseline Multi-Seed: Split {split_title}\n(Barras = Média, Linhas de erro = ±1 Desvio Padrão)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    colors = {'KNN': '#3498db', 'MLP': '#e74c3c'}
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        x = np.arange(len(esm_models))
        width = 0.35
        
        for i, clf in enumerate(['KNN', 'MLP']):
            means = []
            stds = []
            
            for esm in esm_models:
                agg = results['models'][esm]['aggregated']
                means.append(agg[clf][metric]['mean'])
                stds.append(agg[clf][metric]['std'])
            
            offset = -width/2 if i == 0 else width/2
            bars = ax.bar(x + offset, means, width, label=clf, color=colors[clf], alpha=0.8)
            ax.errorbar(x + offset, means, yerr=stds, fmt='none', color='black', capsize=3)
            
            for bar, mean, std in zip(bars, means, stds):
                ax.annotate(f'{mean:.3f}\n±{std:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=7)
        
        ax.set_ylabel(label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_title(label, fontsize=11, fontweight='bold')
        
        if metric == 'mcc':
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylim(0.7, 1.05)
        
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if split_method == 'random':
        output_path = output_dir / "baseline_multiseed_results.png"
    else:
        output_path = output_dir / "stratified_multiseed_results.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Visualização salva em: {output_path}")


def create_comparison_visualization(random_results: Dict, stratified_results: Dict, output_dir: Path):
    """Cria visualização comparativa entre Random e Stratified."""
    
    esm_models = list(random_results['models'].keys())
    model_labels = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 650M'][:len(esm_models)]
    
    # Figura 1: Comparação ROC-AUC por modelo/classificador
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics_to_compare = ['roc_auc', 'f1', 'mcc']
    metric_labels = ['ROC-AUC', 'F1-Score', 'MCC']
    
    for ax_idx, (metric, label) in enumerate(zip(metrics_to_compare, metric_labels)):
        ax = axes[ax_idx]
        
        x = np.arange(len(esm_models))
        width = 0.2
        
        for clf_idx, clf in enumerate(['KNN', 'MLP']):
            # Random
            random_means = [random_results['models'][esm]['aggregated'][clf][metric]['mean'] for esm in esm_models]
            random_stds = [random_results['models'][esm]['aggregated'][clf][metric]['std'] for esm in esm_models]
            
            # Stratified
            strat_means = [stratified_results['models'][esm]['aggregated'][clf][metric]['mean'] for esm in esm_models]
            strat_stds = [stratified_results['models'][esm]['aggregated'][clf][metric]['std'] for esm in esm_models]
            
            offset_random = width * (clf_idx * 2 - 1.5)
            offset_strat = width * (clf_idx * 2 - 0.5)
            
            color_random = '#3498db' if clf == 'KNN' else '#e74c3c'
            color_strat = '#2980b9' if clf == 'KNN' else '#c0392b'
            
            ax.bar(x + offset_random, random_means, width, label=f'{clf} Random', 
                   color=color_random, alpha=0.7, hatch='')
            ax.errorbar(x + offset_random, random_means, yerr=random_stds, 
                       fmt='none', color='black', capsize=2)
            
            ax.bar(x + offset_strat, strat_means, width, label=f'{clf} Stratified', 
                   color=color_strat, alpha=0.9, hatch='//')
            ax.errorbar(x + offset_strat, strat_means, yerr=strat_stds, 
                       fmt='none', color='black', capsize=2)
        
        ax.set_xlabel('Modelo ESM-2', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} (Test Set)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis='y', alpha=0.3)
        
        if metric == 'mcc':
            ax.set_ylim(0.5, 0.9)
        else:
            ax.set_ylim(0.85, 1.0)
    
    plt.suptitle('Comparação: Split Aleatório vs Estratificado (5 seeds)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_random_vs_stratified.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figura 2: Delta (Stratified - Random)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(esm_models))
    width = 0.35
    
    for clf_idx, clf in enumerate(['KNN', 'MLP']):
        deltas = []
        for esm in esm_models:
            random_auc = random_results['models'][esm]['aggregated'][clf]['roc_auc']['mean']
            strat_auc = stratified_results['models'][esm]['aggregated'][clf]['roc_auc']['mean']
            delta = (strat_auc - random_auc) * 100  # Em pontos percentuais
            deltas.append(delta)
        
        offset = -width/2 if clf_idx == 0 else width/2
        color = '#3498db' if clf == 'KNN' else '#e74c3c'
        
        bars = ax.bar(x + offset, deltas, width, label=clf, color=color, alpha=0.8)
        
        for bar, delta in zip(bars, deltas):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax.annotate(f'{delta:+.2f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3 if height >= 0 else -3),
                       textcoords='offset points',
                       ha='center', va=va, fontsize=10, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Modelo ESM-2', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delta ROC-AUC (pontos percentuais)', fontsize=12, fontweight='bold')
    ax.set_title('Delta: Stratified - Random\n(Valores positivos = Estratificado melhor)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_delta.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparação salva em: {output_dir / 'comparison_random_vs_stratified.png'}")
    print(f"📊 Delta salvo em: {output_dir / 'comparison_delta.png'}")


def print_summary_table(results: Dict, seeds: List[int], split_method: str):
    """Imprime tabela resumo."""
    
    split_label = "ALEATÓRIO" if split_method == 'random' else "ESTRATIFICADO"
    
    print("\n" + "=" * 120)
    print(f"📊 RESUMO DOS RESULTADOS - Split {split_label} (Test ROC-AUC)")
    print("=" * 120)
    
    # Cabeçalho
    header = f"{'Modelo ESM-2':<25}"
    for seed in seeds:
        header += f" Seed {seed:<6}"
    header += f" {'Média':<10} {'±Std':<8}"
    print(header)
    print("-" * 120)
    
    for esm_model in results['models']:
        for clf in ['KNN', 'MLP']:
            row = f"{esm_model[:20]:<20} {clf:<4}"
            
            agg = results['models'][esm_model]['aggregated'][clf]['roc_auc']
            
            for val in agg['values']:
                row += f" {val:.4f} "
            
            row += f" {agg['mean']:.4f}    ±{agg['std']:.4f}"
            print(row)
        print()
    
    print("=" * 120)


def print_comparison_table(random_results: Dict, stratified_results: Dict):
    """Imprime tabela comparativa entre Random e Stratified."""
    
    print("\n" + "=" * 100)
    print("📊 COMPARAÇÃO: RANDOM vs STRATIFIED (Test ROC-AUC)")
    print("=" * 100)
    
    header = f"{'Modelo ESM-2':<25} {'Clf':<5} {'Random':<15} {'Stratified':<15} {'Delta':<10}"
    print(header)
    print("-" * 100)
    
    for esm_model in random_results['models']:
        for clf in ['KNN', 'MLP']:
            random_auc = random_results['models'][esm_model]['aggregated'][clf]['roc_auc']
            strat_auc = stratified_results['models'][esm_model]['aggregated'][clf]['roc_auc']
            
            delta = (strat_auc['mean'] - random_auc['mean']) * 100
            
            model_display = esm_model.replace('_UR50D', '') if clf == 'KNN' else ''
            
            row = f"{model_display:<25} {clf:<5} "
            row += f"{random_auc['mean']:.4f}±{random_auc['std']:.4f}  "
            row += f"{strat_auc['mean']:.4f}±{strat_auc['std']:.4f}  "
            row += f"{delta:+.2f}%"
            print(row)
        print()
    
    print("=" * 100)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Baseline Multi-Seed Pipeline')
    parser.add_argument('--embeddings-dir', type=str, 
                       default='results/protein_model_benchmark_non_human_v2')
    parser.add_argument('--output', type=str, default='results/baseline_multiseed')
    parser.add_argument('--n-seeds', type=int, default=5, help='Número de seeds (1-5)')
    parser.add_argument('--split', type=str, choices=['random', 'stratified', 'both'],
                       default='random', help='Método de split')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Pular se resultados já existirem')
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = SEEDS[:args.n_seeds]
    
    random_results = None
    stratified_results = None
    
    total_start = time.time()
    
    # Executar split(s) selecionado(s)
    if args.split in ['random', 'both']:
        random_results = run_pipeline(
            embeddings_dir, output_dir, 'random', 
            args.n_seeds, args.skip_existing
        )
        if random_results.get('models'):  # Se tem dados novos
            create_single_visualization(random_results, output_dir, 'random')
            print_summary_table(random_results, seeds, 'random')
    
    if args.split in ['stratified', 'both']:
        stratified_results = run_pipeline(
            embeddings_dir, output_dir, 'stratified',
            args.n_seeds, args.skip_existing
        )
        if stratified_results.get('models'):  # Se tem dados novos
            create_single_visualization(stratified_results, output_dir, 'stratified')
            print_summary_table(stratified_results, seeds, 'stratified')
    
    # Comparação se ambos foram executados
    if args.split == 'both':
        # Garantir que temos ambos os resultados
        if random_results is None:
            random_results = load_existing_results(output_dir, 'random')
        if stratified_results is None:
            stratified_results = load_existing_results(output_dir, 'stratified')
        
        if random_results and stratified_results:
            create_comparison_visualization(random_results, stratified_results, output_dir)
            print_comparison_table(random_results, stratified_results)
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print(f"⏱️  Tempo total: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"📁 Resultados em: {output_dir}")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
