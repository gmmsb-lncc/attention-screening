#!/usr/bin/env python3
"""
Utilitários para visualização de clusters.

Este módulo contém constantes, funções auxiliares e operações
de carregamento de dados para as visualizações de clustering.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN


# ============================================================================
# CONSTANTES
# ============================================================================

SPLIT_COLORS = {
    0: '#3498db',  # Train - Azul
    1: '#2ecc71',  # Val - Verde
    2: '#e74c3c'   # Test - Vermelho
}

SPLIT_NAMES = {0: 'Train', 1: 'Val', 2: 'Test'}


# ============================================================================
# FUNÇÕES DE CARREGAMENTO
# ============================================================================

def load_embeddings(embeddings_path: Optional[str] = None,
                   protein_path: Optional[str] = None,
                   ligand_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Carrega embeddings de arquivo(s).
    
    Args:
        embeddings_path: Caminho para embeddings concatenados
        protein_path: Caminho para embeddings de proteína
        ligand_path: Caminho para embeddings de ligante
        
    Returns:
        (embeddings, protein_emb, ligand_emb)
    """
    if embeddings_path:
        embeddings = np.load(embeddings_path)
        return embeddings, None, None
    elif protein_path and ligand_path:
        protein_emb = np.load(protein_path)
        ligand_emb = np.load(ligand_path)
        embeddings = np.concatenate([protein_emb, ligand_emb], axis=1)
        return embeddings, protein_emb, ligand_emb
    else:
        raise ValueError("Forneça embeddings_path OU (protein_path + ligand_path)")


def load_labels(labels_path: str, threshold: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega labels originais e cria versão binarizada.
    
    Args:
        labels_path: Caminho para arquivo de labels
        threshold: Threshold para binarização (padrão: 1000 nM)
        
    Returns:
        (labels_originais, labels_binarios)
    """
    labels = np.load(labels_path)
    
    # Detectar se já são binários
    unique_vals = np.unique(labels)
    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
        print(f"✓ Labels já estão binarizados: {np.sum(labels == 1)} ativos, {np.sum(labels == 0)} inativos")
        return labels, labels
    
    # Binarizar se necessário
    binary_labels = (labels <= threshold).astype(int)
    n_active = np.sum(binary_labels == 1)
    n_inactive = np.sum(binary_labels == 0)
    print(f"✓ Labels binarizados (threshold={threshold}): {n_active} ativos, {n_inactive} inativos")
    
    return labels, binary_labels


def load_split_indices(train_path: str, val_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega índices dos splits.
    
    Returns:
        (train_indices, val_indices, test_indices)
    """
    train_indices = np.load(train_path)
    val_indices = np.load(val_path)
    test_indices = np.load(test_path)
    return train_indices, val_indices, test_indices


def create_split_assignment(n_samples: int, 
                           train_indices: np.ndarray,
                           val_indices: np.ndarray, 
                           test_indices: np.ndarray) -> np.ndarray:
    """
    Cria array de atribuição de splits.
    
    Returns:
        Array onde 0=train, 1=val, 2=test
    """
    split_assignment = np.full(n_samples, -1, dtype=int)
    split_assignment[train_indices] = 0
    split_assignment[val_indices] = 1
    split_assignment[test_indices] = 2
    
    if np.any(split_assignment == -1):
        raise ValueError("Alguns samples não foram atribuídos a nenhum split!")
    
    return split_assignment


# ============================================================================
# ANÁLISE E CLUSTERING
# ============================================================================

def analyze_similarity_distribution(embeddings: np.ndarray, 
                                   sample_size: int = 1000) -> Dict[str, float]:
    """
    Analisa distribuição de similaridades dos embeddings.
    
    Args:
        embeddings: Array de embeddings
        sample_size: Número de amostras para análise
        
    Returns:
        Dicionário com estatísticas de similaridade
    """
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample = embeddings[indices]
    else:
        sample = embeddings
    
    # Normalizar para cosseno
    from sklearn.preprocessing import normalize
    sample_norm = normalize(sample, axis=1, norm='l2')
    
    # Calcular similaridades
    similarities = cosine_similarity(sample_norm)
    
    # Pegar apenas triangular superior (sem diagonal)
    mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
    sim_values = similarities[mask]
    
    return {
        'min': float(np.min(sim_values)),
        'max': float(np.max(sim_values)),
        'mean': float(np.mean(sim_values)),
        'median': float(np.median(sim_values)),
        'std': float(np.std(sim_values))
    }


def perform_clustering(embeddings: np.ndarray, 
                      method: str = 'hierarchical',
                      similarity_threshold: float = 0.7,
                      n_clusters: Optional[int] = None,
                      min_samples: int = 3,
                      eps: float = 0.5) -> Tuple[np.ndarray, int]:
    """
    Executa clustering nos embeddings.
    
    Args:
        embeddings: Array de embeddings
        method: 'hierarchical', 'kmeans', ou 'dbscan'
        similarity_threshold: Threshold de similaridade (para hierarchical)
        n_clusters: Número de clusters (para kmeans)
        min_samples: Amostras mínimas (para dbscan)
        eps: Raio epsilon (para dbscan)
        
    Returns:
        (cluster_labels, n_clusters)
    """
    from sklearn.preprocessing import normalize
    
    embeddings_norm = normalize(embeddings, axis=1, norm='l2')
    
    if method == 'hierarchical':
        distance_threshold = 1.0 - similarity_threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='cosine',
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(embeddings_norm)
        
    elif method == 'kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters é obrigatório para kmeans")
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clustering.fit_predict(embeddings_norm)
        
    elif method == 'dbscan':
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings_norm)
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    unique_clusters = np.unique(cluster_labels)
    valid_clusters = unique_clusters[unique_clusters != -1]
    n_clusters_found = len(valid_clusters)
    
    return cluster_labels, n_clusters_found


def calculate_cluster_metrics(embeddings: np.ndarray, 
                             cluster_labels: np.ndarray) -> Dict[str, Any]:
    """
    Calcula métricas de qualidade dos clusters.
    
    Returns:
        Dicionário com métricas
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.preprocessing import normalize
    
    embeddings_norm = normalize(embeddings, axis=1, norm='l2')
    
    # Filtrar ruído (label -1)
    valid_mask = cluster_labels != -1
    if not np.any(valid_mask):
        return {'error': 'Todos os pontos são ruído'}
    
    emb_valid = embeddings_norm[valid_mask]
    labels_valid = cluster_labels[valid_mask]
    
    unique_labels = np.unique(labels_valid)
    if len(unique_labels) < 2:
        return {'error': 'Menos de 2 clusters válidos'}
    
    metrics = {
        'n_clusters': len(unique_labels),
        'n_noise': int(np.sum(~valid_mask)),
        'silhouette_score': float(silhouette_score(emb_valid, labels_valid, metric='cosine')),
        'davies_bouldin_score': float(davies_bouldin_score(emb_valid, labels_valid)),
        'calinski_harabasz_score': float(calinski_harabasz_score(emb_valid, labels_valid)),
    }
    
    # Estatísticas de tamanho dos clusters
    cluster_sizes = [np.sum(labels_valid == c) for c in unique_labels]
    metrics['cluster_size_mean'] = float(np.mean(cluster_sizes))
    metrics['cluster_size_std'] = float(np.std(cluster_sizes))
    metrics['cluster_size_min'] = int(np.min(cluster_sizes))
    metrics['cluster_size_max'] = int(np.max(cluster_sizes))
    
    return metrics


# ============================================================================
# FUNÇÕES AUXILIARES DE PLOTAGEM
# ============================================================================

def calculate_split_sizes(split_assignment: np.ndarray) -> Tuple[int, int, int, int]:
    """Calcula tamanhos dos splits."""
    n_train = np.sum(split_assignment == 0)
    n_val = np.sum(split_assignment == 1)
    n_test = np.sum(split_assignment == 2)
    total = len(split_assignment)
    return n_train, n_val, n_test, total


def setup_axis(ax, xlabel: str, ylabel: str, title: str, add_grid: bool = True):
    """Configura eixo com labels e grid."""
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    if add_grid:
        ax.grid(True, alpha=0.3)
