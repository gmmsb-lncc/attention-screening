#!/usr/bin/env python3
"""
Script para visualização PCA de clusters da estratificação.

Gera visualizações das componentes principais para análise 
da distribuição dos clusters no espaço de embeddings.

IMPORTANTE: Este script usa o MESMO critério de clustering da estratificação:
- Similaridade de cossenos >= 0.7 para agrupar vetores
- Mínimo de 3 pontos para formar um cluster

Usage:
    python scripts/visualize_cluster_pca.py --embeddings path/to/embeddings.npy --output results/
    python scripts/visualize_cluster_pca.py --protein-emb protein.npy --ligand-emb ligand.npy --output results/
    
Opções de visualização:
    --method: pca, tsne, umap (default: pca)
    --n-components: número de componentes (default: 2)
    --perplexity: perplexidade para t-SNE (default: 30)
    --n-neighbors: vizinhos para UMAP (default: 15)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONSTANTES GLOBAIS
# ============================================================================

SPLIT_COLORS = {
    0: '#3498db',  # Train - Azul
    1: '#2ecc71',  # Val - Verde
    2: '#e74c3c'   # Test - Vermelho
}

SPLIT_NAMES = {0: 'Train', 1: 'Val', 2: 'Test'}


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def calculate_split_sizes(split_assignment: np.ndarray) -> Tuple[int, int, int, int]:
    """Calcula tamanhos dos splits (reutilizável)."""
    n_train = np.sum(split_assignment == 0)
    n_val = np.sum(split_assignment == 1)
    n_test = np.sum(split_assignment == 2)
    total = len(split_assignment)
    return n_train, n_val, n_test, total


def setup_axis(ax, xlabel: str, ylabel: str, title: str, add_grid: bool = True):
    """Configura eixo (evita repetição)."""
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    if add_grid:
        ax.grid(True, alpha=0.3)


def plot_split_highlight(ax, embeddings_2d: np.ndarray, split_assignment: np.ndarray, 
                        split_id: int, xlabel: str, ylabel: str):
    """Plota scatter destacando um split específico."""
    mask = split_assignment == split_id
    other_mask = ~mask
    n_split = np.sum(mask)
    total = len(split_assignment)
    
    # Background (outros splits)
    ax.scatter(embeddings_2d[other_mask, 0], embeddings_2d[other_mask, 1],
               c='lightgray', alpha=0.15, s=8, label='Outros')
    
    # Split destacado
    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
               c=SPLIT_COLORS[split_id], alpha=0.7, s=25, 
               edgecolors='white', linewidth=0.5, 
               label=SPLIT_NAMES[split_id])
    
    setup_axis(ax, xlabel, ylabel, 
               f'{SPLIT_NAMES[split_id]} (n={n_split}, {n_split/total:.1%})')
    ax.legend(loc='best', fontsize=9)


# ============================================================================
# FUNÇÕES PRINCIPAIS
# ============================================================================


def load_embeddings(embeddings_path: Optional[str] = None,
                   protein_path: Optional[str] = None,
                   ligand_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Carrega embeddings de arquivo(s).
    
    Args:
        embeddings_path: Caminho para embeddings concatenados
        protein_path: Caminho para embeddings de proteínas
        ligand_path: Caminho para embeddings de ligantes
        
    Returns:
        Tuple (concatenated, protein, ligand)
    """
    protein_emb = None
    ligand_emb = None
    
    if embeddings_path:
        combined = np.load(embeddings_path)
        return combined, None, None
    
    if protein_path and ligand_path:
        protein_emb = np.load(protein_path)
        ligand_emb = np.load(ligand_path)
        combined = np.concatenate([protein_emb, ligand_emb], axis=1)
        return combined, protein_emb, ligand_emb
    
    raise ValueError("Forneça --embeddings ou ambos --protein-emb e --ligand-emb")


def load_labels(labels_path: str, threshold: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega labels e converte para valores numéricos.
    
    Suporta:
    - Array 1D de valores binários (0/1) - retorna diretamente
    - Array 1D de valores numéricos (IC50/Kd/Ki em nM) - binariza
    - Array 2D com colunas [chembl_id, protein, assay_type, value]
    
    Args:
        labels_path: Caminho para arquivo de labels
        threshold: Threshold para binarização (nM), default 1000
        
    Returns:
        Tuple (labels_numéricos, labels_binários)
    """
    labels = np.load(labels_path, allow_pickle=True)
    
    if labels.ndim == 1:
        # Array 1D - verificar se já é binário (0/1)
        unique_vals = np.unique(labels)
        if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
            # Já é binário (0=inativo, 1=ativo)
            binary_labels = labels.astype(int)
            numeric_labels = binary_labels.astype(float)
        else:
            # Array numérico - tentar converter para float e binarizar
            try:
                numeric_labels = labels.astype(float)
                # Ativo (1) se valor < threshold, Inativo (0) se valor >= threshold
                binary_labels = (numeric_labels < threshold).astype(int)
            except ValueError:
                # Se não conseguir, usar índices
                _, numeric_labels = np.unique(labels, return_inverse=True)
                numeric_labels = numeric_labels.astype(float)
                binary_labels = (numeric_labels < threshold).astype(int)
    elif labels.ndim == 2 and labels.shape[1] >= 4:
        # Array 2D com formato [chembl_id, protein, assay_type, value]
        # Extrair coluna de valor (última coluna)
        try:
            numeric_labels = labels[:, -1].astype(float)
            binary_labels = (numeric_labels < threshold).astype(int)
        except ValueError:
            print("⚠️  Não foi possível converter labels para numérico")
            numeric_labels = np.arange(len(labels), dtype=float)
            binary_labels = np.zeros(len(labels), dtype=int)
    else:
        # Tentar usar a primeira coluna ou índices
        try:
            numeric_labels = labels.flatten().astype(float)
            binary_labels = (numeric_labels < threshold).astype(int)
        except ValueError:
            numeric_labels = np.arange(len(labels), dtype=float)
            binary_labels = np.zeros(len(labels), dtype=int)
    
    return numeric_labels, binary_labels


def analyze_similarity_distribution(embeddings: np.ndarray, 
                                   sample_size: int = 1000,
                                   seed: int = 42) -> Dict[str, float]:
    """
    Analisa a distribuição de similaridades de cosseno nos embeddings.
    Retorna estatísticas e recomenda um threshold apropriado.
    
    Args:
        embeddings: Matriz de embeddings
        sample_size: Tamanho da amostra para análise
        seed: Seed para reprodutibilidade
        
    Returns:
        Dicionário com estatísticas de similaridade e threshold recomendado
    """
    np.random.seed(seed)
    
    # Amostra para análise eficiente
    n_samples = min(sample_size, embeddings.shape[0])
    sample_idx = np.random.choice(embeddings.shape[0], n_samples, replace=False)
    sample_emb = embeddings[sample_idx]
    
    # Calcular similaridade
    sim_matrix = cosine_similarity(sample_emb)
    
    # Remover diagonal (auto-similaridade = 1)
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    similarities = sim_matrix[mask]
    
    stats = {
        'min': float(similarities.min()),
        'max': float(similarities.max()),
        'mean': float(similarities.mean()),
        'std': float(similarities.std()),
        'median': float(np.median(similarities)),
        'p10': float(np.percentile(similarities, 10)),
        'p25': float(np.percentile(similarities, 25)),
        'p75': float(np.percentile(similarities, 75)),
        'p90': float(np.percentile(similarities, 90)),
        'p95': float(np.percentile(similarities, 95)),
        'p99': float(np.percentile(similarities, 99)),
    }
    
    # Recomendar threshold baseado na distribuição
    # Se os embeddings são muito homogêneos, recomendar threshold alto
    if stats['min'] > 0.9:
        # Embeddings muito homogêneos - usar P50-P75 como threshold
        stats['recommended_threshold'] = float(np.percentile(similarities, 60))
        stats['homogeneity'] = 'very_high'
        stats['recommendation'] = (
            f"Embeddings altamente homogêneos (min={stats['min']:.3f}). "
            f"Recomendado threshold: {stats['recommended_threshold']:.3f}"
        )
    elif stats['min'] > 0.7:
        # Embeddings moderadamente homogêneos
        stats['recommended_threshold'] = float(np.percentile(similarities, 40))
        stats['homogeneity'] = 'high'
        stats['recommendation'] = (
            f"Embeddings homogêneos (min={stats['min']:.3f}). "
            f"Recomendado threshold: {stats['recommended_threshold']:.3f}"
        )
    else:
        # Embeddings com boa variabilidade - threshold 0.7 é adequado
        stats['recommended_threshold'] = 0.7
        stats['homogeneity'] = 'normal'
        stats['recommendation'] = "Embeddings com boa variabilidade. Threshold 0.7 é adequado."
    
    return stats


def perform_clustering(embeddings: np.ndarray, 
                       algorithm: str = 'hierarchical',
                       similarity_threshold: float = 0.7,
                       min_cluster_size: int = 3,
                       n_clusters: int = 10,
                       **kwargs) -> np.ndarray:
    """
    Realiza clustering nos embeddings usando o MESMO critério da estratificação.
    
    O critério padrão é:
    - Clustering hierárquico com distância de cosseno
    - Similaridade >= 0.7 para agrupar vetores (distance_threshold = 1 - 0.7 = 0.3)
    - Mínimo de 3 pontos para formar um cluster
    
    Args:
        embeddings: Matriz de embeddings
        algorithm: Algoritmo ('hierarchical', 'kmeans', 'dbscan')
        similarity_threshold: Limiar de similaridade de cossenos (default: 0.7)
        min_cluster_size: Tamanho mínimo do cluster (default: 3)
        n_clusters: Número de clusters (para kmeans)
        
    Returns:
        Array com labels dos clusters
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    
    # Calcular matriz de similaridade de cossenos
    print(f"   Calculando similaridade de cossenos...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Converter para distância (distância de cosseno = 1 - similaridade)
    distance_matrix = np.clip(1 - similarity_matrix, 0, 2)
    
    # Converter threshold de similaridade para threshold de distância
    distance_threshold = 1 - similarity_threshold
    
    if algorithm == 'hierarchical':
        # Clustering hierárquico com distância de cosseno
        # Este é o mesmo método usado na estratificação
        print(f"   Usando clustering hierárquico (distance_threshold={distance_threshold:.2f})")
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        labels = clusterer.fit_predict(distance_matrix)
        
        # Filtrar clusters pequenos (marcar como ruído = -1)
        unique_labels, counts = np.unique(labels, return_counts=True)
        small_clusters = unique_labels[counts < min_cluster_size]
        
        for small_cluster in small_clusters:
            labels[labels == small_cluster] = -1
        
        # Renumerar clusters válidos
        valid_labels = labels[labels >= 0]
        if len(valid_labels) > 0:
            unique_valid = np.unique(valid_labels)
            label_map = {old: new for new, old in enumerate(unique_valid)}
            new_labels = np.array([label_map.get(l, -1) if l >= 0 else -1 for l in labels])
            labels = new_labels
            
    elif algorithm == 'dbscan':
        # DBSCAN com distância de cosseno
        eps = kwargs.get('eps', distance_threshold)
        print(f"   Usando DBSCAN (eps={eps:.2f}, min_samples={min_cluster_size})")
        clusterer = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='precomputed')
        labels = clusterer.fit_predict(distance_matrix)
        
    elif algorithm == 'kmeans':
        # K-means (não usa distância de cosseno precomputada)
        print(f"   Usando K-Means (n_clusters={n_clusters})")
        # Normalizar embeddings para aproximar similaridade de cosseno
        embeddings_normalized = normalize(embeddings, norm='l2')
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings_normalized)
    else:
        raise ValueError(f"Algoritmo desconhecido: {algorithm}")
    
    return labels


def compute_pca(embeddings: np.ndarray, 
                n_components: int = 2,
                normalize: bool = True) -> Tuple[np.ndarray, PCA]:
    """
    Calcula PCA nos embeddings.
    
    Args:
        embeddings: Matriz de embeddings
        n_components: Número de componentes principais
        normalize: Se True, normaliza os dados antes do PCA
        
    Returns:
        Tuple (embeddings_reduzidos, pca_model)
    """
    if normalize:
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
    else:
        embeddings_scaled = embeddings
    
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    
    return embeddings_pca, pca


def compute_tsne(embeddings: np.ndarray, 
                 n_components: int = 2,
                 perplexity: int = 30) -> np.ndarray:
    """Calcula t-SNE nos embeddings."""
    from sklearn.manifold import TSNE
    
    # t-SNE é sensível à dimensionalidade - reduzir com PCA primeiro se > 50D
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(embeddings)


def compute_umap(embeddings: np.ndarray, 
                 n_components: int = 2,
                 n_neighbors: int = 15) -> np.ndarray:
    """Calcula UMAP nos embeddings."""
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP não instalado. Execute: pip install umap-learn")
    
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
    return reducer.fit_transform(embeddings)


def plot_cluster_pca(embeddings_2d: np.ndarray,
                     cluster_labels: np.ndarray,
                     pca: Optional[PCA] = None,
                     labels: Optional[np.ndarray] = None,
                     title: str = "Cluster PCA Visualization",
                     output_path: Optional[str] = None,
                     method: str = 'pca') -> plt.Figure:
    """
    Gera visualização 2D dos clusters em layout 2x2.
    
    Args:
        embeddings_2d: Embeddings reduzidos a 2D
        cluster_labels: Labels dos clusters
        pca: Modelo PCA (para exibir variância explicada)
        labels: Labels originais (ativo/inativo)
        title: Título do gráfico
        output_path: Caminho para salvar
        method: Método usado ('pca', 'tsne', 'umap')
        
    Returns:
        Figura matplotlib
    """
    # Layout 2x2 para melhor visualização
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Adicionar nota sobre variância se necessário
    display_title = title
    if pca is not None and method == 'pca':
        var_total = np.sum(pca.explained_variance_ratio_[:2])
        if var_total < 0.3:  # Se PC1+PC2 < 30%
            orig_dim = pca.n_features_in_ if hasattr(pca, 'n_features_in_') else "alta"
            display_title += f'\n(PC1+PC2 explicam {var_total:.1%} - Normal para embeddings de {orig_dim} dimensões altamente homogêneos)'
    
    fig.suptitle(display_title, fontsize=16, fontweight='bold', y=0.98)
    
    # Cores para clusters
    unique_clusters = np.unique(cluster_labels)
    # Separar clusters válidos do ruído (-1)
    valid_clusters = unique_clusters[unique_clusters != -1]
    n_clusters = len(valid_clusters)
    
    # Usar colormap diferente baseado no número de clusters
    if n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    # Criar mapeamento de cluster_id para cor
    color_map = {c: colors[i] for i, c in enumerate(valid_clusters)}
    color_map[-1] = [0.7, 0.7, 0.7, 1.0]  # Cinza para ruído
    
    # =============================================
    # Plot 1 (Superior Esquerdo): Clusters coloridos por ID
    # =============================================
    ax1 = axes[0, 0]
    
    # Plotar ruído primeiro (se existir)
    noise_mask = cluster_labels == -1
    if np.any(noise_mask):
        ax1.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1],
                   c=[color_map[-1]], label='Ruído', alpha=0.3, s=15, marker='x')
    
    # Plotar clusters válidos
    for cluster_id in valid_clusters:
        mask = cluster_labels == cluster_id
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[color_map[cluster_id]], alpha=0.6, s=25)
    
    if pca is not None and method == 'pca':
        var1, var2 = pca.explained_variance_ratio_[:2]
        xlabel = f'PC1 ({var1:.1%} variância)'
        ylabel = f'PC2 ({var2:.1%} variância)'
    else:
        xlabel = f'{method.upper()} Componente 1'
        ylabel = f'{method.upper()} Componente 2'
    
    setup_axis(ax1, xlabel, ylabel, f'Distribuição dos Clusters (n={n_clusters})')
    
    # =============================================
    # Plot 2 (Superior Direito): Labels originais (ativo/inativo)
    # =============================================
    ax2 = axes[0, 1]
    
    if labels is not None:
        unique_labels = np.unique(labels)
        n_unique_labels = len(unique_labels)
        
        if n_unique_labels <= 2:
            # Labels binários - ativo/inativo
            colors_labels = ['#2ecc71', '#e74c3c']  # Verde para ativo, vermelho para inativo
            label_names = {0: 'Inativo', 1: 'Ativo'}
            
            for i, label in enumerate(sorted(unique_labels)):
                mask = labels == label
                ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=colors_labels[int(label)], label=label_names.get(label, f'Label {label}'),
                           alpha=0.6, s=25)
            ax2.legend(loc='upper right', fontsize=10)
            ax2.set_title('Distribuição Ativo/Inativo')
        elif n_unique_labels > 20:
            # Labels contínuos
            scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                 c=labels, cmap='coolwarm', alpha=0.6, s=25)
            plt.colorbar(scatter, ax=ax2, label='Valor', shrink=0.8)
            ax2.set_title('Distribuição por Valor')
        else:
            # Labels discretos moderados
            for label in unique_labels:
                mask = labels == label
                ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           label=f'Label {label}', alpha=0.6, s=25)
            if n_unique_labels <= 10:
                ax2.legend(loc='upper right', fontsize=8)
            ax2.set_title('Distribuição por Label')
    else:
        # Sem labels - mostrar densidade
        ax2.hexbin(embeddings_2d[:, 0], embeddings_2d[:, 1], gridsize=30, 
                   cmap='YlOrRd', mincnt=1)
        plt.colorbar(ax2.collections[0], ax=ax2, label='Densidade', shrink=0.8)
        ax2.set_title('Mapa de Densidade')
    
    setup_axis(ax2, xlabel, ylabel, ax2.get_title())
    
    # =============================================
    # Plot 3 (Inferior Esquerdo): Histograma de tamanhos dos clusters
    # =============================================
    ax3 = axes[1, 0]
    cluster_sizes = [np.sum(cluster_labels == c) for c in valid_clusters]
    
    if n_clusters > 30:
        # Muitos clusters - usar histograma
        ax3.hist(cluster_sizes, bins=min(30, n_clusters//2), color='steelblue', 
                 edgecolor='white', alpha=0.8)
        ax3.set_xlabel('Tamanho do Cluster')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Distribuição de Tamanhos dos Clusters')
        
        # Adicionar estatísticas
        mean_size = np.mean(cluster_sizes)
        median_size = np.median(cluster_sizes)
        ax3.axvline(mean_size, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_size:.1f}')
        ax3.axvline(median_size, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median_size:.1f}')
        ax3.legend(loc='upper right')
    else:
        # Poucos clusters - barras individuais ordenadas por tamanho
        sorted_idx = np.argsort(cluster_sizes)[::-1]
        sorted_sizes = [cluster_sizes[i] for i in sorted_idx]
        sorted_labels = [valid_clusters[i] for i in sorted_idx]
        sorted_colors = [color_map[valid_clusters[i]] for i in sorted_idx]
        
        bars = ax3.bar(range(len(sorted_sizes)), sorted_sizes, color=sorted_colors, 
                       edgecolor='white', alpha=0.8)
        
        # Labels apenas se couberem
        if n_clusters <= 15:
            ax3.set_xticks(range(len(sorted_labels)))
            ax3.set_xticklabels([f'C{c}' for c in sorted_labels], rotation=45, ha='right')
        
        # Adicionar valores nas barras maiores
        mean_size = np.mean(cluster_sizes)
        for i, (bar, size) in enumerate(zip(bars, sorted_sizes)):
            if i < 10 or size > mean_size:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(size), ha='center', va='bottom', fontsize=7)
    
    setup_axis(ax3, 'Cluster (ordenado por tamanho)' if n_clusters <= 30 else 'Tamanho do Cluster',
               'Número de Amostras' if n_clusters <= 30 else 'Frequência',
               'Distribuição de Tamanhos dos Clusters' if n_clusters > 30 else 'Tamanho dos Clusters',
               add_grid=True)
    
    # =============================================
    # Plot 4 (Inferior Direito): Estatísticas e composição dos clusters
    # =============================================
    ax4 = axes[1, 1]
    
    if labels is not None and len(np.unique(labels)) <= 2:
        # Composição ativo/inativo por cluster (top clusters)
        top_n = min(15, n_clusters)
        top_clusters = valid_clusters[np.argsort(cluster_sizes)[::-1][:top_n]]
        
        active_counts = []
        inactive_counts = []
        
        for cluster_id in top_clusters:
            mask = cluster_labels == cluster_id
            cluster_labels_subset = labels[mask]
            active_counts.append(np.sum(cluster_labels_subset == 1))
            inactive_counts.append(np.sum(cluster_labels_subset == 0))
        
        x = np.arange(len(top_clusters))
        width = 0.7
        
        bars1 = ax4.bar(x, active_counts, width, label='Ativo', color='#2ecc71', alpha=0.8)
        bars2 = ax4.bar(x, inactive_counts, width, bottom=active_counts, label='Inativo', color='#e74c3c', alpha=0.8)
        
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'C{c}' for c in top_clusters], rotation=45, ha='right')
        ax4.legend(loc='upper right')
        setup_axis(ax4, 'Top Clusters (por tamanho)', 'Número de Amostras', 
                  'Composição Ativo/Inativo por Cluster')
        
        # Calcular e mostrar proporções
        total_active = np.sum(labels == 1)
        total_inactive = np.sum(labels == 0)
        ax4.text(0.02, 0.98, f'Total: {total_active} ativos, {total_inactive} inativos', 
                transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Sem labels binários - mostrar boxplot de tamanhos ou pie chart
        if n_clusters >= 5:
            # Pie chart com top clusters + outros
            top_n = min(8, n_clusters)
            sorted_idx = np.argsort(cluster_sizes)[::-1]
            top_sizes = [cluster_sizes[i] for i in sorted_idx[:top_n]]
            top_labels_names = [f'C{valid_clusters[i]}' for i in sorted_idx[:top_n]]
            top_colors = [color_map[valid_clusters[i]] for i in sorted_idx[:top_n]]
            
            if n_clusters > top_n:
                others_size = sum(cluster_sizes[i] for i in sorted_idx[top_n:])
                top_sizes.append(others_size)
                top_labels_names.append(f'Outros ({n_clusters - top_n})')
                top_colors.append([0.7, 0.7, 0.7, 1.0])
            
            wedges, texts, autotexts = ax4.pie(top_sizes, labels=top_labels_names, colors=top_colors,
                                               autopct='%1.1f%%', startangle=90, pctdistance=0.75)
            ax4.set_title('Proporção dos Maiores Clusters')
            
            # Ajustar tamanho do texto
            for autotext in autotexts:
                autotext.set_fontsize(8)
            for text in texts:
                text.set_fontsize(9)
        else:
            # Poucos clusters - barras horizontais com percentuais
            sorted_idx = np.argsort(cluster_sizes)
            sorted_sizes = [cluster_sizes[i] for i in sorted_idx]
            sorted_labels = [f'Cluster {valid_clusters[i]}' for i in sorted_idx]
            sorted_colors = [color_map[valid_clusters[i]] for i in sorted_idx]
            
            total = sum(sorted_sizes)
            percentages = [s/total*100 for s in sorted_sizes]
            
            bars = ax4.barh(range(len(sorted_sizes)), percentages, color=sorted_colors, 
                           edgecolor='white', alpha=0.8)
            ax4.set_xlabel('Porcentagem do Total (%)')
            ax4.set_ylabel('Cluster')
            ax4.set_title('Distribuição Percentual')
            ax4.set_yticks(range(len(sorted_labels)))
            ax4.set_yticklabels(sorted_labels)
            
            for i, (bar, pct, size) in enumerate(zip(bars, percentages, sorted_sizes)):
                ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{pct:.1f}% (n={size})', ha='left', va='center', fontsize=8)
            ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualização salva em: {output_path}")
    
    return fig


def plot_pca_variance(pca: PCA, output_path: Optional[str] = None) -> plt.Figure:
    """
    Gera gráfico de variância explicada por componente.
    
    Args:
        pca: Modelo PCA ajustado
        output_path: Caminho para salvar
        
    Returns:
        Figura matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n_components = len(pca.explained_variance_ratio_)
    
    # Plot 1: Variância por componente
    ax1 = axes[0]
    ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_ * 100, 
            color='steelblue', alpha=0.8)
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Variância Explicada (%)')
    ax1.set_title('Variância Explicada por Componente')
    ax1.set_xticks(range(1, n_components + 1))
    
    # Plot 2: Variância acumulada
    ax2 = axes[1]
    cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100
    ax2.plot(range(1, n_components + 1), cumulative_var, 'o-', color='darkgreen')
    ax2.axhline(y=90, color='red', linestyle='--', label='90% variância')
    ax2.axhline(y=95, color='orange', linestyle='--', label='95% variância')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Variância Acumulada (%)')
    ax2.set_title('Variância Acumulada')
    ax2.legend()
    ax2.set_xticks(range(1, n_components + 1))
    ax2.set_ylim(0, 105)
    
    # Encontrar n_componentes para 90% e 95%
    n_90 = np.argmax(cumulative_var >= 90) + 1
    n_95 = np.argmax(cumulative_var >= 95) + 1
    
    fig.suptitle(f'Análise PCA - {n_90} PCs para 90%, {n_95} PCs para 95% variância', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico de variância salvo em: {output_path}")
    
    return fig


def plot_multi_view_comparison(protein_emb: np.ndarray,
                               ligand_emb: np.ndarray,
                               combined_emb: np.ndarray,
                               cluster_labels: np.ndarray,
                               output_path: Optional[str] = None) -> plt.Figure:
    """
    Compara visualizações PCA de proteínas, ligantes e combinados.
    
    Args:
        protein_emb: Embeddings de proteínas
        ligand_emb: Embeddings de ligantes
        combined_emb: Embeddings concatenados
        cluster_labels: Labels dos clusters
        output_path: Caminho para salvar
        
    Returns:
        Figura matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Comparação Multi-View PCA', fontsize=14, fontweight='bold')
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    embeddings_list = [
        ('Proteínas', protein_emb),
        ('Ligantes', ligand_emb),
        ('Combinado', combined_emb)
    ]
    
    for idx, (name, emb) in enumerate(embeddings_list):
        ax = axes[idx]
        emb_2d, pca = compute_pca(emb, n_components=2)
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                      c=[colors[i]], alpha=0.7, s=30)
        
        var1, var2 = pca.explained_variance_ratio_[:2]
        ax.set_xlabel(f'PC1 ({var1:.1%})')
        ax.set_ylabel(f'PC2 ({var2:.1%})')
        ax.set_title(f'{name} (dim={emb.shape[1]})')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparação multi-view salva em: {output_path}")
    
    return fig


def plot_split_distribution(embeddings_2d: np.ndarray,
                            cluster_labels: np.ndarray,
                            split_assignment: np.ndarray,
                            labels: Optional[np.ndarray] = None,
                            pca: Optional[PCA] = None,
                            method: str = 'pca',
                            output_path: Optional[str] = None) -> plt.Figure:
    """
    Visualiza a distribuição dos splits (train/val/test) nos clusters.
    Layout 2x3:
    Linha 1: [Visão Geral] [Distribuição dos Clusters] [Ativo/Inativo]
    Linha 2: [Train]        [Val]                       [Test]
    
    Args:
        embeddings_2d: Embeddings reduzidos a 2D
        cluster_labels: Labels dos clusters
        split_assignment: Array indicando split (0=train, 1=val, 2=test)
        labels: Labels originais (opcional)
        pca: Modelo PCA (opcional)
        method: Método de redução ('pca', 'tsne', 'umap')
        output_path: Caminho para salvar
        
    Returns:
        Figura matplotlib
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.25)
    
    # Criar título com nota sobre variância se necessário
    title = 'Distribuição dos Splits Train/Val/Test nos Clusters'
    if pca is not None and method == 'pca':
        var_total = np.sum(pca.explained_variance_ratio_[:2])
        if var_total < 0.3:  # Se PC1+PC2 < 30%
            orig_dim = pca.n_features_in_ if hasattr(pca, 'n_features_in_') else 1728
            title += f'\n(PC1+PC2: {var_total:.1%} - Normal para {orig_dim}D embeddings homogêneos)'
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
    
    # Usar constantes globais
    n_train, n_val, n_test, total = calculate_split_sizes(split_assignment)
    
    # Estatísticas de clusters
    unique_clusters = np.unique(cluster_labels)
    valid_clusters = unique_clusters[unique_clusters != -1]
    
    # Labels dos eixos
    if pca is not None and method == 'pca':
        var1, var2 = pca.explained_variance_ratio_[:2]
        xlabel = f'PC1 ({var1:.1%})'
        ylabel = f'PC2 ({var2:.1%})'
    else:
        xlabel = f'{method.upper()} 1'
        ylabel = f'{method.upper()} 2'
    
    # =============================================
    # [0][0] Visão Geral
    # =============================================
    ax00 = fig.add_subplot(gs[0, 0])
    
    for split_id in [0, 1, 2]:
        mask = split_assignment == split_id
        ax00.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=SPLIT_COLORS[split_id], label=SPLIT_NAMES[split_id],
                    alpha=0.6, s=25, edgecolors='white', linewidth=0.5)
    
    setup_axis(ax00, xlabel, ylabel, 'Visão Geral: Splits no Espaço 2D')
    ax00.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # Estatísticas
    stats_text = (f"Train: {n_train} ({n_train/total:.1%})\n"
                 f"Val: {n_val} ({n_val/total:.1%})\n"
                 f"Test: {n_test} ({n_test/total:.1%})")
    ax00.text(0.02, 0.98, stats_text, transform=ax00.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # =============================================
    # [0][1] Distribuição dos Clusters por Split
    # =============================================
    ax01 = fig.add_subplot(gs[0, 1])
    
    # Calcular contagens por cluster
    cluster_split_counts = {}
    for cluster_id in valid_clusters:
        mask = cluster_labels == cluster_id
        cluster_splits = split_assignment[mask]
        cluster_split_counts[cluster_id] = {
            'train': np.sum(cluster_splits == 0),
            'val': np.sum(cluster_splits == 1),
            'test': np.sum(cluster_splits == 2)
        }
    
    # Top 15 clusters
    cluster_sizes = [(cid, sum(cluster_split_counts[cid].values())) 
                     for cid in valid_clusters]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    top_n = min(15, len(cluster_sizes))
    top_clusters = [cid for cid, _ in cluster_sizes[:top_n]]
    
    # Stacked bar chart
    train_counts = [cluster_split_counts[cid]['train'] for cid in top_clusters]
    val_counts = [cluster_split_counts[cid]['val'] for cid in top_clusters]
    test_counts = [cluster_split_counts[cid]['test'] for cid in top_clusters]
    
    x = np.arange(len(top_clusters))
    width = 0.75
    
    ax01.bar(x, train_counts, width, label='Train', 
            color=SPLIT_COLORS[0], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax01.bar(x, val_counts, width, bottom=train_counts, label='Val',
            color=SPLIT_COLORS[1], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax01.bar(x, test_counts, width, 
            bottom=np.array(train_counts) + np.array(val_counts),
            label='Test', color=SPLIT_COLORS[2], alpha=0.85, 
            edgecolor='white', linewidth=0.5)
    
    ax01.set_xlabel(f'Top {top_n} Clusters (por tamanho)', fontsize=10)
    ax01.set_ylabel('Amostras', fontsize=10)
    ax01.set_title('Composição dos Clusters', fontsize=11, fontweight='bold')
    ax01.set_xticks(x)
    ax01.set_xticklabels([f'C{cid}' for cid in top_clusters], 
                         rotation=45, ha='right', fontsize=8)
    ax01.legend(loc='upper right', fontsize=9)
    ax01.grid(True, alpha=0.3, axis='y')
    
    # =============================================
    # [0][2] Distribuição Ativo/Inativo por Split
    # =============================================
    ax02 = fig.add_subplot(gs[0, 2])
    
    if labels is not None:
        # Contar ativos/inativos por split
        split_active = {}
        split_inactive = {}
        for split_id in [0, 1, 2]:
            mask = split_assignment == split_id
            split_labels = labels[mask]
            split_active[split_id] = np.sum(split_labels == 1)
            split_inactive[split_id] = np.sum(split_labels == 0)
        
        x_pos = np.arange(3)
        width = 0.35
        
        active_counts = [split_active[0], split_active[1], split_active[2]]
        inactive_counts = [split_inactive[0], split_inactive[1], split_inactive[2]]
        
        bars1 = ax02.bar(x_pos - width/2, active_counts, width, 
                        label='Ativo', color='#27ae60', alpha=0.85, 
                        edgecolor='white', linewidth=0.5)
        bars2 = ax02.bar(x_pos + width/2, inactive_counts, width,
                        label='Inativo', color='#e67e22', alpha=0.85,
                        edgecolor='white', linewidth=0.5)
        
        ax02.set_ylabel('Amostras', fontsize=10)
        ax02.set_title('Distribuição Ativo/Inativo', fontsize=11, fontweight='bold')
        ax02.set_xticks(x_pos)
        ax02.set_xticklabels(['Train', 'Val', 'Test'], fontsize=10)
        ax02.legend(loc='upper right', fontsize=9)
        ax02.grid(True, alpha=0.3, axis='y')
        
        # Adicionar percentuais nas barras
        for i, (act, inact) in enumerate(zip(active_counts, inactive_counts)):
            total_split = act + inact
            if total_split > 0:
                ax02.text(i - width/2, act + 5, f'{act/total_split:.1%}',
                         ha='center', va='bottom', fontsize=8)
                ax02.text(i + width/2, inact + 5, f'{inact/total_split:.1%}',
                         ha='center', va='bottom', fontsize=8)
    else:
        ax02.text(0.5, 0.5, 'Labels não disponíveis', 
                 ha='center', va='center', transform=ax02.transAxes,
                 fontsize=11, style='italic', color='gray')
        ax02.set_title('Distribuição Ativo/Inativo', fontsize=11, fontweight='bold')
    
    # =============================================
    # [1][0] Train Split - Distribuição Espacial
    # =============================================
    ax10 = fig.add_subplot(gs[1, 0])
    plot_split_highlight(ax10, embeddings_2d, split_assignment, 0, xlabel, ylabel)
    
    # =============================================
    # [1][1] Val Split - Distribuição Espacial
    # =============================================
    ax11 = fig.add_subplot(gs[1, 1])
    plot_split_highlight(ax11, embeddings_2d, split_assignment, 1, xlabel, ylabel)
    
    # =============================================
    # [1][2] Test Split - Distribuição Espacial
    # =============================================
    ax12 = fig.add_subplot(gs[1, 2])
    plot_split_highlight(ax12, embeddings_2d, split_assignment, 2, xlabel, ylabel)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualização de splits salva em: {output_path}")
    
    return fig


def plot_cluster_integrity(cluster_labels: np.ndarray,
                           split_assignment: np.ndarray,
                           output_path: Optional[str] = None) -> plt.Figure:
    """
    Verifica e visualiza a integridade dos clusters (cada cluster em um único split).
    
    Args:
        cluster_labels: Labels dos clusters
        split_assignment: Array indicando split (0=train, 1=val, 2=test)
        output_path: Caminho para salvar
        
    Returns:
        Figura matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análise de Integridade dos Clusters', fontsize=16, fontweight='bold')
    
    unique_clusters = np.unique(cluster_labels)
    valid_clusters = unique_clusters[unique_clusters != -1]
    
    # Verificar integridade de cada cluster
    cluster_integrity = {}
    split_violations = []
    
    for cluster_id in valid_clusters:
        mask = cluster_labels == cluster_id
        cluster_splits = split_assignment[mask]
        unique_splits = np.unique(cluster_splits)
        
        cluster_integrity[cluster_id] = {
            'size': len(cluster_splits),
            'splits': unique_splits,
            'is_intact': len(unique_splits) == 1,
            'primary_split': np.bincount(cluster_splits).argmax()
        }
        
        if len(unique_splits) > 1:
            split_violations.append(cluster_id)
    
    # =============================================
    # Plot 1: Matriz de integridade
    # =============================================
    ax1 = axes[0, 0]
    
    # Contar clusters por split
    clusters_per_split = {0: 0, 1: 0, 2: 0}
    for cluster_id, info in cluster_integrity.items():
        if info['is_intact']:
            clusters_per_split[info['primary_split']] += 1
    
    splits = ['Train', 'Val', 'Test']
    counts = [clusters_per_split[0], clusters_per_split[1], clusters_per_split[2]]
    colors = [SPLIT_COLORS[0], SPLIT_COLORS[1], SPLIT_COLORS[2]]
    
    bars = ax1.bar(splits, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    setup_axis(ax1, '', 'Número de Clusters', 'Clusters Íntegros por Split')
    
    # Adicionar valores nas barras
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Adicionar informação de violações
    n_violations = len(split_violations)
    violation_text = f"✅ Clusters íntegros: {len(valid_clusters) - n_violations}/{len(valid_clusters)}"
    if n_violations > 0:
        violation_text += f"\n⚠️ Clusters divididos: {n_violations}"
    
    ax1.text(0.5, 0.95, violation_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='yellow' if n_violations > 0 else 'lightgreen',
                     alpha=0.8))
    
    # =============================================
    # Plot 2: Distribuição de tamanhos por split
    # =============================================
    ax2 = axes[0, 1]
    
    train_cluster_sizes = [info['size'] for cid, info in cluster_integrity.items() 
                           if info['is_intact'] and info['primary_split'] == 0]
    val_cluster_sizes = [info['size'] for cid, info in cluster_integrity.items()
                         if info['is_intact'] and info['primary_split'] == 1]
    test_cluster_sizes = [info['size'] for cid, info in cluster_integrity.items()
                          if info['is_intact'] and info['primary_split'] == 2]
    
    data_to_plot = [train_cluster_sizes, val_cluster_sizes, test_cluster_sizes]
    labels_box = ['Train', 'Val', 'Test']
    colors_box = [SPLIT_COLORS[0], SPLIT_COLORS[1], SPLIT_COLORS[2]]
    
    bp = ax2.boxplot(data_to_plot, labels=labels_box, patch_artist=True,
                     notch=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    setup_axis(ax2, '', 'Tamanho do Cluster', 'Distribuição de Tamanhos dos Clusters por Split')
    
    # =============================================
    # Plot 3: Heatmap de composição (se houver violações)
    # =============================================
    ax3 = axes[1, 0]
    
    if n_violations > 0:
        # Mostrar clusters divididos
        ax3.set_title('⚠️ Clusters com Violação de Integridade', fontsize=12)
        
        violation_data = []
        violation_labels = []
        
        for cluster_id in split_violations[:15]:  # Top 15 violações
            mask = cluster_labels == cluster_id
            cluster_splits = split_assignment[mask]
            
            train_count = np.sum(cluster_splits == 0)
            val_count = np.sum(cluster_splits == 1)
            test_count = np.sum(cluster_splits == 2)
            total = train_count + val_count + test_count
            
            violation_data.append([train_count/total, val_count/total, test_count/total])
            violation_labels.append(f'C{cluster_id}\n(n={total})')
        
        violation_data = np.array(violation_data)
        
        im = ax3.imshow(violation_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        ax3.set_xticks([0, 1, 2])
        ax3.set_xticklabels(['Train', 'Val', 'Test'])
        ax3.set_yticks(range(len(violation_labels)))
        ax3.set_yticklabels(violation_labels, fontsize=8)
        setup_axis(ax3, 'Split', 'Cluster', '⚠️ Clusters com Violação de Integridade', add_grid=False)
        
        # Adicionar valores nas células
        for i in range(len(violation_data)):
            for j in range(3):
                text = ax3.text(j, i, f'{violation_data[i, j]:.1%}',
                               ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im, ax=ax3, label='Proporção')
    else:
        ax3.text(0.5, 0.5, '✅ PERFEITO!\n\nTodos os clusters\nestão íntegros\n\n'
                '(cada cluster 100%\nem um único split)',
                transform=ax3.transAxes, fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('Status de Integridade', fontsize=12)
    
    # =============================================
    # Plot 4: Resumo estatístico
    # =============================================
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calcular estatísticas
    total_samples = len(split_assignment)
    n_train = np.sum(split_assignment == 0)
    n_val = np.sum(split_assignment == 1)
    n_test = np.sum(split_assignment == 2)
    
    avg_cluster_size = np.mean([info['size'] for info in cluster_integrity.values()])
    
    stats_text = f"""
    RESUMO ESTATÍSTICO
    {'='*40}
    
    Total de Amostras: {total_samples:,}
    Total de Clusters: {len(valid_clusters)}
    
    DISTRIBUIÇÃO DOS SPLITS:
    • Train: {n_train:,} ({n_train/total_samples:.1%})
    • Val:   {n_val:,} ({n_val/total_samples:.1%})
    • Test:  {n_test:,} ({n_test/total_samples:.1%})
    
    CLUSTERS POR SPLIT:
    • Train: {clusters_per_split[0]} clusters
    • Val:   {clusters_per_split[1]} clusters
    • Test:  {clusters_per_split[2]} clusters
    
    TAMANHO MÉDIO DOS CLUSTERS:
    • Geral: {avg_cluster_size:.1f} amostras/cluster
    """
    
    if train_cluster_sizes:
        stats_text += f"    • Train: {np.mean(train_cluster_sizes):.1f} amostras/cluster\n"
    if val_cluster_sizes:
        stats_text += f"    • Val:   {np.mean(val_cluster_sizes):.1f} amostras/cluster\n"
    if test_cluster_sizes:
        stats_text += f"    • Test:  {np.mean(test_cluster_sizes):.1f} amostras/cluster\n"
    
    stats_text += f"\n    INTEGRIDADE:\n"
    if n_violations == 0:
        stats_text += f"    ✅ 100% íntegro (sem violações)\n"
    else:
        integrity_pct = (len(valid_clusters) - n_violations) / len(valid_clusters) * 100
        stats_text += f"    ⚠️  {integrity_pct:.1f}% íntegro ({n_violations} violações)\n"
    
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Análise de integridade salva em: {output_path}")
    
    return fig


def calculate_cluster_metrics(embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
    """
    Calcula métricas de qualidade dos clusters.
    
    Args:
        embeddings: Matriz de embeddings
        cluster_labels: Labels dos clusters
        
    Returns:
        Dicionário com métricas
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
    
    if n_clusters < 2:
        return {
            'n_clusters': n_clusters,
            'silhouette_score': None,
            'calinski_harabasz_score': None,
            'davies_bouldin_score': None,
            'warning': 'Menos de 2 clusters - métricas não calculáveis'
        }
    
    # Filtrar ruído se existir
    mask = cluster_labels != -1
    
    metrics = {
        'n_clusters': int(n_clusters),
        'n_samples': int(len(cluster_labels)),
        'n_noise': int(np.sum(cluster_labels == -1)),
        'silhouette_score': float(silhouette_score(embeddings[mask], cluster_labels[mask])),
        'calinski_harabasz_score': float(calinski_harabasz_score(embeddings[mask], cluster_labels[mask])),
        'davies_bouldin_score': float(davies_bouldin_score(embeddings[mask], cluster_labels[mask])),
    }
    
    # Estatísticas de tamanho dos clusters
    unique, counts = np.unique(cluster_labels[mask], return_counts=True)
    metrics['cluster_sizes'] = {int(k): int(v) for k, v in zip(unique, counts)}
    metrics['cluster_size_stats'] = {
        'min': int(counts.min()),
        'max': int(counts.max()),
        'mean': float(counts.mean()),
        'std': float(counts.std())
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Visualização PCA de clusters de estratificação (usando critério de similaridade de cossenos)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CRITÉRIO DE CLUSTERING (mesmo da estratificação):
  - Similaridade de cossenos >= 0.7 para agrupar vetores
  - Mínimo de 3 pontos para formar um cluster
  - Clustering hierárquico com linkage average

Exemplos:
  # PCA básico com clustering hierárquico (padrão - mesmo da estratificação)
  python scripts/visualize_cluster_pca.py --embeddings matrix.npy --output results/

  # Visualizar distribuição dos splits (train/val/test)
  python scripts/visualize_cluster_pca.py \\
      --embeddings matrix.npy \\
      --train-idx train_indices.npy \\
      --val-idx val_indices.npy \\
      --test-idx test_indices.npy \\
      --output results/

  # Usar split_assignment pré-calculado
  python scripts/visualize_cluster_pca.py \\
      --embeddings matrix.npy \\
      --split-assignment splits.npy \\
      --output results/

  # Alterar threshold de similaridade
  python scripts/visualize_cluster_pca.py --embeddings matrix.npy --similarity-threshold 0.8 --output results/

  # PCA com embeddings separados (multi-view)
  python scripts/visualize_cluster_pca.py --protein-emb protein.npy --ligand-emb ligand.npy --output results/

  # t-SNE em vez de PCA
  python scripts/visualize_cluster_pca.py --embeddings matrix.npy --method tsne --output results/
  
  # Visualização completa com labels, splits e métricas
  python scripts/visualize_cluster_pca.py \\
      --embeddings matrix.npy \\
      --labels labels.npy \\
      --train-idx train.npy --val-idx val.npy --test-idx test.npy \\
      --save-metrics --output results/
        """
    )
    
    # Argumentos de entrada
    parser.add_argument('--embeddings', '-e', type=str, help='Caminho para embeddings concatenados (.npy)')
    parser.add_argument('--protein-emb', '-p', type=str, help='Caminho para embeddings de proteínas (.npy)')
    parser.add_argument('--ligand-emb', '-l', type=str, help='Caminho para embeddings de ligantes (.npy)')
    parser.add_argument('--labels', type=str, help='Caminho para labels originais (.npy)')
    parser.add_argument('--cluster-labels', type=str, help='Caminho para labels de clusters existentes (.npy)')
    parser.add_argument('--split-assignment', type=str, help='Caminho para array de splits (0=train, 1=val, 2=test) (.npy)')
    parser.add_argument('--train-idx', type=str, help='Caminho para índices do split train (.npy)')
    parser.add_argument('--val-idx', type=str, help='Caminho para índices do split val (.npy)')
    parser.add_argument('--test-idx', type=str, help='Caminho para índices do split test (.npy)')
    
    # Argumentos de visualização
    parser.add_argument('--method', '-m', type=str, default='pca', choices=['pca', 'tsne', 'umap'],
                       help='Método de redução de dimensionalidade (default: pca)')
    parser.add_argument('--n-components', type=int, default=2, help='Número de componentes (default: 2)')
    parser.add_argument('--perplexity', type=int, default=30, help='Perplexidade para t-SNE (default: 30)')
    parser.add_argument('--n-neighbors', type=int, default=15, help='Vizinhos para UMAP (default: 15)')
    
    # Argumentos de clustering (mesmo critério da estratificação)
    parser.add_argument('--clustering', '-c', type=str, default='hierarchical', 
                       choices=['hierarchical', 'dbscan', 'kmeans'],
                       help='Algoritmo de clustering (default: hierarchical - mesmo da estratificação)')
    parser.add_argument('--similarity-threshold', '-s', type=float, default=0.7, 
                       help='Limiar de similaridade de cossenos (default: 0.7 - mesmo da estratificação)')
    parser.add_argument('--min-cluster-size', type=int, default=3, 
                       help='Tamanho mínimo do cluster (default: 3 - mesmo da estratificação)')
    parser.add_argument('--n-clusters', type=int, default=20, help='Número de clusters para K-Means (default: 20)')
    parser.add_argument('--auto-threshold', action='store_true', 
                       help='Usar automaticamente o threshold recomendado baseado na distribuição de similaridades')
    
    # Argumentos de saída
    parser.add_argument('--output', '-o', type=str, default='.', help='Diretório de saída (default: .)')
    parser.add_argument('--prefix', type=str, default='stratification_clusters', help='Prefixo para arquivos de saída')
    parser.add_argument('--no-show', action='store_true', help='Não exibir gráficos (apenas salvar)')
    parser.add_argument('--save-metrics', action='store_true', help='Salvar métricas em JSON')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.embeddings and not (args.protein_emb and args.ligand_emb):
        parser.error("Forneça --embeddings ou ambos --protein-emb e --ligand-emb")
    
    # Criar diretório de saída
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Visualização PCA de Clusters")
    print("=" * 60)
    
    # Carregar embeddings
    print("\n📂 Carregando embeddings...")
    combined_emb, protein_emb, ligand_emb = load_embeddings(
        args.embeddings, args.protein_emb, args.ligand_emb
    )
    print(f"   Shape: {combined_emb.shape}")
    
    # Analisar distribuição de similaridades SOMENTE se usar algoritmo baseado em threshold
    # K-means NÃO usa threshold de similaridade, então não avisar
    if args.clustering in ['hierarchical', 'dbscan']:
        print("\n📊 Analisando distribuição de similaridades...")
        sim_stats = analyze_similarity_distribution(combined_emb)
        print(f"   Similaridade min: {sim_stats['min']:.4f}")
        print(f"   Similaridade max: {sim_stats['max']:.4f}")
        print(f"   Similaridade média: {sim_stats['mean']:.4f}")
        print(f"   Homogeneidade: {sim_stats['homogeneity']}")
        
        # Usar threshold automático se solicitado ou avisar sobre problema
        if args.auto_threshold:
            args.similarity_threshold = sim_stats['recommended_threshold']
            print(f"   ✅ Usando threshold automático: {args.similarity_threshold:.4f}")
        elif sim_stats['homogeneity'] != 'normal':
            print(f"   ⚠️  {sim_stats['recommendation']}")
            if args.similarity_threshold < sim_stats['recommended_threshold']:
                print(f"   💡 Seu threshold ({args.similarity_threshold}) pode resultar em poucos clusters.")
                print(f"   💡 Use --auto-threshold para ajuste automático")
    
    # Carregar labels originais (se disponíveis)
    labels = None
    binary_labels = None
    if args.labels:
        numeric_labels, binary_labels = load_labels(args.labels)
        labels = binary_labels  # Usar labels binários para visualização
        n_active = np.sum(binary_labels == 1)
        n_inactive = np.sum(binary_labels == 0)
        print(f"\n📋 Labels binários: {n_active} ativos, {n_inactive} inativos (threshold=1000nM)")
    
    # Realizar ou carregar clustering
    if args.cluster_labels:
        print(f"\n📥 Carregando labels de clusters de: {args.cluster_labels}")
        cluster_labels = np.load(args.cluster_labels)
    else:
        print(f"\n🔄 Realizando clustering ({args.clustering})...")
        if args.clustering == 'kmeans':
            print(f"   Algoritmo: K-Means++ (mesmo do pipeline)")
            print(f"   Número de clusters: {args.n_clusters}")
        else:
            print(f"   Critério: similaridade de cossenos >= {args.similarity_threshold:.4f}")
        print(f"   Mínimo de pontos por cluster: {args.min_cluster_size}")
        cluster_labels = perform_clustering(
            combined_emb, 
            algorithm=args.clustering,
            similarity_threshold=args.similarity_threshold,
            min_cluster_size=args.min_cluster_size,
            n_clusters=args.n_clusters
        )
    
    n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
    n_noise = np.sum(cluster_labels == -1)
    print(f"   Clusters válidos: {n_clusters}")
    if n_noise > 0:
        print(f"   Pontos não agrupados (ruído): {n_noise}")
    
    # Calcular métricas
    print("\n📊 Calculando métricas de clustering...")
    metrics = calculate_cluster_metrics(combined_emb, cluster_labels)
    print(f"   Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f}" if metrics.get('silhouette_score') else "   Silhouette Score: N/A")
    print(f"   Calinski-Harabasz Score: {metrics.get('calinski_harabasz_score', 'N/A'):.2f}" if metrics.get('calinski_harabasz_score') else "   Calinski-Harabasz Score: N/A")
    print(f"   Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 'N/A'):.4f}" if metrics.get('davies_bouldin_score') else "   Davies-Bouldin Score: N/A")
    
    if args.save_metrics:
        metrics_path = output_dir / f"{args.prefix}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Métricas salvas em: {metrics_path}")
    
    # Redução de dimensionalidade
    print(f"\n🔄 Calculando {args.method.upper()}...")
    pca = None
    if args.method == 'pca':
        # PCA com todos os componentes para análise de variância
        emb_2d, pca = compute_pca(combined_emb, n_components=min(10, combined_emb.shape[1]))
        emb_2d = emb_2d[:, :2]  # Manter apenas 2D para visualização
        
        # Gerar gráfico de variância
        variance_path = output_dir / f"{args.prefix}_variance.png"
        plot_pca_variance(pca, output_path=str(variance_path))
        
    elif args.method == 'tsne':
        emb_2d = compute_tsne(combined_emb, n_components=2, perplexity=args.perplexity)
    else:  # umap
        emb_2d = compute_umap(combined_emb, n_components=2, n_neighbors=args.n_neighbors)
    
    # Gerar visualização principal
    print("\n📈 Gerando visualizações...")
    main_plot_path = output_dir / f"{args.prefix}_{args.method}.png"
    plot_cluster_pca(
        emb_2d, cluster_labels, pca, labels,
        title=f"Distribuição de Clusters ({args.method.upper()})",
        output_path=str(main_plot_path),
        method=args.method
    )
    
    # Comparação multi-view (se embeddings separados disponíveis)
    if protein_emb is not None and ligand_emb is not None:
        print("\n📈 Gerando comparação multi-view...")
        multiview_path = output_dir / f"{args.prefix}_multiview_{args.method}.png"
        plot_multi_view_comparison(
            protein_emb, ligand_emb, combined_emb, cluster_labels,
            output_path=str(multiview_path)
        )
    
    # Salvar cluster labels
    cluster_labels_path = output_dir / f"{args.prefix}_labels.npy"
    np.save(cluster_labels_path, cluster_labels)
    print(f"\n✅ Labels de clusters salvos em: {cluster_labels_path}")
    
    # =============================================
    # Visualização de Splits (se fornecidos)
    # =============================================
    split_assignment = None
    
    # Opção 1: Carregar split_assignment diretamente
    if args.split_assignment:
        print(f"\n📥 Carregando split assignment de: {args.split_assignment}")
        split_assignment = np.load(args.split_assignment)
    
    # Opção 2: Construir split_assignment a partir de train/val/test indices
    elif args.train_idx and args.val_idx and args.test_idx:
        print("\n📥 Carregando índices de splits...")
        train_idx = np.load(args.train_idx)
        val_idx = np.load(args.val_idx)
        test_idx = np.load(args.test_idx)
        
        # Criar array de split_assignment
        split_assignment = np.full(len(combined_emb), -1, dtype=int)
        split_assignment[train_idx] = 0  # Train
        split_assignment[val_idx] = 1    # Val
        split_assignment[test_idx] = 2   # Test
        
        # Verificar se todos os índices foram atribuídos
        unassigned = np.sum(split_assignment == -1)
        if unassigned > 0:
            print(f"   ⚠️ {unassigned} amostras não atribuídas a nenhum split")
        
        # Salvar split_assignment construído
        split_path = output_dir / f"{args.prefix}_split_assignment.npy"
        np.save(split_path, split_assignment)
        print(f"   ✅ Split assignment salvo em: {split_path}")
    
    # Gerar visualizações de splits se disponíveis
    if split_assignment is not None:
        print("\n📈 Gerando visualizações de splits...")
        
        # Visualização principal de splits
        split_plot_path = output_dir / f"{args.prefix}_splits_{args.method}.png"
        plot_split_distribution(
            emb_2d, cluster_labels, split_assignment, labels, pca, args.method,
            output_path=str(split_plot_path)
        )
        
        # Análise de integridade dos clusters
        integrity_plot_path = output_dir / f"{args.prefix}_cluster_integrity.png"
        plot_cluster_integrity(
            cluster_labels, split_assignment,
            output_path=str(integrity_plot_path)
        )
        
        # Salvar estatísticas de splits em JSON
        if args.save_metrics:
            # Adicionar informações de splits às métricas
            n_train = np.sum(split_assignment == 0)
            n_val = np.sum(split_assignment == 1)
            n_test = np.sum(split_assignment == 2)
            total = len(split_assignment)
            
            metrics['split_statistics'] = {
                'train': {'count': int(n_train), 'ratio': float(n_train / total)},
                'val': {'count': int(n_val), 'ratio': float(n_val / total)},
                'test': {'count': int(n_test), 'ratio': float(n_test / total)}
            }
            
            # Verificar integridade dos clusters
            unique_clusters = np.unique(cluster_labels)
            valid_clusters = unique_clusters[unique_clusters != -1]
            
            violations = []
            for cluster_id in valid_clusters:
                mask = cluster_labels == cluster_id
                cluster_splits = split_assignment[mask]
                unique_splits = np.unique(cluster_splits)
                
                if len(unique_splits) > 1:
                    violations.append({
                        'cluster_id': int(cluster_id),
                        'size': int(np.sum(mask)),
                        'splits': unique_splits.tolist()
                    })
            
            metrics['cluster_integrity'] = {
                'total_clusters': int(len(valid_clusters)),
                'intact_clusters': int(len(valid_clusters) - len(violations)),
                'violated_clusters': int(len(violations)),
                'integrity_ratio': float((len(valid_clusters) - len(violations)) / len(valid_clusters)),
                'violations': violations
            }
            
            # Salvar métricas atualizadas
            metrics_path = output_dir / f"{args.prefix}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\n✅ Métricas com splits salvas em: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("✅ Visualização concluída!")
    print("=" * 60)
    
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
