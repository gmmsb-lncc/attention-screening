#!/usr/bin/env python3
"""
Funções de plotagem para visualização de clusters.

Este módulo contém todas as funções de geração de gráficos
para análise de clusters e splits.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

from viz_utils import (
    SPLIT_COLORS, SPLIT_NAMES,
    calculate_split_sizes, setup_axis
)


# ============================================================================
# REDUÇÃO DE DIMENSIONALIDADE
# ============================================================================

def compute_pca(embeddings: np.ndarray, 
                n_components: int = 2,
                normalize: bool = True) -> Tuple[np.ndarray, PCA]:
    """Aplica PCA nos embeddings."""
    from sklearn.preprocessing import StandardScaler, normalize as sklearn_normalize
    
    if normalize:
        embeddings = sklearn_normalize(embeddings, axis=1, norm='l2')
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    
    return embeddings_pca, pca


def compute_tsne(embeddings: np.ndarray,
                 n_components: int = 2,
                 perplexity: float = 30.0,
                 random_state: int = 42) -> np.ndarray:
    """Aplica t-SNE nos embeddings."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=random_state, n_jobs=-1)
    return tsne.fit_transform(embeddings)


def compute_umap(embeddings: np.ndarray,
                 n_components: int = 2,
                 n_neighbors: int = 15,
                 random_state: int = 42) -> np.ndarray:
    """Aplica UMAP nos embeddings."""
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           random_state=random_state)
        return reducer.fit_transform(embeddings)
    except ImportError:
        raise ImportError("UMAP não instalado. Instale com: pip install umap-learn")


# ============================================================================
# PLOTAGENS PRINCIPAIS
# ============================================================================

def plot_split_highlight(ax, embeddings_2d: np.ndarray, split_assignment: np.ndarray, 
                        split_id: int, xlabel: str, ylabel: str):
    """Plota scatter destacando um split específico."""
    mask = split_assignment == split_id
    other_mask = ~mask
    n_split = np.sum(mask)
    total = len(split_assignment)
    
    # Background
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


def plot_cluster_pca(embeddings_2d: np.ndarray,
                     cluster_labels: np.ndarray,
                     pca: Optional[PCA] = None,
                     labels: Optional[np.ndarray] = None,
                     title: str = "Cluster PCA Visualization",
                     output_path: Optional[str] = None,
                     method: str = 'pca') -> plt.Figure:
    """
    Gera visualização 2D dos clusters em layout 2x2.
    
    Returns:
        Figura matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Título com informação de variância
    display_title = title
    if pca is not None and method == 'pca':
        var_total = np.sum(pca.explained_variance_ratio_[:2])
        if var_total < 0.3:
            orig_dim = pca.n_features_in_ if hasattr(pca, 'n_features_in_') else "alta"
            display_title += f'\n(PC1+PC2 explicam {var_total:.1%} - Normal para {orig_dim}D homogêneos)'
    
    fig.suptitle(display_title, fontsize=16, fontweight='bold', y=0.98)
    
    # Configurar cores dos clusters
    unique_clusters = np.unique(cluster_labels)
    valid_clusters = unique_clusters[unique_clusters != -1]
    n_clusters = len(valid_clusters)
    
    if n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    color_map = {c: colors[i] for i, c in enumerate(valid_clusters)}
    color_map[-1] = [0.7, 0.7, 0.7, 1.0]  # Cinza para ruído
    
    # Configurar labels dos eixos
    if pca is not None and method == 'pca':
        var1, var2 = pca.explained_variance_ratio_[:2]
        xlabel = f'PC1 ({var1:.1%} variância)'
        ylabel = f'PC2 ({var2:.1%} variância)'
    else:
        xlabel = f'{method.upper()} Componente 1'
        ylabel = f'{method.upper()} Componente 2'
    
    # Plot 1: Clusters coloridos
    ax1 = axes[0, 0]
    noise_mask = cluster_labels == -1
    if np.any(noise_mask):
        ax1.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1],
                   c=[color_map[-1]], label='Ruído', alpha=0.3, s=15, marker='x')
    
    for cluster_id in valid_clusters:
        mask = cluster_labels == cluster_id
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[color_map[cluster_id]], alpha=0.6, s=25)
    
    setup_axis(ax1, xlabel, ylabel, f'Distribuição dos Clusters (n={n_clusters})')
    
    # Plot 2: Labels ativo/inativo
    ax2 = axes[0, 1]
    if labels is not None:
        unique_labels = np.unique(labels)
        n_unique_labels = len(unique_labels)
        
        if n_unique_labels <= 2:
            colors_labels = ['#2ecc71', '#e74c3c']
            label_names = {0: 'Inativo', 1: 'Ativo'}
            for i, label in enumerate(sorted(unique_labels)):
                mask = labels == label
                ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=colors_labels[int(label)], 
                           label=label_names.get(label, f'Label {label}'),
                           alpha=0.6, s=25)
            ax2.legend(loc='upper right', fontsize=10)
            ax2.set_title('Distribuição Ativo/Inativo')
        else:
            scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                 c=labels, cmap='coolwarm', alpha=0.6, s=25)
            plt.colorbar(scatter, ax=ax2, label='Valor', shrink=0.8)
            ax2.set_title('Distribuição por Valor')
    else:
        ax2.hexbin(embeddings_2d[:, 0], embeddings_2d[:, 1], gridsize=30, 
                   cmap='YlOrRd', mincnt=1)
        plt.colorbar(ax2.collections[0], ax=ax2, label='Densidade', shrink=0.8)
        ax2.set_title('Mapa de Densidade')
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histograma de tamanhos
    ax3 = axes[1, 0]
    cluster_sizes = [np.sum(cluster_labels == c) for c in valid_clusters]
    
    if n_clusters > 30:
        ax3.hist(cluster_sizes, bins=min(30, n_clusters//2), 
                 color='steelblue', edgecolor='white', alpha=0.8)
        mean_size = np.mean(cluster_sizes)
        median_size = np.median(cluster_sizes)
        ax3.axvline(mean_size, color='red', linestyle='--', linewidth=2, 
                    label=f'Média: {mean_size:.1f}')
        ax3.axvline(median_size, color='orange', linestyle='--', linewidth=2, 
                    label=f'Mediana: {median_size:.1f}')
        ax3.legend(loc='upper right')
    else:
        sorted_idx = np.argsort(cluster_sizes)[::-1]
        sorted_sizes = [cluster_sizes[i] for i in sorted_idx]
        sorted_colors = [color_map[valid_clusters[i]] for i in sorted_idx]
        
        ax3.bar(range(len(sorted_sizes)), sorted_sizes, 
                color=sorted_colors, edgecolor='white', alpha=0.8)
    
    setup_axis(ax3, 'Cluster ID', 'Número de Amostras', 
               'Distribuição de Tamanhos dos Clusters')
    
    # Plot 4: Estatísticas
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    Estatísticas dos Clusters:
    
    • Total de clusters: {n_clusters}
    • Total de amostras: {len(cluster_labels)}
    • Ruído: {np.sum(noise_mask)} ({np.sum(noise_mask)/len(cluster_labels):.1%})
    
    Tamanho dos clusters:
    • Média: {np.mean(cluster_sizes):.1f}
    • Mediana: {np.median(cluster_sizes):.1f}
    • Mín: {np.min(cluster_sizes)}
    • Máx: {np.max(cluster_sizes)}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=12, 
             verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualização PCA salva em: {output_path}")
    
    return fig


def plot_split_distribution(embeddings_2d: np.ndarray,
                           split_assignment: np.ndarray,
                           labels: Optional[np.ndarray] = None,
                           pca: Optional[PCA] = None,
                           title: str = "Distribuição dos Splits",
                           output_path: Optional[str] = None,
                           method: str = 'pca') -> plt.Figure:
    """Visualização consolidada da distribuição dos splits em grid 2x3."""
    
    n_train, n_val, n_test, total = calculate_split_sizes(split_assignment)
    
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Configurar labels dos eixos
    if pca is not None and method == 'pca':
        var1, var2 = pca.explained_variance_ratio_[:2]
        xlabel = f'PC1 ({var1:.1%} var)'
        ylabel = f'PC2 ({var2:.1%} var)'
    else:
        xlabel = f'{method.upper()} 1'
        ylabel = f'{method.upper()} 2'
    
    # [0][0] Visão geral
    ax00 = fig.add_subplot(gs[0, 0])
    for split_id in [0, 1, 2]:
        mask = split_assignment == split_id
        ax00.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=SPLIT_COLORS[split_id], alpha=0.6, s=20, 
                    label=SPLIT_NAMES[split_id])
    setup_axis(ax00, xlabel, ylabel, 'Visão Geral: Splits no Espaço 2D')
    ax00.legend(loc='best', fontsize=10)
    
    # [0][1] Barras de contagem
    ax01 = fig.add_subplot(gs[0, 1])
    splits = ['Train', 'Val', 'Test']
    counts = [n_train, n_val, n_test]
    bars = ax01.bar(splits, counts, color=[SPLIT_COLORS[i] for i in range(3)],
                    edgecolor='white', alpha=0.8, width=0.6)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax01.text(bar.get_x() + bar.get_width()/2, height + 50,
                 f'{count}\n({count/total:.1%})',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    setup_axis(ax01, '', 'Número de Amostras', 'Contagem por Split', add_grid=False)
    ax01.set_ylim(0, max(counts) * 1.15)
    
    # [0][2] Distribuição ativo/inativo
    ax02 = fig.add_subplot(gs[0, 2])
    if labels is not None:
        width = 0.35
        x = np.arange(3)
        
        for i, split_id in enumerate([0, 1, 2]):
            mask = split_assignment == split_id
            split_labels = labels[mask]
            n_active = np.sum(split_labels == 1)
            n_inactive = np.sum(split_labels == 0)
            total_split = len(split_labels)
            
            ax02.bar(i - width/2, n_active, width, label='Ativo' if i == 0 else '',
                    color='#2ecc71', alpha=0.8)
            ax02.bar(i + width/2, n_inactive, width, label='Inativo' if i == 0 else '',
                    color='#e74c3c', alpha=0.8)
            
            ax02.text(i - width/2, n_active + 5, f'{n_active/total_split:.1%}',
                     ha='center', va='bottom', fontsize=8)
            ax02.text(i + width/2, n_inactive + 5, f'{n_inactive/total_split:.1%}',
                     ha='center', va='bottom', fontsize=8)
        
        ax02.set_xticks(x)
        ax02.set_xticklabels(splits)
        ax02.legend(loc='upper right', fontsize=9)
        setup_axis(ax02, '', 'Contagem', 'Distribuição Ativo/Inativo por Split', add_grid=False)
    else:
        ax02.text(0.5, 0.5, 'Labels não disponíveis', 
                 ha='center', va='center', transform=ax02.transAxes,
                 fontsize=11, style='italic', color='gray')
        ax02.set_title('Distribuição Ativo/Inativo', fontsize=11, fontweight='bold')
    
    # [1][0-2] Scatter plots individuais
    for i, split_id in enumerate([0, 1, 2]):
        ax = fig.add_subplot(gs[1, i])
        plot_split_highlight(ax, embeddings_2d, split_assignment, split_id, xlabel, ylabel)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualização de splits salva em: {output_path}")
    
    return fig


def plot_cluster_integrity(cluster_labels: np.ndarray,
                          split_assignment: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          title: str = "Integridade dos Clusters",
                          output_path: Optional[str] = None) -> plt.Figure:
    """Visualiza integridade dos clusters em relação aos splits."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    unique_clusters = np.unique(cluster_labels)
    valid_clusters = unique_clusters[unique_clusters != -1]
    n_clusters = len(valid_clusters)
    
    # [0][0] Composição dos clusters por split
    ax1 = axes[0, 0]
    cluster_split_matrix = np.zeros((n_clusters, 3))
    
    for i, cluster_id in enumerate(valid_clusters):
        mask = cluster_labels == cluster_id
        for split_id in range(3):
            cluster_split_matrix[i, split_id] = np.sum((cluster_labels == cluster_id) & 
                                                       (split_assignment == split_id))
    
    if n_clusters <= 30:
        x = np.arange(n_clusters)
        width = 0.25
        
        for split_id in range(3):
            ax1.bar(x + split_id * width, cluster_split_matrix[:, split_id],
                   width, label=SPLIT_NAMES[split_id], 
                   color=SPLIT_COLORS[split_id], alpha=0.8)
        
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([str(c) for c in valid_clusters[:n_clusters]], 
                           rotation=45 if n_clusters > 15 else 0, fontsize=8)
    else:
        im = ax1.imshow(cluster_split_matrix.T, aspect='auto', cmap='YlOrRd')
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['Train', 'Val', 'Test'])
        plt.colorbar(im, ax=ax1, label='Contagem')
    
    setup_axis(ax1, 'Cluster ID', 'Contagem', 'Composição dos Clusters por Split')
    ax1.legend(loc='upper right')
    
    # [0][1] Pureza dos clusters (labels)
    ax2 = axes[0, 1]
    if labels is not None:
        cluster_purity = []
        for cluster_id in valid_clusters:
            mask = cluster_labels == cluster_id
            cluster_labs = labels[mask]
            if len(cluster_labs) > 0:
                purity = max(np.sum(cluster_labs == 0), np.sum(cluster_labs == 1)) / len(cluster_labs)
                cluster_purity.append(purity)
            else:
                cluster_purity.append(0)
        
        ax2.hist(cluster_purity, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        mean_purity = np.mean(cluster_purity)
        ax2.axvline(mean_purity, color='red', linestyle='--', linewidth=2,
                   label=f'Média: {mean_purity:.2f}')
        ax2.legend()
        setup_axis(ax2, 'Pureza', 'Frequência', 'Distribuição de Pureza dos Clusters')
    else:
        ax2.text(0.5, 0.5, 'Labels não disponíveis', ha='center', va='center',
                transform=ax2.transAxes, fontsize=11, style='italic', color='gray')
    
    # [1][0] Splits por cluster (porcentagem)
    ax3 = axes[1, 0]
    cluster_split_pct = cluster_split_matrix / cluster_split_matrix.sum(axis=1, keepdims=True) * 100
    
    if n_clusters <= 30:
        for split_id in range(3):
            ax3.bar(x + split_id * width, cluster_split_pct[:, split_id],
                   width, label=SPLIT_NAMES[split_id],
                   color=SPLIT_COLORS[split_id], alpha=0.8)
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([str(c) for c in valid_clusters[:n_clusters]],
                           rotation=45 if n_clusters > 15 else 0, fontsize=8)
    else:
        im = ax3.imshow(cluster_split_pct.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Train', 'Val', 'Test'])
        plt.colorbar(im, ax=ax3, label='%')
    
    setup_axis(ax3, 'Cluster ID', '% do Cluster', 'Composição Percentual dos Clusters')
    ax3.legend(loc='upper right')
    
    # [1][1] Estatísticas
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Contar clusters dominados por cada split
    dominant_splits = np.argmax(cluster_split_pct, axis=1)
    n_train_dom = np.sum(dominant_splits == 0)
    n_val_dom = np.sum(dominant_splits == 1)
    n_test_dom = np.sum(dominant_splits == 2)
    
    stats_text = f"""
    Estatísticas de Integridade:
    
    Total de clusters: {n_clusters}
    
    Clusters dominados por cada split:
    • Train: {n_train_dom} ({n_train_dom/n_clusters:.1%})
    • Val: {n_val_dom} ({n_val_dom/n_clusters:.1%})
    • Test: {n_test_dom} ({n_test_dom/n_clusters:.1%})
    """
    
    if labels is not None:
        stats_text += f"""
    
    Pureza dos clusters (labels):
    • Média: {np.mean(cluster_purity):.2%}
    • Mediana: {np.median(cluster_purity):.2%}
    • Mín: {np.min(cluster_purity):.2%}
    • Máx: {np.max(cluster_purity):.2%}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11,
             verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualização de integridade salva em: {output_path}")
    
    return fig


def plot_pca_variance(pca: PCA, output_path: Optional[str] = None) -> plt.Figure:
    """Plota variância explicada do PCA."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual
    ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Variância Explicada')
    ax1.set_title('Variância por Componente')
    ax1.grid(True, alpha=0.3)
    
    # Cumulativa
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, 'o-', color='steelblue', linewidth=2)
    ax2.axhline(0.9, color='red', linestyle='--', label='90%')
    ax2.axhline(0.95, color='orange', linestyle='--', label='95%')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Variância Explicada Acumulada')
    ax2.set_title('Variância Cumulativa')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico de variância salvo em: {output_path}")
    
    return fig
