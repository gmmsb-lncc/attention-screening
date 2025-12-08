#!/usr/bin/env python3
"""
Script para visualização PCA de clusters da estratificação.

Este script modularizado usa:
- viz_utils.py: funções auxiliares e carregamento de dados
- viz_plots.py: funções de plotagem

Usage:
    python scripts/visualize_cluster_pca.py --embeddings path/to/embeddings.npy --output results/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos auxiliares
from scripts.viz_utils import (
    load_embeddings, load_labels, load_split_indices, create_split_assignment,
    analyze_similarity_distribution, perform_clustering, calculate_cluster_metrics
)

from scripts.viz_plots import (
    compute_pca, compute_tsne, compute_umap,
    plot_cluster_pca, plot_split_distribution, plot_cluster_integrity, plot_pca_variance
)


def main():
    parser = argparse.ArgumentParser(
        description='Visualização PCA de clusters de estratificação',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # PCA básico com clustering
  python scripts/visualize_cluster_pca.py --embeddings matrix.npy --output results/

  # Visualizar distribuição dos splits
  python scripts/visualize_cluster_pca.py \\
      --embeddings matrix.npy \\
      --train-idx train_indices.npy --val-idx val_indices.npy --test-idx test_indices.npy \\
      --output results/

  # Com labels e métricas
  python scripts/visualize_cluster_pca.py \\
      --embeddings matrix.npy --labels labels.npy \\
      --train-idx train.npy --val-idx val.npy --test-idx test.npy \\
      --save-metrics --output results/
        """
    )
    
    # Argumentos de entrada
    parser.add_argument('--embeddings', '-e', type=str, 
                       help='Caminho para embeddings concatenados (.npy)')
    parser.add_argument('--protein-emb', '-p', type=str, 
                       help='Caminho para embeddings de proteínas (.npy)')
    parser.add_argument('--ligand-emb', '-l', type=str, 
                       help='Caminho para embeddings de ligantes (.npy)')
    parser.add_argument('--labels', type=str, 
                       help='Caminho para labels originais (.npy)')
    parser.add_argument('--cluster-labels', type=str, 
                       help='Caminho para labels de clusters existentes (.npy)')
    parser.add_argument('--split-assignment', type=str, 
                       help='Array de splits (0=train, 1=val, 2=test) (.npy)')
    parser.add_argument('--train-idx', type=str, 
                       help='Índices do split train (.npy)')
    parser.add_argument('--val-idx', type=str, 
                       help='Índices do split val (.npy)')
    parser.add_argument('--test-idx', type=str, 
                       help='Índices do split test (.npy)')
    
    # Visualização
    parser.add_argument('--method', '-m', type=str, default='pca', 
                       choices=['pca', 'tsne', 'umap'],
                       help='Método de redução de dimensionalidade (default: pca)')
    parser.add_argument('--n-components', type=int, default=2, 
                       help='Número de componentes (default: 2)')
    parser.add_argument('--perplexity', type=int, default=30, 
                       help='Perplexidade para t-SNE (default: 30)')
    parser.add_argument('--n-neighbors', type=int, default=15, 
                       help='Vizinhos para UMAP (default: 15)')
    
    # Clustering
    parser.add_argument('--clustering', '-c', type=str, default='hierarchical', 
                       choices=['hierarchical', 'dbscan', 'kmeans'],
                       help='Algoritmo de clustering (default: hierarchical)')
    parser.add_argument('--similarity-threshold', '-s', type=float, default=0.7, 
                       help='Threshold de similaridade (default: 0.7)')
    parser.add_argument('--min-cluster-size', type=int, default=3, 
                       help='Tamanho mínimo do cluster (default: 3)')
    parser.add_argument('--n-clusters', type=int, default=20, 
                       help='Número de clusters para K-Means (default: 20)')
    
    # Saída
    parser.add_argument('--output', '-o', type=str, default='.', 
                       help='Diretório de saída (default: .)')
    parser.add_argument('--prefix', type=str, default='stratification_clusters', 
                       help='Prefixo para arquivos de saída')
    parser.add_argument('--no-show', action='store_true', 
                       help='Não exibir gráficos (apenas salvar)')
    parser.add_argument('--save-metrics', action='store_true', 
                       help='Salvar métricas em JSON')
    
    args = parser.parse_args()
    
    # Validação
    if not args.embeddings and not (args.protein_emb and args.ligand_emb):
        parser.error("Forneça --embeddings ou ambos --protein-emb e --ligand-emb")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Visualização PCA de Clusters")
    print("=" * 60)
    
    # Carregar dados
    print("\n📂 Carregando embeddings...")
    combined_emb, protein_emb, ligand_emb = load_embeddings(
        args.embeddings, args.protein_emb, args.ligand_emb
    )
    print(f"✓ Shape: {combined_emb.shape}")
    
    # Labels
    labels_orig, binary_labels = None, None
    if args.labels:
        print("\n📂 Carregando labels...")
        labels_orig, binary_labels = load_labels(args.labels)
    
    # Splits
    split_assignment = None
    if args.split_assignment:
        print("\n📂 Carregando split assignment...")
        split_assignment = np.load(args.split_assignment)
    elif args.train_idx and args.val_idx and args.test_idx:
        print("\n📂 Carregando splits (train/val/test)...")
        train_idx, val_idx, test_idx = load_split_indices(
            args.train_idx, args.val_idx, args.test_idx
        )
        split_assignment = create_split_assignment(
            len(combined_emb), train_idx, val_idx, test_idx
        )
        print(f"✓ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Análise de similaridade
    print("\n🔍 Analisando distribuição de similaridades...")
    sim_stats = analyze_similarity_distribution(combined_emb)
    print(f"✓ Similaridade: min={sim_stats['min']:.4f}, mean={sim_stats['mean']:.4f}, max={sim_stats['max']:.4f}")
    
    # Clustering
    cluster_labels = None
    if args.cluster_labels:
        print("\n📂 Carregando cluster labels...")
        cluster_labels = np.load(args.cluster_labels)
    else:
        print(f"\n🔬 Executando clustering ({args.clustering})...")
        cluster_labels, n_clusters = perform_clustering(
            combined_emb,
            method=args.clustering,
            similarity_threshold=args.similarity_threshold,
            n_clusters=args.n_clusters,
            min_samples=args.min_cluster_size
        )
        print(f"✓ {n_clusters} clusters encontrados")
    
    # Redução dimensional
    print(f"\n📉 Aplicando {args.method.upper()}...")
    embeddings_2d, pca_model = None, None
    
    if args.method == 'pca':
        embeddings_2d, pca_model = compute_pca(combined_emb, args.n_components)
        var_ratio = pca_model.explained_variance_ratio_
        print(f"✓ Variância explicada: PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}")
    elif args.method == 'tsne':
        embeddings_2d = compute_tsne(combined_emb, args.n_components, args.perplexity)
    elif args.method == 'umap':
        embeddings_2d = compute_umap(combined_emb, args.n_components, args.n_neighbors)
    
    # Visualizações
    print("\n🎨 Gerando visualizações...")
    
    # 1. Visualização PCA principal
    output_pca = output_dir / f"{args.prefix}_{args.method}.png"
    plot_cluster_pca(
        embeddings_2d, cluster_labels, pca_model, binary_labels,
        title=f"Visualização {args.method.upper()} dos Clusters",
        output_path=str(output_pca),
        method=args.method
    )
    
    # 2. Distribuição de splits (se disponível)
    if split_assignment is not None:
        output_splits = output_dir / f"{args.prefix}_splits_{args.method}.png"
        plot_split_distribution(
            embeddings_2d, split_assignment, binary_labels, pca_model,
            title="Distribuição dos Splits no Espaço Reduzido",
            output_path=str(output_splits),
            method=args.method
        )
    
    # 3. Integridade dos clusters (se splits disponíveis)
    if split_assignment is not None:
        output_integrity = output_dir / f"{args.prefix}_cluster_integrity.png"
        plot_cluster_integrity(
            cluster_labels, split_assignment, binary_labels,
            title="Análise de Integridade dos Clusters",
            output_path=str(output_integrity)
        )
    
    # 4. Variância do PCA
    if pca_model is not None and args.method == 'pca':
        output_var = output_dir / f"{args.prefix}_pca_variance.png"
        plot_pca_variance(pca_model, str(output_var))
    
    # Métricas
    if args.save_metrics:
        print("\n💾 Salvando métricas...")
        metrics = calculate_cluster_metrics(combined_emb, cluster_labels)
        metrics['similarity_stats'] = sim_stats
        
        if args.method == 'pca' and pca_model is not None:
            metrics['pca_variance_ratio'] = pca_model.explained_variance_ratio_.tolist()
        
        metrics_file = output_dir / f"{args.prefix}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Métricas salvas em: {metrics_file}")
    
    print("\n✅ Visualizações concluídas!")
    print(f"📁 Arquivos salvos em: {output_dir}")
    
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
