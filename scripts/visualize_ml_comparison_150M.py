#!/usr/bin/env python3
"""
Compara performance dos 12 ML models através de todos os modelos de proteína.
Gera heatmaps, boxplots e rankings estatísticos.

Versão modularizada seguindo princípios SOLID, KISS e Clean Code.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict

from visualization import (
    load_classification_metrics,
    load_regression_metrics,
    setup_plot_style
)
from visualization.plot_heatmaps import (
    plot_classification_heatmaps,
    plot_regression_heatmaps
)
from visualization.plot_statistics import (
    plot_classification_boxplot,
    plot_regression_boxplot,
    create_ranking_table
)


# Lista de modelos de proteína (excluindo boltz2)
PROTEIN_MODELS = [
    'esm2_t6_8M_UR50D',
    'esm2_t12_35M_UR50D',
    'esm2_t30_150M_UR50D',
    'esm2_t33_650M_UR50D',
    'esm2_t36_3B_UR50D',
    'esmc-300m-2024-12',
    'esmc-600m-2024-12',
]


def load_all_protein_models(base_dir: Path) -> tuple[Dict, Dict]:
    """
    Carrega métricas de todos os modelos de proteína.
    
    Args:
        base_dir: Diretório base com resultados
        
    Returns:
        Tuple com (all_clf_test, all_reg_test)
    """
    all_clf_test = {}
    all_reg_test = {}
    
    print("📊 Carregando métricas de todos os modelos de proteína...")
    
    for protein_model in PROTEIN_MODELS:
        model_path = base_dir / protein_model
        
        if not model_path.exists():
            print(f"   ⚠️  Pulando {protein_model} (não encontrado)")
            continue
        
        # Carregar métricas
        _, clf_test = load_classification_metrics(model_path)
        _, reg_test = load_regression_metrics(model_path)
        
        all_clf_test[protein_model] = clf_test
        all_reg_test[protein_model] = reg_test
        
        print(f"   ✅ {protein_model}: {len(clf_test)} clf, {len(reg_test)} reg")
    
    return all_clf_test, all_reg_test


def create_figure_layout():
    """
    Cria layout da figura com grid 3x2.
    
    Returns:
        Tuple com (fig, axes_dict)
    """
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    axes = {
        # Heatmaps (primeira e segunda linhas)
        'roc_heatmap': fig.add_subplot(gs[0, 0]),
        'f1_heatmap': fig.add_subplot(gs[0, 1]),
        'mae_heatmap': fig.add_subplot(gs[1, 0]),
        'r2_heatmap': fig.add_subplot(gs[1, 1]),
        
        # Boxplots (terceira linha)
        'clf_boxplot': fig.add_subplot(gs[2, 0]),
        'reg_boxplot': fig.add_subplot(gs[2, 1]),
    }
    
    return fig, axes


def create_ranking_figure(all_clf_test: Dict, all_reg_test: Dict, output_path: Path):
    """
    Cria figura separada com tabela de ranking.
    
    Args:
        all_clf_test: Métricas de classificação
        all_reg_test: Métricas de regressão
        output_path: Caminho de saída
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    create_ranking_table(all_clf_test, all_reg_test, PROTEIN_MODELS, ax)
    
    fig.suptitle('ML Models Ranking: Average Performance Across All Protein Models',
                 fontsize=14, fontweight='bold')
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Ranking salvo: {output_path}")
    plt.close()


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Compara ML models através de todos os modelos de proteína'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='results/protein_model_benchmark_non_human_v2',
        help='Diretório base com resultados dos modelos'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/results_150M',
        help='Diretório de saída para as visualizações'
    )
    
    args = parser.parse_args()
    
    # Setup
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not base_dir.exists():
        print(f"❌ Erro: Diretório não encontrado: {base_dir}")
        return
    
    # Configurar estilo
    setup_plot_style()
    
    # Carregar todas as métricas
    all_clf_test, all_reg_test = load_all_protein_models(base_dir)
    
    if not all_clf_test or not all_reg_test:
        print("❌ Erro: Nenhuma métrica carregada")
        return
    
    print(f"\n📈 Gerando visualizações comparativas...")
    
    # Criar figura principal
    fig, axes = create_figure_layout()
    
    # Plotar heatmaps
    plot_classification_heatmaps(
        all_clf_test, PROTEIN_MODELS,
        axes['roc_heatmap'], axes['f1_heatmap']
    )
    plot_regression_heatmaps(
        all_reg_test, PROTEIN_MODELS,
        axes['mae_heatmap'], axes['r2_heatmap']
    )
    
    # Plotar boxplots
    plot_classification_boxplot(all_clf_test, PROTEIN_MODELS, axes['clf_boxplot'])
    plot_regression_boxplot(all_reg_test, PROTEIN_MODELS, axes['reg_boxplot'])
    
    # Título principal
    fig.suptitle(
        'ML Models Comparison: Performance Across All Protein Models',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    # Salvar figura principal
    output_path = output_dir / "ml_comparison_all_proteins.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comparação salva: {output_path}")
    plt.close()
    
    # Criar figura de ranking
    ranking_path = output_dir / "ml_ranking_all_proteins.png"
    create_ranking_figure(all_clf_test, all_reg_test, ranking_path)
    
    print(f"\n✅ Todas as visualizações foram geradas com sucesso!")


if __name__ == '__main__':
    main()
