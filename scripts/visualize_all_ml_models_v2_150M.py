#!/usr/bin/env python3
"""
Visualiza todos os 12 modelos ML (classificação e regressão) para um modelo de proteína.
Compara métricas de validação vs teste para detectar overfitting.

Versão modularizada seguindo princípios SOLID, KISS e Clean Code.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from visualization import (
    load_all_metrics,
    plot_classification_metrics,
    plot_regression_metrics,
    create_summary_table,
    setup_plot_style
)


def create_figure_layout():
    """
    Cria layout da figura com grid 3x3.
    
    Returns:
        Tuple com (fig, axes_dict)
    """
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    axes = {
        # Classification (primeira linha)
        'roc': fig.add_subplot(gs[0, 0]),
        'f1': fig.add_subplot(gs[0, 1]),
        'acc': fig.add_subplot(gs[0, 2]),
        
        # MCC + Regression (segunda linha)
        'mcc': fig.add_subplot(gs[1, 0]),
        'mae': fig.add_subplot(gs[1, 1]),
        'r2': fig.add_subplot(gs[1, 2]),
        
        # RMSE + Scatter + Summary (terceira linha)
        'rmse': fig.add_subplot(gs[2, 0]),
        'scatter': fig.add_subplot(gs[2, 1]),
        'summary': fig.add_subplot(gs[2, 2]),
    }
    
    return fig, axes


def validate_model_path(base_path: Path) -> bool:
    """
    Valida se o caminho do modelo existe.
    
    Args:
        base_path: Caminho base do modelo
        
    Returns:
        True se válido, False caso contrário
    """
    if not base_path.exists():
        print(f"❌ Erro: Diretório não encontrado: {base_path}")
        return False
    return True


def save_visualization(fig: plt.Figure, output_path: Path, model_name: str):
    """
    Salva a visualização em arquivo.
    
    Args:
        fig: Figura do matplotlib
        output_path: Caminho do arquivo de saída
        model_name: Nome do modelo (para título)
    """
    fig.suptitle(
        f'ML Models Benchmark: {model_name}\n(Validation vs Test Performance)',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualização salva: {output_path}")
    plt.close()


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Visualiza todos os 12 modelos ML para um modelo de proteína'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Nome do modelo de proteína (ex: esm2_t30_150M_UR50D)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/results_150M',
        help='Diretório de saída para as visualizações'
    )
    
    args = parser.parse_args()
    
    # Setup
    base_path = Path(f"results/protein_model_benchmark_non_human_v2/{args.model}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validar
    if not validate_model_path(base_path):
        return
    
    # Configurar estilo
    setup_plot_style()
    
    # Carregar métricas
    print(f"📊 Carregando métricas de: {args.model}")
    clf_val, clf_test, reg_val, reg_test = load_all_metrics(args.model, base_path)
    
    print(f"   ✅ Classification: {len(clf_val)} modelos (val), {len(clf_test)} modelos (test)")
    print(f"   ✅ Regression: {len(reg_val)} modelos (val), {len(reg_test)} modelos (test)")
    
    # Criar figura
    fig, axes = create_figure_layout()
    
    # Plotar métricas
    print(f"📈 Gerando visualizações...")
    plot_classification_metrics(
        clf_val, clf_test,
        axes['roc'], axes['f1'], axes['acc'], axes['mcc']
    )
    plot_regression_metrics(
        reg_val, reg_test,
        axes['mae'], axes['r2'], axes['rmse'], axes['scatter']
    )
    create_summary_table(
        clf_val, clf_test, reg_val, reg_test,
        axes['summary']
    )
    
    # Salvar
    output_path = output_dir / f"all_ml_models_{args.model}.png"
    save_visualization(fig, output_path, args.model)


if __name__ == '__main__':
    main()
