#!/usr/bin/env python3
"""
Script para visualizações avançadas de comparação de modelos.

Gera gráficos científicos sofisticados incluindo radar, heatmap, 
trade-offs, Pareto e distribuições.

Usage:
    python scripts/advanced_analysis.py --files results/*.json
    python scripts/advanced_analysis.py --files file1.json file2.json --output results/advanced
"""

import argparse
import sys
from pathlib import Path

# Adicionar path para imports
sys.path.insert(0, str(Path(__file__).parent))

from visualization.data_loader import load_results_from_files
from visualization.metrics_extractor import MetricsExtractor, calculate_overall_score
from visualization.advanced_plots import AdvancedPlotter


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description='Visualizações avançadas de comparação de modelos'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        required=True,
        help='Arquivos JSON com resultados'
    )
    parser.add_argument(
        '--output',
        default='results/advanced_analysis',
        help='Diretório de saída'
    )
    
    return parser.parse_args()


def main():
    """Função principal."""
    # Parse argumentos
    args = parse_arguments()
    
    print("🔬 Análise Avançada de Modelos de Proteína")
    print("=" * 50)
    
    # 1. Carregar dados
    print("\n📂 Carregando resultados...")
    results = load_results_from_files(args.files)
    
    if not results:
        print("❌ Nenhum resultado válido carregado!")
        return 1
    
    print(f"✅ {len(results)} modelos carregados")
    
    # 2. Extrair métricas
    print("\n📊 Extraindo métricas...")
    extractor = MetricsExtractor(results)
    
    classification_data = extractor.extract_classification_metrics()
    regression_data = extractor.extract_regression_metrics()
    embedding_data = extractor.extract_embedding_info()
    
    # Calcular scores gerais
    overall_data = []
    for i in range(len(classification_data)):
        model_name = classification_data[i]['Model']
        overall_score = calculate_overall_score(
            classification_data[i],
            regression_data[i]
        )
        overall_data.append({
            'Model': model_name,
            'Overall_Score': overall_score,
            'Total_Dim': embedding_data[i]['Total_Dim']
        })
    
    print(f"✅ Métricas extraídas")
    
    # 3. Gerar visualizações avançadas
    print("\n📈 Gerando visualizações avançadas...")
    output_dir = Path(args.output)
    plotter = AdvancedPlotter(output_dir)
    
    plot_files = []
    
    # Plot 1: Radar Chart
    print("  → Criando radar chart...")
    path = plotter.create_radar_chart(classification_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 2: Heatmap
    print("  → Criando heatmap...")
    path = plotter.create_heatmap(classification_data, regression_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 3: Trade-off Analysis
    print("  → Criando análise de trade-offs...")
    path = plotter.create_tradeoff_scatter(classification_data, regression_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 4: Pareto Chart
    print("  → Criando gráfico de Pareto...")
    path = plotter.create_pareto_chart(overall_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 5: Regression Correlation Heatmap
    print("  → Criando heatmap de correlação (regressão)...")
    path = plotter.create_regression_correlation_heatmap(regression_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 6: Distribution - Classification
    print("  → Criando distribuição de classificação...")
    path = plotter.create_distribution_boxplot(results, 'classification')
    if path:
        plot_files.append(path)
        print(f"    ✓ {path.name}")
    
    # Plot 7: Distribution - Regression
    print("  → Criando distribuição de regressão...")
    path = plotter.create_distribution_boxplot(results, 'regression')
    if path:
        plot_files.append(path)
        print(f"    ✓ {path.name}")
    
    # Resumo final
    print("\n" + "=" * 50)
    print("✅ Análise avançada concluída!")
    print(f"\n📁 Resultados salvos em: {output_dir}")
    print(f"📊 {len(plot_files)} visualizações avançadas geradas")
    
    # Insights
    print("\n💡 Insights:")
    best = max(overall_data, key=lambda x: x['Overall_Score'])
    most_efficient = min(overall_data, key=lambda x: x['Total_Dim'])
    
    print(f"  🏆 Melhor performance: {best['Model']} ({best['Overall_Score']:.2f})")
    print(f"  ⚡ Mais eficiente: {most_efficient['Model']} ({most_efficient['Total_Dim']} dims)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
