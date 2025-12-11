#!/usr/bin/env python3
"""
Script principal para comparação básica de modelos de proteína.

Este script utiliza módulos separados seguindo princípios SOLID e Clean Code.
Gera visualizações básicas e relatório de comparação.

Usage:
    python scripts/compare_models_v2.py --files results/*.json
    python scripts/compare_models_v2.py --files file1.json file2.json --output results/comparison
"""

import argparse
import sys
from pathlib import Path

# Adicionar path para imports
sys.path.insert(0, str(Path(__file__).parent))

from visualization.data_loader import load_results_from_files
from visualization.metrics_extractor import MetricsExtractor, calculate_overall_score
from visualization.basic_plots import BasicPlotter
from visualization.report_generator import ReportGenerator, save_dataframes_to_csv


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description='Comparação de modelos de proteína'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        required=True,
        help='Arquivos JSON com resultados'
    )
    parser.add_argument(
        '--output',
        default='results/protein_model_comparison',
        help='Diretório de saída'
    )
    
    return parser.parse_args()


def main():
    """Função principal."""
    # Parse argumentos
    args = parse_arguments()
    
    print("🔬 Comparação de Modelos de Proteína")
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
    
    print(f"✅ Métricas extraídas para {len(overall_data)} modelos")
    
    # 3. Gerar visualizações (em processos separados para evitar conflitos matplotlib)
    print("\n📈 Gerando visualizações...")
    output_dir = Path(args.output)
    plotter = BasicPlotter(output_dir)
    
    plot_files = []
    
    # Plot 1: Classificação
    print("  → Classificação...")
    path = plotter.plot_classification_comparison(classification_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 2: Regressão  
    print("  → Regressão...")
    path = plotter.plot_regression_comparison(regression_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 3: Embeddings
    print("  → Embeddings...")
    path = plotter.plot_embedding_dimensions(embedding_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 4: Ranking
    print("  → Ranking...")
    path = plotter.plot_overall_ranking(overall_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # 4. Salvar dados em CSV
    print("\n💾 Salvando dados em CSV...")
    save_dataframes_to_csv(output_dir, classification_data, 
                          regression_data, overall_data)
    print("  ✓ classification_metrics.csv")
    print("  ✓ regression_metrics.csv")
    print("  ✓ overall_scores.csv")
    
    # 5. Gerar relatório
    print("\n📝 Gerando relatório...")
    report_gen = ReportGenerator(output_dir)
    report_path = report_gen.generate_summary(
        classification_data,
        regression_data,
        embedding_data,
        overall_data
    )
    print(f"  ✓ {report_path.name}")
    
    # Resumo final
    print("\n" + "=" * 50)
    print("✅ Análise concluída!")
    print(f"\n📁 Resultados salvos em: {output_dir}")
    print(f"📊 {len(plot_files)} gráficos gerados")
    print(f"📄 1 relatório markdown gerado")
    
    # Melhor modelo
    best = max(overall_data, key=lambda x: x['Overall_Score'])
    print(f"\n🏆 Melhor modelo: {best['Model']} (Score: {best['Overall_Score']:.2f})")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
