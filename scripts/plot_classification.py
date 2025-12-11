#!/usr/bin/env python3
"""
Script modular simplificado - gera apenas plots de classificação.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.data_loader import load_results_from_files
from visualization.metrics_extractor import MetricsExtractor
from visualization.basic_plots import BasicPlotter


def main(files, output_dir):
    results = load_results_from_files(files)
    extractor = MetricsExtractor(results)
    classification_data = extractor.extract_classification_metrics()
    
    plotter = BasicPlotter(Path(output_dir))
    return plotter.plot_classification_comparison(classification_data)


if __name__ == '__main__':
    import sys
    files = sys.argv[1:-1]
    output = sys.argv[-1]
    path = main(files, output)
    print(f"✓ {path}")
