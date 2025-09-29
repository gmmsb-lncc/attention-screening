"""
Database Module - Comprehensive data analysis and processing for molecular databases.

This module provides modular tools for molecular data analysis including:
- Comparative analysis between datasets
- Molecular clustering and similarity analysis  
- Descriptor calculation and analysis
- Data cleaning and standardization
- Class balance analysis and stratification

The module maintains compatibility with original scripts while providing
a clean, modular architecture for extensibility and maintenance.
"""

# Core functionality
from .core import DatabaseConfig, BaseAnalyzer
from .core.exceptions import DatabaseError, AnalysisError, ProcessingError

# Processing modules
from .processing import MolecularClusterer, MolecularDescriptors, DataCleaner

# Analysis modules  
from .analysis import ComparativeAnalyzer, BalanceChecker

__version__ = "1.0.0"

__all__ = [
    # Core
    'DatabaseConfig',
    'BaseAnalyzer',
    'DatabaseError',
    'AnalysisError', 
    'ProcessingError',
    
    # Processing
    'MolecularClusterer',
    'MolecularDescriptors',
    'DataCleaner',
    
    # Analysis
    'ComparativeAnalyzer',
    'BalanceChecker'
]


def create_analyzer(analyzer_type: str, **kwargs):
    """
    Factory function to create analyzers.
    
    Args:
        analyzer_type: Type of analyzer ('comparative', 'balance', 'cluster', 'descriptors', 'cleaner')
        **kwargs: Arguments to pass to analyzer constructor
        
    Returns:
        Analyzer instance
    """
    analyzers = {
        'comparative': ComparativeAnalyzer,
        'balance': BalanceChecker,
        'cluster': MolecularClusterer,
        'descriptors': MolecularDescriptors,
        'cleaner': DataCleaner
    }
    
    if analyzer_type not in analyzers:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}. Available: {list(analyzers.keys())}")
    
    return analyzers[analyzer_type](**kwargs)


def get_default_config():
    """Get default database configuration."""
    return DatabaseConfig()


# Convenience functions for common operations
def quick_comparative_analysis(human_file: str, non_human_file: str, output_dir: str = "output"):
    """
    Quick comparative analysis between human and non-human datasets.
    
    Args:
        human_file: Path to human kinase data
        non_human_file: Path to non-human kinase data  
        output_dir: Output directory for results
        
    Returns:
        Analysis results dictionary
    """
    config = DatabaseConfig({'output_dir': output_dir})
    analyzer = ComparativeAnalyzer(config)
    analyzer.load_datasets(human_file, non_human_file)
    results = analyzer.analyze()
    
    # Generate visualizations
    analyzer.plot_activity_distributions(f"{output_dir}/activity_distributions.png")
    analyzer.plot_kinase_overlap(f"{output_dir}/kinase_overlap.png")
    analyzer.generate_comparison_report(f"{output_dir}/comparison_report.json")
    
    return results


def quick_balance_analysis(data_file: str, thresholds: list = None, output_dir: str = "output"):
    """
    Quick class balance analysis.
    
    Args:
        data_file: Path to data file
        thresholds: Activity thresholds to analyze
        output_dir: Output directory for results
        
    Returns:
        Balance analysis results
    """
    config = DatabaseConfig({'output_dir': output_dir})
    if thresholds:
        config.set('activity_thresholds', thresholds)
    
    checker = BalanceChecker(config, data_file)
    results = checker.analyze()
    
    # Generate visualization
    checker.plot_class_distribution(f"{output_dir}/balance_analysis.png")
    
    return results


def quick_clustering_analysis(data_file: str, similarity_threshold: float = 0.8, output_dir: str = "output"):
    """
    Quick molecular clustering analysis.
    
    Args:
        data_file: Path to SMILES data file
        similarity_threshold: Tanimoto similarity threshold
        output_dir: Output directory for results
        
    Returns:
        Clustering results
    """
    config = DatabaseConfig({
        'similarity_threshold': similarity_threshold,
        'output_dir': output_dir
    })
    
    clusterer = MolecularClusterer(config, data_file)
    clusterer.load_smiles_data()
    results = clusterer.analyze()
    
    # Generate visualization
    clusterer.plot_clusters(f"{output_dir}/molecular_clusters.png")
    clusterer.save_clusters(f"{output_dir}/clusters.pkl")
    
    return results
