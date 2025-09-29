"""
Comparative Analysis Module.

Provides functionality for comparing datasets, particularly human vs non-human
kinase data, with statistical analysis and visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
import os
from pathlib import Path

# Add the database directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_analyzer import BaseAnalyzer
from core.config import DatabaseConfig
from core.exceptions import AnalysisError


class ComparativeAnalyzer(BaseAnalyzer):
    """
    Comparative analysis between datasets.
    
    This class provides functionality to compare datasets, particularly
    human vs non-human kinase data. Maintains compatibility with the
    original comparative_analysis.py functionality.
    """

    def __init__(self, config: DatabaseConfig = None):
        """
        Initialize the comparative analyzer.
        
        Args:
            config: DatabaseConfig instance
        """
        super().__init__(config)
        self.df_human = None
        self.df_non_human = None

    def load_datasets(self, human_file: str, non_human_file: str) -> tuple:
        """
        Load human and non-human datasets.
        
        Args:
            human_file: Path to human kinase data file
            non_human_file: Path to non-human kinase data file
            
        Returns:
            Tuple of (human_df, non_human_df)
        """
        print("Loading datasets...")
        
        # Verify files exist
        if not os.path.exists(human_file):
            raise AnalysisError(f"Human file not found: {human_file}")
        
        if not os.path.exists(non_human_file):
            raise AnalysisError(f"Non-human file not found: {non_human_file}")
        
        # Load datasets
        self.df_human = pd.read_csv(human_file, sep='\\t')
        self.df_non_human = pd.read_csv(non_human_file, sep='\\t')
        
        print(f"Human kinases: {len(self.df_human)} records")
        print(f"Non-human kinases: {len(self.df_non_human)} records")
        
        # Store in results
        self._results['human_records'] = len(self.df_human)
        self._results['non_human_records'] = len(self.df_non_human)
        
        return self.df_human, self.df_non_human

    def basic_statistics(self) -> dict:
        """
        Calculate basic statistics for both datasets.
        
        Returns:
            Dictionary with basic statistics
        """
        if self.df_human is None or self.df_non_human is None:
            raise AnalysisError("Datasets not loaded. Call load_datasets() first.")
        
        print("\\n=== BASIC STATISTICS ===")
        
        # Number of unique compounds
        human_compounds = self.df_human['molregno'].nunique()
        non_human_compounds = self.df_non_human['molregno'].nunique()
        
        print(f"Unique compounds - Human: {human_compounds}")
        print(f"Unique compounds - Non-human: {non_human_compounds}")
        
        # Number of unique kinases
        human_kinases = self.df_human['target_kinase'].nunique()
        non_human_kinases = self.df_non_human['target_kinase'].nunique()
        
        print(f"Unique kinases - Human: {human_kinases}")
        print(f"Unique kinases - Non-human: {non_human_kinases}")
        
        # Most common organisms (for non-human)
        print("\\nMost common organisms (non-human):")
        organism_counts = self.df_non_human['organism'].value_counts().head(10)
        for organism, count in organism_counts.items():
            print(f"  {organism}: {count}")
        
        # Standard value types
        print("\\nStandard value types:")
        print("Human:", self.df_human['standard_type'].value_counts().to_dict())
        print("Non-human:", self.df_non_human['standard_type'].value_counts().to_dict())
        
        # Store results
        stats = {
            'human_compounds': human_compounds,
            'non_human_compounds': non_human_compounds,
            'human_kinases': human_kinases,
            'non_human_kinases': non_human_kinases,
            'top_organisms': organism_counts.to_dict(),
            'human_value_types': self.df_human['standard_type'].value_counts().to_dict(),
            'non_human_value_types': self.df_non_human['standard_type'].value_counts().to_dict()
        }
        
        self._results['basic_stats'] = stats
        return stats

    def activity_distribution_analysis(self) -> dict:
        """
        Analyze activity value distributions.
        
        Returns:
            Dictionary with activity distribution analysis
        """
        if self.df_human is None or self.df_non_human is None:
            raise AnalysisError("Datasets not loaded. Call load_datasets() first.")
        
        print("\\n=== ACTIVITY DISTRIBUTION ANALYSIS ===")
        
        # Convert to pIC50 scale
        self.df_human['pIC50'] = -np.log10(self.df_human['standard_value'] * 1e-9)
        self.df_non_human['pIC50'] = -np.log10(self.df_non_human['standard_value'] * 1e-9)
        
        # Calculate statistics
        human_stats = {
            'mean': self.df_human['pIC50'].mean(),
            'median': self.df_human['pIC50'].median(),
            'std': self.df_human['pIC50'].std(),
            'min': self.df_human['pIC50'].min(),
            'max': self.df_human['pIC50'].max()
        }
        
        non_human_stats = {
            'mean': self.df_non_human['pIC50'].mean(),
            'median': self.df_non_human['pIC50'].median(),
            'std': self.df_non_human['pIC50'].std(),
            'min': self.df_non_human['pIC50'].min(),
            'max': self.df_non_human['pIC50'].max()
        }
        
        print("pIC50 Statistics:")
        print(f"Human - Mean: {human_stats['mean']:.2f}, Median: {human_stats['median']:.2f}, Std: {human_stats['std']:.2f}")
        print(f"Non-human - Mean: {non_human_stats['mean']:.2f}, Median: {non_human_stats['median']:.2f}, Std: {non_human_stats['std']:.2f}")
        
        # Store results
        activity_analysis = {
            'human_pIC50_stats': human_stats,
            'non_human_pIC50_stats': non_human_stats
        }
        
        self._results['activity_analysis'] = activity_analysis
        return activity_analysis

    def plot_activity_distributions(self, output_path: str = None) -> None:
        """
        Plot activity distributions for both datasets.
        
        Args:
            output_path: Path to save the plot
        """
        if 'activity_analysis' not in self._results:
            self.activity_distribution_analysis()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot histograms
        axes[0].hist(self.df_human['pIC50'].dropna(), bins=50, alpha=0.7, 
                    label='Human', density=True, color='blue')
        axes[0].hist(self.df_non_human['pIC50'].dropna(), bins=50, alpha=0.7, 
                    label='Non-human', density=True, color='red')
        axes[0].set_xlabel('pIC50')
        axes[0].set_ylabel('Density')
        axes[0].set_title('pIC50 Distribution Comparison')
        axes[0].legend()
        
        # Plot box plots
        data_for_box = [
            self.df_human['pIC50'].dropna(),
            self.df_non_human['pIC50'].dropna()
        ]
        axes[1].boxplot(data_for_box, labels=['Human', 'Non-human'])
        axes[1].set_ylabel('pIC50')
        axes[1].set_title('pIC50 Box Plot Comparison')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        
        if self.config.get('save_plots', True):
            plt.show()

    def kinase_distribution_analysis(self) -> dict:
        """
        Analyze kinase distribution between datasets.
        
        Returns:
            Dictionary with kinase distribution analysis
        """
        if self.df_human is None or self.df_non_human is None:
            raise AnalysisError("Datasets not loaded. Call load_datasets() first.")
        
        print("\\n=== KINASE DISTRIBUTION ANALYSIS ===")
        
        # Top kinases in each dataset
        human_top_kinases = self.df_human['target_kinase'].value_counts().head(10)
        non_human_top_kinases = self.df_non_human['target_kinase'].value_counts().head(10)
        
        print("Top 10 kinases in human dataset:")
        for kinase, count in human_top_kinases.items():
            print(f"  {kinase}: {count}")
        
        print("\\nTop 10 kinases in non-human dataset:")
        for kinase, count in non_human_top_kinases.items():
            print(f"  {kinase}: {count}")
        
        # Common kinases
        human_kinases = set(self.df_human['target_kinase'].unique())
        non_human_kinases = set(self.df_non_human['target_kinase'].unique())
        
        common_kinases = human_kinases.intersection(non_human_kinases)
        human_only = human_kinases - non_human_kinases
        non_human_only = non_human_kinases - human_kinases
        
        print(f"\\nKinase overlap analysis:")
        print(f"  Common kinases: {len(common_kinases)}")
        print(f"  Human-only kinases: {len(human_only)}")
        print(f"  Non-human-only kinases: {len(non_human_only)}")
        
        # Store results
        kinase_analysis = {
            'human_top_kinases': human_top_kinases.to_dict(),
            'non_human_top_kinases': non_human_top_kinases.to_dict(),
            'common_kinases': list(common_kinases),
            'human_only_kinases': list(human_only),
            'non_human_only_kinases': list(non_human_only),
            'kinase_counts': {
                'common': len(common_kinases),
                'human_only': len(human_only),
                'non_human_only': len(non_human_only)
            }
        }
        
        self._results['kinase_analysis'] = kinase_analysis
        return kinase_analysis

    def plot_kinase_overlap(self, output_path: str = None) -> None:
        """
        Plot kinase overlap visualization.
        
        Args:
            output_path: Path to save the plot
        """
        if 'kinase_analysis' not in self._results:
            self.kinase_distribution_analysis()
        
        kinase_counts = self._results['kinase_analysis']['kinase_counts']
        
        # Create Venn diagram-like bar plot
        categories = ['Human Only', 'Common', 'Non-human Only']
        counts = [kinase_counts['human_only'], kinase_counts['common'], kinase_counts['non_human_only']]
        colors = ['lightblue', 'purple', 'lightcoral']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, counts, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.xlabel('Kinase Categories')
        plt.ylabel('Number of Kinases')
        plt.title('Kinase Distribution: Human vs Non-human Datasets')
        plt.grid(axis='y', alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        
        if self.config.get('save_plots', True):
            plt.show()

    def generate_comparison_report(self, output_path: str = None) -> dict:
        """
        Generate comprehensive comparison report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Complete comparison report
        """
        print("Generating comprehensive comparison report...")
        
        # Run all analyses
        basic_stats = self.basic_statistics()
        activity_analysis = self.activity_distribution_analysis()
        kinase_analysis = self.kinase_distribution_analysis()
        
        # Compile report
        report = {
            'summary': {
                'human_records': self._results['human_records'],
                'non_human_records': self._results['non_human_records'],
                'total_records': self._results['human_records'] + self._results['non_human_records']
            },
            'basic_statistics': basic_stats,
            'activity_analysis': activity_analysis,
            'kinase_analysis': kinase_analysis
        }
        
        # Save report if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to: {output_path}")
        
        self._results['comparison_report'] = report
        return report

    def analyze(self) -> dict:
        """
        Perform complete comparative analysis.
        
        Returns:
            Dictionary with analysis results
        """
        if self.df_human is None or self.df_non_human is None:
            raise AnalysisError("Datasets not loaded. Call load_datasets() first.")
        
        # Generate comprehensive report
        self.generate_comparison_report()
        
        return self._results


def main():
    """
    Main function to run comparative analysis (compatibility with original script).
    """
    # Default file paths (adjust as needed)
    human_file = "kinase_human_compounds.tsv"
    non_human_file = "kinase_non_human_compounds.tsv"
    
    if not os.path.exists(human_file) or not os.path.exists(non_human_file):
        print("Warning: Default data files not found.")
        print(f"Looking for: {human_file} and {non_human_file}")
        print("Please provide correct file paths.")
        return
    
    # Create analyzer and run analysis
    analyzer = ComparativeAnalyzer()
    analyzer.load_datasets(human_file, non_human_file)
    results = analyzer.analyze()
    
    # Generate visualizations
    analyzer.plot_activity_distributions("activity_distributions.png")
    analyzer.plot_kinase_overlap("kinase_overlap.png")
    
    # Save report
    analyzer.generate_comparison_report("comparison_report.json")
    
    print("\\nComparative analysis completed!")
    

if __name__ == "__main__":
    main()
