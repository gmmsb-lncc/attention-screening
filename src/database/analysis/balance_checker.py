"""
Balance Checker Module.

Provides functionality for checking class balance in datasets,
analyzing activity thresholds, and generating balance metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import entropy

import sys
import os
from pathlib import Path

# Add the database directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_analyzer import BaseAnalyzer
from core.config import DatabaseConfig
from core.exceptions import AnalysisError


class BalanceChecker(BaseAnalyzer):
    """
    Check class balance in molecular datasets.
    
    This class provides functionality to analyze class balance based on
    activity thresholds, calculate entropy metrics, and generate visualizations.
    Based on the original analysisLLM.ipynb functionality.
    """

    def __init__(self, config: DatabaseConfig = None, filepath: str = None):
        """
        Initialize the balance checker.
        
        Args:
            config: DatabaseConfig instance
            filepath: Path to data file
        """
        super().__init__(config)
        self.filepath = filepath
        self.activity_thresholds = config.activity_thresholds if config else [1000, 10000]

    def load_dataset(self) -> pd.DataFrame:
        """
        Load dataset for balance analysis.
        
        Returns:
            Loaded DataFrame
        """
        if not self.filepath:
            raise AnalysisError("No file path provided")
        
        # Load only necessary columns for efficiency
        columns = ['chembl_id', 'molregno', 'target_kinase', 'canonical_smiles',
                  'standard_value', 'standard_type', 'kinase_group']
        
        try:
            self._data = pd.read_csv(self.filepath, sep='\\t', usecols=columns, low_memory=False)
        except ValueError:
            # If some columns don't exist, load all and filter later
            self._data = pd.read_csv(self.filepath, sep='\\t', low_memory=False)
        
        return self._data

    def prepare_labels(self, activity_threshold: float) -> pd.DataFrame:
        """
        Prepare activity labels based on threshold.
        
        Args:
            activity_threshold: Threshold value for active/inactive classification (nM)
            
        Returns:
            DataFrame with prepared labels
        """
        if self._data is None:
            raise AnalysisError("No data loaded. Call load_dataset() first.")
        
        # Ensure standard_value is numeric
        self._data['standard_value'] = pd.to_numeric(self._data['standard_value'], errors='coerce')
        
        # Create labels based on threshold
        self._data['label'] = self._data['standard_value'].apply(
            lambda x: 'active' if pd.notna(x) and x < activity_threshold else 'inactive'
        )
        
        return self._data

    def filter_relevant_groups(self) -> pd.DataFrame:
        """
        Filter data to include only relevant kinase groups.
        
        Returns:
            Filtered DataFrame
        """
        if self._data is None:
            raise AnalysisError("No data loaded. Call load_dataset() first.")
        
        # Define relevant kinase groups
        relevant_kinase_groups = [
            'AGC', 'Other', 'TKL', 'CAMK', 'CK1', 'CMGC', 'STE', 'Tyrosine',
            'Transferase', 'Serine', 'Nuclear', 'Lyase'
        ]
        
        if 'kinase_group' in self._data.columns:
            self._data = self._data[self._data['kinase_group'].isin(relevant_kinase_groups)]
        
        return self._data

    def analyze_class_balance(self, activity_threshold: float) -> tuple:
        """
        Analyze class balance for a specific activity threshold.
        
        Args:
            activity_threshold: Activity threshold for classification
            
        Returns:
            Tuple of (data, class_ratio, entropy, coefficient_of_variation)
        """
        # Prepare labels
        self.prepare_labels(activity_threshold)
        
        # Filter relevant groups if kinase_group column exists
        if 'kinase_group' in self._data.columns:
            self.filter_relevant_groups()
        
        # Calculate class balance metrics
        if 'kinase_group' in self._data.columns:
            class_counts = self._data.groupby(['kinase_group', 'label']).size().unstack(fill_value=0)
        else:
            class_counts = self._data['label'].value_counts()
        
        # Calculate metrics
        class_ratio = self._calculate_class_ratio()
        entropy_val = self._calculate_entropy()
        cv = self._calculate_coefficient_of_variation()
        
        return self._data.copy(), class_ratio, entropy_val, cv

    def _calculate_class_ratio(self) -> float:
        """Calculate class ratio (active/inactive)."""
        label_counts = self._data['label'].value_counts()
        active_count = label_counts.get('active', 0)
        inactive_count = label_counts.get('inactive', 0)
        
        if inactive_count == 0:
            return float('inf')
        return active_count / inactive_count

    def _calculate_entropy(self) -> float:
        """Calculate entropy of class distribution."""
        label_counts = self._data['label'].value_counts()
        total = len(self._data)
        
        if total == 0:
            return 0
        
        probabilities = label_counts / total
        return entropy(probabilities, base=2)

    def _calculate_coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation for kinase groups."""
        if 'kinase_group' not in self._data.columns:
            return 0
        
        group_counts = self._data['kinase_group'].value_counts()
        if len(group_counts) == 0:
            return 0
        
        mean_count = group_counts.mean()
        if mean_count == 0:
            return 0
        
        return group_counts.std() / mean_count

    def run_analysis_for_threshold(self, activity_threshold: float) -> dict:
        """
        Run complete analysis for a single threshold.
        
        Args:
            activity_threshold: Activity threshold to analyze
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing threshold: {activity_threshold} nM")
        
        data, class_ratio, entropy_val, cv = self.analyze_class_balance(activity_threshold)
        
        # Calculate additional statistics
        total_samples = len(data)
        active_samples = len(data[data['label'] == 'active'])
        inactive_samples = len(data[data['label'] == 'inactive'])
        
        results = {
            'threshold': activity_threshold,
            'data': data,
            'total_samples': total_samples,
            'active_samples': active_samples,
            'inactive_samples': inactive_samples,
            'class_ratio': class_ratio,
            'entropy': entropy_val,
            'coefficient_of_variation': cv,
            'active_percentage': (active_samples / total_samples * 100) if total_samples > 0 else 0
        }
        
        return results

    def compare_thresholds(self, thresholds: list = None) -> dict:
        """
        Compare multiple activity thresholds.
        
        Args:
            thresholds: List of thresholds to compare
            
        Returns:
            Dictionary with comparison results
        """
        if thresholds is None:
            thresholds = self.activity_thresholds
        
        comparison_results = {}
        threshold_data = {}
        
        for threshold in thresholds:
            # Reload data for each threshold to avoid conflicts
            self.load_dataset()
            results = self.run_analysis_for_threshold(threshold)
            
            comparison_results[f'{threshold}_nM'] = {
                'threshold': threshold,
                'class_ratio': results['class_ratio'],
                'entropy': results['entropy'],
                'coefficient_of_variation': results['coefficient_of_variation'],
                'active_percentage': results['active_percentage'],
                'total_samples': results['total_samples']
            }
            
            # Store data for plotting
            results['data']['threshold'] = f'{threshold}_nM'
            threshold_data[f'{threshold}_nM'] = results['data']
        
        self._results['comparison'] = comparison_results
        self._results['threshold_data'] = threshold_data
        
        return comparison_results

    def plot_class_distribution(self, output_path: str = None) -> None:
        """
        Plot class distribution comparison between thresholds.
        
        Args:
            output_path: Path to save the plot
        """
        if 'threshold_data' not in self._results:
            raise AnalysisError("No threshold data available. Run compare_thresholds() first.")
        
        threshold_data = self._results['threshold_data']
        
        # Combine data from all thresholds
        combined_data = pd.concat(threshold_data.values(), ignore_index=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Bar plot of class counts
        sns.countplot(data=combined_data, x='threshold', hue='label', ax=axes[0,0])
        axes[0,0].set_title('Class Distribution by Threshold')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Box plot of standard values
        if 'standard_value' in combined_data.columns:
            sns.boxplot(data=combined_data, x='threshold', y='standard_value', hue='label', ax=axes[0,1])
            axes[0,1].set_title('Standard Value Distribution by Threshold')
            axes[0,1].set_yscale('log')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Kinase group distribution (if available)
        if 'kinase_group' in combined_data.columns:
            kinase_data = combined_data.groupby(['threshold', 'kinase_group', 'label']).size().reset_index(name='count')
            sns.barplot(data=kinase_data, x='kinase_group', y='count', hue='threshold', ax=axes[1,0])
            axes[1,0].set_title('Distribution by Kinase Group')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Metrics comparison
        comparison = self._results['comparison']
        metrics_data = []
        for threshold, data in comparison.items():
            metrics_data.extend([
                {'threshold': threshold, 'metric': 'Class Ratio', 'value': data['class_ratio']},
                {'threshold': threshold, 'metric': 'Entropy', 'value': data['entropy']},
                {'threshold': threshold, 'metric': 'CV', 'value': data['coefficient_of_variation']}
            ])
        
        metrics_df = pd.DataFrame(metrics_data)
        sns.barplot(data=metrics_df, x='metric', y='value', hue='threshold', ax=axes[1,1])
        axes[1,1].set_title('Balance Metrics Comparison')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        
        if self.config.get('save_plots', True):
            plt.show()

    def get_balance_summary(self) -> dict:
        """
        Get summary of balance analysis.
        
        Returns:
            Dictionary with balance summary
        """
        if 'comparison' not in self._results:
            raise AnalysisError("No comparison results available. Run compare_thresholds() first.")
        
        comparison = self._results['comparison']
        
        summary = {
            'thresholds_analyzed': list(comparison.keys()),
            'best_balance_threshold': None,
            'metrics_summary': {}
        }
        
        # Find threshold with best balance (closest to 1.0 class ratio and highest entropy)
        best_score = -1
        best_threshold = None
        
        for threshold, data in comparison.items():
            # Score based on how close class ratio is to 1.0 and entropy value
            ratio_score = 1 / (1 + abs(data['class_ratio'] - 1.0))
            entropy_score = data['entropy'] / 1.0  # Normalize entropy (max is ~1 for binary)
            combined_score = (ratio_score + entropy_score) / 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_threshold = threshold
        
        summary['best_balance_threshold'] = best_threshold
        summary['best_balance_score'] = best_score
        
        # Add metrics summary
        for metric in ['class_ratio', 'entropy', 'coefficient_of_variation']:
            values = [data[metric] for data in comparison.values()]
            summary['metrics_summary'][metric] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        self._results['summary'] = summary
        return summary

    def analyze(self) -> dict:
        """
        Perform complete balance analysis.
        
        Returns:
            Dictionary with analysis results
        """
        if self._data is None:
            self.load_dataset()
        
        # Compare thresholds
        self.compare_thresholds()
        
        # Get summary
        self.get_balance_summary()
        
        return self._results


def main():
    """
    Main function for balance checking (compatibility with original notebook).
    """
    # Default file path
    filepath = './filtered_dataset.tsv'
    
    if not os.path.exists(filepath):
        print(f"Warning: Default data file not found: {filepath}")
        print("Please provide correct file path.")
        return
    
    # Create balance checker
    checker = BalanceChecker(filepath=filepath)
    results = checker.analyze()
    
    # Generate plots
    checker.plot_class_distribution("class_distribution_analysis.png")
    
    # Print summary
    summary = checker.get_balance_summary()
    print("\\n=== BALANCE ANALYSIS SUMMARY ===")
    print(f"Best balanced threshold: {summary['best_balance_threshold']}")
    print(f"Balance score: {summary['best_balance_score']:.3f}")
    
    print("\\nBalance analysis completed!")


if __name__ == "__main__":
    main()
