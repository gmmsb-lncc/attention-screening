"""
Molecular Descriptors Module.

Provides functionality for calculating molecular descriptors from SMILES strings
using RDKit, including parallel processing capabilities.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import seaborn as sns

import sys
import os
from pathlib import Path

# Add the database directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_analyzer import BaseAnalyzer
from core.config import DatabaseConfig
from core.exceptions import ProcessingError


class MolecularDescriptors(BaseAnalyzer):
    """
    Calculate molecular descriptors for SMILES data.
    
    This class provides functionality to calculate common molecular descriptors
    including MW, LogP, HBD, HBA, TPSA, and NRB. Maintains compatibility with
    the original descriptors.py functionality.
    """

    def __init__(self, config: DatabaseConfig = None, data_path: str = None, batch_size: int = None):
        """
        Initialize the molecular descriptors calculator.
        
        Args:
            config: DatabaseConfig instance
            data_path: Path to data file
            batch_size: Batch size for processing
        """
        super().__init__(config)
        self.data_path = data_path
        self.batch_size = batch_size or self.config.batch_size
        self.descriptor_data = None

    def load_smiles_data(self) -> pd.DataFrame:
        """
        Load SMILES data from file.
        
        Returns:
            Loaded DataFrame
        """
        if not self.data_path:
            raise ProcessingError("No data path provided")
        
        self.load_data(self.data_path)
        return self._data

    @staticmethod
    def calculate_descriptors_for_smiles(smiles_list: list) -> list:
        """
        Calculate descriptors for a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of descriptor dictionaries
        """
        results = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    results.append({
                        'canonical_smiles': smiles,
                        'MW': Descriptors.MolWt(mol),
                        'LogP': Descriptors.MolLogP(mol),
                        'HBD': Descriptors.NumHDonors(mol),
                        'HBA': Descriptors.NumHAcceptors(mol),
                        'TPSA': Descriptors.TPSA(mol),
                        'NRB': Descriptors.NumRotatableBonds(mol)
                    })
                else:
                    # Add entry with None values for failed SMILES
                    results.append({
                        'canonical_smiles': smiles,
                        'MW': None, 'LogP': None, 'HBD': None,
                        'HBA': None, 'TPSA': None, 'NRB': None
                    })
            except Exception as e:
                print(f"Error processing SMILES: {smiles}: {e}")
                # Add entry with None values for errored SMILES
                results.append({
                    'canonical_smiles': smiles,
                    'MW': None, 'LogP': None, 'HBD': None,
                    'HBA': None, 'TPSA': None, 'NRB': None
                })
        return results

    def compute_descriptors(self, smiles_column: str = 'canonical_smiles', 
                          use_parallel: bool = True) -> pd.DataFrame:
        """
        Compute molecular descriptors for all molecules.
        
        Args:
            smiles_column: Column containing SMILES strings
            use_parallel: Whether to use parallel processing
            
        Returns:
            DataFrame with calculated descriptors
        """
        if self._data is None:
            raise ProcessingError("No data loaded. Call load_smiles_data() first.")
        
        # Get unique SMILES to avoid duplicate calculations
        smiles_data = self._data[smiles_column].dropna().unique()
        
        if use_parallel and self.config.get('use_parallel', True):
            results = self._compute_parallel(smiles_data)
        else:
            results = self._compute_sequential(smiles_data)
        
        # Convert results to DataFrame and merge with original data
        descriptors_df = pd.DataFrame(results)
        
        # Remove entries with all None descriptors (failed calculations)
        descriptors_df = descriptors_df.dropna(subset=['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'NRB'], how='all')
        
        # Merge with original data to preserve additional columns
        merge_columns = [smiles_column]
        if 'kinase_group' in self._data.columns:
            merge_columns.append('kinase_group')
        if 'count_kinase_group' in self._data.columns:
            merge_columns.append('count_kinase_group')
        
        self.descriptor_data = pd.merge(
            descriptors_df, 
            self._data[merge_columns].drop_duplicates(), 
            left_on='canonical_smiles',
            right_on=smiles_column, 
            how='left'
        )
        
        self._results['descriptor_data'] = self.descriptor_data
        return self.descriptor_data

    def _compute_parallel(self, smiles_data: list) -> list:
        """Compute descriptors using parallel processing."""
        # Split data into chunks
        chunks = [smiles_data[i:i + self.batch_size] 
                 for i in range(0, len(smiles_data), self.batch_size)]
        
        results = []
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            for chunk_result in tqdm(
                executor.map(self.calculate_descriptors_for_smiles, chunks),
                total=len(chunks),
                desc="Calculating descriptors"
            ):
                results.extend(chunk_result)
        
        return results

    def _compute_sequential(self, smiles_data: list) -> list:
        """Compute descriptors sequentially."""
        return self.calculate_descriptors_for_smiles(smiles_data)

    def save_descriptors(self, output_path: str) -> None:
        """
        Save descriptor data to file.
        
        Args:
            output_path: Path to save the descriptor data
        """
        if self.descriptor_data is None:
            raise ProcessingError("No descriptor data to save. Run compute_descriptors() first.")
        
        self.descriptor_data.to_csv(output_path, sep='\\t', index=False)

    def plot_histograms(self, additional_data_path: str = None, output_path: str = None) -> None:
        """
        Plot histograms of molecular descriptors.
        
        Args:
            additional_data_path: Path to additional data for comparison
            output_path: Path to save the plot
        """
        if self.descriptor_data is None:
            raise ProcessingError("No descriptor data available. Run compute_descriptors() first.")
        
        # Load additional data if provided
        additional_data = None
        if additional_data_path:
            try:
                additional_data = pd.read_csv(additional_data_path, sep='\\t')
            except Exception as e:
                print(f"Warning: Could not load additional data from {additional_data_path}: {e}")
        
        # Set up the plot
        fig, axs = plt.subplots(3, 2, figsize=(13, 13))
        axs = axs.flatten()
        
        # Descriptor names from config
        descriptors = self.config.descriptor_names
        
        for i, descriptor in enumerate(descriptors):
            if descriptor in self.descriptor_data.columns:
                # Plot main data
                axs[i].hist(self.descriptor_data[descriptor].dropna(), 
                           bins=50, alpha=0.7, label='Current Data', density=True)
                
                # Plot additional data if available
                if additional_data is not None and descriptor in additional_data.columns:
                    axs[i].hist(additional_data[descriptor].dropna(), 
                               bins=50, alpha=0.7, label='Additional Data', density=True)
                
                axs[i].set_title(f'{descriptor} Distribution')
                axs[i].set_xlabel(descriptor)
                axs[i].set_ylabel('Density')
                if additional_data is not None:
                    axs[i].legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        
        if self.config.get('save_plots', True):
            plt.show()

    def get_descriptor_statistics(self) -> dict:
        """
        Get statistical summary of descriptors.
        
        Returns:
            Dictionary with descriptor statistics
        """
        if self.descriptor_data is None:
            raise ProcessingError("No descriptor data available. Run compute_descriptors() first.")
        
        stats = {}
        descriptors = self.config.descriptor_names
        
        for descriptor in descriptors:
            if descriptor in self.descriptor_data.columns:
                data = self.descriptor_data[descriptor].dropna()
                stats[descriptor] = {
                    'count': len(data),
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'median': data.median(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75)
                }
        
        self._results['statistics'] = stats
        return stats

    def plot_correlation_matrix(self, output_path: str = None) -> None:
        """
        Plot correlation matrix of descriptors.
        
        Args:
            output_path: Path to save the plot
        """
        if self.descriptor_data is None:
            raise ProcessingError("No descriptor data available. Run compute_descriptors() first.")
        
        # Select only descriptor columns
        descriptor_cols = [col for col in self.config.descriptor_names 
                          if col in self.descriptor_data.columns]
        
        if len(descriptor_cols) < 2:
            raise ProcessingError("Need at least 2 descriptors to plot correlation matrix.")
        
        # Calculate correlation matrix
        corr_matrix = self.descriptor_data[descriptor_cols].corr()
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Molecular Descriptors Correlation Matrix')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        
        if self.config.get('save_plots', True):
            plt.show()

    def analyze(self) -> dict:
        """
        Perform complete descriptor analysis.
        
        Returns:
            Dictionary with analysis results
        """
        if self._data is None:
            raise ProcessingError("No data loaded. Call load_smiles_data() first.")
        
        # Compute descriptors
        self.compute_descriptors()
        
        # Get statistics
        self.get_descriptor_statistics()
        
        return self._results
