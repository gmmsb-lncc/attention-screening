"""
Data Cleaning Module.

Provides functionality for removing redundancies, standardizing SMILES,
and cleaning molecular data.
"""

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import SaltRemover

import sys
import os
from pathlib import Path

# Add the database directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_analyzer import BaseAnalyzer
from core.config import DatabaseConfig
from core.exceptions import ProcessingError


class DataCleaner(BaseAnalyzer):
    """
    Clean and standardize molecular data.
    
    This class provides functionality to remove salts, canonicalize SMILES,
    and remove duplicate entries. Maintains compatibility with the original
    remove_redundance.py functionality.
    """

    def __init__(self, config: DatabaseConfig = None, input_file_path: str = None, 
                 output_directory: str = None):
        """
        Initialize the data cleaner.
        
        Args:
            config: DatabaseConfig instance
            input_file_path: Path to input data file
            output_directory: Directory to save cleaned data
        """
        super().__init__(config)
        self.input_file_path = input_file_path
        self.output_directory = output_directory or self.config.get('output_dir', 'output')
        self.remover = SaltRemover.SaltRemover()
        self._cleaned_data = None

    def load_input_data(self) -> pd.DataFrame:
        """
        Load input data for cleaning.
        
        Returns:
            Loaded DataFrame
        """
        if not self.input_file_path:
            raise ProcessingError("No input file path provided")
        
        self.load_data(self.input_file_path)
        return self._data

    def remove_salts_and_canonicalize(self, smiles: str) -> str:
        """
        Remove salts and canonicalize SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Cleaned and canonicalized SMILES string
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Remove salts
                salt_free_mol = self.remover.StripMol(mol)
                if salt_free_mol:
                    # Canonicalize
                    salt_free_smiles = Chem.MolToSmiles(salt_free_mol, canonical=True)
                    return salt_free_smiles
            return smiles
        except Exception:
            # Return original SMILES if processing fails
            return smiles

    def process_smiles_data(self, smiles_column: str = 'canonical_smiles') -> pd.DataFrame:
        """
        Process and clean SMILES data.
        
        Args:
            smiles_column: Column containing SMILES strings
            
        Returns:
            DataFrame with cleaned SMILES
        """
        if self._data is None:
            raise ProcessingError("No data loaded. Call load_input_data() first.")
        
        # Apply cleaning with progress bar
        tqdm.pandas(desc="Removing salts and canonicalizing SMILES")
        self._data[smiles_column] = self._data[smiles_column].progress_apply(
            self.remove_salts_and_canonicalize
        )
        
        return self._data

    def remove_duplicate_smiles(self, smiles_column: str = 'canonical_smiles') -> pd.DataFrame:
        """
        Remove duplicate SMILES entries.
        
        Args:
            smiles_column: Column containing SMILES strings
            
        Returns:
            DataFrame with unique SMILES
        """
        if self._data is None:
            raise ProcessingError("No data processed. Call process_smiles_data() first.")
        
        initial_count = len(self._data)
        
        # Remove duplicates based on SMILES
        self._data = self._data.drop_duplicates(subset=[smiles_column])
        
        final_count = len(self._data)
        removed_count = initial_count - final_count
        
        print(f"Removed {removed_count} duplicate entries. {final_count} unique entries remaining.")
        
        self._results['initial_count'] = initial_count
        self._results['final_count'] = final_count
        self._results['removed_duplicates'] = removed_count
        
        return self._data

    def remove_invalid_smiles(self, smiles_column: str = 'canonical_smiles') -> pd.DataFrame:
        """
        Remove entries with invalid SMILES.
        
        Args:
            smiles_column: Column containing SMILES strings
            
        Returns:
            DataFrame with valid SMILES only
        """
        if self._data is None:
            raise ProcessingError("No data available. Load data first.")
        
        initial_count = len(self._data)
        
        # Function to check if SMILES is valid
        def is_valid_smiles(smiles):
            if pd.isna(smiles) or smiles == '':
                return False
            try:
                mol = Chem.MolFromSmiles(smiles)
                return mol is not None
            except:
                return False
        
        # Filter valid SMILES
        valid_mask = self._data[smiles_column].apply(is_valid_smiles)
        self._data = self._data[valid_mask]
        
        final_count = len(self._data)
        removed_count = initial_count - final_count
        
        print(f"Removed {removed_count} invalid SMILES entries. {final_count} valid entries remaining.")
        
        if 'removed_invalid' not in self._results:
            self._results['removed_invalid'] = 0
        self._results['removed_invalid'] += removed_count
        
        return self._data

    def standardize_data(self, smiles_column: str = 'canonical_smiles',
                        remove_duplicates: bool = True,
                        remove_invalid: bool = True) -> pd.DataFrame:
        """
        Complete data standardization pipeline.
        
        Args:
            smiles_column: Column containing SMILES strings
            remove_duplicates: Whether to remove duplicate entries
            remove_invalid: Whether to remove invalid SMILES
            
        Returns:
            Cleaned and standardized DataFrame
        """
        if self._data is None:
            raise ProcessingError("No data loaded. Call load_input_data() first.")
        
        print("Starting data standardization...")
        
        # Process SMILES (remove salts and canonicalize)
        self.process_smiles_data(smiles_column)
        
        # Remove invalid SMILES if requested
        if remove_invalid:
            self.remove_invalid_smiles(smiles_column)
        
        # Remove duplicates if requested
        if remove_duplicates:
            self.remove_duplicate_smiles(smiles_column)
        
        self._cleaned_data = self._data.copy()
        self._results['cleaned_data'] = self._cleaned_data
        
        print("Data standardization completed.")
        return self._cleaned_data

    def get_cleaning_summary(self) -> dict:
        """
        Get summary of cleaning operations.
        
        Returns:
            Dictionary with cleaning statistics
        """
        summary = {
            'initial_count': self._results.get('initial_count', 0),
            'final_count': self._results.get('final_count', 0),
            'removed_duplicates': self._results.get('removed_duplicates', 0),
            'removed_invalid': self._results.get('removed_invalid', 0),
            'data_reduction_percent': 0
        }
        
        if summary['initial_count'] > 0:
            reduction = summary['initial_count'] - summary['final_count']
            summary['data_reduction_percent'] = (reduction / summary['initial_count']) * 100
        
        return summary

    def save_cleaned_data(self, output_filename: str = 'cleaned_data.tsv') -> str:
        """
        Save cleaned data to file.
        
        Args:
            output_filename: Name of output file
            
        Returns:
            Path to saved file
        """
        if self._cleaned_data is None:
            raise ProcessingError("No cleaned data to save. Run standardize_data() first.")
        
        import os
        os.makedirs(self.output_directory, exist_ok=True)
        
        output_path = os.path.join(self.output_directory, output_filename)
        self._cleaned_data.to_csv(output_path, sep='\\t', index=False)
        
        print(f"Cleaned data saved to: {output_path}")
        return output_path

    def analyze(self) -> dict:
        """
        Perform complete data cleaning analysis.
        
        Returns:
            Dictionary with cleaning results
        """
        if self._data is None:
            raise ProcessingError("No data loaded. Call load_input_data() first.")
        
        # Standardize data
        self.standardize_data()
        
        # Get summary
        summary = self.get_cleaning_summary()
        self._results['summary'] = summary
        
        return self._results

    @classmethod
    def clean_file(cls, input_path: str, output_path: str, 
                   config: DatabaseConfig = None) -> dict:
        """
        Convenience method to clean a file in one operation.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            config: Optional configuration
            
        Returns:
            Cleaning summary dictionary
        """
        cleaner = cls(config=config, input_file_path=input_path)
        cleaner.load_input_data()
        cleaner.standardize_data()
        
        # Extract directory and filename from output_path
        import os
        output_dir = os.path.dirname(output_path)
        output_file = os.path.basename(output_path)
        
        cleaner.output_directory = output_dir
        cleaner.save_cleaned_data(output_file)
        
        return cleaner.get_cleaning_summary()
