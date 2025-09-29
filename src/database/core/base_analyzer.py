"""
Base analyzer class for database analysis operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
from .config import DatabaseConfig
from .exceptions import AnalysisError


class BaseAnalyzer(ABC):
    """
    Abstract base class for database analyzers.
    
    Provides common functionality for loading data, configuration management,
    and standardized interface for analysis operations.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config: Optional DatabaseConfig instance
        """
        self.config = config or DatabaseConfig()
        self._data = None
        self._results = {}
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get loaded data."""
        return self._data
    
    @property
    def results(self) -> Dict[str, Any]:
        """Get analysis results."""
        return self._results
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Default parameters for TSV files (common in this project)
            default_kwargs = {'sep': '\t', 'low_memory': False}
            default_kwargs.update(kwargs)
            
            self._data = pd.read_csv(file_path, **default_kwargs)
            return self._data
        except Exception as e:
            raise AnalysisError(f"Failed to load data from {file_path}: {e}")
    
    def validate_data(self, required_columns: list = None) -> bool:
        """
        Validate loaded data.
        
        Args:
            required_columns: List of required column names
            
        Returns:
            True if data is valid
            
        Raises:
            AnalysisError: If data validation fails
        """
        if self._data is None:
            raise AnalysisError("No data loaded. Call load_data() first.")
        
        if self._data.empty:
            raise AnalysisError("Loaded data is empty.")
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in self._data.columns]
            if missing_cols:
                raise AnalysisError(f"Missing required columns: {missing_cols}")
        
        return True
    
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        Perform analysis operation.
        
        Returns:
            Dictionary with analysis results
        """
        pass
    
    def save_results(self, output_path: str, format: str = 'csv') -> None:
        """
        Save analysis results to file.
        
        Args:
            output_path: Output file path
            format: Output format ('csv', 'tsv', 'json')
        """
        if not self._results:
            raise AnalysisError("No results to save. Run analyze() first.")
        
        try:
            if format.lower() in ['csv', 'tsv']:
                sep = '\t' if format.lower() == 'tsv' else ','
                if isinstance(self._results, dict) and 'data' in self._results:
                    self._results['data'].to_csv(output_path, sep=sep, index=False)
                else:
                    # Convert dict results to DataFrame
                    pd.DataFrame([self._results]).to_csv(output_path, sep=sep, index=False)
            elif format.lower() == 'json':
                import json
                with open(output_path, 'w') as f:
                    json.dump(self._results, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            raise AnalysisError(f"Failed to save results to {output_path}: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of analysis results.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'data_shape': self._data.shape if self._data is not None else None,
            'has_results': bool(self._results),
            'config': self.config.to_dict()
        }
        return summary
