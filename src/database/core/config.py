"""
Database Configuration Module.

Manages configuration settings for database analysis and processing operations.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class DatabaseConfig:
    """
    Configuration manager for database operations.
    
    Provides centralized configuration for file paths, processing parameters,
    and analysis settings while maintaining compatibility with existing scripts.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with optional custom settings.
        
        Args:
            config_dict: Optional dictionary of configuration overrides
        """
        # Default configuration
        self._config = {
            # File paths
            'base_dir': '.',
            'data_dir': 'data',
            'output_dir': 'output',
            
            # Processing parameters
            'batch_size': 1000,
            'num_workers': None,  # Auto-detect
            'similarity_threshold': 0.8,
            
            # Analysis parameters
            'activity_thresholds': [1000, 10000],  # nM
            'descriptor_names': ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'NRB'],
            
            # Visualization
            'figure_size': (12, 8),
            'dpi': 300,
            'save_plots': True,
            
            # Performance
            'use_parallel': True,
            'memory_efficient': True,
            'cache_enabled': True
        }
        
        # Apply custom configuration
        if config_dict:
            self._config.update(config_dict)
        
        # Auto-detect number of workers
        if self._config['num_workers'] is None:
            self._config['num_workers'] = max(1, os.cpu_count() // 2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        self._config.update(config_dict)
    
    @property
    def batch_size(self) -> int:
        """Get batch size for processing."""
        return self._config['batch_size']
    
    @property
    def num_workers(self) -> int:
        """Get number of parallel workers."""
        return self._config['num_workers']
    
    @property
    def similarity_threshold(self) -> float:
        """Get similarity threshold for clustering."""
        return self._config['similarity_threshold']
    
    @property
    def activity_thresholds(self) -> list:
        """Get activity thresholds for classification."""
        return self._config['activity_thresholds']
    
    @property
    def descriptor_names(self) -> list:
        """Get list of molecular descriptors to calculate."""
        return self._config['descriptor_names']
    
    def get_data_path(self, filename: str) -> Path:
        """Get full path to data file."""
        return Path(self._config['base_dir']) / self._config['data_dir'] / filename
    
    def get_output_path(self, filename: str) -> Path:
        """Get full path to output file."""
        output_dir = Path(self._config['base_dir']) / self._config['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
